#!/usr/bin/env python3
"""Generate LLM predictions for one fold and score them with the strict metric.

The generation path is MLX (Apple silicon) so the same code works for a local
4-bit Qwen model with or without a LoRA adapter.  Scoring reuses the project's
strict quadruple evaluator, which makes the number directly comparable with the
BERT-route experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from opinion_mining.analysis import save_candidates
from opinion_mining.baseline import Candidate
from opinion_mining.data import load_train_data
from opinion_mining.folds import load_assignment_records
from opinion_mining.llm_parse import LLMParseError, parse_llm_output
from opinion_mining.metrics import strict_f1
from opinion_mining.llm_prompt import SYSTEM_PROMPT, user_prompt


ROOT = Path(__file__).resolve().parents[1]


def _load_mlx(model_path: str, adapter_path: str | None):
    from mlx_lm import load

    return load(model_path, adapter_path=adapter_path)


def _build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply the chat template, disabling hybrid thinking when supported."""
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def generate_answers(
    model_path: str,
    adapter_path: str | None,
    rows,
    *,
    max_tokens: int,
    device_temperature: float = 0.0,
    limit: int | None = None,
    on_progress=None,
) -> dict[int, str]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_mlx(model_path, adapter_path)
    sampler = make_sampler(temp=device_temperature)
    answers: dict[int, str] = {}
    subset = rows[:limit] if limit else rows
    for index, row in enumerate(subset, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
        ]
        prompt = _build_prompt(tokenizer, messages)
        started = time.time()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
        answers[row.id] = text
        if on_progress is not None:
            on_progress(index, len(subset), row.id, text, time.time() - started)
    return answers


def run_test_split(args) -> dict[str, object]:
    import hashlib

    from opinion_mining.data import load_test_reviews
    from opinion_mining.submission import validate_submission, write_submission

    test_rows = load_test_reviews(ROOT / "artifacts/data/test/TEST/Test_reviews.csv")
    subset = test_rows[: args.limit] if args.limit else test_rows
    by_id = {row.id: row for row in subset}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[int, list[Candidate]] = {}
    parse_failures = 0
    answers_path = args.output_dir / "raw_answers.jsonl"
    with answers_path.open("w", encoding="utf-8") as handle:
        def on_progress(index: int, total: int, row_id: int, text: str, seconds: float) -> None:
            nonlocal parse_failures
            try:
                quads = parse_llm_output(text, by_id[row_id].text)
            except LLMParseError:
                quads = []
                parse_failures += 1
            candidates[row_id] = [Candidate(quad, 1.0, ("llm",)) for quad in quads]
            handle.write(json.dumps({"id": row_id, "answer": text, "quadruples": [asdict(quad) for quad in quads]}, ensure_ascii=False) + "\n")
            handle.flush()

        generate_answers(args.model, args.adapter_path, subset, max_tokens=args.max_tokens, on_progress=on_progress)

    save_candidates(args.output_dir / "candidates.json", candidates)
    predictions = {row_id: {item.quadruple for item in items} for row_id, items in candidates.items()}
    csv_path = args.output_dir / "Result.csv"
    write_submission(csv_path, test_rows, predictions)
    report = validate_submission(csv_path, test_rows)
    summary = {
        "split": "test",
        "model": args.model,
        "adapter_path": args.adapter_path,
        "answered_ids": len(predictions),
        "limit": args.limit,
        "parse_failures": parse_failures,
        "predicted_quadruples": sum(len(values) for values in predictions.values()),
        "rows": report.rows,
        "empty_ids": report.empty_ids,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "csv": str(csv_path),
    }
    (args.output_dir / "score.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--split", choices=("fold", "test"), default="fold")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--valid-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.split == "test":
        predictions = run_test_split(args)
        print(json.dumps(predictions, ensure_ascii=False), flush=True)
        return 0

    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    by_id = {row.id: row for row in rows}
    assignments = load_assignment_records(ROOT / "artifacts/reports/fold_assignments_seed42.json", n_splits=args.n_splits, seed=args.seed)
    fold = next(item for item in assignments if int(item["fold"]) == args.valid_fold)
    valid_rows = [by_id[int(value)] for value in fold["valid_ids"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_dir / "raw_answers.jsonl"
    candidates: dict[int, list[Candidate]] = {}
    parse_failures = 0
    with answers_path.open("w", encoding="utf-8") as handle:
        def on_progress(index: int, total: int, row_id: int, text: str, seconds: float) -> None:
            nonlocal parse_failures
            try:
                quads = parse_llm_output(text, by_id[row_id].text)
            except LLMParseError:
                quads = []
                parse_failures += 1
            candidates[row_id] = [Candidate(quad, 1.0, ("llm",)) for quad in quads]
            handle.write(json.dumps({"id": row_id, "answer": text, "quadruples": [asdict(quad) for quad in quads]}, ensure_ascii=False) + "\n")
            handle.flush()
            if not args.quiet:
                print(json.dumps({"i": index, "total": total, "id": row_id, "pred": len(quads), "sec": round(seconds, 2)}, ensure_ascii=False), flush=True)

        generate_answers(args.model, args.adapter_path, valid_rows, max_tokens=args.max_tokens, limit=args.limit, on_progress=on_progress)

    save_candidates(args.output_dir / "candidates.json", candidates)
    scored_ids = sorted(candidates)
    gold = {row_id: set(by_id[row_id].labels) for row_id in scored_ids}
    predictions = {row_id: {item.quadruple for item in candidates[row_id]} for row_id in scored_ids}
    state_counts = {"explicit": 0, "implicit-A": 0, "implicit-O": 0}
    for values in predictions.values():
        for quad in values:
            if quad.aspect == "_":
                state_counts["implicit-A"] += 1
            elif quad.opinion == "_":
                state_counts["implicit-O"] += 1
            else:
                state_counts["explicit"] += 1
    score = strict_f1(gold, predictions)
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "valid_fold": args.valid_fold,
        "n_splits": args.n_splits,
        "scored_ids": len(scored_ids),
        "parse_failures": parse_failures,
        "predicted_quadruples": sum(len(values) for values in predictions.values()),
        "gold_quadruples": sum(len(values) for values in gold.values()),
        "state_counts": state_counts,
        "score": asdict(score),
    }
    (args.output_dir / "score.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
