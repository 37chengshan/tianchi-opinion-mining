#!/usr/bin/env python3
"""Build chat-format JSONL for LoRA fine-tuning from the canonical fold splits.

Key differences from the public reference recipe for this competition:
- implicit-O quadruples (``opinion="_"``) are kept instead of dropped, because
  the competition gold contains them and our B2 ablation showed they are
  recoverable.
- the supervised target is a strict JSON object with a fixed key order, which
  keeps parsing deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from opinion_mining.data import load_train_data
from opinion_mining.llm_prompt import chat_record
from opinion_mining.folds import load_assignment_records
from opinion_mining.submission import CATEGORIES, POLARITIES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWS = ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv"
DEFAULT_LABELS = ROOT / "artifacts/data/train/TRAIN/Train_labels.csv"
DEFAULT_ASSIGNMENTS = ROOT / "artifacts/reports/fold_assignments_seed42.json"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build(
    output_dir: Path,
    *,
    reviews_path: Path = DEFAULT_REVIEWS,
    labels_path: Path = DEFAULT_LABELS,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    n_splits: int = 3,
    valid_fold: int = 1,
    seed: int = 42,
    full: bool = False,
) -> dict[str, object]:
    rows = load_train_data(reviews_path, labels_path)
    by_id = {row.id: row for row in rows}
    if full:
        train_ids = [row.id for row in rows]
        valid_ids = [row.id for row in rows[:64]]
        meta_extra = {"mode": "full_train"}
    else:
        assignments = load_assignment_records(assignments_path, n_splits=n_splits, seed=seed)
        fold = next((item for item in assignments if int(item["fold"]) == valid_fold), None)
        if fold is None:
            raise ValueError(f"fold {valid_fold} not present in {assignments_path}")
        valid_ids = [int(value) for value in fold["valid_ids"]]
        train_ids = [int(row_id) for item in assignments if int(item["fold"]) != valid_fold for row_id in item["valid_ids"]]
        meta_extra = {"mode": "fold_holdout", "n_splits": n_splits, "valid_fold": valid_fold, "seed": seed}
    missing = sorted(set(train_ids + valid_ids) - set(by_id))
    if missing:
        raise ValueError(f"assignment references unknown review ids: {missing[:5]}")

    train_records = [chat_record(by_id[row_id], supervised=True) for row_id in train_ids]
    valid_records = [chat_record(by_id[row_id], supervised=True) for row_id in valid_ids]
    test_records = [chat_record(by_id[row_id], supervised=False) for row_id in valid_ids]
    counts = {
        "train.jsonl": _write_jsonl(output_dir / "train.jsonl", train_records),
        "valid.jsonl": _write_jsonl(output_dir / "valid.jsonl", valid_records),
        "test.jsonl": _write_jsonl(output_dir / "test.jsonl", test_records),
    }
    implicit_o = sum(1 for row_id in train_ids for quad in by_id[row_id].labels if quad.opinion == "_")
    implicit_a = sum(1 for row_id in train_ids for quad in by_id[row_id].labels if quad.aspect == "_")
    meta = {
        "name": output_dir.name,
        "train_rows": len(train_ids),
        "valid_rows": len(valid_ids),
        "train_implicit_aspect": implicit_a,
        "train_implicit_opinion": implicit_o,
        "assignments": str(assignments_path),
        "categories": sorted(CATEGORIES),
        "polarities": sorted(POLARITIES),
        **meta_extra,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": meta, "files": counts}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/llm_data/fold1_3fold")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--valid-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="train on every labelled row (final submission model)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = build(args.output_dir, n_splits=args.n_splits, valid_fold=args.valid_fold, seed=args.seed, full=args.full)
    print(json.dumps(result["files"], ensure_ascii=False), flush=True)
    print(json.dumps(result["meta"], ensure_ascii=False), flush=True)
