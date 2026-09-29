#!/usr/bin/env python3
"""Read-only candidate ceiling and error-bucket audit for Grid V1 OOF maps."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import load_candidates
from opinion_mining.data import Quadruple, load_train_data
from opinion_mining.metrics import strict_f1

BUCKETS = ("explicit_span", "implicit-A", "implicit-O", "pair", "category", "polarity")


def _quad_set(items: Iterable[object]) -> set[Quadruple]:
    return {item.quadruple if hasattr(item, "quadruple") else item for item in items}  # type: ignore[misc]


def _shares_anchor(gold: Quadruple, candidate: Quadruple) -> bool:
    return gold.aspect == candidate.aspect or gold.opinion == candidate.opinion


def _bucket(gold: Quadruple, candidates: set[Quadruple]) -> str:
    """Assign one actionable reason to a missed gold quadruple."""
    if gold in candidates:
        return "covered"
    same_pair = {q for q in candidates if (q.aspect, q.opinion) == (gold.aspect, gold.opinion)}
    if gold.aspect == "_" and not same_pair:
        return "implicit-A"
    if gold.opinion == "_" and not same_pair:
        return "implicit-O"
    if gold.aspect != "_" and gold.opinion != "_" and not same_pair:
        return "explicit_span" if any(_shares_anchor(gold, c) for c in candidates) else "pair"
    if same_pair and not any(q.category == gold.category for q in same_pair):
        return "category"
    if same_pair:
        return "polarity"
    return "pair"


def audit_candidate_map(
    candidates: Mapping[int, Iterable[object]],
    gold: Mapping[int, Iterable[Quadruple]],
    *,
    name: str = "model",
) -> dict[str, object]:
    """Return a JSON-serialisable audit using the repository's strict tuple semantics."""
    gold_sets = {int(rid): set(values) for rid, values in gold.items()}
    candidate_sets = {int(rid): _quad_set(values) for rid, values in candidates.items()}
    covered = sum(len(values & candidate_sets.get(rid, set())) for rid, values in gold_sets.items())
    gold_n = sum(len(values) for values in gold_sets.values())
    candidate_n = sum(len(values) for values in candidate_sets.values())
    pair_gold_keys: set[tuple[int, str, str]] = set()
    pair_hit_keys: set[tuple[int, str, str]] = set()
    implicit_gold_keys: set[tuple[int, str, str]] = set()
    implicit_pair_hit_keys: set[tuple[int, str, str]] = set()
    pair_quadruple_hit = 0
    buckets = {key: Counter(fn=0, fp=0, recoverable_tp=0) for key in BUCKETS}
    category_confusion: Counter[tuple[str, str]] = Counter()
    polarity_confusion: Counter[tuple[str, str]] = Counter()

    for rid, gold_row in gold_sets.items():
        cand_row = candidate_sets.get(rid, set())
        for q in gold_row:
            same_pair = [c for c in cand_row if (c.aspect, c.opinion) == (q.aspect, q.opinion)]
            pair_key = (rid, q.aspect, q.opinion)
            pair_gold_keys.add(pair_key)
            if same_pair:
                pair_hit_keys.add(pair_key)
                pair_quadruple_hit += 1
            if q.aspect == "_" or q.opinion == "_":
                implicit_gold_keys.add(pair_key)
                if same_pair:
                    implicit_pair_hit_keys.add(pair_key)
            if q in cand_row:
                continue
            kind = _bucket(q, cand_row)
            buckets[kind]["fn"] += 1
            # A missing pair cannot be recovered by ranking/calibration.
            buckets[kind]["recoverable_tp"] += int(bool(same_pair))
            for c in same_pair:
                if c.category != q.category:
                    category_confusion[(q.category, c.category)] += 1
                if c.polarity != q.polarity:
                    polarity_confusion[(q.polarity, c.polarity)] += 1
        for c in cand_row - gold_row:
            same_pair_gold = [q for q in gold_row if (q.aspect, q.opinion) == (c.aspect, c.opinion)]
            if same_pair_gold:
                # Attribute this FP to the most specific label mismatch.
                kind = "polarity" if any(q.category == c.category for q in same_pair_gold) else "category"
            elif c.aspect == "_":
                kind = "implicit-A"
            elif c.opinion == "_":
                kind = "implicit-O"
            elif c.aspect != "_" and c.opinion != "_":
                kind = "explicit_span" if any(_shares_anchor(c, q) for q in gold_row) else "pair"
            else:
                kind = "pair"
            buckets[kind]["fp"] += 1

    score = strict_f1(gold_sets, candidate_sets)
    bucket_payload = {
        key: {
            "fn": int(value["fn"]),
            "fp": int(value["fp"]),
            "theoretical_recoverable_tp": int(value["recoverable_tp"]),
        }
        for key, value in buckets.items()
    }
    return {
        "model": name,
        "score": {"precision": score.precision, "recall": score.recall, "f1": score.f1, "correct": score.correct, "predicted": score.predicted, "gold": score.gold},
        "candidate_count": candidate_n,
        "gold_count": gold_n,
        "candidate_oracle_recall": covered / gold_n if gold_n else 0.0,
        "candidate_oracle_hit": covered,
        "pair_gold_count": len(pair_gold_keys),
        "pair_hit_count": len(pair_hit_keys),
        "pair_recall": len(pair_hit_keys) / len(pair_gold_keys) if pair_gold_keys else 0.0,
        "pair_oracle": pair_quadruple_hit / gold_n if gold_n else 0.0,
        "pair_oracle_weighting": "quadruple_weighted",
        "implicit_gold_count": len(implicit_gold_keys),
        "implicit_pair_hit_count": len(implicit_pair_hit_keys),
        "implicit_recall": len(implicit_pair_hit_keys) / len(implicit_gold_keys) if implicit_gold_keys else 0.0,
        "implicit_oracle": len(implicit_pair_hit_keys) / len(implicit_gold_keys) if implicit_gold_keys else 0.0,
        "implicit_oracle_weighting": "unique_pair",
        "buckets": bucket_payload,
        "category_confusion": {f"{a}->{b}": n for (a, b), n in sorted(category_confusion.items())},
        "polarity_confusion": {f"{a}->{b}": n for (a, b), n in sorted(polarity_confusion.items())},
        "largest_error_buckets": [key for key, _ in sorted(bucket_payload.items(), key=lambda item: (-item[1]["fn"], item[0]))[:3]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_train_data(args.reviews, args.labels)
    gold = {row.id: row.labels for row in rows}
    reports = {}
    for spec in args.candidates:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            parser.error("--candidates must be NAME=PATH")
        reports[name] = audit_candidate_map(load_candidates(path), gold, name=name)
    payload = {"models": reports}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
