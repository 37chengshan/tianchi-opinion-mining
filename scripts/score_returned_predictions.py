#!/usr/bin/env python3
"""Score cloud-returned predictions with the repository's strict metric.

    PYTHONPATH=src python3 scripts/score_returned_predictions.py \
        --predictions predictions_fold1.jsonl --n-splits 3 --valid-fold 1

The file must contain one JSON object per line with `id` and `quadruples`
(list of [aspect, opinion, category, polarity]).  Scoring uses the same
canonical fold assignment as every other experiment, so numbers are directly
comparable with the BERT route (WWM fold-1 reference: 0.7442).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from opinion_mining.analysis import save_candidates
from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple, load_train_data
from opinion_mining.folds import load_assignment_records
from opinion_mining.metrics import strict_f1


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--valid-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-out", type=Path, default=None, help="optional path to save candidates for fusion")
    args = parser.parse_args()

    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    by_id = {row.id: row for row in rows}
    assignments = load_assignment_records(ROOT / "artifacts/reports/fold_assignments_seed42.json", n_splits=args.n_splits, seed=args.seed)
    fold_ids = next(
        ([int(value) for value in item["valid_ids"]] for item in assignments if int(item["fold"]) == args.valid_fold),
        None,
    )
    if fold_ids is None:
        raise SystemExit(f"fold {args.valid_fold} not found for n_splits={args.n_splits}")

    predictions: dict[int, set[Quadruple]] = {}
    known = {int(value) for value in fold_ids}
    unknown_ids: list[int] = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        review_id = payload.get("id")
        if review_id is None:
            continue
        review_id = int(review_id)
        if review_id not in known:
            unknown_ids.append(review_id)
            continue
        quads = {
            Quadruple(str(item[0]), str(item[1]), str(item[2]), str(item[3]))
            for item in payload.get("quadruples") or []
        }
        predictions[review_id] = quads

    gold = {row_id: set(by_id[row_id].labels) for row_id in fold_ids}
    missing = sorted(set(fold_ids) - set(predictions))
    score = strict_f1(gold, predictions)
    report = {
        "predictions": str(args.predictions),
        "n_splits": args.n_splits,
        "valid_fold": args.valid_fold,
        "scored_ids": len(predictions),
        "missing_ids": len(missing),
        "unknown_ids": len(unknown_ids),
        "predicted_quadruples": sum(len(values) for values in predictions.values()),
        "gold_quadruples": sum(len(values) for values in gold.values()),
        "score": asdict(score),
    }
    if args.candidates_out:
        save_candidates(
            args.candidates_out,
            {row_id: [Candidate(quad, 1.0, ("llm",)) for quad in predictions.get(row_id, set())] for row_id in fold_ids},
        )
        report["candidates_out"] = str(args.candidates_out)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
