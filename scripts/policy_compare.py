#!/usr/bin/env python3
"""Compare simple combinable decision policies for several candidate maps.

All thresholds come from fold-safe cross-fitting (fitted on the other folds),
so every reported number is directly comparable with the repository's strict
OOF numbers.  Policies evaluated:

- ``<source>``: that model alone.
- ``union``: accept a quadruple when any model accepts it (recall-oriented).
- ``intersection``: accept only when every model accepts it (precision-oriented).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data
from opinion_mining.folds import load_assignment_records
from opinion_mining.metrics import strict_f1
from opinion_mining.pipeline import crossfit_threshold_score, threshold_predictions


ROOT = Path(__file__).resolve().parents[1]


def _fold_predictions(mapping, gold, folds, report):
    """Apply each fold's cross-fitted thresholds to that fold's ids."""
    predictions: dict[int, set] = {}
    for fold_index, fold in enumerate(report["folds"], start=1):
        valid_ids = {int(value) for value in folds[fold_index - 1]}
        subset = {row_id: mapping[row_id] for row_id in valid_ids if row_id in mapping}
        fold_predictions = threshold_predictions(
            subset,
            float(fold["threshold"]),
            float(fold["implicit_threshold"]),
            state_thresholds=fold.get("state_thresholds"),
        )
        predictions.update(fold_predictions)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="name=path/to/oof_candidates.json")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sources = []
    for item in args.source:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--source must be name=path, got {item!r}")
        sources.append((name, Path(path)))

    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    assignments = load_assignment_records(ROOT / "artifacts/reports/fold_assignments_seed42.json", n_splits=args.n_splits, seed=args.seed)
    folds = [[int(value) for value in item["valid_ids"]] for item in assignments]
    all_ids = {row_id for fold in folds for row_id in fold}
    gold = {row.id: set(row.labels) for row in rows if row.id in all_ids}

    per_source = {}
    policies = {}
    source_predictions = {}
    for name, path in sources:
        mapping = load_candidates(path)
        report = crossfit_threshold_score(mapping, gold, folds)
        predictions = _fold_predictions(mapping, gold, folds, report)
        source_predictions[name] = predictions
        score = strict_f1(gold, predictions)
        per_source[name] = {
            "path": str(path),
            "f1": score.f1,
            "precision": score.precision,
            "recall": score.recall,
            "tp": score.correct,
            "predicted": score.predicted,
        }
        policies[name] = {"f1": score.f1, "precision": score.precision, "recall": score.recall, "tp": score.correct, "predicted": score.predicted}

    row_ids = sorted(gold)
    union = {row_id: set().union(*[pred.get(row_id, set()) for pred in source_predictions.values()]) for row_id in row_ids}
    intersection = {
        row_id: set.intersection(*[pred.get(row_id, set()) for pred in source_predictions.values()]) for row_id in row_ids
    }
    for label, predictions in (("union", union), ("intersection", intersection)):
        score = strict_f1(gold, predictions)
        policies[label] = {"f1": score.f1, "precision": score.precision, "recall": score.recall, "tp": score.correct, "predicted": score.predicted}

    tp_sets = {
        name: {row_id for row_id, values in predictions.items() for value in values if value in gold.get(row_id, set())}
        for name, predictions in source_predictions.items()
    }
    tp_overlap = {}
    names = list(tp_sets)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left, right = names[left_index], names[right_index]
            shared = len(tp_sets[left] & tp_sets[right])
            tp_overlap[f"{left}|{right}"] = {
                "shared_tp": shared,
                "left_only_tp": len(tp_sets[left] - tp_sets[right]),
                "right_only_tp": len(tp_sets[right] - tp_sets[left]),
                "jaccard": shared / max(1, len(tp_sets[left] | tp_sets[right])),
            }

    payload = {"n_splits": args.n_splits, "sources": per_source, "policies": policies, "tp_overlap": tp_overlap}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
