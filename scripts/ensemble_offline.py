#!/usr/bin/env python3
"""Fold-safe score-level ensembling of several candidate maps.

Threshold selection never sees the fold it scores: for every fold the
thresholds are fitted on the remaining folds' candidates and applied to the
held-out fold.  The reported number is therefore comparable to the strict OOF
numbers used elsewhere in this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from opinion_mining.analysis import load_candidates, merge_candidate_maps
from opinion_mining.data import load_train_data
from opinion_mining.folds import load_assignment_records
from opinion_mining.metrics import strict_f1
from opinion_mining.pipeline import crossfit_threshold_score, select_threshold, threshold_predictions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = ROOT / "artifacts/reports/fold_assignments_seed42.json"


def _gold(fold_ids: dict[int, list[int]], *, reviews_path: Path | None = None, labels_path: Path | None = None):
    rows = load_train_data(
        reviews_path or ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        labels_path or ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    all_ids = {row_id for ids in fold_ids.values() for row_id in ids}
    return {row.id: set(row.labels) for row in rows if row.id in all_ids}


def evaluate_ensemble(
    sources: list[tuple[str, Path]],
    *,
    weights: list[float] | None,
    bonus: float,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    n_splits: int = 3,
    seed: int = 42,
    reviews_path: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, object]:
    assignments = load_assignment_records(assignments_path, n_splits=n_splits, seed=seed)
    folds = [[int(value) for value in item["valid_ids"]] for item in assignments]
    maps = [load_candidates(path) for _, path in sources]
    merged = merge_candidate_maps(maps, weights=weights, average_present=True, bonus=bonus)
    gold = _gold(
        {item["fold"]: [int(v) for v in item["valid_ids"]] for item in assignments},
        reviews_path=reviews_path,
        labels_path=labels_path,
    )

    crossfit = crossfit_threshold_score(merged, gold, folds)
    in_sample = select_threshold(merged, gold)
    per_source = {}
    for (name, path), mapping in zip(sources, maps):
        source_crossfit = crossfit_threshold_score(mapping, gold, folds)
        per_source[name] = {
            "path": str(path),
            "crossfit_f1": source_crossfit["score"].f1,
            "in_sample_f1_not_strict": in_sample and select_threshold(mapping, gold)["score"].f1,
        }

    overlap = {"tp_intersection": 0, "union_tp": 0}
    if len(maps) >= 2:
        gold_sets = gold
        pred_sets = [
            {
                row_id: {
                    item.quadruple
                    for item in mapping.get(row_id, [])
                    if item.score >= float(crossfit["folds"][0]["threshold"])
                }
                for row_id in gold_sets
            }
            for mapping in maps
        ]
        for row_id, truth in gold_sets.items():
            hits = [truth & pred.get(row_id, set()) for pred in pred_sets]
            overlap["union_tp"] += len(set().union(*hits)) if hits else 0
            if hits:
                overlap["tp_intersection"] += len(set.intersection(*hits))

    return {
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "weights": weights,
        "bonus": bonus,
        "n_splits": n_splits,
        "merged_candidates": sum(len(values) for values in merged.values()),
        "crossfit": {
            "f1": crossfit["score"].f1,
            "precision": crossfit["score"].precision,
            "recall": crossfit["score"].recall,
            "folds": [
                {
                    "fold": item["fold"],
                    "threshold": item["threshold"],
                    "implicit_threshold": item["implicit_threshold"],
                    "f1": item["score"].f1,
                }
                for item in crossfit["folds"]
            ],
        },
        "in_sample_reference_not_strict": {
            "f1": in_sample["score"].f1,
            "threshold": in_sample["threshold"],
            "implicit_threshold": in_sample["implicit_threshold"],
        },
        "per_source": per_source,
        "overlap": overlap,
    }


def evaluate_fold_subset(
    sources: list[tuple[str, Path]],
    *,
    weights: list[float] | None,
    bonus: float,
    fold: int,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    n_splits: int = 3,
    seed: int = 42,
    reviews_path: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, object]:
    """Score only one fold's ids with thresholds fitted on the other folds.

    Used when one source (for example an LLM screen) only produced predictions
    for a single fold: the thresholds still come from the other folds, so the
    number stays leakage-free.
    """
    assignments = load_assignment_records(assignments_path, n_splits=n_splits, seed=seed)
    folds = [[int(value) for value in item["valid_ids"]] for item in assignments]
    target_ids = [int(value) for item in assignments if int(item["fold"]) == fold for value in item["valid_ids"]]
    maps = [load_candidates(path) for _, path in sources]
    merged = merge_candidate_maps(maps, weights=weights, average_present=True, bonus=bonus)
    gold_all = _gold(
        {item["fold"]: [int(v) for v in item["valid_ids"]] for item in assignments},
        reviews_path=reviews_path,
        labels_path=labels_path,
    )
    crossfit = crossfit_threshold_score(merged, gold_all, folds)
    fold_report = next(item for item in crossfit["folds"] if item["fold"] == fold)
    subset = {row_id: merged[row_id] for row_id in target_ids if row_id in merged}
    predictions = threshold_predictions(
        subset,
        float(fold_report["threshold"]),
        float(fold_report["implicit_threshold"]),
        state_thresholds=fold_report.get("state_thresholds"),
    )
    score = strict_f1({row_id: gold_all[row_id] for row_id in target_ids}, predictions)
    covered = [row_id for row_id in target_ids if row_id in merged]
    return {
        "fold": fold,
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "weights": weights,
        "bonus": bonus,
        "coverage": {"target_ids": len(target_ids), "with_candidates": len(covered)},
        "thresholds": {
            "explicit": fold_report["threshold"],
            "implicit": fold_report["implicit_threshold"],
            "state_thresholds": fold_report.get("state_thresholds"),
        },
        "score": {"f1": score.f1, "precision": score.precision, "recall": score.recall, "correct": score.correct, "predicted": score.predicted, "gold": score.gold},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="name=path/to/oof_candidates.json")
    parser.add_argument("--weights", default=None, help="comma separated, one per source")
    parser.add_argument("--bonus", type=float, default=0.0)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--fold-only", type=int, default=None, help="score only this fold (e.g. LLM single-fold screen)")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sources = []
    for item in args.source:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--source must be name=path, got {item!r}")
        sources.append((name, Path(path)))
    weights = [float(value) for value in args.weights.split(",")] if args.weights else None
    if args.fold_only is not None:
        report = evaluate_fold_subset(sources, weights=weights, bonus=args.bonus, fold=args.fold_only, n_splits=args.n_splits)
    else:
        report = evaluate_ensemble(sources, weights=weights, bonus=args.bonus, n_splits=args.n_splits)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
