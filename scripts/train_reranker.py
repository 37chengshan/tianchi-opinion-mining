#!/usr/bin/env python3
"""Fold-safe candidate re-ranker over one or more BERT candidate maps.

Protocol (no fold ever contributes to its own model or threshold):
1. For target fold k, the remaining folds are split again: each one is held out
   once to produce out-of-fold probabilities, which calibrate the decision
   threshold.
2. A final model is fit on all remaining folds and applied to fold k with the
   calibrated threshold.

The strict F1 over all folds is directly comparable with the repository's other
OOF numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from opinion_mining.analysis import load_candidates
from opinion_mining.data import ReviewExample, load_train_data
from opinion_mining.folds import load_assignment_records
from opinion_mining.metrics import strict_f1
from opinion_mining.priors import LabelPriors, append_prior_features, prior_feature_names
from opinion_mining.reranker import build_rows, feature_names, labels_for, matrix


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Split:
    train_matrix: np.ndarray
    train_labels: np.ndarray
    test_matrix: np.ndarray
    test_rows: list


def _make_model(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _feature_matrix(rows, priors: LabelPriors | None):
    if priors is None:
        return matrix(rows)
    values = append_prior_features(rows, priors)
    return np.asarray(values, dtype=np.float32) if values else np.zeros((0, 0), dtype=np.float32)


def _threshold_for(probabilities: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    total_gold = int(labels.sum())
    best_f1, best_threshold = 0.0, 0.5
    cumulative = np.cumsum(sorted_labels)
    for index in range(len(sorted_labels)):
        tp = int(cumulative[index])
        predicted = index + 1
        if predicted == 0 or total_gold == 0:
            continue
        precision = tp / predicted
        recall = tp / total_gold
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(probabilities[order[index]])
    return best_threshold


def run(
    sources: Sequence[Path],
    *,
    n_splits: int = 3,
    seed: int = 42,
    max_per_review: int | None = 80,
    use_priors: bool = False,
) -> dict[str, object]:
    rows_data = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    reviews: dict[int, ReviewExample] = {row.id: row for row in rows_data}
    assignments = load_assignment_records(ROOT / "artifacts/reports/fold_assignments_seed42.json", n_splits=n_splits, seed=seed)
    fold_ids = [[int(value) for value in item["valid_ids"]] for item in assignments]

    maps = [load_candidates(path) for path in sources]
    if max_per_review is not None:
        trimmed = []
        for mapping in maps:
            limited = {}
            for review_id, items in mapping.items():
                ordered = sorted(items, key=lambda item: -item.score)
                limited[review_id] = ordered[:max_per_review]
            trimmed.append(limited)
        maps = trimmed

    all_rows = build_rows(maps, reviews)
    gold = {row.id: set(row.labels) for row in rows_data}
    feature_count = len(feature_names(len(sources)))
    by_fold: dict[int, list] = {index: [] for index in range(1, n_splits + 1)}
    fold_of = {}
    for index, ids in enumerate(fold_ids, start=1):
        for review_id in ids:
            fold_of[review_id] = index
    for row in all_rows:
        fold = fold_of.get(row.review_id)
        if fold is not None:
            by_fold[fold].append(row)

    ceiling_hits = sum(1 for row in all_rows if row.quadruple in gold.get(row.review_id, set()))
    total_gold = sum(len(values) for values in gold.values())

    fold_reports = []
    predictions: dict[int, set] = {}
    for target in sorted(by_fold):
        others = [fold for fold in sorted(by_fold) if fold != target]
        calibration_probabilities: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
        for held_out in others:
            fit_rows = [row for fold in others if fold != held_out for row in by_fold[fold]]
            fit_ids = [review_id for fold in others if fold != held_out for review_id in fold_ids[fold - 1]]
            hold_rows = by_fold[held_out]
            if not fit_rows or not hold_rows:
                continue
            model = _make_model(seed)
            fit_priors = LabelPriors.from_labels({row_id: gold[row_id] for row_id in fit_ids}) if use_priors else None
            model.fit(_feature_matrix(fit_rows, fit_priors), labels_for(fit_rows, gold))
            calibration_probabilities.append(model.predict_proba(_feature_matrix(hold_rows, fit_priors))[:, 1])
            calibration_labels.append(labels_for(hold_rows, gold))
        probabilities_all = np.concatenate(calibration_probabilities)
        labels_all = np.concatenate(calibration_labels)
        threshold = _threshold_for(probabilities_all, labels_all)

        fit_rows = [row for fold in others for row in by_fold[fold]]
        fit_ids = [review_id for fold in others for review_id in fold_ids[fold - 1]]
        target_rows = by_fold[target]
        model = _make_model(seed)
        final_priors = LabelPriors.from_labels({row_id: gold[row_id] for row_id in fit_ids}) if use_priors else None
        model.fit(_feature_matrix(fit_rows, final_priors), labels_for(fit_rows, gold))
        probabilities = model.predict_proba(_feature_matrix(target_rows, final_priors))[:, 1]
        for row, probability in zip(target_rows, probabilities):
            if probability >= threshold:
                predictions.setdefault(row.review_id, set()).add(row.quadruple)
        fold_score = strict_f1(
            {review_id: gold[review_id] for review_id in fold_ids[target - 1]},
            {review_id: predictions.get(review_id, set()) for review_id in fold_ids[target - 1]},
        )
        fold_reports.append(
            {
                "fold": target,
                "threshold": threshold,
                "candidates": len(target_rows),
                "f1": fold_score.f1,
                "precision": fold_score.precision,
                "recall": fold_score.recall,
                "tp": fold_score.correct,
                "predicted": fold_score.predicted,
            }
        )

    overall = strict_f1(gold, {review_id: predictions.get(review_id, set()) for review_id in gold})
    return {
        "sources": [str(path) for path in sources],
        "n_splits": n_splits,
        "max_per_review": max_per_review,
        "feature_count": feature_count + (len(prior_feature_names()) if use_priors else 0),
        "use_priors": use_priors,
        "candidate_rows": len(all_rows),
        "candidate_recall_ceiling": ceiling_hits / max(1, total_gold),
        "oracle_f1_if_perfect_rank": 2 * ceiling_hits / max(1, total_gold + ceiling_hits),
        "threshold_policy": "nested out-of-fold calibration on the remaining folds",
        "score": {"f1": overall.f1, "precision": overall.precision, "recall": overall.recall, "tp": overall.correct, "predicted": overall.predicted, "gold": overall.gold},
        "folds": fold_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-review", type=int, default=80)
    parser.add_argument("--use-priors", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run(list(args.source), n_splits=args.n_splits, seed=args.seed, max_per_review=args.max_per_review, use_priors=args.use_priors)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("score", "candidate_recall_ceiling", "oracle_f1_if_perfect_rank", "candidate_rows", "folds")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
