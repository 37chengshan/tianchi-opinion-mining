#!/usr/bin/env python3
"""Calibrate per-type/category/polarity thresholds on saved strict OOF."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import load_candidates, merge_candidate_maps
from opinion_mining.data import load_test_reviews, load_train_data
from opinion_mining.metrics import strict_f1
from opinion_mining.folds import fixed_splits
from opinion_mining.submission import validate_submission, write_submission


initial_thresholds = {"threshold": 0.727, "implicit_threshold": 0.503}


def load_data():
    train_root = ROOT / "artifacts" / "data" / "train" / "TRAIN"
    test_root = ROOT / "artifacts" / "data" / "test" / "TEST"
    train_zip = Path("/Users/cc/Downloads/初赛训练数据 2019-08-01.zip")
    test_zip = Path("/Users/cc/Downloads/初赛测试数据 2019-08-15.zip")
    if not (train_root / "Train_reviews.csv").exists():
        with zipfile.ZipFile(train_zip) as archive:
            archive.extractall(ROOT / "artifacts" / "data" / "train")
    if not (test_root / "Test_reviews.csv").exists():
        with zipfile.ZipFile(test_zip) as archive:
            archive.extractall(ROOT / "artifacts" / "data" / "test")
    return load_train_data(train_root / "Train_reviews.csv", train_root / "Train_labels.csv"), load_test_reviews(test_root / "Test_reviews.csv")


def group_key(item):
    quad = item.quadruple
    return ("implicit" if quad.aspect == "_" else "explicit", quad.category, quad.polarity)


def calibrate(oof, gold, initial):
    by_group = defaultdict(list)
    for rid, items in oof.items():
        for item in items:
            by_group[group_key(item)].append((int(rid), item))
    curves = {}
    for group, items in by_group.items():
        buckets = defaultdict(list)
        for rid, item in items:
            buckets[round(float(item.score), 3)].append((rid, item))
        curve = {1.0: (0, 0)}
        predicted = correct = 0
        for threshold in sorted(buckets, reverse=True):
            for rid, item in buckets[threshold]:
                predicted += 1
                correct += int(item.quadruple in gold[rid])
            curve[threshold] = (predicted, correct)
        curve[0.0] = (predicted, correct)
        curves[group] = curve

    gold_count = sum(len(values) for values in gold.values())
    thresholds = {group: (initial["implicit_threshold"] if group[0] == "implicit" else initial["threshold"]) for group in curves}

    def counts(group, threshold):
        if threshold in curves[group]:
            return curves[group][threshold]
        # All thresholds are rounded before entering the curve.
        available = [value for value in curves[group] if value >= threshold]
        return curves[group][min(available)] if available else (0, 0)

    state = {group: counts(group, threshold) for group, threshold in thresholds.items()}
    predicted = sum(pair[0] for pair in state.values())
    correct = sum(pair[1] for pair in state.values())

    def score(predicted_count, correct_count):
        precision = correct_count / predicted_count if predicted_count else 0.0
        recall = correct_count / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, precision, recall

    for _ in range(8):
        changed = False
        for group in sorted(curves):
            old_predicted, old_correct = state[group]
            other_predicted = predicted - old_predicted
            other_correct = correct - old_correct
            current_threshold = thresholds[group]
            current_score = (*score(predicted, correct), -current_threshold)
            best_threshold = current_threshold
            best_counts = state[group]
            for threshold, pair in curves[group].items():
                candidate_score = (*score(other_predicted + pair[0], other_correct + pair[1]), -threshold)
                if candidate_score > current_score:
                    current_score = candidate_score
                    best_threshold = threshold
                    best_counts = pair
            if best_threshold != current_threshold:
                thresholds[group] = best_threshold
                state[group] = best_counts
                predicted = other_predicted + best_counts[0]
                correct = other_correct + best_counts[1]
                changed = True
        if not changed:
            break
    return thresholds, score(predicted, correct), correct, predicted, gold_count


def apply_thresholds(candidates, thresholds):
    output = {}
    for rid, items in candidates.items():
        # Test rows can contain a group absent from the OOF candidate map.  A
        # global/type fallback is safer than silently dropping every item.
        output[rid] = {
            item.quadruple
            for item in items
            if item.score >= thresholds.get(
                group_key(item),
                initial_thresholds["implicit_threshold"]
                if group_key(item)[0] == "implicit"
                else initial_thresholds["threshold"],
            )
        }
    return output


def shrink_thresholds(thresholds, oof, gold, initial, *, min_gold: int = 60):
    """Shrink sparse group optima toward the explicit/implicit baseline."""
    gold_support = defaultdict(int)
    for values in gold.values():
        for quad in values:
            gold_support[("implicit" if quad.aspect == "_" else "explicit", quad.category, quad.polarity)] += 1
    output = {}
    for group, value in thresholds.items():
        base = initial["implicit_threshold"] if group[0] == "implicit" else initial["threshold"]
        weight = min(1.0, gold_support.get(group, 0) / float(min_gold))
        output[group] = round(base + weight * (float(value) - base), 3)
    return output


def lofo_score(oof, gold, train_rows, initial):
    """Evaluate calibration with thresholds fit only on other folds."""
    fold_by_id = {}
    splits = fixed_splits(train_rows, n_splits=5, seed=42)
    for fold, (_, valid_indices) in enumerate(splits, start=1):
        for index in valid_indices:
            fold_by_id[train_rows[int(index)].id] = fold
    predictions = {}
    fold_scores = []
    for fold in range(1, 6):
        train_ids = {rid for rid in oof if fold_by_id.get(rid) != fold}
        valid_ids = {rid for rid in oof if fold_by_id.get(rid) == fold}
        fit_oof = {rid: oof[rid] for rid in train_ids}
        fit_gold = {rid: gold[rid] for rid in train_ids}
        thresholds, _, _, _, _ = calibrate(fit_oof, fit_gold, initial)
        thresholds = shrink_thresholds(thresholds, fit_oof, fit_gold, initial)
        valid_candidates = {rid: oof[rid] for rid in valid_ids}
        valid_predictions = apply_thresholds(valid_candidates, thresholds)
        predictions.update(valid_predictions)
        fold_score = strict_f1({rid: gold[rid] for rid in valid_ids}, valid_predictions)
        fold_scores.append({"fold": fold, "f1": fold_score.f1, "precision": fold_score.precision, "recall": fold_score.recall, "correct": fold_score.correct, "predicted": fold_score.predicted, "gold": fold_score.gold})
    score = strict_f1({rid: gold[rid] for rid in oof}, predictions)
    return score, fold_scores


def main():
    train, test = load_data()
    gold = {row.id: set(row.labels) for row in train}
    oof = load_candidates(ROOT / "artifacts/experiments/neural_5fold_confirm/oof_candidates.json")
    threshold_payload = json.loads((ROOT / "artifacts/experiments/neural_5fold_confirm/threshold.json").read_text(encoding="utf-8"))
    initial = {"threshold": float(threshold_payload["threshold"]), "implicit_threshold": float(threshold_payload["implicit_threshold"])}
    global initial_thresholds
    initial_thresholds = initial
    lofo, lofo_folds = lofo_score(oof, gold, train, initial)
    thresholds, oof_score, correct, predicted, gold_count = calibrate(oof, gold, initial)
    thresholds = shrink_thresholds(thresholds, oof, gold, initial)
    full_predictions = apply_thresholds(oof, thresholds)
    full_score = strict_f1(gold, full_predictions)
    test_maps = [load_candidates(path) for path in sorted((ROOT / "artifacts/experiments/neural_5fold_confirm").glob("fold_*/test_candidates.json"))]
    test_candidates = merge_candidate_maps(test_maps, average_present=True, bonus=0.015)
    predictions = apply_thresholds(test_candidates, thresholds)
    output = ROOT / "artifacts/submissions/Result_group_calibrated.csv"
    report = write_submission(output, test, predictions)
    validation = validate_submission(output, test)
    raw = output.read_bytes()
    result = {
        "calibration": "5-fold LOFO group calibration with sparse-group shrinkage",
        "initial_threshold": initial,
        "group_thresholds": {"|".join(group): value for group, value in sorted(thresholds.items())},
        "lofo_oof": {"f1": lofo.f1, "precision": lofo.precision, "recall": lofo.recall, "correct": lofo.correct, "predicted": lofo.predicted, "gold": lofo.gold, "folds": lofo_folds},
        "full_oof_after_shrink": {"f1": full_score.f1, "precision": full_score.precision, "recall": full_score.recall, "correct": full_score.correct, "predicted": full_score.predicted, "gold": full_score.gold},
        "uncalibrated_in_sample_reference": {"f1": oof_score[0], "precision": oof_score[1], "recall": oof_score[2], "correct": correct, "predicted": predicted, "gold": gold_count},
        "submission": {"path": str(output), "rows": validation.rows, "predicted_quadruples": validation.predicted_quadruples, "empty_ids": validation.empty_ids, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "validator": "passed"},
    }
    report_path = ROOT / "artifacts/reports/group_calibration_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
