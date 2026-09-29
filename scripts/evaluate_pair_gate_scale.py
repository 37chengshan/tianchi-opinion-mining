#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data
from opinion_mining.ensemble import filter_predictions_by_pair_support
from opinion_mining.metrics import strict_f1
from opinion_mining.pipeline import select_threshold, threshold_predictions


DEFAULT_SCALES = (0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)


def _score_dict(score):
    return {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "correct": score.correct,
        "predicted": score.predicted,
        "gold": score.gold,
    }


def _subset(mapping, ids):
    return {rid: mapping[rid] for rid in ids if rid in mapping}


def _threshold(candidates, selected):
    return threshold_predictions(
        candidates,
        float(selected["threshold"]),
        float(selected["implicit_threshold"]),
        state_thresholds=selected.get("state_thresholds"),
    )


def evaluate(base_candidates, verifier_candidates, gold, folds, scales=DEFAULT_SCALES):
    all_ids = set(gold)
    predictions = {}
    fold_reports = []
    for fold_index, valid_ids_raw in enumerate(folds, start=1):
        valid_ids = set(int(value) for value in valid_ids_raw)
        fit_ids = all_ids - valid_ids
        fit_gold = _subset(gold, fit_ids)
        valid_gold = _subset(gold, valid_ids)
        base_fit = _subset(base_candidates, fit_ids)
        base_valid = _subset(base_candidates, valid_ids)
        verifier_fit = _subset(verifier_candidates, fit_ids)
        verifier_valid = _subset(verifier_candidates, valid_ids)

        base_selected = select_threshold(base_fit, fit_gold, separate_implicit=True)
        verifier_selected = select_threshold(verifier_fit, fit_gold, separate_implicit=True)
        base_fit_predictions = _threshold(base_fit, base_selected)
        base_valid_predictions = _threshold(base_valid, base_selected)
        verifier_gates = verifier_selected["state_thresholds"]

        scale_trials = []
        best = None
        for scale in scales:
            fit_filtered = filter_predictions_by_pair_support(
                base_fit_predictions,
                verifier_fit,
                verifier_gates,
                gate_scale=float(scale),
            )
            score = strict_f1(fit_gold, fit_filtered)
            trial = {"scale": float(scale), "score": _score_dict(score)}
            scale_trials.append(trial)
            key = (score.f1, score.precision, score.recall, float(scale))
            if best is None or key > best[0]:
                best = (key, float(scale))

        selected_scale = best[1]
        valid_filtered = filter_predictions_by_pair_support(
            base_valid_predictions,
            verifier_valid,
            verifier_gates,
            gate_scale=selected_scale,
        )
        predictions.update(valid_filtered)
        valid_score = strict_f1(valid_gold, valid_filtered)
        fold_reports.append(
            {
                "fold": fold_index,
                "selected_scale": selected_scale,
                "base_state_thresholds": base_selected["state_thresholds"],
                "verifier_state_thresholds": verifier_gates,
                "fit_trials": scale_trials,
                "valid_score": _score_dict(valid_score),
            }
        )

    crossfit_score = strict_f1(gold, predictions)

    base_selected = select_threshold(base_candidates, gold, separate_implicit=True)
    verifier_selected = select_threshold(verifier_candidates, gold, separate_implicit=True)
    base_predictions = _threshold(base_candidates, base_selected)
    verifier_gates = verifier_selected["state_thresholds"]
    full_trials = []
    best_full = None
    for scale in scales:
        filtered = filter_predictions_by_pair_support(
            base_predictions,
            verifier_candidates,
            verifier_gates,
            gate_scale=float(scale),
        )
        score = strict_f1(gold, filtered)
        full_trials.append({"scale": float(scale), "score": _score_dict(score)})
        key = (score.f1, score.precision, score.recall, float(scale))
        if best_full is None or key > best_full[0]:
            best_full = (key, float(scale), score)

    return {
        "crossfit_score": _score_dict(crossfit_score),
        "folds": fold_reports,
        "full_oof": {
            "selected_scale": best_full[1],
            "score": _score_dict(best_full[2]),
            "base_state_thresholds": base_selected["state_thresholds"],
            "verifier_state_thresholds": verifier_gates,
            "trials": full_trials,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/experiments/grid_wwm_3fold_20260928/oof_candidates.json")
    parser.add_argument("--verifier", type=Path, default=ROOT / "artifacts/experiments/grid_wwm_v21_grid_redecode_3fold/oof_candidates.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/experiments/grid_wwm_v2code_3fold/fold_assignments.json")
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/pair_gate_scale_nested.json")
    args = parser.parse_args()

    rows = load_train_data(args.reviews, args.labels)
    gold = {row.id: set(row.labels) for row in rows}
    base = load_candidates(args.base)
    verifier = load_candidates(args.verifier)
    fold_payload = json.loads(args.folds.read_text(encoding="utf-8"))
    folds = [item["valid_ids"] for item in fold_payload]
    report = evaluate(base, verifier, gold, folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
