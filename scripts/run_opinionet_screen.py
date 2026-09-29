#!/usr/bin/env python3
"""Run the OpinioNet-style one-stage head on fixed folds for backbone screening."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

from opinion_mining.analysis import error_analysis, merge_candidate_maps, save_candidates
from opinion_mining.anchor_config import AnchorV1Config
from opinion_mining.data import load_test_reviews, load_train_data
from opinion_mining.folds import load_assignment_records, splits_from_assignments
from opinion_mining.metrics import strict_f1
from opinion_mining.opinionet_trainer import OpinionetTrainer
from opinion_mining.pipeline import select_threshold, threshold_predictions
from opinion_mining.resource_guard import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/experiments/opinionet_rbt3_3fold"
ANCHOR_REPORT = ROOT / "artifacts/experiments/anchor_rbt3_v1_3fold/report.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(
    output_dir: Path = RUN,
    *,
    model_name: str = "hfl/rbt3",
    trainable_layers: int = 2,
    max_length: int = 96,
    batch_size: int = 4,
    gradient_accumulation: int = 2,
    epochs: int = 4,
    name: str | None = None,
) -> dict[str, object]:
    train = load_train_data(ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv", ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    test = load_test_reviews(ROOT / "artifacts/data/test/TEST/Test_reviews.csv")
    assignments_path = ROOT / "artifacts/reports/fold_assignments_seed42.json"
    assignments = load_assignment_records(assignments_path, n_splits=3, seed=42)
    splits = splits_from_assignments(train, assignments, n_splits=3, seed=42)
    base = AnchorV1Config()
    config = replace(
        base,
        model_name=model_name,
        trainable_layers=trainable_layers,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
    ).to_neural_config()
    run_name = name or f"opinionet_{model_name.split('/')[-1]}_3fold"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", {"name": run_name, "config": asdict(config), "head": "OpinionetHead", "fold_assignments": str(assignments_path)})
    _write_json(output_dir / "fold_assignments.json", assignments)
    oof: dict[int, list] = {}
    test_maps: list[dict[int, list]] = []
    fold_reports = []
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_config = replace(config, seed=config.seed + fold * 1009)
        train_rows = [train[int(index)] for index in train_indices]
        valid_rows = [train[int(index)] for index in valid_indices]
        started = time.time()
        trainer = OpinionetTrainer(fold_config, guard=ResourceGuard())
        result = trainer.train_fold(train_rows, valid_rows, test, output_dir=output_dir / f"fold_{fold}")
        oof.update(result["candidates"])
        test_maps.append(result["test_candidates"])
        save_candidates(output_dir / f"fold_{fold}" / "test_candidates.json", result["test_candidates"])
        valid_predictions = threshold_predictions(result["candidates"], 0.12, 0.10)
        fold_score = strict_f1({row.id: row.labels for row in valid_rows}, valid_predictions)
        fold_reports.append({"fold": fold, "train": len(train_rows), "valid": len(valid_rows), "seconds": time.time() - started, "score_at_monitor_threshold": asdict(fold_score), "history": result["history"]})
        del result, trainer
        gc.collect()
    save_candidates(output_dir / "oof_candidates.json", oof)
    gold = {row.id: set(row.labels) for row in train}
    selected = select_threshold(oof, gold)
    test_candidates = merge_candidate_maps(test_maps, average_present=True, bonus=0.015)
    save_candidates(output_dir / "test_candidates.json", test_candidates)
    test_predictions = threshold_predictions(test_candidates, selected["threshold"], selected["implicit_threshold"], state_thresholds=selected.get("state_thresholds"))
    anchor = json.loads(ANCHOR_REPORT.read_text(encoding="utf-8"))
    report = {
        "name": run_name,
        "model": model_name,
        "stage": "opinionet_screen",
        "config": asdict(config),
        "threshold": selected["threshold"],
        "implicit_threshold": selected["implicit_threshold"],
        "state_thresholds": selected.get("state_thresholds"),
        "score": asdict(selected["score"]),
        "folds": fold_reports,
        "error_analysis": error_analysis(gold, selected["predictions"]),
        "test_ids": len(test_predictions),
        "test_predicted_quadruples": sum(len(values) for values in test_predictions.values()),
        "anchor_v1_f1": anchor["score"]["f1"],
        "delta_vs_anchor_v1": float(selected["score"].f1) - float(anchor["score"]["f1"]),
        "control_gate": {"allowed_drop": 0.002, "passed": float(selected["score"].f1) >= float(anchor["score"]["f1"]) - 0.002},
    }
    _write_json(output_dir / "threshold.json", {"threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "state_thresholds": selected.get("state_thresholds"), "score": asdict(selected["score"]), "folds": fold_reports, "config": asdict(config)})
    _write_json(output_dir / "report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="hfl/rbt3")
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    default_dir = ROOT / f"artifacts/experiments/opinionet_{args.model_name.split('/')[-1]}_3fold"
    result = run(
        args.output_dir or default_dir,
        model_name=args.model_name,
        trainable_layers=args.trainable_layers,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        name=args.name,
    )
    print(json.dumps({"score": result["score"], "control_gate": result["control_gate"]}, ensure_ascii=False))
