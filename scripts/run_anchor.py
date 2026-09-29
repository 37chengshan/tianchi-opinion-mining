#!/usr/bin/env python3
"""Train and evaluate the evidence-backed Anchor v1 on fixed three-fold splits."""

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
from opinion_mining.neural import NeuralTrainer
from opinion_mining.pipeline import select_threshold, threshold_predictions
from opinion_mining.resource_guard import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_REVIEWS = ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv"
DEFAULT_TRAIN_LABELS = ROOT / "artifacts/data/train/TRAIN/Train_labels.csv"
DEFAULT_TEST_REVIEWS = ROOT / "artifacts/data/test/TEST/Test_reviews.csv"
DEFAULT_ASSIGNMENTS = ROOT / "artifacts/reports/fold_assignments_seed42.json"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/anchor_rbt3_v1_3fold"
ISOLATED_REPORTS = {
    "anchor_b1_offsets_3fold": ROOT / "artifacts/experiments/anchor_b1_offsets_3fold/report.json",
    "anchor_b2_implicit_o_3fold": ROOT / "artifacts/experiments/anchor_b2_implicit_o_3fold/report.json",
    "anchor_b3_multirelation_3fold": ROOT / "artifacts/experiments/anchor_b3_multirelation_3fold/report.json",
}


def _make_live_writer(output_dir: Path, *, name: str, folds_total: int, epochs_total: int):
    live_path = output_dir / "live_progress.json"
    state = {"updated": None, "history": []}

    def write(payload: dict[str, object]) -> None:
        state["updated"] = time.time()
        state["history"].append({"at": time.time(), **payload})
        state["history"] = state["history"][-40:]
        fold = int(payload.get("fold") or 1)
        epoch = int(payload.get("epoch") or 0)
        steps = int(payload.get("steps") or 0)
        step = int(payload.get("step") or 0)
        epoch_fraction = (step / steps) if steps else 0.0
        overall = ((fold - 1) + (max(0, epoch - 1) + epoch_fraction) / max(1, epochs_total)) / max(1, folds_total)
        record = {
            "name": name,
            "fold": fold,
            "folds_total": folds_total,
            "epoch": epoch,
            "epochs_total": epochs_total,
            "step": step,
            "steps_in_epoch": steps,
            "loss": payload.get("loss"),
            "f1": payload.get("f1"),
            "overall_percent": round(100.0 * overall, 2),
            "updated_at": time.strftime("%H:%M:%S"),
            "updated_ts": state["updated"],
        }
        _write_json(live_path, {**record, "history": state["history"]})

    return write


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-reviews", type=Path, default=DEFAULT_TRAIN_REVIEWS)
    parser.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--test-reviews", type=Path, default=DEFAULT_TEST_REVIEWS)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--trainable-layers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--fgm-epsilon", type=float, default=None)
    return parser.parse_args()


def run(
    config: AnchorV1Config,
    *,
    train_reviews: Path,
    train_labels: Path,
    test_reviews: Path,
    assignments_path: Path,
    output_dir: Path,
    name: str | None = None,
    n_splits: int = 3,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_train_data(train_reviews, train_labels)
    test_rows = load_test_reviews(test_reviews)
    assignments = load_assignment_records(assignments_path, n_splits=n_splits, seed=config.seed)
    splits = splits_from_assignments(train_rows, assignments, n_splits=n_splits, seed=config.seed)
    neural_config = config.to_neural_config()
    run_name = name or f"anchor_{config.model_name.split('/')[-1]}_3fold"
    _write_json(
        output_dir / "config.json",
        {
            "name": run_name,
            "config": config.as_dict(),
            "config_hash": config.config_hash(),
            "fold_assignments": str(assignments_path),
            "composition": "B1+B2+B3 only; no architecture/loss/threshold change",
        },
    )
    _write_json(output_dir / "fold_assignments.json", assignments)

    live_write = _make_live_writer(output_dir, name=run_name, folds_total=len(splits), epochs_total=neural_config.epochs)
    oof: dict[int, list] = {}
    test_maps: list[dict[int, list]] = []
    fold_reports: list[dict[str, object]] = []
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_config = replace(neural_config, seed=neural_config.seed + fold * 1009)
        train_fold = [train_rows[int(index)] for index in train_indices]
        valid_fold = [train_rows[int(index)] for index in valid_indices]
        guard = ResourceGuard()
        started = time.time()
        print(json.dumps({"fold": fold, "train": len(train_fold), "valid": len(valid_fold)}, ensure_ascii=False), flush=True)
        trainer = NeuralTrainer(fold_config, guard=guard)
        result = trainer.train_fold(
            train_fold,
            valid_fold,
            test_rows,
            output_dir=output_dir / f"fold_{fold}",
            on_update=lambda payload, fold=fold: live_write({"fold": fold, **payload}),
        )
        oof.update(result.candidates)
        test_maps.append(result.test_candidates)
        save_candidates(output_dir / f"fold_{fold}" / "test_candidates.json", result.test_candidates)
        valid_predictions = threshold_predictions(result.candidates, 0.12, 0.10)
        fold_score = strict_f1(
            {row.id: row.labels for row in valid_fold},
            valid_predictions,
        )
        fold_reports.append(
            {
                "fold": fold,
                "train": len(train_fold),
                "valid": len(valid_fold),
                "seconds": time.time() - started,
                "monitor_history": result.history,
                "candidate_score_not_thresholded": asdict(fold_score),
            }
        )
        del result, trainer, guard
        gc.collect()

    save_candidates(output_dir / "oof_candidates.json", oof)
    gold = {row.id: set(row.labels) for row in train_rows}
    selected = select_threshold(oof, gold)
    predictions = selected["predictions"]
    score = selected["score"]
    test_candidates = merge_candidate_maps(test_maps, average_present=True, bonus=0.015)
    save_candidates(output_dir / "test_candidates.json", test_candidates)
    test_predictions = threshold_predictions(
        test_candidates,
        float(selected["threshold"]),
        float(selected["implicit_threshold"]),
        state_thresholds=selected.get("state_thresholds"),
    )
    isolated_scores = {}
    for name, path in ISOLATED_REPORTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        isolated_scores[name] = {
            "f1": payload["score"]["f1"],
            "precision": payload["score"]["precision"],
            "recall": payload["score"]["recall"],
            "error_analysis": payload.get("error_analysis", {}),
        }
    best_isolated_name = max(isolated_scores, key=lambda name: float(isolated_scores[name]["f1"]))
    best_isolated_f1 = float(isolated_scores[best_isolated_name]["f1"])
    report = {
        "name": run_name,
        "model": "mps_rbt3_span_relation",
        "stage": "composition",
        "config": config.as_dict(),
        "config_hash": config.config_hash(),
        "threshold": selected["threshold"],
        "implicit_threshold": selected["implicit_threshold"],
        "state_thresholds": selected.get("state_thresholds"),
        "score": asdict(score),
        "folds": fold_reports,
        "error_analysis": error_analysis(gold, predictions),
        "test_ids": len(test_predictions),
        "test_predicted_quadruples": sum(len(values) for values in test_predictions.values()),
        "isolated_scores": isolated_scores,
        "best_isolated": {"name": best_isolated_name, "f1": best_isolated_f1},
        "delta_vs_best_isolated": float(score.f1) - best_isolated_f1,
        "composition_gate": {
            "max_allowed_drop": 0.002,
            "passed": float(score.f1) >= best_isolated_f1 - 0.002,
            "promotion_candidate": "anchor_rbt3_v1" if float(score.f1) >= best_isolated_f1 - 0.002 else best_isolated_name,
        },
    }
    _write_json(output_dir / "threshold.json", {"threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "state_thresholds": selected.get("state_thresholds"), "score": asdict(score), "folds": fold_reports, "config": config.as_dict()})
    _write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    args = _parse_args()
    base = AnchorV1Config()
    overrides = {}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.trainable_layers is not None:
        overrides["trainable_layers"] = args.trainable_layers
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.head_learning_rate is not None:
        overrides["head_learning_rate"] = args.head_learning_rate
    if args.use_ema:
        overrides["use_ema"] = True
    if args.ema_decay is not None:
        overrides["ema_decay"] = args.ema_decay
    if args.fgm_epsilon is not None:
        overrides["fgm_epsilon"] = args.fgm_epsilon
    config = replace(base, **overrides) if overrides else base
    output_dir = args.output_dir
    if output_dir == DEFAULT_OUTPUT and overrides:
        output_dir = ROOT / f"artifacts/experiments/anchor_{config.model_name.split('/')[-1]}_3fold"
    report = run(
        config,
        train_reviews=args.train_reviews,
        train_labels=args.train_labels,
        test_reviews=args.test_reviews,
        assignments_path=args.fold_assignments,
        output_dir=output_dir,
        name=args.name,
        n_splits=args.n_splits,
    )
    print(json.dumps({"name": report["name"], "f1": report["score"]["f1"], "best_isolated": report["best_isolated"], "composition_gate": report["composition_gate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
