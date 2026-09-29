#!/usr/bin/env python3
from __future__ import annotations

"""Run the compact relation-grid screening pipeline.

The default command is plan-only. ``--dummy`` and ``--dummy-train`` are
bounded CPU smoke paths. A real run requires ``--train``, uses only locally
cached backbones, writes fold-exclusive OOF evidence, and can optionally
write a validated test candidate. This script never uploads to Tianchi and
never writes a leaderboard score.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import time
import webbrowser
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from opinion_mining.analysis import merge_candidate_maps, save_candidates
from opinion_mining.data import load_test_reviews, load_train_data
from opinion_mining.dashboard_runtime import DashboardWriter
from opinion_mining.folds import fixed_splits
from opinion_mining.grid_trainer import (
    MODEL_ALIASES,
    GridTrainConfig,
    ResourcePressureRestart,
    dummy_smoke,
    dummy_train_smoke,
    fixed_screen_plan,
    train_grid_fold,
    write_json,
)
from opinion_mining.metrics import Score, strict_f1
from opinion_mining.pipeline import crossfit_threshold_score, select_threshold, threshold_predictions
from opinion_mining.resource_guard import ResourceGuard
from opinion_mining.submission import validate_submission, write_submission
from submission_manifest import ManifestError, create_candidate_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, smoke-test, or train the compact relation grid.")
    parser.add_argument("--model-name", default="rbt3", help="rbt3, macbert, wwm, or a local model reference")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for this run's artifacts")
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--test-reviews", type=Path, default=ROOT / "artifacts/data/test/TEST/Test_reviews.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="prefix rows for plan/smoke only; 0 keeps all rows")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3, choices=(3, 5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relation-rank", type=int, default=32)
    parser.add_argument("--trainable-layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--head-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-span-length", type=int, default=12)
    parser.add_argument("--span-top-k", type=int, default=16)
    parser.add_argument("--relation-top-k", type=int, default=1)
    parser.add_argument("--nms-iou", type=float, default=0.8)
    parser.add_argument("--implicit-loss-weight", type=float, default=None, help="1.0, 1.5, or 2.0; extra weight on implicit-relation BCE")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--include-test", action="store_true", help="predict all test rows and write a validated candidate")
    parser.add_argument("--promote-root", action="store_true", help="copy the validated candidate to repository Result.csv")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:18765/")
    parser.add_argument("--no-open-dashboard", action="store_true")
    parser.add_argument("--train", action="store_true", help="run real fold-local training; default is plan-only")
    parser.add_argument("--download-missing-backbone", action="store_true", help="prefetch a missing Hugging Face backbone before training; training itself remains local-files-only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dummy", action="store_true", help="one dummy CPU batch")
    mode.add_argument("--dummy-train", action="store_true", help="one synthetic CPU fold with checkpoint")
    return parser


def _score_dict(score: Score | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(score, Score):
        return {"precision": float(score.precision), "recall": float(score.recall), "f1": float(score.f1), "correct": int(score.correct), "predicted": int(score.predicted), "gold": int(score.gold)}
    return dict(score)


def _snapshot_dict(snapshot: Any) -> dict[str, Any]:
    return dict(snapshot.__dict__) if hasattr(snapshot, "__dict__") else dict(snapshot)


def _open_dashboard(url: str) -> None:
    try:
        webbrowser.open(url, new=0, autoraise=True)
    except Exception:
        pass


def _default_layers(model_name: str) -> int:
    resolved = MODEL_ALIASES.get(model_name.strip().lower(), model_name).lower()
    return 3 if resolved.endswith("rbt3") else 12


def _default_batch(model_name: str) -> int:
    resolved = MODEL_ALIASES.get(model_name.strip().lower(), model_name).lower()
    return 4 if resolved.endswith("rbt3") else 2


def _prefetch_backbone(model_name: str) -> str | None:
    """Ensure a remote Hugging Face backbone is present before local-only training."""
    resolved = MODEL_ALIASES.get(model_name.strip().lower(), model_name)
    if Path(resolved).expanduser().exists():
        return str(Path(resolved).expanduser())
    from huggingface_hub import snapshot_download

    # The repository also exposes Flax and TensorFlow weights.  They are not
    # used by the PyTorch/MPS trainer and can triple both download time and
    # cache footprint on the 16GB machine.
    return snapshot_download(
        repo_id=resolved,
        allow_patterns=[
            "config.json", "pytorch_model.bin", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.txt", "merges.txt", "sentencepiece.bpe.model",
        ],
    )


def _load_historical_champion(root: Path) -> float | None:
    """Recover the best known local F1 across previous dashboard/autopilot runs."""
    values: list[float] = []
    for path in (
        root / "dashboard" / "state.json",
        root / "artifacts" / "reports" / "autopilot_progress.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            metric = payload.get("metrics", {}).get("champion_f1")
            if metric is not None:
                try:
                    values.append(float(metric))
                except (TypeError, ValueError):
                    pass
            for row in payload.get("trials", []) or []:
                try:
                    values.append(float(row.get("f1")))
                except (TypeError, ValueError, AttributeError):
                    pass
    return max(values) if values else None


def _progress_payload(
    *,
    fold: int,
    total_folds: int,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    elapsed_seconds: float,
) -> dict[str, float | int]:
    """Compute step/epoch/fold/overall progress and a simple ETA."""
    total_folds = max(1, int(total_folds))
    total_epochs = max(1, int(total_epochs))
    total_steps = max(1, int(total_steps))
    fold = min(total_folds, max(1, int(fold)))
    epoch = min(total_epochs, max(1, int(epoch)))
    step = min(total_steps, max(0, int(step)))
    step_fraction = step / total_steps
    fold_fraction = ((epoch - 1) + step_fraction) / total_epochs
    overall_fraction = ((fold - 1) + fold_fraction) / total_folds
    eta_seconds = 0.0
    if overall_fraction > 0 and elapsed_seconds > 0:
        eta_seconds = max(0.0, elapsed_seconds * (1.0 - overall_fraction) / overall_fraction)
    return {
        "fold": fold,
        "total_folds": total_folds,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "step_percent": step_fraction * 100.0,
        "epoch_percent": step_fraction * 100.0,
        "fold_percent": fold_fraction * 100.0,
        "overall_percent": overall_fraction * 100.0,
        "eta_seconds": eta_seconds,
    }


def _format_training_step_event(
    *,
    fold: int,
    total_folds: int,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float | None,
    resource: dict[str, Any] | None,
) -> str:
    parts = [
        f"Fold {fold}/{total_folds}",
        f"Epoch {epoch}/{total_epochs}",
        f"Step {step}/{total_steps}",
    ]
    if loss is not None:
        parts.append(f"loss {float(loss):.5f}")
    resource = resource or {}
    if resource.get("mps_driver_gb") is not None:
        parts.append(f"MPS {float(resource['mps_driver_gb']):.2f}GB")
    if resource.get("available_gb") is not None:
        parts.append(f"available {float(resource['available_gb']):.2f}GB")
    return " · ".join(parts)


def _config_from_args(args: argparse.Namespace) -> GridTrainConfig:
    return GridTrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size if args.batch_size is not None else _default_batch(args.model_name),
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        relation_rank=args.relation_rank,
        trainable_layers=args.trainable_layers if args.trainable_layers is not None else _default_layers(args.model_name),
        seed=args.seed,
        device=args.device,
        n_splits=args.folds,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        max_span_length=args.max_span_length,
        span_top_k=args.span_top_k,
        relation_top_k=args.relation_top_k,
        nms_iou=args.nms_iou,
        **({"implicit_loss_weight": args.implicit_loss_weight} if args.implicit_loss_weight is not None else {}),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_rows < 0 or args.max_length < 3:
        raise SystemExit("--max-rows must be >= 0 and --max-length must be >= 3")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.gradient_accumulation < 1 or args.epochs < 1 or args.relation_rank < 1:
        raise SystemExit("--gradient-accumulation, --epochs, and --relation-rank must be positive")
    if args.trainable_layers is not None and args.trainable_layers < 0:
        raise SystemExit("--trainable-layers must be >= 0")
    if args.max_span_length < 1 or args.span_top_k < 1 or args.relation_top_k < 1:
        raise SystemExit("span and relation top-k values must be positive")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise SystemExit("--nms-iou must be in [0, 1]")
    if args.budget_hours <= 0:
        raise SystemExit("--budget-hours must be positive")
    if args.promote_root and not args.include_test:
        raise SystemExit("--promote-root requires --include-test")


def _write_plan(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    guard = ResourceGuard()
    rows = load_train_data(args.reviews, args.labels)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if len(rows) < args.folds:
        raise SystemExit(f"at least {args.folds} rows are required")
    plan = fixed_screen_plan(rows, config, guard)
    plan["known_model_aliases"] = sorted(MODEL_ALIASES)
    plan["fixed_split_seed"] = args.seed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "screening_plan.json", plan)
    write_json(args.output_dir / "oof.json", {"status": "not_run", "records": [], "metrics": "not_computed"})
    write_json(args.output_dir / "checkpoint.json", {"status": "not_created", "reason": "train_flag_not_set", "metrics": "not_computed"})
    print(json.dumps({"status": "plan_only", "rows": len(rows), "folds": args.folds, "output_dir": str(args.output_dir), "metrics": "not_computed"}, ensure_ascii=False))
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    guard = ResourceGuard(memory_budget_gb=12.0, soft_rss_gb=10.0, hard_rss_gb=12.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical_champion = _load_historical_champion(ROOT)
    dashboard = DashboardWriter(ROOT / "dashboard", budget_seconds=args.budget_hours * 3600.0)
    if historical_champion is not None:
        dashboard.update(metrics={"champion_f1": historical_champion})
    if not args.no_open_dashboard:
        _open_dashboard(args.dashboard_url)

    preflight = _snapshot_dict(guard.snapshot())
    write_json(args.output_dir / "resource_preflight.json", preflight)
    dashboard.resource(preflight)
    dashboard.update(status="running", stage=f"GRID_{args.folds}FOLD_PRECHECK", message="relation-grid preflight complete")
    if (
        preflight.get("pressure") == "critical"
        or float(preflight.get("memory_percent") or 0.0) >= 98.0
        or float(preflight.get("tracked_gb") or 0.0) >= guard.hard_rss_gb
        or float(preflight.get("swap_growth_gb") or 0.0) >= 1.5
    ):
        message = "resource gate: critical pressure, 98% system memory, hard tracked budget, or rapid swap growth"
        dashboard.update(status="paused", stage="RESOURCE_WAIT", message=message)
        write_json(args.output_dir / "resource_gate.json", {"status": "paused", "reason": message, "snapshot": preflight})
        print(json.dumps({"status": "paused", "reason": message, "snapshot": preflight}, ensure_ascii=False))
        return 3

    prefetch_path = None
    if args.download_missing_backbone:
        dashboard.update(status="running", stage="BACKBONE_PREFETCH", message=f"prefetching {args.model_name} before local-only training")
        prefetch_path = _prefetch_backbone(args.model_name)
        write_json(args.output_dir / "backbone_prefetch.json", {"model_name": args.model_name, "path": prefetch_path, "status": "complete"})

    rows = load_train_data(args.reviews, args.labels)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if len(rows) < args.folds:
        raise SystemExit(f"at least {args.folds} rows are required")
    test_rows = load_test_reviews(args.test_reviews) if args.include_test else []
    splits = fixed_splits(rows, n_splits=args.folds, seed=args.seed)
    write_json(args.output_dir / "fold_assignments.json", [{"fold": fold, "train_ids": [rows[i].id for i in train_idx], "valid_ids": [rows[i].id for i in valid_idx], "seed": args.seed, "n_splits": args.folds} for fold, (train_idx, valid_idx) in enumerate(splits, start=1)])

    gold = {row.id: set(row.labels) for row in rows}
    oof_candidates: dict[int, list[Any]] = {}
    test_maps: list[dict[int, list[Any]]] = []
    fold_reports: list[dict[str, Any]] = []
    started = time.time()
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_config = replace(config, seed=args.seed + fold * 1009)
        fold_dir = args.output_dir / f"fold_{fold}"
        trial_name = f"grid_{args.model_name}_{args.folds}fold"
        fold_config_text = json.dumps(asdict(fold_config), ensure_ascii=False, sort_keys=True)
        dashboard.update(
            status="running",
            stage=f"GRID_{args.folds}FOLD",
            message=f"training fold {fold}/{args.folds}",
            current={
                "trial": trial_name,
                "model_name": args.model_name,
                "fold": fold,
                "folds": args.folds,
                "epoch": 0,
                "epochs": fold_config.epochs,
                "step": 0,
                "steps": None,
                "progress_percent": (fold - 1) / args.folds * 100.0,
                "fold_percent": 0.0,
                "epoch_percent": 0.0,
                "phase": "fold_start",
                "runtime_config": fold_config_text,
                "config": fold_config_text,
            },
        )
        dashboard.progress({
            "timestamp": time.time(), "trial": trial_name, "model_name": args.model_name,
            "fold": fold, "folds": args.folds, "epoch": 0, "epochs": fold_config.epochs,
            "step": 0, "steps": 0, "progress_percent": (fold - 1) / args.folds * 100.0,
            "fold_percent": 0.0, "epoch_percent": 0.0, "phase": "fold_start",
            "runtime_config": fold_config_text, "runtime_config_data": asdict(fold_config),
        })
        dashboard.event(f"开始 fold {fold}/{args.folds} · 等待第一轮 step telemetry")

        def on_update(payload: dict[str, Any], *, fold_number: int = fold) -> None:
            resource = payload.get("resource")
            if resource:
                dashboard.resource(resource)
            if "f1" in payload:
                dashboard.metric(precision=float(payload.get("precision") or 0.0), recall=float(payload.get("recall") or 0.0), f1=float(payload.get("f1") or 0.0))
            runtime_config = payload.get("runtime_config") or asdict(fold_config)
            runtime_config_text = json.dumps(runtime_config, ensure_ascii=False, sort_keys=True)
            epoch = max(1, int(payload.get("epoch") or 1))
            step = int(payload.get("step") or 0)
            steps = max(1, int(payload.get("steps") or 1))
            epochs = max(1, int(runtime_config.get("epochs", fold_config.epochs)))
            progress = _progress_payload(
                fold=fold_number,
                total_folds=args.folds,
                epoch=epoch,
                total_epochs=epochs,
                step=step,
                total_steps=steps,
                elapsed_seconds=max(0.0, time.time() - started),
            )
            progress_record = {
                "timestamp": time.time(), "trial": trial_name, "model_name": args.model_name,
                "fold": fold_number, "folds": args.folds, "epoch": epoch, "epochs": epochs,
                "step": step, "steps": steps, "progress_percent": progress["overall_percent"],
                "overall_percent": progress["overall_percent"], "step_percent": progress["step_percent"],
                "fold_percent": progress["fold_percent"], "epoch_percent": progress["epoch_percent"],
                "eta_seconds": progress["eta_seconds"],
                "phase": payload.get("phase", "step"), "runtime_config": runtime_config_text,
                "runtime_config_data": runtime_config,
            }
            dashboard.progress(progress_record)
            if "loss" in payload:
                dashboard.loss({
                    "timestamp": time.time(), "trial": trial_name, "fold": fold_number,
                    "epoch": epoch, "step": step, "steps": steps, "loss": payload.get("loss"),
                    "grid_loss": payload.get("grid_loss"), "boundary_loss": payload.get("boundary_loss"),
                    "phase": payload.get("phase", "step"), "runtime_config": runtime_config,
                })
                if payload.get("phase", "step") == "step" and step > 0 and step % 100 == 0:
                    dashboard.event(_format_training_step_event(
                        fold=fold_number,
                        total_folds=args.folds,
                        epoch=epoch,
                        total_epochs=epochs,
                        step=step,
                        total_steps=steps,
                        loss=payload.get("loss"),
                        resource=resource if isinstance(resource, dict) else None,
                    ))
            dashboard.update(current={
                "trial": trial_name, "model_name": args.model_name, "fold": fold_number,
                "folds": args.folds, "epoch": epoch, "epochs": epochs, "step": step,
                "steps": steps, "progress_percent": progress["overall_percent"],
                "overall_percent": progress["overall_percent"], "step_percent": progress["step_percent"],
                "fold_percent": progress["fold_percent"], "epoch_percent": progress["epoch_percent"],
                "eta_seconds": progress["eta_seconds"], "phase": payload.get("phase", "step"),
                "loss": payload.get("loss"), "runtime_config": runtime_config_text, "config": runtime_config_text,
            })
            if payload.get("phase") == "epoch_complete":
                dashboard.event(f"fold {fold_number}/{args.folds} · epoch {epoch}/{epochs} 完成 · F1 {float(payload.get('f1') or 0.0):.4f}")

        active_fold_config = fold_config
        restart_count = 0
        while True:
            try:
                result = train_grid_fold(
                    [rows[i] for i in train_idx],
                    [rows[i] for i in valid_idx],
                    active_fold_config,
                    guard=guard,
                    test_rows=test_rows if args.include_test else None,
                    output_dir=fold_dir,
                    on_update=on_update,
                    select_best_epoch=False,
                )
                break
            except ResourcePressureRestart as exc:
                restart_count += 1
                if restart_count > 2:
                    raise
                recommendation = exc.recommendation
                active_fold_config = replace(
                    active_fold_config,
                    batch_size=int(recommendation.get("batch_size", active_fold_config.batch_size)),
                    gradient_accumulation=int(recommendation.get("gradient_accumulation", active_fold_config.gradient_accumulation)),
                    max_length=int(recommendation.get("max_length", active_fold_config.max_length)),
                    trainable_layers=int(recommendation.get("trainable_layers", active_fold_config.trainable_layers)),
                )
                dashboard.update(
                    status="running",
                    stage=f"GRID_{args.folds}FOLD_RESOURCE_RESTART",
                    message=f"fold {fold} restarting with safer runtime config",
                    current={"trial": trial_name, "model_name": args.model_name, "fold": fold, "folds": args.folds, "epoch": 0, "epochs": active_fold_config.epochs, "step": 0, "steps": None, "phase": "resource_restart", "runtime_config": json.dumps(asdict(active_fold_config), ensure_ascii=False, sort_keys=True), "config": json.dumps(asdict(active_fold_config), ensure_ascii=False, sort_keys=True)},
                )
                dashboard.event(f"fold {fold}/{args.folds} 触发资源降级并重启 · batch={active_fold_config.batch_size} · max_length={active_fold_config.max_length}")
            except Exception as exc:
                dashboard.update(status="failed", stage="GRID_FAILED", message=f"fold {fold} failed: {type(exc).__name__}: {exc}")
                write_json(args.output_dir / "fold_failure.json", {"fold": fold, "error_type": type(exc).__name__, "error": str(exc)})
                raise

        valid_candidates = result["candidates"]
        overlap = set(oof_candidates).intersection(valid_candidates)
        if overlap:
            raise RuntimeError(f"OOF id overlap detected at fold {fold}: {sorted(overlap)[:5]}")
        oof_candidates.update(valid_candidates)
        if args.include_test:
            test_maps.append(result["test_candidates"])
        fold_gold = {rows[i].id: rows[i].labels for i in valid_idx}
        fold_monitor_selected = select_threshold(valid_candidates, fold_gold, separate_implicit=True)
        monitor = fold_monitor_selected["score"]
        fold_report = {
            "fold": fold,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "candidate_count": sum(len(values) for values in valid_candidates.values()),
            "monitor": _score_dict(monitor),
            "monitor_calibration": "fold_local_four_state_display_only",
            "monitor_state_thresholds": fold_monitor_selected.get("state_thresholds", {}),
            "history": result["history"],
            "checkpoint": result["checkpoint"],
            "config": result["config"],
        }
        fold_reports.append(fold_report)
        dashboard.trial({"name": f"grid_{args.model_name}_fold_{fold}", "model": "relation_grid", "stage": f"{args.folds}-fold", **_score_dict(monitor), "fold": fold, "seconds": time.time() - started, "config": json.dumps(result["config"], ensure_ascii=False, sort_keys=True)})
        last_history = result.get("history")[-1] if result.get("history") else {}
        runtime_config_text = json.dumps(result.get("config") or {}, ensure_ascii=False, sort_keys=True)
        dashboard.progress({
            "timestamp": time.time(), "trial": trial_name, "model_name": args.model_name,
            "fold": fold, "folds": args.folds, "epoch": int(last_history.get("epoch") or fold_config.epochs),
            "epochs": int(result.get("config", {}).get("epochs", fold_config.epochs)), "step": int(last_history.get("step") or 0),
            "steps": int(last_history.get("steps") or last_history.get("step") or 0),
            "progress_percent": fold / args.folds * 100.0, "fold_percent": 100.0, "epoch_percent": 100.0,
            "phase": "fold_complete", "runtime_config": runtime_config_text, "runtime_config_data": result.get("config") or {},
        })
        dashboard.update(counts={"completed": fold}, message=f"fold {fold}/{args.folds} 完成 · 等待下一 fold" if fold < args.folds else f"fold {fold}/{args.folds} 完成 · 开始 OOF 校准")
        dashboard.event(f"完成 fold {fold}/{args.folds} · strict monitor F1 {monitor.f1:.4f}")
        write_json(fold_dir / "fold_report.json", fold_report)

    if set(oof_candidates) != set(gold):
        missing = sorted(set(gold) - set(oof_candidates))
        extra = sorted(set(oof_candidates) - set(gold))
        raise RuntimeError(f"OOF coverage mismatch; missing={missing[:5]} extra={extra[:5]}")

    threshold_result = select_threshold(oof_candidates, gold, separate_implicit=True)
    full_fit_score_payload = _score_dict(threshold_result["score"])
    crossfit_payload: dict[str, Any] | None = None
    if args.folds == 5:
        valid_id_folds = [[rows[i].id for i in valid_idx] for _, valid_idx in splits]
        crossfit = crossfit_threshold_score(oof_candidates, gold, valid_id_folds, separate_implicit=True)
        score_payload = _score_dict(crossfit["score"])
        crossfit_payload = {
            "score": score_payload,
            "folds": [
                {
                    **{key: value for key, value in item.items() if key != "score"},
                    "score": _score_dict(item["score"]),
                }
                for item in crossfit["folds"]
            ],
        }
        calibration_name = "5-fold cross-fit four-state threshold calibration"
    else:
        score_payload = full_fit_score_payload
        calibration_name = "3-fold screening four-state threshold fit on aggregate fold-exclusive OOF"
    threshold_payload = {
        "threshold": float(threshold_result["threshold"]),
        "implicit_threshold": float(threshold_result["implicit_threshold"]),
        "state_thresholds": {key: float(value) for key, value in threshold_result.get("state_thresholds", {}).items()},
        "full_oof_fit_score": full_fit_score_payload,
        "reported_score": score_payload,
        "calibration": calibration_name,
        "crossfit": crossfit_payload,
    }
    save_candidates(args.output_dir / "oof_candidates.json", oof_candidates)
    oof_report = {"status": "completed", "model_name": args.model_name, "resolved_model_name": MODEL_ALIASES.get(args.model_name.lower(), args.model_name), "folds": args.folds, "fold_reports": fold_reports, "threshold": threshold_payload, "metrics": score_payload, "rows": len(rows), "gold_quadruples": sum(len(values) for values in gold.values()), "oof_candidate_count": sum(len(values) for values in oof_candidates.values()), "elapsed_seconds": time.time() - started}
    oof_path = args.output_dir / "oof_report.json"
    write_json(oof_path, oof_report)
    config_payload = {"config": asdict(config), "fold_assignments": [{"fold": fold, "train_ids": [rows[i].id for i in train_idx], "valid_ids": [rows[i].id for i in valid_idx]} for fold, (train_idx, valid_idx) in enumerate(splits, start=1)], "threshold": threshold_payload, "resource_preflight": preflight, "screening_only": args.folds == 3, "leaderboard_score": None}
    config_path = args.output_dir / "config.json"
    write_json(config_path, config_payload)
    dashboard.metric(
        precision=score_payload["precision"],
        recall=score_payload["recall"],
        f1=score_payload["f1"],
        champion_f1=max(float(score_payload["f1"]), float(historical_champion or 0.0)),
    )
    dashboard.progress({
        "timestamp": time.time(), "trial": f"grid_{args.model_name}_{args.folds}fold", "model_name": args.model_name,
        "fold": args.folds, "folds": args.folds, "epoch": args.epochs, "epochs": args.epochs,
        "step": 0, "steps": 0, "progress_percent": 100.0, "overall_percent": 100.0,
        "step_percent": 100.0, "fold_percent": 100.0, "epoch_percent": 100.0,
        "eta_seconds": 0.0, "phase": "oof_calibration", "runtime_config": json.dumps(asdict(config), ensure_ascii=False, sort_keys=True),
        "runtime_config_data": asdict(config),
    })
    dashboard.event(f"全部 {args.folds} folds 完成 · 进入 {calibration_name}")

    submission_payload: dict[str, Any] = {"status": "not_written", "leaderboard_score": None}
    if args.include_test:
        merged_test = merge_candidate_maps(test_maps, average_present=True)
        test_predictions = threshold_predictions(
            merged_test,
            float(threshold_result["threshold"]),
            float(threshold_result["implicit_threshold"]),
            state_thresholds=threshold_result.get("state_thresholds"),
        )
        candidate_path = args.output_dir / "Result.csv"
        report = write_submission(candidate_path, test_rows, test_predictions)
        validated = validate_submission(candidate_path, test_rows)
        if report != validated:
            raise RuntimeError("submission round-trip validation mismatch")
        candidate_id = f"grid-{args.model_name}-{args.folds}fold-{int(time.time())}"
        try:
            manifest = create_candidate_manifest(candidate_path, oof_path, config_path, args.output_dir / "online_loop", candidate_dir=args.output_dir / "online_loop" / "candidates", candidate_id=candidate_id, expected_test_ids=[row.id for row in test_rows])
        except ManifestError as exc:
            raise RuntimeError(f"local manifest creation failed: {exc}") from exc
        submission_payload = {"status": "validated_local_candidate", "candidate_id": candidate_id, "path": str(candidate_path), "rows": validated.rows, "predicted_quadruples": validated.predicted_quadruples, "empty_ids": validated.empty_ids, "manifest_path": manifest["manifest_path"], "queue_path": manifest["queue_path"], "leaderboard_score": None}
        if args.promote_root:
            root_result = ROOT / "Result.csv"
            shutil.copyfile(candidate_path, root_result)
            validate_submission(root_result, test_rows)
            submission_payload["promoted_root"] = str(root_result)
    dashboard.final(metrics=score_payload, submission=submission_payload, message="relation-grid OOF complete; local candidate validated" if args.include_test else "relation-grid OOF complete")
    write_json(args.output_dir / "final_report.json", {"oof": oof_report, "submission": submission_payload, "leaderboard_score": None})
    print(json.dumps({"status": "completed", "metrics": score_payload, "threshold": threshold_payload, "submission": submission_payload, "output_dir": str(args.output_dir)}, ensure_ascii=False, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    if args.dummy_train:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        smoke = dummy_train_smoke(max_length=min(args.max_length, 32), relation_rank=args.relation_rank, output_dir=args.output_dir)
        write_json(args.output_dir / "dummy_train.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False))
        return 0
    if args.dummy:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        smoke = dummy_smoke(max_length=min(args.max_length, 32))
        write_json(args.output_dir / "dummy_smoke.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False))
        return 0
    if not args.train:
        return _write_plan(args)
    return _run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
