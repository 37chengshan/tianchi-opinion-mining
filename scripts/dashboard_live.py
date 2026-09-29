#!/usr/bin/env python3
"""Regenerate dashboard/state.json from the live experiment directory.

The autopilot writes its own dashboard state; runs started directly through
run_anchor.py do not, so this script derives the same schema from the fold
history files and the runtime resource guard.  Run it in a loop while training.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from opinion_mining.resource_guard import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
EXPERIMENTS = ROOT / "artifacts/experiments"

# Key results kept for the leaderboard panel (score fields are strict metrics).
LEADERBOARD = [
    {"name": "anchor_wwm_5fold", "model": "hfl/chinese-roberta-wwm-ext", "stage": "online_submitted", "f1": 0.754295, "precision": 0.760656, "recall": 0.748040, "leaderboard_score": 0.7583, "note": "当前线上最佳 candidate-84e0474e404e-ac9a430bf2db"},
    {"name": "anchor_macbert_5fold", "model": "hfl/chinese-macbert-base", "stage": "running", "f1": None, "precision": None, "recall": None, "leaderboard_score": None, "note": "与 WWM 5 折融合候选"},
    {"name": "anchor_wwm_long_3fold", "model": "hfl/chinese-roberta-wwm-ext", "stage": "running", "f1": None, "precision": None, "recall": None, "leaderboard_score": None, "note": "8 epochs / lr 1.2e-5，验证欠训练假设"},
    {"name": "anchor_wwm_5fold_prev", "model": "hfl/chinese-roberta-wwm-ext", "stage": "completed", "f1": 0.754295, "precision": 0.760656, "recall": 0.748040, "leaderboard_score": 0.7583, "note": "线上 0.7583"},
    {"name": "anchor_wwm_3fold", "model": "hfl/chinese-roberta-wwm-ext", "stage": "completed", "f1": 0.745873, "precision": 0.783308, "recall": 0.711852, "leaderboard_score": None, "note": "3 折筛选"},
    {"name": "anchor_macbert_3fold", "model": "hfl/chinese-macbert-base", "stage": "completed", "f1": 0.735749, "precision": None, "recall": None, "leaderboard_score": None, "note": "同折内 F1"},
    {"name": "anchor_rbt3_v1_3fold", "model": "hfl/rbt3", "stage": "completed", "f1": 0.714031, "precision": 0.736033, "recall": 0.693305, "leaderboard_score": None, "note": "B1+B2+B3 组合控制组"},
    {"name": "neural_5fold_confirm", "model": "hfl/rbt3", "stage": "online_champion_legacy", "f1": 0.720898, "precision": 0.745888, "recall": 0.697527, "leaderboard_score": 0.7162020541, "note": "旧冠军（已保留）"},
]

KEEP_TRIALS = {
    "anchor_macbert_5fold",
    "anchor_wwm_long_3fold",
    "anchor_wwm_long_5fold",
    "anchor_wwm_large_3fold",
    "anchor_wwm_5fold",
    "anchor_macbert_3fold",
    "anchor_wwm_3fold",
}


MARK_FILE = DASHBOARD / "progress_marks.json"


def _load_marks() -> dict[str, float]:
    data = _read_json(MARK_FILE) or {}
    return {str(key): float(value) for key, value in data.items()} if isinstance(data, dict) else {}


def _save_marks(marks: dict[str, float]) -> None:
    MARK_FILE.write_text(json.dumps(marks, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _trial_progress(trial_dir: Path, marks: dict[str, float], now: float) -> dict[str, object] | None:
    folds = sorted(p for p in trial_dir.glob("fold_*") if p.is_dir())
    if not folds:
        live_only = _read_json(trial_dir / "live_progress.json")
        if not isinstance(live_only, dict):
            return None
        config = _read_json(trial_dir / "config.json") or {}
        config = config.get("config", config)
        return {
            "name": trial_dir.name,
            "folds_done": max(0, int(live_only.get("fold", 1)) - 1),
            "folds_started": int(live_only.get("folds_total", 1)),
            "last_fold": {"fold": int(live_only.get("fold", 1)), "epochs": int(live_only.get("epoch", 0)), "f1": live_only.get("f1"), "precision": None, "recall": None},
            "fold_stats": [],
            "loss": [],
            "config": config,
            "epochs_total": int(live_only.get("epochs_total", 0)),
            "epochs_done": max(0, (int(live_only.get("fold", 1)) - 1) * int(live_only.get("epochs_total", 0)) + int(live_only.get("epoch", 1)) - 1),
            "epoch_seconds": None,
            "started_at": (trial_dir / "config.json").stat().st_mtime if (trial_dir / "config.json").exists() else None,
            "config_mtime": (trial_dir / "config.json").stat().st_mtime if (trial_dir / "config.json").exists() else None,
            "mtime": max((path.stat().st_mtime for path in trial_dir.rglob("*") if path.is_file()), default=now),
        }
    fold_stats = []
    loss_records = []
    epoch_marks: list[float] = []
    for index, fold_dir in enumerate(folds, start=1):
        history = _read_json(fold_dir / "history.json") or []
        if not history:
            continue
        last = history[-1]
        fold_stats.append({"fold": index, "epochs": len(history), "f1": last.get("f1"), "precision": last.get("precision"), "recall": last.get("recall")})
        for record in history:
            key = f"{trial_dir.name}|{index}|{record.get('epoch')}"
            marks.setdefault(key, now)
            epoch_marks.append(marks[key])
            loss_records.append(
                {
                    "trial": trial_dir.name,
                    "fold": index,
                    "epoch": record.get("epoch"),
                    "at": time.strftime("%H:%M", time.localtime(marks[key])),
                    "loss": record.get("loss"),
                    "f1": record.get("f1"),
                    "precision": record.get("precision"),
                    "recall": record.get("recall"),
                }
            )
    if not fold_stats:
        return None
    config = _read_json(trial_dir / "config.json") or {}
    config = config.get("config", config)
    epochs_total = int(config.get("epochs") or 0)
    durations = [b - a for a, b in zip(epoch_marks, epoch_marks[1:]) if b > a]
    epoch_seconds = sorted(durations)[len(durations) // 2] if durations else None
    config_mtime = (trial_dir / "config.json").stat().st_mtime if (trial_dir / "config.json").exists() else None
    epochs_done_so_far = sum(item["epochs"] for item in fold_stats)
    if not epoch_seconds and config_mtime and epochs_done_so_far:
        epoch_seconds = max(1.0, (time.time() - config_mtime) / epochs_done_so_far)
    epochs_done = sum(item["epochs"] for item in fold_stats)
    return {
        "name": trial_dir.name,
        "folds_done": len(fold_stats),
        "folds_started": len(folds),
        "last_fold": fold_stats[-1],
        "fold_stats": fold_stats,
        "loss": loss_records,
        "config": config,
        "epochs_total": epochs_total,
        "epochs_done": epochs_done,
        "epoch_seconds": epoch_seconds,
        "started_at": config_mtime,
        "config_mtime": config_mtime,
        "mtime": max(path.stat().st_mtime for path in trial_dir.rglob("*") if path.is_file()),
    }



FRIENDLY = {
    "anchor_macbert_5fold": ("MacBERT 5 折训练", 5),
    "anchor_wwm_long_5fold": ("WWM 5 折长训练（10 轮）", 5),
    "anchor_wwm_3fold": ("WWM 3 折筛选", 3),
    "anchor_macbert_3fold": ("MacBERT 3 折筛选", 3),
    "anchor_wwm_large_3fold": ("WWM-large 3 折筛选", 3),
    "anchor_wwm_long_3fold": ("WWM 长训练（8 epochs）3 折", 3),
    "anchor_wwm_5fold": ("WWM 5 折训练", 5),
    "anchor_wwm_large_3fold": ("WWM-large 3 折筛选", 3),
}


def _fmt_minutes(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "—"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} 分钟"
    return f"{minutes // 60} 小时 {minutes % 60} 分钟"


def _trial_elapsed_seconds(trials: list[dict]) -> float:
    starts = []
    for trial in trials:
        config_path = EXPERIMENTS / str(trial["name"]) / "config.json"
        if config_path.exists():
            starts.append(config_path.stat().st_mtime)
    return max(0.0, time.time() - min(starts)) if starts else 0.0


def _write_history_feed(live: dict | None, trials: list[dict]) -> None:
    """Feed the UI's live-stream panel with real step records."""
    records: list[dict[str, object]] = []
    if live is not None:
        data = live["data"]
        for item in data.get("history") or []:
            records.append(
                {
                    "trial": data.get("name"),
                    "fold": item.get("fold"),
                    "epoch": item.get("epoch"),
                    "step": item.get("step"),
                    "steps": item.get("steps"),
                    "at": time.strftime("%H:%M:%S", time.localtime(float(item.get("at", 0)))),
                    "loss": item.get("loss"),
                    "span_loss": item.get("span_loss"),
                    "relation_loss": item.get("relation_loss"),
                }
            )
    for trial in trials:
        for item in trial["loss"]:
            records.append(
                {
                    "trial": trial["name"],
                    "fold": item.get("fold"),
                    "epoch": item.get("epoch"),
                    "step": None,
                    "steps": None,
                    "at": item.get("at"),
                    "loss": item.get("loss"),
                    "f1": item.get("f1"),
                }
            )
    payload = {"loss": records[-400:], "f1": [{"at": r.get("at"), "value": r.get("f1")} for r in records if r.get("f1") is not None][-400:]}
    (DASHBOARD / "history.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _simple_view(trials: list[dict], now: float) -> dict[str, object]:
    runs = []
    total_remaining = 0.0
    for trial in trials:
        label, folds_total = FRIENDLY.get(trial["name"], (trial["name"], max(1, trial["folds_done"])))
        epochs_total = trial["epochs_total"] or 0
        steps_total = folds_total * epochs_total if epochs_total else folds_total
        steps_done = trial["epochs_done"]
        percent = round(100.0 * steps_done / steps_total, 1) if steps_total else 0.0
        remaining_steps = max(0, steps_total - steps_done)
        epoch_seconds = trial["epoch_seconds"] or 0.0
        remaining_seconds = remaining_steps * epoch_seconds if epoch_seconds else None
        if remaining_seconds:
            total_remaining += remaining_seconds
        last = trial["last_fold"]
        runs.append(
            {
                "label": label,
                "percent": percent,
                "steps_done": steps_done,
                "steps_total": steps_total,
                "folds_done": trial["folds_done"],
                "folds_total": folds_total,
                "current_fold_epoch": last["epochs"],
                "epochs_per_fold": epochs_total,
                "minutes_per_epoch": round(epoch_seconds / 60.0, 1) if epoch_seconds else None,
                "remaining": _fmt_minutes(remaining_seconds),
                "eta_clock": time.strftime("%H:%M", time.localtime(now + remaining_seconds)) if remaining_seconds else "—",
                "latest_monitor_f1": round(last["f1"], 4) if last.get("f1") is not None else None,
                "status": "完成" if percent >= 100 else "训练中",
            }
        )
    runs.sort(key=lambda item: (item["status"] == "完成", -item["percent"]))
    return {
        "headline": "线上最好成绩 0.7583（已提交）；目标是 0.80",
        "remaining_total": _fmt_minutes(total_remaining) if total_remaining else "—",
        "eta_clock": time.strftime("%H:%M", time.localtime(now + total_remaining)) if total_remaining else "—",
        "runs": runs,
        "best": {
            "local": 0.754295,
            "online": 0.7583,
            "online_note": "candidate-84e0474e404e-ac9a430bf2db",
        },
        "up_next": [
            "MacBERT 5 折完成 → 与 WWM 5 折融合，若本地分更高就重新提交",
            "长训练（8 epochs）验证欠训练假设；若更高则替换主模型",
            "之后：large 大模型 / TAPT 领域自适应，继续冲 0.80",
        ],
    }



def _live_progress(trial_dir: Path) -> dict[str, object] | None:
    data = _read_json(trial_dir / "live_progress.json")
    if not isinstance(data, dict):
        return None
    history = [item for item in (data.get("history") or []) if isinstance(item, dict)]
    rate = None
    rates: list[float] = []
    for previous, current in zip(history, history[1:]):
        if (previous.get("fold"), previous.get("epoch")) != (current.get("fold"), current.get("epoch")):
            continue
        span = float(current.get("at", 0)) - float(previous.get("at", 0))
        steps = int(current.get("step", 0)) - int(previous.get("step", 0))
        if span > 0 and steps > 0:
            rates.append(span / steps)
    if rates:
        rates.sort()
        rate = rates[len(rates) // 2]
    return {"data": data, "seconds_per_step": rate, "updated_ts": data.get("updated_ts")}


def build_state() -> dict[str, object]:
    marks = _load_marks()
    trials = []
    for trial_dir in sorted(EXPERIMENTS.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name not in KEEP_TRIALS:
            continue
        progress = _trial_progress(trial_dir, marks, time.time())
        if progress:
            trials.append(progress)
    trials.sort(key=lambda item: item["mtime"])

    events = []
    now = time.strftime("%H:%M:%S")
    for trial in trials:
        events.append(f"{now} {trial['name']}: {trial['folds_done']}/{trial['folds_started']} folds, last fold epoch {trial['last_fold']['epochs']} f1={trial['last_fold']['f1']:.4f}" if trial["last_fold"].get("f1") is not None else f"{now} {trial['name']}: running")

    live = None
    fresh_after = time.time() - 300.0
    for trial in reversed(trials):
        candidate = _live_progress(EXPERIMENTS / trial["name"])
        if not candidate:
            continue
        updated_ts = float(candidate.get("updated_ts") or 0.0)
        if updated_ts < fresh_after:
            continue
        live = {"trial": trial, **candidate}
        break
    active = trials[-1] if trials else None
    guard = ResourceGuard()
    try:
        resource = asdict(guard.snapshot())
    except Exception:
        resource = {}
    if active is not None:
        last_history = _read_json(EXPERIMENTS / active["name"] / f"fold_{active['folds_done']}" / "history.json") or []
        if last_history and isinstance(last_history[-1].get("resource"), dict):
            resource.update({key: value for key, value in last_history[-1]["resource"].items() if value is not None})

    current = {}
    loss = []
    metrics = {"precision": None, "recall": None, "f1": None, "champion_f1": 0.7583}
    if active is not None:
        last = active["last_fold"]
        folds_total = 5 if "5fold" in active["name"] else 3
        epochs_total = active["epochs_total"] or 0
        steps_total = folds_total * epochs_total if epochs_total else folds_total
        percent = round(100.0 * active["epochs_done"] / steps_total, 1) if steps_total else 0.0
        current = {
            "trial": active["name"],
            "model_name": active["config"].get("model_name"),
            "fold": last["fold"],
            "folds": folds_total,
            "epoch": last["epochs"],
            "epochs": epochs_total,
            "progress_percent": percent,
            "overall_percent": percent,
            "steps_done": active["epochs_done"],
            "steps_total": steps_total,
            "minutes_per_epoch": round(active["epoch_seconds"] / 60.0, 1) if active["epoch_seconds"] else None,
            "phase": "training",
            "config": json.dumps(active["config"], ensure_ascii=False),
            "f1": last.get("f1"),
            "precision": last.get("precision"),
            "recall": last.get("recall"),
        }
        metrics.update({"precision": last.get("precision"), "recall": last.get("recall"), "f1": last.get("f1")})
        loss = active["loss"][-400:]

    simple = _simple_view(trials, time.time())
    now_ts = time.time()

    def _fields(*, trial, fold, folds_total, epoch, epochs_total, step, steps_in_epoch, epoch_seconds, monitor_f1=None, config=None):
        epoch_fraction = (step / steps_in_epoch) if steps_in_epoch else 0.0
        fold_fraction = (max(0, epoch - 1) + epoch_fraction) / max(1, epochs_total)
        overall_fraction = ((fold - 1) + fold_fraction) / max(1, folds_total)
        done_epochs = (fold - 1) * epochs_total + (epoch - 1) + epoch_fraction
        remaining_epochs = max(0.0, folds_total * epochs_total - done_epochs)
        remaining_seconds = remaining_epochs * epoch_seconds if epoch_seconds else None
        return {
            "trial": trial,
            "model_name": (config or {}).get("model_name"),
            "fold": fold,
            "folds": folds_total,
            "epoch": epoch,
            "epochs": epochs_total,
            "step": step,
            "steps": steps_in_epoch,
            "epochs_done": round(done_epochs, 2),
            "epochs_total_all": folds_total * epochs_total,
            "fold_percent": round(100.0 * fold_fraction, 1),
            "epoch_percent": round(100.0 * epoch_fraction, 1),
            "step_percent": round(100.0 * epoch_fraction, 1),
            "progress_percent": round(100.0 * overall_fraction, 2),
            "overall_percent": round(100.0 * overall_fraction, 2),
            "eta_seconds": remaining_seconds,
            "minutes_per_epoch": round(epoch_seconds / 60.0, 1) if epoch_seconds else None,
            "eta_clock": time.strftime("%H:%M", time.localtime(now_ts + remaining_seconds)) if remaining_seconds else None,
            "loss": None,
            "phase": "training",
            "config": json.dumps(config, ensure_ascii=False) if config else None,
            "monitor_f1": monitor_f1,
        }

    progress_fields = None
    if live is not None:
        data = live["data"]
        steps_in_epoch = int(data.get("steps_in_epoch") or 1)
        progress_fields = _fields(
            trial=data.get("name"),
            fold=int(data.get("fold") or 1),
            folds_total=int(data.get("folds_total") or 1),
            epoch=int(data.get("epoch") or 1),
            epochs_total=int(data.get("epochs_total") or 1),
            step=int(data.get("step") or 0),
            steps_in_epoch=steps_in_epoch,
            epoch_seconds=(live["seconds_per_step"] or 0.0) * max(1, steps_in_epoch),
            monitor_f1=data.get("f1"),
            config=live["trial"]["config"],
        )
        progress_fields["loss"] = data.get("loss")
        current = progress_fields
        metrics["current_state_f1"] = data.get("f1")
    elif active is not None:
        folds_total = 5 if "5fold" in active["name"] else 3
        epochs_total = active["epochs_total"] or 0
        epoch_seconds = active["epoch_seconds"] or 0.0
        total_epochs_all = max(1, folds_total * epochs_total)
        recorded_epochs = float(active["epochs_done"])
        # Interpolate between finished epochs so the bar moves continuously
        # instead of jumping once per epoch.
        trial_files = [
            path.stat().st_mtime
            for path in (EXPERIMENTS / active["name"]).rglob("*")
            if path.is_file() and path.name != "live_progress.json"
        ]
        mark_ts = [value for key, value in marks.items() if key.startswith(f"{active['name']}|")]
        last_record_ts = max(trial_files + mark_ts) if (trial_files or mark_ts) else 0.0
        ordered = sorted(set(mark_ts))
        deltas = sorted({round(b - a, 2) for a, b in zip(ordered, ordered[1:]) if b > a})
        interval = deltas[len(deltas) // 2] if deltas else (epoch_seconds or None)
        inflight = ((time.time() - last_record_ts) / interval) if (interval and last_record_ts) else 0.0
        inflight = max(0.0, min(1.0, inflight))
        estimated_epochs = recorded_epochs + inflight
        progress_epochs = min(total_epochs_all, max(recorded_epochs, estimated_epochs))
        fold_index = min(folds_total, int(progress_epochs // max(1, epochs_total)) + 1)
        epoch_in_fold = int(progress_epochs % max(1, epochs_total)) + 1
        remaining_seconds = max(0.0, total_epochs_all - progress_epochs) * epoch_seconds if epoch_seconds else None
        progress_fields = {
            "trial": active["name"],
            "model_name": active["config"].get("model_name"),
            "fold": fold_index,
            "folds": folds_total,
            "epoch": epoch_in_fold,
            "epochs": epochs_total,
            "step": None,
            "steps": None,
            "epochs_done": round(progress_epochs, 2),
            "epochs_total_all": total_epochs_all,
            "fold_percent": round(100.0 * ((progress_epochs % max(1, epochs_total)) / max(1, epochs_total)), 1),
            "epoch_percent": round(100.0 * ((progress_epochs % 1) if progress_epochs % 1 else 1.0), 1),
            "step_percent": None,
            "progress_percent": round(100.0 * progress_epochs / total_epochs_all, 2),
            "overall_percent": round(100.0 * progress_epochs / total_epochs_all, 2),
            "eta_seconds": remaining_seconds,
            "minutes_per_epoch": round(epoch_seconds / 60.0, 1) if epoch_seconds else None,
            "eta_clock": time.strftime("%H:%M", time.localtime(now_ts + remaining_seconds)) if remaining_seconds else None,
            "loss": None,
            "phase": "training",
            "config": json.dumps(active["config"], ensure_ascii=False),
            "monitor_f1": active["last_fold"].get("f1"),
        }
        current = progress_fields

    if progress_fields is not None:
        label = FRIENDLY.get(str(progress_fields["trial"]), (str(progress_fields["trial"]), progress_fields["folds"]))[0]
        eta_text = progress_fields.get("eta_clock") or "—"
        current["message"] = (
            f"{label}：第 {progress_fields['fold']}/{progress_fields['folds']} 折 · "
            f"第 {progress_fields['epoch']}/{progress_fields['epochs']} 轮 · "
            + (f"第 {progress_fields['step']}/{progress_fields['steps']} 步 · " if progress_fields.get('step') is not None else "")
            + f"总进度 {progress_fields['overall_percent']}% · "
            f"约 {progress_fields['minutes_per_epoch'] or '—'} 分钟/轮 · 预计 {eta_text} 完成"
        )
        run = next((item for item in simple["runs"] if item["label"] == label), None)
        if run is not None:
            run["percent"] = progress_fields["overall_percent"]
            run["live_step"] = f"{progress_fields['step']}/{progress_fields['steps']}"
            if progress_fields.get("eta_seconds"):
                run["remaining"] = _fmt_minutes(progress_fields["eta_seconds"])
                run["eta_clock"] = eta_text
        if progress_fields.get("eta_seconds"):
            simple["remaining_total"] = _fmt_minutes(progress_fields["eta_seconds"])
            simple["eta_clock"] = eta_text

    _write_history_feed(live, trials)
    _save_marks(marks)

    return {
        "status": "running",
        "simple": simple,
        "stage": "TONIGHT_RUN",
        "message": (current.get("message") if current else "等待训练写入状态") or "等待训练写入状态",
        "elapsed_seconds": _trial_elapsed_seconds([active]) if active is not None else 0.0,
        "budget_seconds": 0.0,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current": current,
        "progress": dict(current, timestamp=time.time()),
        "metrics": metrics,
        "resources": resource,
        "resource": [resource] if resource else [],
        "counts": {},
        "leaderboard": LEADERBOARD,
        "folds": [],
        "loss": loss,
        "submission": {
            "candidate_id": "candidate-84e0474e404e-ac9a430bf2db",
            "path": str(ROOT / "artifacts/submissions/candidates/anchor_wwm_5fold/Result.csv"),
            "sha256": "84e0474e404e89ef324d5649bbc28ee21cb6e0522c0151e9f891eadecb7c48e8",
            "leaderboard_score": 0.7583,
        },
        "events": events[-40:],
    }


def main() -> int:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    target = DASHBOARD / "state.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(build_state(), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"updated": str(target), "at": datetime.now().strftime("%H:%M:%S")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
