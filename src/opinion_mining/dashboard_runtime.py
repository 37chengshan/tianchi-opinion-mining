from __future__ import annotations

"""Atomic JSON state protocol used by the browser dashboard."""

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class DashboardWriter:
    def __init__(self, root: str | Path, *, budget_seconds: float):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.history_path = self.root / "history.json"
        previous_state: dict[str, Any] = {}
        previous_history: dict[str, list[Any]] = {}
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_state = loaded
        except (OSError, ValueError, TypeError):
            pass
        try:
            loaded = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_history = {
                    str(key): list(value) for key, value in loaded.items() if isinstance(value, list)
                }
        except (OSError, ValueError, TypeError):
            pass
        self.started_at = time.time()
        self.budget_seconds = budget_seconds
        self.lock = threading.Lock()
        previous_metrics = previous_state.get("metrics") if isinstance(previous_state.get("metrics"), dict) else {}
        previous_events = previous_state.get("events") if isinstance(previous_state.get("events"), list) else []
        previous_leaderboard = previous_state.get("leaderboard") if isinstance(previous_state.get("leaderboard"), list) else []
        previous_submission = previous_state.get("submission")
        self.state: dict[str, Any] = {
            "status": "ready",
            "stage": "PRECHECK",
            "message": "Dashboard ready. Autopilot has not started yet.",
            "elapsed_seconds": 0,
            "budget_seconds": budget_seconds,
            "current": {
                "trial": None,
                "model_name": None,
                "fold": None,
                "folds": None,
                "epoch": None,
                "epochs": None,
                "step": None,
                "steps": None,
                "progress_percent": 0.0,
                "overall_percent": 0.0,
                "step_percent": 0.0,
                "fold_percent": 0.0,
                "epoch_percent": 0.0,
                "eta_seconds": None,
                "phase": "waiting",
                "runtime_config": None,
                "config": "waiting",
            },
            "progress": {
                "fold": 0,
                "folds": 0,
                "epoch": 0,
                "epochs": 0,
                "step": 0,
                "steps": 0,
                "progress_percent": 0.0,
                "overall_percent": 0.0,
                "step_percent": 0.0,
                "fold_percent": 0.0,
                "epoch_percent": 0.0,
                "eta_seconds": None,
                "phase": "waiting",
            },
            "metrics": {"precision": None, "recall": None, "f1": None, "champion_f1": previous_metrics.get("champion_f1")},
            "resources": {"rss_gb": None, "root_rss_gb": None, "process_count": None, "tracked_gb": None, "available_gb": None, "memory_percent": None, "pressure": "normal", "pressure_free_percent": None, "swap_used_gb": None, "swap_total_gb": None, "swap_growth_gb": None, "compressed_gb": None, "mps_allocated_gb": None, "mps_driver_gb": None, "mps_status": "unknown", "mps_error": None, "memory_budget_gb": 12.0},
            "counts": {"completed": 0, "pruned": 0, "oom": 0, "failed": 0},
            "leaderboard": previous_leaderboard[-80:],
            "folds": [],
            "loss": [],
            "submission": previous_submission,
            "events": [*previous_events[-119:], "Dashboard initialized for a new run; previous history preserved."],
        }
        self.history: dict[str, list[Any]] = {
            "resource": list(previous_history.get("resource", []))[-2000:],
            "f1": list(previous_history.get("f1", []))[-2000:],
            "loss": list(previous_history.get("loss", []))[-4000:],
            "progress": list(previous_history.get("progress", []))[-6000:],
            "trials": list(previous_history.get("trials", []))[-2000:],
        }
        self.flush()

    def flush(self) -> None:
        with self.lock:
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def update(self, **changes: Any) -> None:
        with self.lock:
            for key, value in changes.items():
                if isinstance(value, dict) and isinstance(self.state.get(key), dict):
                    self.state[key].update(_jsonable(value))
                else:
                    self.state[key] = _jsonable(value)
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)

    def event(self, message: str) -> None:
        with self.lock:
            events = self.state.setdefault("events", [])
            events.append(f"{time.strftime('%H:%M:%S')} {message}")
            self.state["events"] = events[-120:]
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)

    def resource(self, snapshot: Any) -> None:
        item = _jsonable(snapshot)
        with self.lock:
            self.state["resources"].update(item)
            self.history.setdefault("resource", []).append(item)
            self.history["resource"] = self.history["resource"][-2000:]
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    def metric(self, *, precision: float | None, recall: float | None, f1: float | None, champion_f1: float | None = None) -> None:
        with self.lock:
            metrics = self.state.setdefault("metrics", {})
            metrics.update({"precision": precision, "recall": recall, "f1": f1})
            if champion_f1 is not None:
                metrics["champion_f1"] = champion_f1
            self.history.setdefault("f1", []).append({"time": time.time(), "f1": f1 or 0.0, "champion_f1": metrics.get("champion_f1") or 0.0})
            self.history["f1"] = self.history["f1"][-2000:]
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    def loss(self, record: dict[str, Any]) -> None:
        """Append one real training-step/epoch loss record for live charts."""
        item = _jsonable(record)
        with self.lock:
            values = self.history.setdefault("loss", [])
            values.append(item)
            self.history["loss"] = values[-4000:]
            self.state["loss"] = self.history["loss"][-200:]
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    def progress(self, record: dict[str, Any]) -> None:
        """Persist replayable fold/epoch/step progress for the live UI."""
        item = _jsonable(record)
        with self.lock:
            self.state["progress"] = item
            current = self.state.setdefault("current", {})
            for key in (
                "trial", "model_name", "fold", "folds", "epoch", "epochs", "step", "steps",
                "progress_percent", "overall_percent", "step_percent", "fold_percent", "epoch_percent",
                "eta_seconds", "phase", "runtime_config", "config",
            ):
                if key in item:
                    current[key] = item[key]
            values = self.history.setdefault("progress", [])
            values.append(item)
            self.history["progress"] = values[-6000:]
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    def trial(self, record: dict[str, Any]) -> None:
        with self.lock:
            leaderboard = self.state.setdefault("leaderboard", [])
            leaderboard.append(_jsonable(record))
            leaderboard.sort(key=lambda row: float(row.get("f1") or 0.0), reverse=True)
            self.state["leaderboard"] = leaderboard[:80]
            self.history.setdefault("trials", []).append(_jsonable(record))
            self.history["trials"] = self.history["trials"][-2000:]
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)

    def final(self, *, metrics: dict[str, Any], submission: dict[str, Any], message: str) -> None:
        with self.lock:
            self.state["status"] = "completed"
            self.state["stage"] = "FINAL"
            self.state["message"] = message
            self.state["metrics"].update(_jsonable(metrics))
            self.state["submission"] = _jsonable(submission)
            self.state["elapsed_seconds"] = max(0.0, time.time() - self.started_at)
            self._atomic_write(self.state_path, self.state)
            self._atomic_write(self.history_path, self.history)
