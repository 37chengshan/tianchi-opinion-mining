#!/usr/bin/env python3
"""Sidecar macOS resource monitor for an already-running training process.

PyTorch MPS counters are process-local, so this sidecar never overwrites the
MPS fields written by the trainer.  It only corrects process-tree RSS and
system-memory fields while a long-running job is active.  The in-process
ResourceGuard remains the authoritative source for MPS allocation counters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.resource_guard import read_macos_memory_stats


def process_tree(pid: int) -> tuple[float, float, int]:
    root = psutil.Process(pid)
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    seen: set[int] = set()
    total = 0
    root_rss = 0
    count = 0
    for process in processes:
        try:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            rss = int(process.memory_info().rss)
        except (psutil.Error, OSError):
            continue
        total += rss
        count += 1
        if process.pid == pid:
            root_rss = rss
    return total / 2**30, root_rss / 2**30, count


def system_stats() -> dict[str, float | str | None]:
    return read_macos_memory_stats()


def atomic_write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".monitor.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_state(state_path: Path, pid: int) -> None:
    tree_rss, root_rss, process_count = process_tree(pid)
    virtual = psutil.virtual_memory()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    resources = state.setdefault("resources", {})
    previous_swap = resources.get("swap_used_gb")
    system = system_stats()
    if isinstance(previous_swap, (int, float)) and isinstance(system.get("swap_used_gb"), (int, float)):
        system["swap_growth_gb"] = float(system["swap_used_gb"]) - float(previous_swap)
    resources.update(
        {
            "pid": pid,
            "rss_gb": tree_rss,
            "root_rss_gb": root_rss,
            "process_count": process_count,
            "available_gb": virtual.available / 2**30,
            "memory_percent": float(virtual.percent),
            **system,
        }
    )
    # MPS counters are process-local.  A sidecar must preserve the latest
    # in-process values and only refresh system/process-tree telemetry.
    resources.setdefault("mps_available", None)
    resources.setdefault("mps_status", "unknown")
    resources.setdefault("mps_allocated_gb", None)
    resources.setdefault("mps_driver_gb", None)
    driver = resources.get("mps_driver_gb")
    resources["tracked_gb"] = tree_rss + (float(driver) if driver is not None else 0.0)
    resources["telemetry_source"] = "resource_monitor:process_tree+macos_system; mps=in_process_snapshot"
    state["resources"] = resources
    atomic_write(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--state", type=Path, default=Path("dashboard/state.json"))
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    state_path = args.state.resolve()
    while True:
        try:
            process = psutil.Process(args.pid)
            if not process.is_running():
                break
            update_state(state_path, args.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, json.JSONDecodeError):
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
