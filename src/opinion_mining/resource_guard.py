from __future__ import annotations

"""Conservative macOS/MPS resource telemetry and automatic downshifts."""

from dataclasses import asdict, dataclass
import os
import re
import subprocess
import time
from typing import Any

import psutil
import torch


@dataclass
class ResourceSnapshot:
    timestamp: float
    pid: int
    rss_gb: float
    root_rss_gb: float
    process_count: int
    tracked_gb: float
    available_gb: float
    memory_percent: float
    pressure: str
    pressure_free_percent: float | None
    swap_used_gb: float | None
    swap_total_gb: float | None
    swap_growth_gb: float | None
    compressed_gb: float | None
    mps_allocated_gb: float | None
    mps_driver_gb: float | None
    mps_available: bool | None
    mps_status: str
    mps_error: str | None
    memory_budget_gb: float


def _size_to_gb(value: str, unit: str) -> float:
    factors = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    return float(value) * factors[unit.upper()]


def _parse_swapusage(text: str) -> tuple[float | None, float | None]:
    matches = {
        key: re.search(rf"{key}\s*=\s*([0-9.]+)\s*([KMGT])", text, flags=re.IGNORECASE)
        for key in ("total", "used")
    }
    if not all(matches.values()):
        return None, None
    return tuple(_size_to_gb(match.group(1), match.group(2)) for match in (matches["used"], matches["total"]))


def _parse_compressed_gb(text: str) -> float | None:
    page = re.search(r"page size of\s*(\d+)\s*bytes", text, flags=re.IGNORECASE)
    compressed = re.search(r"Pages stored in compressor:\s*(\d+)", text, flags=re.IGNORECASE)
    if not page or not compressed:
        return None
    return int(page.group(1)) * int(compressed.group(1)) / 2**30


def read_macos_memory_stats() -> dict[str, float | str | None]:
    """Read system memory telemetry without inventing values when unavailable."""

    result: dict[str, float | str | None] = {
        "pressure": "unknown",
        "pressure_free_percent": None,
        "swap_used_gb": None,
        "swap_total_gb": None,
        "compressed_gb": None,
    }
    try:
        pressure = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=2)
        text = pressure.stdout + pressure.stderr
        match = re.search(
            r"(?:system-wide\s+)?memory\s+free\s+percentage\s*:\s*([0-9.]+)%",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            free_percent = float(match.group(1))
            result["pressure_free_percent"] = free_percent
            result["pressure"] = "critical" if free_percent <= 10 else "warn" if free_percent <= 20 else "normal"
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass

    try:
        swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=2)
        used, total = _parse_swapusage(swap.stdout + swap.stderr)
        result["swap_used_gb"] = used
        result["swap_total_gb"] = total
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass

    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
        result["compressed_gb"] = _parse_compressed_gb(vm.stdout + vm.stderr)
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass
    return result


class ResourceGuard:
    def __init__(self, *, memory_budget_gb: float = 12.0, soft_rss_gb: float = 10.0, hard_rss_gb: float = 12.0):
        self.memory_budget_gb = memory_budget_gb
        self.soft_rss_gb = soft_rss_gb
        self.hard_rss_gb = hard_rss_gb
        self._last_swap_used_gb: float | None = None

    @staticmethod
    def _mac_memory_stats() -> dict[str, float | str | None]:
        """Read macOS pressure, swap, and compressor counters when available.

        PyTorch's MPS counters are process-local, so these system counters are
        intentionally kept separate.  The distinction makes it possible to
        tell whether a low Python RSS is hiding unified-memory pressure.
        """
        return read_macos_memory_stats()

    @staticmethod
    def _pressure() -> str:
        # Kept as a small compatibility helper for callers that only need the
        # categorical state.  snapshot() also exposes the parsed percentage.
        return str(ResourceGuard._mac_memory_stats()["pressure"])

    @staticmethod
    def _process_tree_rss() -> tuple[float, float, int]:
        """Return (tree RSS, root RSS, process count) in GiB.

        DataLoader workers and tokenizer helpers may be children of the
        training process.  Summing their RSS gives a conservative process
        footprint; the root value remains visible for diagnosis.
        """
        root = psutil.Process(os.getpid())
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
            if process.pid == root.pid:
                root_rss = rss
        return total / 2**30, root_rss / 2**30, count

    @staticmethod
    def _mps_stats() -> tuple[float | None, float | None, bool | None, str, str | None]:
        if not hasattr(torch, "mps"):
            return None, None, None, "unavailable", "torch.mps is missing"
        try:
            available = bool(torch.backends.mps.is_available())
        except Exception as error:  # monitoring must not break training
            return None, None, None, "error", f"availability check: {type(error).__name__}: {error}"
        if not available:
            return None, None, False, "unavailable", "torch.backends.mps.is_available() is false"
        current_fn = getattr(torch.mps, "current_allocated_memory", None)
        driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
        if current_fn is None or driver_fn is None:
            return None, None, True, "unsupported", "PyTorch MPS memory APIs are unavailable"
        try:
            return float(current_fn()) / 2**30, float(driver_fn()) / 2**30, True, "ok", None
        except Exception as error:  # RuntimeError and version-specific backend errors
            return None, None, True, "error", f"MPS counter read: {type(error).__name__}: {error}"

    def snapshot(self) -> ResourceSnapshot:
        virtual = psutil.virtual_memory()
        tree_rss, root_rss, process_count = self._process_tree_rss()
        system = self._mac_memory_stats()
        mps_allocated, mps_driver, mps_available, mps_status, mps_error = self._mps_stats()
        swap_used = system["swap_used_gb"]
        swap_growth = None
        if isinstance(swap_used, (int, float)) and self._last_swap_used_gb is not None:
            swap_growth = float(swap_used) - self._last_swap_used_gb
        if isinstance(swap_used, (int, float)):
            self._last_swap_used_gb = float(swap_used)
        # MPS driver allocations live in unified memory and may not be present
        # in Python RSS.  Keep the sum as a conservative guard metric while
        # exposing both components separately in the Dashboard.
        tracked = tree_rss + (mps_driver or 0.0)
        return ResourceSnapshot(
            timestamp=time.time(),
            pid=os.getpid(),
            rss_gb=tree_rss,
            root_rss_gb=root_rss,
            process_count=process_count,
            tracked_gb=tracked,
            available_gb=virtual.available / 2**30,
            memory_percent=float(virtual.percent),
            pressure=str(system["pressure"]),
            pressure_free_percent=system["pressure_free_percent"],
            swap_used_gb=swap_used,
            swap_total_gb=system["swap_total_gb"],
            swap_growth_gb=swap_growth,
            compressed_gb=system["compressed_gb"],
            mps_allocated_gb=mps_allocated,
            mps_driver_gb=mps_driver,
            mps_available=mps_available,
            mps_status=mps_status,
            mps_error=mps_error,
            memory_budget_gb=self.memory_budget_gb,
        )

    def _recommend_from_snapshot(self, config: Any, current: Any) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        # On unified memory, available system memory is the first control
        # signal.  RSS/MPS thresholds cap the process footprint but never
        # override a system-level pressure signal.
        swap_growth = current.swap_growth_gb or 0.0
        # Keep working near 95% system usage when pressure is normal, but
        # reserve a hard stop for a critically small reserve or rapidly
        # growing swap.  The trainer checks the same decision at step level.
        severe = (
            current.pressure == "critical"
            or current.tracked_gb >= self.hard_rss_gb
            or current.memory_percent >= 98.0
            or swap_growth >= 1.5
        )
        pressured = (
            severe
            or current.pressure == "warn"
            or current.tracked_gb >= self.soft_rss_gb
            or current.memory_percent >= 95.0
            or swap_growth >= 1.0
        )
        if severe:
            changes["batch_size"] = 1
            changes["max_length"] = min(int(getattr(config, "max_length", 96)), 80)
            changes["trainable_layers"] = min(int(getattr(config, "trainable_layers", 2)), 1)
            changes["gradient_accumulation"] = max(4, int(getattr(config, "gradient_accumulation", 2)))
        elif current.memory_percent >= 95.0 or current.tracked_gb >= self.soft_rss_gb or swap_growth >= 1.0:
            changes["batch_size"] = 1
            changes["max_length"] = min(int(getattr(config, "max_length", 128)), 96)
            changes["trainable_layers"] = min(int(getattr(config, "trainable_layers", 3)), 6)
            changes["gradient_accumulation"] = max(4, int(getattr(config, "gradient_accumulation", 2)))
        elif current.memory_percent >= 90.0 or current.tracked_gb >= 8.0 or current.available_gb < 1.5:
            changes["batch_size"] = min(2, int(getattr(config, "batch_size", 4)))
            changes["max_length"] = min(int(getattr(config, "max_length", 128)), 112)
            changes["trainable_layers"] = min(int(getattr(config, "trainable_layers", 3)), 8)
            changes["gradient_accumulation"] = max(3, int(getattr(config, "gradient_accumulation", 2)))
        elif pressured:
            changes["batch_size"] = min(2, int(getattr(config, "batch_size", 4)))
            changes["max_length"] = int(getattr(config, "max_length", 128))
            changes["trainable_layers"] = int(getattr(config, "trainable_layers", 3))
            changes["gradient_accumulation"] = max(2, int(getattr(config, "gradient_accumulation", 2)))
        else:
            changes["batch_size"] = int(getattr(config, "batch_size", 4))
            changes["max_length"] = int(getattr(config, "max_length", 128))
            changes["trainable_layers"] = int(getattr(config, "trainable_layers", 3))
            changes["gradient_accumulation"] = int(getattr(config, "gradient_accumulation", 2))
        if pressured or severe:
            self.release_cache()
        return changes

    def recommend(self, config: Any) -> dict[str, Any]:
        return self._recommend_from_snapshot(config, self.snapshot())

    def runtime_decision(self, config: Any) -> dict[str, Any]:
        """Return one atomic decision suitable for mid-epoch resource checks."""
        current = self.snapshot()
        swap_growth = current.swap_growth_gb or 0.0
        severe = (
            current.pressure == "critical"
            or current.tracked_gb >= self.hard_rss_gb
            or current.memory_percent >= 98.0
            or swap_growth >= 1.5
        )
        return {
            "restart_required": bool(severe),
            "reason": "red_memory_pressure" if severe else "continue",
            "recommendation": self._recommend_from_snapshot(config, current),
            "snapshot": current,
        }

    @staticmethod
    def release_cache() -> None:
        import gc

        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot())
