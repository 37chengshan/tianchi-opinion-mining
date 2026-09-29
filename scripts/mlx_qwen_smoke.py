#!/usr/bin/env python3
"""Safe preflight and import/config smoke for the Qwen3-4B MLX challenger.

This entry point deliberately has no model-loading, generation, training,
download, installation, or MLX tensor operations.  It first inspects module
availability, local model caches, and host memory.  Only when all hard gates
pass does it import ``mlx``/``mlx_lm`` and build the LoRA argument parser.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"
GIB = 1024 ** 3
MIN_TOTAL_GIB = 16.0
RED_AVAILABLE_GIB = 3.0
GREEN_AVAILABLE_GIB = 5.0


def _module_available(name: str) -> bool:
    """Check a module spec without importing or initializing the module."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _run_text(command: Sequence[str], timeout: float = 1.5) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _memory_snapshot() -> Dict[str, Any]:
    """Read memory using OS facilities without importing MLX or allocating tensors."""

    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    source: List[str] = []

    if platform.system() == "Darwin":
        total_bytes = _parse_int(_run_text(("sysctl", "-n", "hw.memsize")))
        if total_bytes is not None:
            source.append("sysctl:hw.memsize")

        vm_stat = _run_text(("vm_stat",))
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_match.group(1)) if page_match else None
        if page_size is not None:
            pages: Dict[str, int] = {}
            for line in vm_stat.splitlines():
                match = re.match(r"^Pages ([^:]+):\s+(\d+)", line)
                if match:
                    pages[match.group(1).strip().lower()] = int(match.group(2))
            # This is an intentionally conservative approximation.  It does
            # not query MPS and is used only to stop unsafe smoke attempts.
            available_pages = sum(
                pages.get(name, 0)
                for name in ("free", "inactive", "speculative")
            )
            if available_pages:
                available_bytes = available_pages * page_size
                source.append("vm_stat:free+inactive+speculative")

    if total_bytes is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            page_size = physical_pages = available_pages = 0
        if page_size and physical_pages:
            total_bytes = page_size * physical_pages
            source.append("sysconf:physical")
        if page_size and available_pages:
            available_bytes = page_size * available_pages
            source.append("sysconf:available")

    def gib(value: Optional[int]) -> Optional[float]:
        return round(value / GIB, 2) if value is not None else None

    available_gib = gib(available_bytes)
    if available_gib is None:
        band = "unknown"
    elif available_gib < RED_AVAILABLE_GIB:
        band = "red"
    elif available_gib < GREEN_AVAILABLE_GIB:
        band = "yellow"
    else:
        band = "green"

    return {
        "total_gib": gib(total_bytes),
        "available_gib_approx": available_gib,
        "available_band": band,
        "source": source,
    }


def _model_files_present(path: Path) -> bool:
    """Recognize a local MLX model without reading any weight contents."""

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    has_weights = (path / "model.safetensors").is_file()
    if not has_weights:
        has_weights = (path / "model.safetensors.index.json").is_file()
    if not has_weights:
        try:
            has_weights = any(
                item.is_file() and item.suffix == ".safetensors"
                for item in path.iterdir()
            )
        except OSError:
            has_weights = False
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    return has_weights and has_tokenizer


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            result.append(Path(key))
    return result


def _cache_roots() -> List[Path]:
    roots: List[Path] = []
    for variable in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        roots.append(Path(xdg_cache).expanduser() / "huggingface" / "hub")

    roots.extend(
        (
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "Library" / "Caches" / "huggingface" / "hub",
        )
    )
    return _unique_paths(roots)


def _model_candidates(model_ref: str, explicit_cache: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if explicit_cache:
        candidates.append(Path(explicit_cache).expanduser())

    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        candidates.append(model_path)

    repo_root = Path(__file__).resolve().parents[1]
    model_name = model_ref.rsplit("/", 1)[-1]
    cache_key = "models--" + model_ref.replace("/", "--")
    for root in (repo_root, Path.cwd(), *_cache_roots()):
        candidates.extend(
            (
                root / model_ref,
                root / model_name,
                root / cache_key,
            )
        )
        snapshot_root = root / cache_key / "snapshots"
        try:
            candidates.extend(
                child for child in snapshot_root.iterdir() if child.is_dir()
            )
        except OSError:
            pass
    return _unique_paths(candidates)


def _model_cache_check(model_ref: str, explicit_cache: Optional[str]) -> Dict[str, Any]:
    for candidate in _model_candidates(model_ref, explicit_cache):
        if _model_files_present(candidate):
            return {
                "requested": model_ref,
                "cached": True,
                "path": str(candidate),
            }
    return {
        "requested": model_ref,
        "cached": False,
        "path": None,
    }


def _parser_option_strings(parser: Any) -> set[str]:
    options: set[str] = set()
    for action in getattr(parser, "_actions", ()):
        options.update(getattr(action, "option_strings", ()))
    return options


def _safe_version(module: Any) -> Optional[str]:
    value = getattr(module, "__version__", None)
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    return text[:80] or None


def _config_smoke(model_ref: str) -> Dict[str, Any]:
    """Import packages and build a parser; never load a model or create tensors."""

    mlx = importlib.import_module("mlx")
    mlx_lm = importlib.import_module("mlx_lm")
    lora = importlib.import_module("mlx_lm.lora")

    build_parser = getattr(lora, "build_parser", None)
    if not callable(build_parser):
        return {
            "passed": False,
            "reason": "mlx_lm_lora_has_no_build_parser",
        }

    parser = build_parser()
    expected = (
        "--model",
        "--batch-size",
        "--grad-accumulation-steps",
        "--max-seq-length",
        "--num-layers",
        "--grad-checkpoint",
    )
    available_options = _parser_option_strings(parser)
    missing_options = [name for name in expected if name not in available_options]
    if missing_options:
        return {
            "passed": False,
            "reason": "mlx_lm_lora_parser_missing_options",
            "missing_options": missing_options,
        }

    # Parsing these values validates the intended future configuration while
    # deliberately omitting --train, --test, generation, and all model I/O.
    parsed = parser.parse_args(
        [
            "--model",
            model_ref,
            "--batch-size",
            "1",
            "--grad-accumulation-steps",
            "8",
            "--max-seq-length",
            "512",
            "--num-layers",
            "4",
            "--grad-checkpoint",
        ]
    )
    return {
        "passed": True,
        "imports": {
            "mlx": _safe_version(mlx),
            "mlx_lm": _safe_version(mlx_lm),
            "mlx_lm.lora": True,
        },
        "config": {
            "model": getattr(parsed, "model", model_ref),
            "batch_size": getattr(parsed, "batch_size", None),
            "grad_accumulation_steps": getattr(parsed, "grad_accumulation_steps", None),
            "max_seq_length": getattr(parsed, "max_seq_length", None),
            "num_layers": getattr(parsed, "num_layers", None),
            "grad_checkpoint": bool(getattr(parsed, "grad_checkpoint", False)),
        },
        "model_loaded": False,
        "generated": False,
        "trained": False,
        "mlx_tensor_operations": False,
        "mps_allocations_attempted": False,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Qwen3-4B MLX preflight; never downloads, loads, trains, or generates."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MLX_QWEN_MODEL") or DEFAULT_MODEL,
        help="A local model path or model id used for cache lookup only.",
    )
    parser.add_argument(
        "--model-cache",
        default=None,
        help="Optional explicit local model directory to inspect.",
    )
    return parser.parse_args()


def _payload(
    status: str,
    reason: str,
    checks: Dict[str, Any],
    **extra: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "mlx_qwen_smoke",
        "status": status,
        "reason": reason,
        "checks": checks,
    }
    result.update(extra)
    return result


def main() -> int:
    args = _arguments()

    # Keep this order intentional: all preflight checks happen before either
    # MLX package is imported.  Cache lookup is filesystem-only and never asks
    # Hugging Face for metadata or weights.
    dependencies = {
        "mlx": _module_available("mlx"),
        "mlx_lm": _module_available("mlx_lm"),
    }
    cache = _model_cache_check(args.model, args.model_cache)
    memory = _memory_snapshot()
    host = {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
    }
    checks = {
        "host": host,
        "dependencies": dependencies,
        "model_cache": cache,
        "memory": memory,
    }

    if host["system"] != "Darwin" or str(host["machine"]).lower() not in {
        "arm64",
        "aarch64",
    }:
        return _emit(
            _payload(
                "skip",
                "unsupported_apple_silicon_host",
                checks,
                next_action="Run this challenger only on an Apple Silicon macOS host.",
            )
        )

    total_gib = memory.get("total_gib")
    if total_gib is not None and total_gib < MIN_TOTAL_GIB:
        return _emit(
            _payload(
                "skip",
                "host_memory_below_16_gib_gate",
                checks,
                next_action="Keep the Qwen3-4B challenger queued for a host with at least 16 GiB unified memory.",
            )
        )
    if total_gib is None or memory.get("available_gib_approx") is None:
        return _emit(
            _payload(
                "queue",
                "memory_measurement_unavailable",
                checks,
                next_action="Re-run when the host memory check is available; no MLX import was attempted.",
            )
        )
    if float(memory["available_gib_approx"]) < RED_AVAILABLE_GIB:
        return _emit(
            _payload(
                "skip",
                "available_memory_below_red_gate",
                checks,
                next_action="Stop and free memory; do not start a Qwen process while available memory is below 3 GiB.",
            )
        )

    missing = [name for name, present in dependencies.items() if not present]
    if missing:
        return _emit(
            _payload(
                "queue",
                "missing_dependency",
                checks,
                missing_dependencies=missing,
                next_action="Use an already prepared environment and re-run; this entry point never installs packages.",
            )
        )
    if not cache["cached"]:
        return _emit(
            _payload(
                "queue",
                "model_not_cached",
                checks,
                next_action="Make the 4-bit model available locally, then re-run; this entry point never downloads it.",
            )
        )

    try:
        smoke = _config_smoke(str(cache["path"]))
    except Exception as exc:  # Keep the JSON free of paths or remote error text.
        return _emit(
            _payload(
                "skip",
                "import_config_smoke_failed",
                checks,
                error_type=type(exc).__name__,
                next_action="Inspect the local MLX/MLX-LM environment outside this entry point, then re-run.",
            )
        )

    if not smoke.get("passed"):
        return _emit(
            _payload(
                "skip",
                str(smoke.get("reason", "config_smoke_failed")),
                checks,
                smoke=smoke,
            )
        )
    return _emit(
        _payload(
            "ready",
            "import_config_smoke_passed",
            checks,
            smoke=smoke,
            next_action="Only an explicitly approved, separately monitored training run may follow this smoke.",
        )
    )


def _emit(result: Dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Preserve the machine-readable contract even if an OS-specific probe
        # fails unexpectedly.  Do not include exception text, which can carry
        # paths or remote configuration details.
        raise SystemExit(
            _emit(
                {
                    "schema_version": 1,
                    "tool": "mlx_qwen_smoke",
                    "status": "skip",
                    "reason": "preflight_error",
                    "error_type": type(exc).__name__,
                }
            )
        )
