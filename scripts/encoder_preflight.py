#!/usr/bin/env python3
"""Read-only preflight for the strong Chinese encoders and MLX challenger.

The default path only inspects local files, installed distributions, and host
capabilities.  ``--probe-load`` is an explicit opt-in: it loads only a model
directory that this script has already proved complete and local, with
``local_files_only=True``, then deletes the objects immediately.

This script never calls the Hub, downloads, installs, trains, writes model
artifacts, or starts a generation process.  It emits one JSON document so the
result can be copied into a research note without turning an assumption into
an availability claim.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
GIB = 1024 ** 3
MIN_TOTAL_GIB = 16.0
RED_AVAILABLE_GIB = 3.0
GREEN_AVAILABLE_GIB = 5.0

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "macbert_base": {
        "model_id": "hfl/chinese-macbert-base",
        "model_card": "https://huggingface.co/hfl/chinese-macbert-base",
    },
    "roberta_wwm_ext_base": {
        "model_id": "hfl/chinese-roberta-wwm-ext",
        "model_card": "https://huggingface.co/hfl/chinese-roberta-wwm-ext",
    },
    "current_baseline": {
        "model_id": "hfl/rbt3",
        "model_card": "https://huggingface.co/hfl/rbt3",
    },
}

PACKAGE_SPECS: Dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece",
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}


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


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _gib(value: Optional[int]) -> Optional[float]:
    return round(value / GIB, 2) if value is not None else None


def _module_status(import_name: str, distribution: str) -> Dict[str, Any]:
    try:
        present = importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        present = False
    version: Optional[str] = None
    if present:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {"present": present, "version": version, "distribution": distribution}


def _memory_snapshot() -> Dict[str, Any]:
    """Read host memory without allocating tensors or importing MLX."""

    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    sources: List[str] = []
    free_percent: Optional[float] = None
    swap_used_gib: Optional[float] = None
    swap_total_gib: Optional[float] = None

    if platform.system() == "Darwin":
        total_bytes = _parse_int(_run_text(("sysctl", "-n", "hw.memsize")))
        if total_bytes is not None:
            sources.append("sysctl:hw.memsize")

        vm_stat = _run_text(("vm_stat",))
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_match.group(1)) if page_match else None
        if page_size is not None:
            pages: Dict[str, int] = {}
            for line in vm_stat.splitlines():
                match = re.match(r"^Pages ([^:]+):\s+(\d+)", line)
                if match:
                    pages[match.group(1).strip().lower()] = int(match.group(2))
            available_pages = sum(
                pages.get(name, 0) for name in ("free", "inactive", "speculative")
            )
            if available_pages:
                available_bytes = available_pages * page_size
                sources.append("vm_stat:free+inactive+speculative")

        pressure = _run_text(("memory_pressure", "-Q"))
        match = re.search(r"free percentage:\s*(\d+)%", pressure, flags=re.I)
        if match:
            free_percent = float(match.group(1))

        swap = _run_text(("sysctl", "vm.swapusage"))
        total_match = re.search(r"total\s*=\s*([0-9.]+)M", swap)
        used_match = re.search(r"used\s*=\s*([0-9.]+)M", swap)
        if total_match and used_match:
            swap_total_gib = round(float(total_match.group(1)) / 1024, 2)
            swap_used_gib = round(float(used_match.group(1)) / 1024, 2)

    if total_bytes is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            page_size = physical_pages = available_pages = 0
        if page_size and physical_pages:
            total_bytes = page_size * physical_pages
            sources.append("sysconf:physical")
        if page_size and available_pages:
            available_bytes = page_size * available_pages
            sources.append("sysconf:available")

    available_gib = _gib(available_bytes)
    if available_gib is None:
        band = "unknown"
    elif available_gib < RED_AVAILABLE_GIB:
        band = "red"
    elif available_gib < GREEN_AVAILABLE_GIB:
        band = "yellow"
    else:
        band = "green"

    return {
        "total_gib": _gib(total_bytes),
        "available_gib_approx": available_gib,
        "available_band": band,
        "memory_pressure_free_percent": free_percent,
        "swap_used_gib": swap_used_gib,
        "swap_total_gib": swap_total_gib,
        "source": sources,
    }


def _torch_mps_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "torch_imported_for_check": False,
        "version": None,
        "built": None,
        "available": None,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        return status
    status["torch_imported_for_check"] = True
    status["version"] = str(getattr(torch, "__version__", "unknown"))
    try:
        status["built"] = bool(torch.backends.mps.is_built())
        status["available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:
        status["error_type"] = type(exc).__name__
    return status


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(Path(key))
    return result


def _cache_roots() -> List[Path]:
    roots: List[Path] = []
    for variable in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        roots.append(Path(xdg_cache) / "huggingface" / "hub")
    roots.extend(
        (
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "Library" / "Caches" / "huggingface" / "hub",
        )
    )
    return _unique_paths(roots)


def _candidate_paths(model_id: str, explicit_cache: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if explicit_cache:
        candidates.append(Path(explicit_cache))

    model_path = Path(model_id).expanduser()
    if model_path.exists():
        candidates.append(model_path)

    repo_root = Path(__file__).resolve().parents[1]
    model_name = model_id.rsplit("/", 1)[-1]
    cache_key = "models--" + model_id.replace("/", "--")
    for root in (repo_root, Path.cwd(), *_cache_roots()):
        candidates.extend((root / model_id, root / model_name, root / cache_key))
        snapshot_root = root / cache_key / "snapshots"
        try:
            candidates.extend(child for child in snapshot_root.iterdir() if child.is_dir())
        except OSError:
            pass
    return _unique_paths(candidates)


def _file_inventory(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        return {"directory": False, "complete": False, "files": []}
    required = {"config": (path / "config.json").is_file()}
    tokenizer_names = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.txt",
        "vocab.json",
    )
    required["tokenizer"] = any((path / name).is_file() for name in tokenizer_names)
    weight_names = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    required["weights"] = any((path / name).is_file() for name in weight_names)
    if not required["weights"]:
        try:
            required["weights"] = any(
                item.is_file() and item.suffix == ".safetensors"
                for item in path.iterdir()
            )
        except OSError:
            required["weights"] = False
    visible_files: List[str] = []
    try:
        visible_files = sorted(item.name for item in path.iterdir() if item.is_file())
    except OSError:
        pass
    return {
        "directory": True,
        "complete": all(required.values()),
        "required": required,
        "files": visible_files[:80],
    }


def _config_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys = (
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
    )
    return {key: config[key] for key in keys if key in config}


def _model_cache_check(
    model_id: str,
    model_card: str,
    explicit_cache: Optional[str] = None,
) -> Dict[str, Any]:
    paths = _candidate_paths(model_id, explicit_cache)
    partial: List[Dict[str, Any]] = []
    for path in paths:
        inventory = _file_inventory(path)
        if not inventory.get("directory"):
            continue
        if inventory.get("complete"):
            return {
                "model_id": model_id,
                "model_card": model_card,
                "state": "complete",
                "cached": True,
                "path": str(path),
                "inventory": inventory,
                "config": _config_summary(path),
            }
        partial.append({"path": str(path), "inventory": inventory})
    return {
        "model_id": model_id,
        "model_card": model_card,
        "state": "partial" if partial else "absent",
        "cached": False,
        "path": None,
        "partial_candidates": partial[:8],
    }


def _probe_load(record: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """Load one already-cached local model, then release it immediately."""

    result: Dict[str, Any] = {
        "requested": record.get("model_id"),
        "attempted": False,
        "released": False,
        "device": device,
    }
    path_text = record.get("path")
    if not record.get("cached") or not path_text:
        result["reason"] = "model_not_cached"
        return result

    result["attempted"] = True
    model: Any = None
    tokenizer: Any = None
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        local_path = str(Path(str(path_text)).resolve())
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            use_fast=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            local_path,
            local_files_only=True,
        )
        if device != "cpu":
            model.to(device)
        result["passed"] = True
        result["hidden_size"] = int(getattr(model.config, "hidden_size", 0))
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:
        result["passed"] = False
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:240]
    finally:
        del model
        del tokenizer
        gc.collect()
        result["released"] = True
        try:
            import torch  # type: ignore

            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
                result["mps_empty_cache_called"] = True
        except Exception as exc:
            result["release_error_type"] = type(exc).__name__
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only strong encoder preflight; no download, install, or training."
    )
    parser.add_argument(
        "--probe-load",
        action="store_true",
        help="Attempt AutoModel loading only for complete local caches, then release immediately.",
    )
    parser.add_argument(
        "--probe-device",
        choices=("cpu", "mps"),
        default="cpu",
        help="Device for an explicit probe-load; CPU is the safe default.",
    )
    parser.add_argument(
        "--model-cache",
        default=None,
        help="Optional complete local directory for the single --model-id being checked.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model id to pair with --model-cache; defaults to all tracked models.",
    )
    return parser.parse_args()


def _emit(payload: Dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    args = _arguments()
    if args.model_cache and not args.model_id:
        raise SystemExit("--model-cache requires --model-id so the local directory cannot be mislabelled")

    specs = MODEL_SPECS
    if args.model_id:
        matches = [spec for spec in MODEL_SPECS.values() if spec["model_id"] == args.model_id]
        spec = matches[0] if matches else {"model_id": args.model_id, "model_card": ""}
        specs = {"requested": spec}

    packages = {
        key: _module_status(import_name, distribution)
        for key, (import_name, distribution) in (
            (key, (key, distribution)) for key, distribution in PACKAGE_SPECS.items()
        )
    }
    memory = _memory_snapshot()
    host = {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
    }
    mps = _torch_mps_status()

    cache: Dict[str, Dict[str, Any]] = {}
    for name, spec in specs.items():
        cache[name] = _model_cache_check(
            spec["model_id"],
            spec["model_card"],
            explicit_cache=args.model_cache if args.model_id == spec["model_id"] else None,
        )

    probe: List[Dict[str, Any]] = []
    if args.probe_load:
        total_gib = memory.get("total_gib")
        available_gib = memory.get("available_gib_approx")
        if total_gib is not None and total_gib < MIN_TOTAL_GIB:
            probe = [
                {
                    "attempted": False,
                    "reason": "host_memory_below_16_gib_gate",
                    "released": True,
                }
            ]
        elif available_gib is not None and available_gib < RED_AVAILABLE_GIB:
            probe = [
                {
                    "attempted": False,
                    "reason": "available_memory_below_red_gate",
                    "released": True,
                }
            ]
        else:
            for record in cache.values():
                probe.append(_probe_load(record, args.probe_device))

    target_names = {"macbert_base", "roberta_wwm_ext_base"}
    target_cache = [cache[name] for name in cache if name in target_names]
    missing_packages = [
        name for name in ("torch", "transformers") if not packages[name]["present"]
    ]
    target_cached = any(record.get("cached") for record in target_cache)
    probe_failed = any(item.get("attempted") and not item.get("passed") for item in probe)

    if missing_packages:
        status = "queue"
        reason = "missing_encoder_dependency"
    elif not target_cached:
        status = "queue"
        reason = "strong_encoder_models_not_cached"
    elif not bool(mps.get("available")):
        status = "queue"
        reason = "mps_unavailable_cpu_only"
    elif probe_failed:
        status = "queue"
        reason = "cached_model_probe_failed"
    else:
        status = "ready"
        reason = "preflight_checks_passed"

    return _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "tool": "encoder_preflight",
            "status": status,
            "reason": reason,
            "host": host,
            "memory": memory,
            "mps": mps,
            "dependencies": packages,
            "models": cache,
            "probe_load": {
                "requested": bool(args.probe_load),
                "device": args.probe_device,
                "results": probe,
                "network_accessed": False,
                "downloads_attempted": False,
                "training_started": False,
            },
            "gates": {
                "total_memory_at_least_16_gib": memory.get("total_gib") is not None
                and float(memory["total_gib"]) >= MIN_TOTAL_GIB,
                "available_memory_red_gate_gib": RED_AVAILABLE_GIB,
                "available_memory_green_gate_gib": GREEN_AVAILABLE_GIB,
                "transformers_and_torch_present": not missing_packages,
                "at_least_one_target_encoder_cached": target_cached,
                "mps_available": bool(mps.get("available")),
                "probe_failed": probe_failed,
            },
            "next_action": (
                "Make a target encoder available locally, then re-run; this script will not download it."
                if not target_cached
                else "Use an explicitly reviewed fold-local experiment; this preflight does not start training."
            ),
        }
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            _emit(
                {
                    "schema_version": SCHEMA_VERSION,
                    "tool": "encoder_preflight",
                    "status": "skip",
                    "reason": "preflight_error",
                    "error_type": type(exc).__name__,
                }
            )
        )
