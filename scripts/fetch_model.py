#!/usr/bin/env python3
"""Fetch a pretrained backbone from ModelScope (Alibaba Cloud 魔搭) or Hugging Face.

Why this exists
---------------
The competition runs on Tianchi (Alibaba Cloud) and the backbones used by this
repository are mirrored on ModelScope. Fetching through ModelScope keeps the
reproduction path inside the Alibaba Cloud ecosystem (domestic direct download,
no proxy) while the Hugging Face route stays available for other environments.
The two sources are the same upstream weights; this script only changes the
transport and caches them in one local directory.

Usage
-----
    PYTHONPATH=src python3 scripts/fetch_model.py --model hfl/rbt3 --out models/rbt3
    PYTHONPATH=src python3 scripts/fetch_model.py --model hfl/chinese-roberta-wwm-ext \\
        --source modelscope --out models/wwm
    PYTHONPATH=src python3 scripts/fetch_model.py --model Qwen/Qwen3-4B --source modelscope

It prints a JSON summary so a run can be pasted into an experiment log.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Verified ModelScope mirrors of the backbones this repository trains.
# Keys are Hugging Face ids, values are ModelScope ids.
MODELSCOPE_MIRRORS = {
    "hfl/rbt3": "dienstag/rbt3",
    "hfl/chinese-roberta-wwm-ext": "dienstag/chinese-roberta-wwm-ext",
}

# Namespaces that publish under the same id on both hubs.
PASSTHROUGH_PREFIXES = ("Qwen/", "Qwen2/", "Qwen3/")

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".h5", ".msgpack")


def modelscope_id_for(model: str, explicit: str | None = None) -> str:
    """Return the ModelScope id for a Hugging Face id.

    Raises ValueError when no mirror is known, so the caller never silently
    downloads an unrelated repository.
    """
    if explicit:
        return explicit
    if model in MODELSCOPE_MIRRORS:
        return MODELSCOPE_MIRRORS[model]
    if model.startswith(PASSTHROUGH_PREFIXES):
        return model
    raise ValueError(
        f"no ModelScope mirror known for {model!r}; pass --modelscope-id to override"
    )


def inspect_dir(path: pathlib.Path) -> dict:
    """Summarize a downloaded model directory and assert it is loadable."""
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    weights = [p for p in files if p.suffix in WEIGHT_SUFFIXES]
    has_config = (path / "config.json").exists()
    has_tokenizer = any(
        (path / name).exists()
        for name in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "spiece.model")
    )
    total_bytes = sum(p.stat().st_size for p in files)
    problems = []
    if not has_config:
        problems.append("config.json missing")
    if not weights:
        problems.append("no weight file found")
    if not has_tokenizer:
        problems.append("no tokenizer file found")
    return {
        "path": str(path),
        "file_count": len(files),
        "weight_files": [p.name for p in weights],
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "has_config": has_config,
        "has_tokenizer": has_tokenizer,
        "problems": problems,
    }


def fetch_hf(model: str, out: pathlib.Path) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, local_dir=str(out))


def fetch_modelscope(model: str, out: pathlib.Path) -> str:
    from modelscope import snapshot_download

    return snapshot_download(model, local_dir=str(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model id as used by the training scripts")
    parser.add_argument("--source", choices=("modelscope", "hf"), default="modelscope",
                        help="transport; default modelscope (Alibaba Cloud 魔搭)")
    parser.add_argument("--modelscope-id", default=None, help="override the ModelScope repo id")
    parser.add_argument("--out", default=None, help="target directory (default models/<name>)")
    parser.add_argument("--force", action="store_true", help="re-download even when present")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out) if args.out else pathlib.Path("models") / args.model.split("/")[-1]

    summary: dict = {"model": args.model, "source": args.source, "out": str(out)}
    try:
        resolved = modelscope_id_for(args.model, args.modelscope_id) if args.source == "modelscope" else args.model
    except ValueError as exc:
        print(json.dumps({**summary, "status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    summary["resolved_id"] = resolved

    if out.exists() and not args.force:
        report = inspect_dir(out)
        status = "cached" if not report["problems"] else "cached-incomplete"
        print(json.dumps({**summary, "status": status, **report}, ensure_ascii=False, indent=2))
        return 0 if not report["problems"] else 1

    try:
        if args.source == "modelscope":
            path = fetch_modelscope(resolved, out)
        else:
            path = fetch_hf(resolved, out)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as JSON
        print(json.dumps({**summary, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]},
                         ensure_ascii=False, indent=2))
        return 1

    report = inspect_dir(pathlib.Path(path))
    payload = {**summary, "status": "downloaded" if not report["problems"] else "downloaded-incomplete", **report}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
