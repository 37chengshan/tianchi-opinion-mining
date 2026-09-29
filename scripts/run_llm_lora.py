#!/usr/bin/env python3
"""Launch mlx-lm LoRA/QLoRA training for the ACOS chat dataset.

Hyperparameters mirror the public reference recipe that reached 0.75 with
Qwen3-4B on this competition (LoRA rank 16 / alpha 32 / dropout 0.05,
lr 1e-4 cosine with warmup, batch 4 x grad-accum 8, 3 epochs), adapted to
MLX 4-bit training on one Apple GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "artifacts/llm_data/fold1_3fold"
DEFAULT_ADAPTERS = ROOT / "artifacts/llm_adapters"


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(args.data_dir),
        "--adapter-path",
        str(args.adapter_dir),
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation-steps",
        str(args.grad_accumulation),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.learning_rate),
        "--num-layers",
        str(args.num_layers),
        "--max-seq-length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--save-every",
        str(args.save_every),
        "--val-batches",
        str(args.val_batches),
    ]
    if args.mask_prompt:
        command.append("--mask-prompt")
    if args.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def _train_lines(data_dir: Path) -> int:
    path = data_dir / "train.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--iters", type=int, default=None, help="override the epoch-derived iteration count")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-report", type=int, default=20)
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--mask-prompt", action="store_true", default=True)
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.adapter_dir is None:
        args.adapter_dir = DEFAULT_ADAPTERS / f"{args.model.split('/')[-1]}_{args.data_dir.name}"
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    train_lines = _train_lines(args.data_dir)
    if args.iters is None:
        args.iters = max(1, int(-(-train_lines * args.epochs // args.batch_size)))
    command = build_command(args)
    meta = {
        "command": command,
        "model": args.model,
        "data_dir": str(args.data_dir),
        "adapter_dir": str(args.adapter_dir),
        "train_lines": train_lines,
        "effective_samples": int(args.iters * args.batch_size),
        "epochs": args.epochs,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "max_seq_length": args.max_seq_length,
            "mask_prompt": args.mask_prompt,
            "grad_checkpoint": args.grad_checkpoint,
        },
    }
    (args.adapter_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    mlx_lora = shutil.which("mlx_lm.lora")
    completed = subprocess.run(command, check=False)
    print(json.dumps({"exit_code": completed.returncode, "mlx_lm_lora_on_path": mlx_lora}), flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
