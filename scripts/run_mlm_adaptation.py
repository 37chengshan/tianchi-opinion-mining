#!/usr/bin/env python3
"""Task-adaptive MLM continuation (TAPT) for the competition backbone.

Runs masked-language-model continuation on the competition's own training
review text.  This is the safe half of the locked plan's Task 5: no external
corpus and no test text are used, so no rules gate is needed.  The adapted
checkpoint is written to a self-contained directory that the existing grid CLI
can consume via ``--model-name <dir>``; training itself still runs with
``local_files_only=True``.

The script is independent of ``grid_model.py`` / ``grid_trainer.py``: it only
reads review text and never touches relation heads.  It writes its own report
and never restarts the dashboard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.data import load_test_reviews
from opinion_mining.dashboard_runtime import DashboardWriter
from opinion_mining.resource_guard import ResourceGuard

MODEL_ALIASES = {
    "macbert": "hfl/chinese-macbert-base",
    "wwm": "hfl/chinese-roberta-wwm-ext",
    "rbt3": "hfl/rbt3",
}


def resolve_model_name(name: str) -> str:
    candidate = MODEL_ALIASES.get(name.strip().lower(), name)
    path = Path(candidate).expanduser()
    return str(path) if path.exists() else candidate


def build_corpus(rows: Sequence[Any]) -> list[str]:
    """Return non-empty review texts in a stable order.

    Only the competition's own training reviews are used, so no external-data
    or test-text rule check is required.
    """

    corpus: list[str] = []
    for row in rows:
        text = getattr(row, "text", None)
        if not isinstance(text, str):
            continue
        cleaned = text.strip()
        if cleaned:
            corpus.append(cleaned)
    return corpus


class _MaskedTextDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, texts: Sequence[str], tokenizer: Any, max_length: int):
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self.encodings = [
            tokenizer(text, truncation=True, max_length=max_length, padding=False)
            for text in texts
        ]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.encodings[index]
        return {key: list(value) for key, value in item.items()}


@dataclass
class _Report:
    model_name: str
    resolved_model_name: str
    corpus_size: int
    corpus_characters: int
    max_length: int
    mlm_probability: float
    learning_rate: float
    epochs_requested: int
    epochs_completed: int
    steps: int
    batch_size: int
    gradient_accumulation: int
    device: str
    losses: list[float]
    peak_memory: dict[str, float | None]
    minimum_available_gb: float | None
    elapsed_seconds: float
    output_dir: str

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        losses = payload["losses"]
        payload["initial_loss"] = losses[0] if losses else None
        payload["final_loss"] = losses[-1] if losses else None
        payload["mean_last_10_loss"] = (
            sum(losses[-10:]) / len(losses[-10:]) if losses else None
        )
        return payload


def _collate(batch: list[dict[str, list[int]]], tokenizer: Any, mlm_probability: float) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch, then mask.

    Implemented locally instead of via ``DataCollatorForLanguageModeling`` so
    the 80/10/10 masking split is explicit and testable.
    """

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a pad token")
    width = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    token_types: list[list[int]] | None = [] if any("token_type_ids" in item for item in batch) else None
    for item in batch:
        ids = list(item["input_ids"])
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append(list(item["attention_mask"]) + [0] * pad)
        if token_types is not None:
            types = list(item.get("token_type_ids", [0] * len(ids)))
            token_types.append(types + [0] * pad)
    tensor_ids = torch.tensor(input_ids, dtype=torch.long)
    tensor_mask = torch.tensor(attention, dtype=torch.long)

    labels = tensor_ids.clone()
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, pad_id}
    probability = torch.full(labels.shape, float(mlm_probability))
    probability[tensor_mask == 0] = 0.0
    for token_id in special:
        if token_id is not None:
            probability[labels == token_id] = 0.0
    selected = torch.bernoulli(probability).bool()
    labels[~selected] = -100

    # 80% mask, 10% random token, 10% keep the original id.
    #
    # The 80/10/10 split needs two independent draws: first pick which of the
    # selected positions become mask, then split the remainder evenly between
    # random replacement and keeping the original id.
    replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & selected
    remainder = selected & ~replace
    randomise = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & remainder
    mask_token = tokenizer.mask_token_id
    if mask_token is None:
        raise ValueError("tokenizer must define a mask token")
    tensor_ids = torch.where(replace, torch.full_like(tensor_ids, mask_token), tensor_ids)
    if int(randomise.sum()) > 0:
        # ``len(tokenizer)`` is the number of usable ids.  ``tokenizer.vocab_size``
        # can include unused padding rows whose embeddings are untrained, and
        # sampling them produces non-finite activations under fp32 MPS.
        high = int(len(tokenizer))
        if high < 1:
            raise ValueError("tokenizer reports an empty vocabulary")
        random_values = torch.randint(low=0, high=high, size=tensor_ids.shape, dtype=torch.long)
        tensor_ids = torch.where(randomise, random_values, tensor_ids)

    payload = {"input_ids": tensor_ids, "attention_mask": tensor_mask, "labels": labels}
    if token_types is not None:
        payload["token_type_ids"] = torch.tensor(token_types, dtype=torch.long)
    return payload


def _snapshot_dict(snapshot: Any) -> dict[str, Any]:
    return dict(snapshot.__dict__) if hasattr(snapshot, "__dict__") else dict(snapshot)


def train_mlm(
    *,
    model_name: str,
    output_dir: Path,
    reviews_path: Path,
    epochs: int,
    max_length: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    mlm_probability: float,
    device: str,
    seed: int,
    max_steps: int,
    budget_minutes: float,
    log_every: int = 20,
    checkpoint_every: int = 200,
    trainable_layers: int = 0,
    dashboard: DashboardWriter | None = None,
    tokenizer: Any | None = None,
    model: Any | None = None,
    rows: Sequence[Any] | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    started = time.time()
    torch.manual_seed(seed)
    random.seed(seed)

    resolved = resolve_model_name(model_name)
    examples = list(rows) if rows is not None else load_test_reviews(reviews_path)
    corpus = build_corpus(examples)
    if not corpus:
        raise SystemExit("no training reviews found for MLM adaptation")

    target_device = torch.device(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True, local_files_only=True)
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(resolved, local_files_only=True)

    # Short-range domain adaptation does not need to move every layer, and on a
    # 16GB unified-memory machine freezing the bottom layers also removes their
    # AdamW momentum from the device.  ``0`` keeps the historical full tune.
    if trainable_layers > 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        backbone = getattr(model, "bert", None) or getattr(model, "roberta", None)
        encoder = getattr(backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            for layer in list(encoder.layer)[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        # The MLM head is the objective's own readout; it must always train.
        for name, parameter in model.named_parameters():
            if name.startswith("cls.") or "predictions" in name:
                parameter.requires_grad = True
    model.to(target_device)
    model.train()

    dataset = _MaskedTextDataset(corpus, tokenizer, max_length)
    guard = ResourceGuard()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise SystemExit("no trainable parameters after applying --trainable-layers")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.01)
    losses: list[float] = []
    peak: dict[str, float | None] = {"tracked_gb": None, "mps_driver_gb": None}
    minimum_available: float | None = None
    steps = 0
    epochs_completed = 0
    active_batch = max(1, batch_size)
    stop_reason = "completed"
    skipped_steps = 0
    learning_rate_now = float(learning_rate)
    best_loss: float | None = None
    checkpoints_written: list[str] = []

    if dashboard is not None:
        dashboard.update(
            status="running",
            stage="TAPT_MLM",
            message=f"task-adaptive MLM on {len(corpus)} training reviews",
            current={
                "trial": f"tapt_{model_name}",
                "model_name": model_name,
                "fold": 1,
                "folds": 1,
                "epoch": 0,
                "epochs": epochs,
                "step": 0,
                "steps": 0,
                "phase": "tapt_start",
                "runtime_config": json.dumps(
                    {
                        "model_name": model_name,
                        "max_length": max_length,
                        "batch_size": batch_size,
                        "gradient_accumulation": gradient_accumulation,
                        "learning_rate": learning_rate,
                        "mlm_probability": mlm_probability,
                        "epochs": epochs,
                        "device": device,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        )
        dashboard.event(f"TAPT 启动 · {model_name} · {len(corpus)} 条训练评论 · 目标 {epochs} epochs")

    for epoch in range(1, epochs + 1):
        # A resource downshift restarts the loader at a smaller batch instead of
        # abandoning the epoch, so an adaptation run keeps making progress on a
        # memory-constrained machine.
        while True:
            # ``downshift`` is set inside the loader loop when the guard asks
            # for a smaller batch, and consumed by the check below.  Resetting
            # it here keeps the next rebuild from looping forever.
            downshift = False
            loader = DataLoader(
                dataset,
                batch_size=active_batch,
                shuffle=True,
                num_workers=0,
                collate_fn=lambda batch: _collate(batch, tokenizer, mlm_probability),
            )
            optimizer.zero_grad(set_to_none=True)
            for raw_batch in loader:
                batch = {key: value.to(target_device) for key, value in raw_batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    # MPS can emit non-finite activations when unified memory
                    # gets tight.  Skip the poisoned step, free cache, and halve
                    # the LR instead of aborting a multi-hour adaptation run.
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    guard.release_cache()
                    learning_rate_now = max(1.0e-6, learning_rate_now * 0.5)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate_now
                    if dashboard is not None:
                        dashboard.event(
                            f"跳过非有限 loss step {steps + 1} · 清缓存 · lr 降为 {learning_rate_now:.2e}"
                        )
                    if skipped_steps >= 20:
                        raise FloatingPointError("MLM loss stayed non-finite after 20 recoveries")
                    continue
                (loss / max(1, gradient_accumulation)).backward()
                if (steps + 1) % max(1, gradient_accumulation) == 0:
                    # ``clip_grad_norm_`` returns a non-finite norm for poisoned
                    # gradients and then silently keeps them; the following step
                    # would write NaN into the weights permanently.  Inspect the
                    # gradients first and drop the update instead of the run.
                    grads_finite = all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                    if grads_finite:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        weights_finite = all(
                            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
                        )
                        if not weights_finite:
                            raise FloatingPointError("optimizer step produced non-finite weights")
                    else:
                        skipped_steps += 1
                        learning_rate_now = max(1.0e-6, learning_rate_now * 0.5)
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate_now
                        if dashboard is not None:
                            dashboard.event(
                                f"跳过非有限梯度 step {steps + 1} · lr 降为 {learning_rate_now:.2e}"
                            )
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
                steps += 1

                # MPS keeps freed activations in its driver pool; without a
                # periodic flush the pool grows until unified memory pressure
                # starts producing non-finite activations.  Flush often: the
                # reviews are short, so the flush cost is negligible.
                if steps % 10 == 0:
                    guard.release_cache()

                # Persist progress so a late crash cannot discard a long run.
                if checkpoint_every and steps % checkpoint_every == 0:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    step_dir = output_dir / f"step_{steps:06d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(step_dir)
                    tokenizer.save_pretrained(step_dir)
                    checkpoints_written.append(str(step_dir))
                    rolling = losses[-50:]
                    rolling_mean = sum(rolling) / len(rolling) if rolling else float("inf")
                    if best_loss is None or rolling_mean < best_loss:
                        best_loss = rolling_mean
                        best_dir = output_dir / "best"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                    if dashboard is not None:
                        dashboard.event(
                            f"TAPT checkpoint step {steps} · rolling loss {rolling_mean:.4f} · 已保存 {step_dir.name}"
                        )

                if steps % log_every == 0:
                    snapshot = _snapshot_dict(guard.snapshot())
                    tracked = snapshot.get("tracked_gb")
                    driver = snapshot.get("mps_driver_gb")
                    available = snapshot.get("available_gb")
                    if isinstance(tracked, (int, float)):
                        peak["tracked_gb"] = max(peak["tracked_gb"] or 0.0, float(tracked))
                    if isinstance(driver, (int, float)):
                        peak["mps_driver_gb"] = max(peak["mps_driver_gb"] or 0.0, float(driver))
                    if isinstance(available, (int, float)):
                        minimum_available = available if minimum_available is None else min(minimum_available, float(available))
                    print(
                        f"epoch {epoch}/{epochs} step {steps} loss {losses[-1]:.4f} "
                        f"avail {available:.2f}GB batch {active_batch}",
                        flush=True,
                    )
                    recommendation = guard.recommend(type("Cfg", (), {"batch_size": active_batch, "max_length": max_length})())
                    suggested = int(recommendation.get("batch_size", active_batch))
                    if suggested < active_batch:
                        active_batch = suggested
                        downshift = True
                        if dashboard is not None:
                            dashboard.event(f"资源降级 · batch 降为 {active_batch} · 重建 DataLoader 继续")
                        break

                    # Adaptation is a luxury: stop cleanly while the adapted
                    # weights are still savable instead of letting unified
                    # memory run dry and poisoning the optimizer with NaN.
                    pressure = str(getattr(guard.snapshot(), "pressure", "normal"))
                    filled = bool(isinstance(available, (int, float)) and available < 1.2) or pressure == "critical"
                    if active_batch <= 1 and filled:
                        stop_reason = "memory_guard"
                        if dashboard is not None:
                            dashboard.event(
                                f"内存守护触发 · batch 1 且 available {available:.2f}GB / pressure {pressure} · 保存已适配权重后收尾"
                            )
                        break

                    if dashboard is not None:
                        epochs_done = steps / max(1, len(dataset) // max(1, active_batch))
                        total_epochs_est = max(1.0, float(epochs))
                        fraction = min(0.999, ((epoch - 1) + epochs_done) / total_epochs_est)
                        dashboard.progress(
                            {
                                "timestamp": time.time(),
                                "trial": f"tapt_{model_name}",
                                "model_name": model_name,
                                "fold": 1,
                                "folds": 1,
                                "epoch": epoch,
                                "epochs": epochs,
                                "step": steps,
                                "steps": len(dataset) // max(1, active_batch),
                                "progress_percent": fraction * 100.0,
                                "overall_percent": fraction * 100.0,
                                "fold_percent": fraction * 100.0,
                                "epoch_percent": epochs_done * 100.0,
                                "phase": "tapt_step",
                            }
                        )
                        dashboard.loss(
                            {
                                "timestamp": time.time(),
                                "trial": f"tapt_{model_name}",
                                "epoch": epoch,
                                "step": steps,
                                "loss": float(losses[-1]),
                                "phase": "tapt_step",
                            }
                        )
                        snapshot_for_ui = _snapshot_dict(guard.snapshot())
                        dashboard.resource(snapshot_for_ui)

                    if max_steps and steps >= max_steps:
                        stop_reason = "max_steps"
                        break
                    if budget_minutes and (time.time() - started) >= budget_minutes * 60.0:
                        stop_reason = "budget"
                        break
            if downshift:
                continue
            break
        epochs_completed = epoch
        if dashboard is not None:
            dashboard.event(
                f"TAPT epoch {epoch}/{epochs} 完成 · loss {losses[-1]:.4f} · 累计 {steps} steps"
            )
        if stop_reason in {"max_steps", "budget", "memory_guard"}:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = _Report(
        model_name=model_name,
        resolved_model_name=resolved,
        corpus_size=len(corpus),
        corpus_characters=sum(len(text) for text in corpus),
        max_length=max_length,
        mlm_probability=mlm_probability,
        learning_rate=learning_rate,
        epochs_requested=epochs,
        epochs_completed=epochs_completed,
        steps=steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        device=str(target_device),
        losses=losses,
        peak_memory=peak,
        minimum_available_gb=minimum_available,
        elapsed_seconds=time.time() - started,
        output_dir=str(output_dir),
    ).as_dict()
    report["stop_reason"] = stop_reason
    report["skipped_steps"] = skipped_steps
    report["final_learning_rate"] = learning_rate_now
    report["best_rolling_loss"] = best_loss
    report["checkpoints"] = checkpoints_written
    # The final directory always holds the last state, so a caller that only
    # wants "the adapted model" never has to know about step checkpoints.
    report["final_dir"] = str(output_dir)
    report["best_dir"] = str(output_dir / "best") if (output_dir / "best").exists() else None
    (output_dir / "adaptation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if dashboard is not None:
        dashboard.final(
            metrics={},
            submission={
                "status": "adapted_checkpoint",
                "path": str(output_dir),
                "model_name": model_name,
                "steps": steps,
                "final_loss": report["final_loss"],
            },
            message=f"TAPT {model_name} 完成 · {steps} steps · final loss {report['final_loss']}",
        )
        dashboard.event(f"适配权重已保存到 {output_dir}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="wwm", help="alias or local model reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0 keeps the full run; small values are smoke tests")
    parser.add_argument("--budget-minutes", type=float, default=45.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=200, help="0 disables intermediate checkpoints")
    parser.add_argument(
        "--trainable-layers",
        type=int,
        default=0,
        help="train only the top N encoder layers plus the MLM head; 0 trains every layer",
    )
    parser.add_argument("--dashboard-root", type=Path, default=None, help="separate state directory; keeps the grid console untouched")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:18766/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs < 1 or args.max_length < 8 or args.batch_size < 1:
        raise SystemExit("--epochs, --max-length and --batch-size must be positive")
    if not 0.0 < args.mlm_probability < 1.0:
        raise SystemExit("--mlm-probability must be between 0 and 1")
    report = train_mlm(
        model_name=args.model_name,
        output_dir=args.output_dir,
        reviews_path=args.reviews,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        mlm_probability=args.mlm_probability,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps,
        budget_minutes=args.budget_minutes,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        trainable_layers=args.trainable_layers,
        dashboard=(
            DashboardWriter(args.dashboard_root, budget_seconds=args.budget_minutes * 60.0)
            if args.dashboard_root is not None
            else None
        ),
    )
    print(json.dumps({k: report[k] for k in ("steps", "epochs_completed", "initial_loss", "final_loss", "mean_last_10_loss", "elapsed_seconds", "output_dir", "stop_reason")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
