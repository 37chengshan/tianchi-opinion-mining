#!/usr/bin/env python3
"""Self-contained LoRA SFT for the ACOS task on a ROCm/CUDA GPU.

Trains on the repository's chat JSONL (system+user prompt, JSON answer) with
prompt tokens masked out, mirroring the public recipe: r=16, alpha=32,
dropout=0.05, lr=1e-4 cosine with warmup, 3 epochs, bf16.

    /opt/venv/bin/python train_cloud_lora.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --data-dir llm_data/fold1_3fold \
        --out /workspace/checkpoints/qwen25_14b_fold1 \
        --epochs 3 --cutoff 1024 --batch-size 1 --grad-accum 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def _encode(tokenizer, record: dict, cutoff: int) -> dict | None:
    messages = record.get("messages") or []
    if len(messages) < 3:
        return None
    prompt_messages = [item for item in messages if item.get("role") != "assistant"]
    answer = next(item["content"] for item in messages if item.get("role") == "assistant")
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + answer + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=cutoff)
    input_ids = full["input_ids"]
    labels = list(input_ids)
    mask_until = min(len(prompt_ids), len(labels))
    for index in range(mask_until):
        labels[index] = -100
    if all(value == -100 for value in labels):
        return None
    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--cutoff", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "model": args.model,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "cutoff": args.cutoff,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (args.out / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class ChatDataset(Dataset):
        def __init__(self, path: Path):
            raw = _read_jsonl(path, args.limit)
            self.items = [item for item in (_encode(tokenizer, record, args.cutoff) for record in raw) if item]
            print(json.dumps({"dataset": str(path), "usable": len(self.items)}), flush=True)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict:
            item = self.items[index]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }

    def collate(batch: list[dict]) -> dict:
        width = max(item["input_ids"].numel() for item in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            pad = width - item["input_ids"].numel()
            out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=tokenizer.pad_token_id or 0))
            out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            out["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        return {key: torch.stack(value) for key, value in out.items()}

    train_set = ChatDataset(args.data_dir / "train.jsonl")
    valid_path = args.data_dir / "valid.jsonl"
    valid_set = ChatDataset(valid_path) if valid_path.exists() else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=max(1, total_steps))
    model.train()
    log_path = args.out / "train_log.jsonl"
    step = 0
    micro = 0
    running_loss = 0.0
    running_count = 0
    started = time.time()
    optim.zero_grad(set_to_none=True)
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, int(math.ceil(args.epochs)) + 1):
            for batch in train_loader:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                running_loss += float(loss.detach()) * args.grad_accum
                running_count += 1
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_steps == 0:
                        elapsed = time.time() - started
                        record = {
                            "epoch": epoch,
                            "step": step,
                            "total_steps": total_steps,
                            "loss": round(running_loss / max(1, running_count), 4),
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_s": round(elapsed, 1),
                            "eta_s": round(elapsed / max(1, step) * max(0, total_steps - step), 1),
                        }
                        print(json.dumps(record), flush=True)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        running_loss = 0.0
                        running_count = 0
                    if args.save_steps and step % args.save_steps == 0:
                        model.save_pretrained(args.out / f"step_{step:06d}")
                        print(json.dumps({"saved": str(args.out / f'step_{step:06d}')}), flush=True)
            if valid_set is not None and len(valid_set):
                model.eval()
                totals = 0.0
                count = 0
                with torch.no_grad():
                    for batch in DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate):
                        batch = {key: value.to(model.device) for key, value in batch.items()}
                        totals += float(model(**batch).loss)
                        count += 1
                model.train()
                print(json.dumps({"epoch": epoch, "valid_loss": round(totals / max(1, count), 4)}), flush=True)
    model.save_pretrained(args.out)
    (args.out / "run_meta.json").write_text(json.dumps({**run_meta, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "final_step": step}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "adapter": str(args.out), "steps": step}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
