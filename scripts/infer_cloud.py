#!/usr/bin/env python3
"""Cloud-side inference for a LoRA/merged Qwen model on the ACOS task.

Runs on the GPU box (transformers + peft).  Uses the repository's prompt
contract and parser so predictions are directly comparable with local runs:

    PYTHONPATH=src python3 infer_cloud.py \
        --model saves/acos_qwen25_7b_lora \
        --data llm_data/fold1_3fold/test.jsonl \
        --out predictions_fold1.jsonl

Each output line: {"id": <review id>, "answer": "<raw text>", "quadruples": [...]}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import time


ID_PATTERN = re.compile(r"评论ID[:：]\s*(\d+)")


def _review_id(prompt: str) -> int | None:
    match = ID_PATTERN.search(prompt)
    return int(match.group(1)) if match else None


def _load_records(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def _split_messages(record: dict) -> tuple[str, str]:
    messages = record.get("messages") or []
    system = next((item["content"] for item in messages if item.get("role") == "system"), "")
    user = next((item["content"] for item in messages if item.get("role") == "user"), "")
    return system, user


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged model dir or base model dir")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--data", type=Path, required=True, help="test.jsonl with prompt-only chat records")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="skip model loading; exercise the parse/write path")
    args = parser.parse_args()

    from opinion_mining.llm_parse import LLMParseError, parse_llm_output

    records = _load_records(args.data, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = model = None
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    parse_failures = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            system, user = _split_messages(record)
            started = time.time()
            if args.dry_run:
                answer = json.dumps({"quadruples": []}, ensure_ascii=False)
            else:
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                answer = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            review_id = _review_id(user)
            try:
                quads = [list(quad) for quad in parse_llm_output(answer, _review_text(user))]
            except LLMParseError:
                quads = []
                parse_failures += 1
            handle.write(json.dumps({"id": review_id, "answer": answer, "quadruples": quads}, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"i": index, "total": len(records), "id": review_id, "pred": len(quads), "sec": round(time.time() - started, 2)}, ensure_ascii=False), flush=True)

    print(json.dumps({"out": str(args.out), "records": len(records), "parse_failures": parse_failures}, ensure_ascii=False), flush=True)
    return 0


def _review_text(user_prompt: str) -> str:
    marker = "评论文本:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1].split("\n", 1)[0].strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
