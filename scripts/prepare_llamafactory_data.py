#!/usr/bin/env python3
"""Convert the repository's chat JSONL into LLaMA-Factory sharegpt data.

Output: <out-dir>/acos_train.json / acos_eval.json + dataset_info.json snippet,
ready for `llamafactory-cli train` with `template: qwen`-style chat templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sharegpt(record: dict) -> dict | None:
    messages = record.get("messages") or []
    if len(messages) < 2:
        return None
    system = next((item["content"] for item in messages if item.get("role") == "system"), "")
    user = next((item["content"] for item in messages if item.get("role") == "user"), "")
    answer = next((item["content"] for item in messages if item.get("role") == "assistant"), None)
    conversations = [
        {"from": "human", "value": (system + "\n\n" + user).strip() if system else user},
    ]
    if answer is not None:
        conversations.append({"from": "gpt", "value": answer})
    return {"conversations": conversations}


def convert_file(source: Path, target: Path) -> int:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = _sharegpt(json.loads(line))
        if item:
            rows.append(item)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="acos_quadruple")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train", "valid", "test"):
        source = args.data_dir / f"{name}.jsonl"
        if source.exists():
            counts[name] = convert_file(source, args.out_dir / f"acos_{name}.json")
    info = {
        args.dataset_name: {
            "file_name": "acos_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        },
        f"{args.dataset_name}_eval": {
            "file_name": "acos_valid.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        },
    }
    (args.out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "counts": counts, "dataset_info": str(args.out_dir / "dataset_info.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
