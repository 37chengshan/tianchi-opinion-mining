#!/usr/bin/env python3
"""Measure decoder-width candidate oracle recall from existing checkpoints.

The V1 error audit showed candidate oracle recall around 0.875, below the
locked plan's 0.90 gate.  This sweep re-decodes saved fold checkpoints with
wider span/pair/relation budgets and reports whether the missing coverage is
a decoder-budget problem or a training problem.  It never trains and never
writes outside ``--output``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.baseline import Candidate
from opinion_mining.analysis import save_candidates
from opinion_mining.data import ReviewExample, load_train_data
from opinion_mining.metrics import strict_f1
from opinion_mining.grid_model import decode_grid_candidates
from opinion_mining.grid_trainer import (
    CompactGridEncoderAdapter,
    GridTrainConfig,
    _valid_surface_mask,
    load_cached_tokenizer,
    resolve_model_name,
)
from opinion_mining.neural import _ReviewDataset, _collate, choose_device
from opinion_mining.pipeline import select_threshold, threshold_predictions


def _oracle(candidates: dict[int, list[Candidate]], gold: dict[int, set[Any]]) -> dict[str, Any]:
    hit = 0
    total = 0
    predicted = 0
    for rid, labels in gold.items():
        total += len(labels)
        quads = {item.quadruple for item in candidates.get(rid, [])}
        predicted += len(quads)
        hit += len(quads & set(labels))
    return {
        "oracle_hit": hit,
        "oracle_gold": total,
        "oracle_recall": hit / total if total else 0.0,
        "predicted_quadruples": predicted,
    }


def decode_fold(
    run_dir: Path,
    fold: int,
    rows: Sequence[ReviewExample],
    *,
    device: str,
    span_top_k: int,
    max_span_length: int,
    pair_top_k: int,
    relation_top_k: int,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint = torch.load(run_dir / f"fold_{fold}" / "model.pt", map_location="cpu", weights_only=False)
    config = GridTrainConfig(**checkpoint["config"])
    assignment = json.loads((run_dir / "fold_assignments.json").read_text(encoding="utf-8"))[fold - 1]
    valid_ids = set(int(value) for value in assignment["valid_ids"])
    valid_rows = [row for row in rows if row.id in valid_ids]
    gold = {row.id: set(row.labels) for row in valid_rows}

    target_device = torch.device(choose_device(device))
    tokenizer = load_cached_tokenizer(resolve_model_name(config.model_name))
    adapter = CompactGridEncoderAdapter.from_pretrained(
        config.model_name,
        relation_rank=config.relation_rank,
        trainable_layers=config.trainable_layers,
    ).to(target_device)
    # V1 checkpoints predate the V2 auxiliary category/polarity heads; the
    # decoder below only consumes the relation grid, so absent heads are fine.
    missing, unexpected = adapter.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    adapter_v2_missing_keys = [key for key in missing if not key.startswith(("category.", "polarity."))]
    if adapter_v2_missing_keys:
        raise RuntimeError(f"missing checkpoint keys: {adapter_v2_missing_keys[:5]}")
    adapter.eval()

    dataset = _ReviewDataset(valid_rows, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    result: dict[int, list[Candidate]] = {}
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {key: (value.to(target_device) if isinstance(value, torch.Tensor) else value) for key, value in raw_batch.items()}
            outputs = adapter(batch)
            cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items() if isinstance(value, torch.Tensor)}
            size = len(raw_batch["pairs"])
            for local in range(size):
                row = valid_rows[offset + local]
                feature = dataset.features[offset + local]
                valid = _valid_surface_mask(feature, row.text, raw_batch["attention_mask"][local])
                result[row.id] = decode_grid_candidates(
                    cpu_outputs["aspect_start"][local],
                    cpu_outputs["aspect_end"][local],
                    cpu_outputs["opinion_start"][local],
                    cpu_outputs["opinion_end"][local],
                    cpu_outputs["relation"][local],
                    valid_mask=valid,
                    top_k_spans=span_top_k,
                    top_k_pairs=pair_top_k,
                    relation_top_k=relation_top_k,
                    max_span_length=max_span_length,
                    nms_iou=config.nms_iou,
                    text=row.text,
                    offsets=feature.offsets,
                )
            offset += size
    del adapter
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"fold": fold, "valid_rows": len(valid_rows), "candidates": result, "gold": gold, **_oracle(result, gold)}


CONFIGS: dict[str, dict[str, Any]] = {
    "v1_repro": dict(span_top_k=8, max_span_length=10, pair_top_k=64, relation_top_k=2),
    "wide": dict(span_top_k=16, max_span_length=12, pair_top_k=256, relation_top_k=4),
    "very_wide": dict(span_top_k=24, max_span_length=14, pair_top_k=512, relation_top_k=8),
    "wide_span2": dict(span_top_k=16, max_span_length=12, pair_top_k=256, relation_top_k=2),
    "wide_span2_ml14": dict(span_top_k=20, max_span_length=14, pair_top_k=400, relation_top_k=2),
    "narrow_ml12": dict(span_top_k=8, max_span_length=12, pair_top_k=64, relation_top_k=2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--configs", default="v1_repro,wide,very_wide")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-candidates", type=Path, help="write the OOF candidate map for the named config")
    parser.add_argument("--save-candidates-config", default="wide")
    parser.add_argument("--score", action="store_true", help="run strict-F1 threshold calibration on the decoded OOF")
    args = parser.parse_args()

    rows = load_train_data(args.reviews, args.labels)
    report: dict[str, Any] = {"run_dir": str(args.run_dir), "folds": args.folds, "configs": {}, "started_at": time.time()}
    for name in [item.strip() for item in args.configs.split(",") if item.strip()]:
        if name not in CONFIGS:
            parser.error(f"unknown config {name}; known: {sorted(CONFIGS)}")
        settings = CONFIGS[name]
        fold_reports = [decode_fold(args.run_dir, fold, rows, device=args.device, batch_size=args.batch_size, **settings) for fold in range(1, args.folds + 1)]
        if args.save_candidates and name == args.save_candidates_config:
            merged: dict[int, list[Candidate]] = {}
            for item in fold_reports:
                merged.update(item["candidates"])
            args.save_candidates.parent.mkdir(parents=True, exist_ok=True)
            save_candidates(args.save_candidates, merged)
            print(json.dumps({"saved_candidates": str(args.save_candidates), "rows": len(merged)}, ensure_ascii=False), flush=True)
        if args.score:
            merged = {}
            for item in fold_reports:
                merged.update(item["candidates"])
            gold_all = {}
            for item in fold_reports:
                gold_all.update(item["gold"])
            calibration = select_threshold(merged, gold_all, separate_implicit=True)
            scored = strict_f1(gold_all, threshold_predictions(merged, float(calibration["threshold"]), float(calibration["implicit_threshold"])))
            print(json.dumps({"scored": name, "threshold": calibration["threshold"], "implicit_threshold": calibration["implicit_threshold"], "precision": scored.precision, "recall": scored.recall, "f1": scored.f1}, ensure_ascii=False), flush=True)
        hits = sum(item["oracle_hit"] for item in fold_reports)
        gold = sum(item["oracle_gold"] for item in fold_reports)
        serialisable = [{key: value for key, value in item.items() if key not in ("candidates", "gold")} for item in fold_reports]
        report["configs"][name] = {
            "settings": settings,
            "folds": serialisable,
            "aggregate": {"oracle_hit": hits, "oracle_gold": gold, "oracle_recall": hits / gold if gold else 0.0, "predicted_quadruples": sum(item["predicted_quadruples"] for item in fold_reports)},
        }
        print(json.dumps({name: report["configs"][name]["aggregate"]}, ensure_ascii=False), flush=True)
    report["finished_at"] = time.time()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
