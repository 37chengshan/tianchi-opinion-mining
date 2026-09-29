#!/usr/bin/env python3
"""Re-decode saved Grid V2 checkpoints with an explicit relation score source.

This is an offline V2.1 experiment.  It does not train or alter a checkpoint.
The V2 trainer emits both the original anchor grid logits and the full-span
pair logits.  The current V2 decoder uses only the latter for the relation
class score.  This script makes that choice explicit so the two paths can be
compared on the same saved fold checkpoints.

The script writes OOF/test candidate maps, full-OOF screening calibration,
and fold-exclusive cross-fit calibration.  It never uploads a submission.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import load_candidates, merge_candidate_maps, save_candidates
from opinion_mining.data import ReviewExample, load_test_reviews, load_train_data
from opinion_mining.grid_model import (
    GridCandidate,
    constrained_topk_span_decode,
    grid_candidates_to_candidates,
    nms_grid_candidates,
)
from opinion_mining.grid_trainer import (
    CompactGridEncoderAdapter,
    GridTrainConfig,
    _valid_surface_mask,
    load_cached_tokenizer,
    resolve_model_name,
)
from opinion_mining.metrics import strict_f1
from opinion_mining.neural import _ReviewDataset, _collate, choose_device
from opinion_mining.pipeline import crossfit_threshold_score, select_threshold, threshold_predictions
from opinion_mining.submission import write_submission


MODES = ("pair", "grid", "mean", "geom", "pair07")


def _score_dict(score: Any) -> dict[str, Any]:
    return {
        "precision": float(score.precision),
        "recall": float(score.recall),
        "f1": float(score.f1),
        "correct": int(score.correct),
        "predicted": int(score.predicted),
        "gold": int(score.gold),
    }


def _relation_score(mode: str, pair: float, grid: float) -> float:
    if mode == "pair":
        return pair
    if mode == "grid":
        return grid
    if mode == "mean":
        return (pair + grid) / 2.0
    if mode == "geom":
        return max(pair * grid, 0.0) ** 0.5
    if mode == "pair07":
        return (pair ** 0.7) * (grid ** 0.3)
    raise ValueError(f"unknown relation score mode: {mode}")


def _decode_rows(
    adapter: CompactGridEncoderAdapter,
    tokenizer: Any,
    rows: list[ReviewExample],
    config: GridTrainConfig,
    device: torch.device,
    mode: str,
) -> dict[int, list[Any]]:
    if not rows:
        return {}
    adapter.eval()
    dataset = _ReviewDataset(rows, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    result: dict[int, list[Any]] = {}
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            outputs = adapter(batch)
            cpu_outputs = {
                key: value.detach().cpu()
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor)
            }
            batch_size = len(raw_batch["pairs"])
            for local_index in range(batch_size):
                row = rows[offset + local_index]
                feature = dataset.features[offset + local_index]
                valid = _valid_surface_mask(
                    feature,
                    row.text,
                    raw_batch["attention_mask"][local_index],
                )
                aspect_spans = constrained_topk_span_decode(
                    cpu_outputs["aspect_start"][local_index],
                    cpu_outputs["aspect_end"][local_index],
                    valid_mask=valid,
                    top_k=config.span_top_k,
                    max_span_length=config.max_span_length,
                    allow_implicit=True,
                    objectiveness_logits=cpu_outputs["objectiveness"][local_index],
                    implicit_logit=cpu_outputs["implicit_aspect_presence"][local_index],
                )
                opinion_spans = constrained_topk_span_decode(
                    cpu_outputs["opinion_start"][local_index],
                    cpu_outputs["opinion_end"][local_index],
                    valid_mask=valid,
                    top_k=config.span_top_k,
                    max_span_length=config.max_span_length,
                    allow_implicit=True,
                    objectiveness_logits=cpu_outputs["objectiveness"][local_index],
                    implicit_logit=cpu_outputs["implicit_opinion_presence"][local_index],
                )
                pair_items: list[tuple[Any, Any]] = []
                pair_rows: list[list[int]] = []
                for aspect in aspect_spans:
                    for opinion in opinion_spans:
                        if (
                            not aspect.implicit
                            and not opinion.implicit
                            and max(aspect.start, opinion.start) <= min(aspect.end, opinion.end)
                        ):
                            continue
                        pair_items.append((aspect, opinion))
                        pair_rows.append(
                            [
                                0,
                                -1 if aspect.implicit else aspect.start,
                                -1 if aspect.implicit else aspect.end,
                                -1 if opinion.implicit else opinion.start,
                                -1 if opinion.implicit else opinion.end,
                            ]
                        )

                refined: list[GridCandidate] = []
                if pair_rows:
                    pair_tensor = torch.tensor(pair_rows, dtype=torch.long, device=device)
                    hidden_row = outputs["hidden"][local_index : local_index + 1]
                    pair_prob = torch.sigmoid(
                        adapter.relation_head.pair_logits(hidden_row, pair_tensor)
                    ).detach().cpu()
                    validity_prob = torch.sigmoid(
                        adapter.relation_head.pair_validity_logits(hidden_row, pair_tensor)
                    ).detach().cpu()
                    grid_prob = torch.sigmoid(cpu_outputs["relation"][local_index])
                    for pair_index, (aspect, opinion) in enumerate(pair_items):
                        aspect_anchor = 0 if aspect.implicit else aspect.start
                        opinion_anchor = 0 if opinion.implicit else opinion.start
                        grid_row = grid_prob[aspect_anchor, opinion_anchor]
                        values, class_ids = torch.topk(
                            pair_prob[pair_index],
                            k=min(max(1, config.relation_top_k), pair_prob.shape[-1]),
                        )
                        span_score = max(0.0, float(aspect.score * opinion.score)) ** 0.5
                        pair_gate = float(validity_prob[pair_index])
                        for pair_value, class_id in zip(values.tolist(), class_ids.tolist()):
                            pair_value = float(pair_value)
                            grid_value = float(grid_row[int(class_id)])
                            relation_value = _relation_score(mode, pair_value, grid_value)
                            refined.append(
                                GridCandidate(
                                    aspect,
                                    opinion,
                                    int(class_id),
                                    span_score * pair_gate * relation_value,
                                )
                            )
                kept = nms_grid_candidates(
                    refined,
                    iou_threshold=config.nms_iou,
                    max_keep=max(config.span_top_k * config.span_top_k, 16),
                )
                result[row.id] = grid_candidates_to_candidates(
                    kept,
                    text=row.text,
                    offsets=feature.offsets,
                    source=f"relation_grid_v21_{mode}",
                )
            offset += batch_size
    return result


def _load_adapter(checkpoint: Path, device: torch.device) -> tuple[CompactGridEncoderAdapter, GridTrainConfig, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = GridTrainConfig(**payload["config"])
    tokenizer = load_cached_tokenizer(resolve_model_name(config.model_name))
    adapter = CompactGridEncoderAdapter.from_pretrained(
        config.model_name,
        relation_rank=config.relation_rank,
        trainable_layers=config.trainable_layers,
    ).to(device)
    missing, unexpected = adapter.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    allowed_missing = [key for key in missing if not key.startswith(("category.", "polarity."))]
    if allowed_missing:
        raise RuntimeError(f"missing checkpoint keys: {allowed_missing[:5]}")
    adapter.eval()
    return adapter, config, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="grid")
    parser.add_argument("--folds", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--reviews", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/data/train/TRAIN/Train_labels.csv")
    parser.add_argument("--test-reviews", type=Path, default=ROOT / "artifacts/data/test/TEST/Test_reviews.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_train_data(args.reviews, args.labels)
    test_rows = load_test_reviews(args.test_reviews) if args.include_test else []
    assignments = json.loads((args.run_dir / "fold_assignments.json").read_text(encoding="utf-8"))
    device = torch.device(choose_device(args.device))
    oof: dict[int, list[Any]] = {}
    test_maps: list[dict[int, list[Any]]] = []
    fold_ids: list[list[int]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in range(1, args.folds + 1):
        assignment = assignments[fold - 1]
        valid_ids = {int(value) for value in assignment["valid_ids"]}
        valid_rows = [row for row in train_rows if row.id in valid_ids]
        fold_ids.append(sorted(valid_ids))
        adapter, config, tokenizer = _load_adapter(args.run_dir / f"fold_{fold}" / "model.pt", device)
        valid_candidates = _decode_rows(adapter, tokenizer, valid_rows, config, device, args.mode)
        oof.update(valid_candidates)
        test_candidates = _decode_rows(adapter, tokenizer, test_rows, config, device, args.mode) if test_rows else {}
        if test_candidates:
            test_maps.append(test_candidates)
        fold_reports.append({
            "fold": fold,
            "valid_rows": len(valid_rows),
            "candidate_count": sum(len(items) for items in valid_candidates.values()),
            "test_candidate_count": sum(len(items) for items in test_candidates.values()),
            "runtime_config": asdict(config),
        })
        del adapter
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    gold = {row.id: set(row.labels) for row in train_rows}
    selected = select_threshold(oof, gold, separate_implicit=True)
    crossfit = crossfit_threshold_score(oof, gold, fold_ids, separate_implicit=True)
    save_candidates(args.output_dir / "oof_candidates.json", oof)
    crossfit_folds = []
    for fold_report in crossfit["folds"]:
        item = dict(fold_report)
        item["score"] = _score_dict(item["score"])
        crossfit_folds.append(item)
    report: dict[str, Any] = {
        "status": "completed",
        "source_run": str(args.run_dir),
        "mode": args.mode,
        "folds": fold_reports,
        "threshold": {
            "threshold": float(selected["threshold"]),
            "implicit_threshold": float(selected["implicit_threshold"]),
            "state_thresholds": {key: float(value) for key, value in selected["state_thresholds"].items()},
            "full_oof_fit_score": _score_dict(selected["score"]),
            "crossfit_score": _score_dict(crossfit["score"]),
            "crossfit_folds": crossfit_folds,
            "calibration": "full OOF screening plus fold-exclusive cross-fit report",
        },
    }
    if test_maps:
        test_candidates = merge_candidate_maps(test_maps, average_present=True)
        save_candidates(args.output_dir / "test_candidates.json", test_candidates)
        predictions = threshold_predictions(
            test_candidates,
            float(selected["threshold"]),
            float(selected["implicit_threshold"]),
            state_thresholds=selected["state_thresholds"],
        )
        submission = write_submission(args.output_dir / "Result.csv", test_rows, predictions)
        report["submission"] = asdict(submission)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
