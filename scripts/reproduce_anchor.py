#!/usr/bin/env python3
"""Replay the frozen RBT3 five-fold champion without starting a new training run.

The historical fold checkpoints are verified against the frozen configuration,
while their archived test candidate maps are used as the deterministic inference
cache.  This keeps the reproduction bounded and byte-for-byte comparable with
the submitted Result.csv even when the original raw test zip is unavailable.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from opinion_mining.analysis import load_candidates, save_candidates, merge_candidate_maps
from opinion_mining.data import Quadruple
from opinion_mining.pipeline import threshold_predictions
from opinion_mining.submission import EMPTY


EXPECTED_FOLD_COUNT = 5


class AnchorReproductionError(RuntimeError):
    """Raised when the frozen archive cannot reproduce the champion."""


@dataclass(frozen=True)
class AnchorFold:
    fold: int
    checkpoint: str
    test_candidates: str


@dataclass(frozen=True)
class AnchorManifest:
    manifest_path: Path = field(repr=False, compare=False)
    candidate_id: str
    expected_sha256: str
    expected_bytes: int
    expected_rows: int
    expected_ids: int
    expected_predicted_quadruples: int
    expected_empty_ids: int
    local_oof_f1: float
    online_f1: float
    explicit_threshold: float
    implicit_threshold: float
    aggregation_bonus: float
    average_present: bool
    repository_root: str
    expected_source_csv: str
    merged_candidates_path: str
    folds: tuple[AnchorFold, ...]
    config: dict[str, Any]

    @property
    def root(self) -> Path:
        root = (self.manifest_path.parent / self.repository_root).resolve()
        if not root.exists():
            raise AnchorReproductionError(f"repository root does not exist: {root}")
        return root

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate


@dataclass(frozen=True)
class ReproductionReport:
    fold_count: int
    ids: int
    rows: int
    predicted_quadruples: int
    empty_ids: int
    bytes: int
    sha256: str


def load_manifest(path: str | Path) -> AnchorManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = tuple(
        AnchorFold(
            fold=int(item["fold"]),
            checkpoint=str(item["checkpoint"]),
            test_candidates=str(item["test_candidates"]),
        )
        for item in payload["folds"]
    )
    if tuple(item.fold for item in folds) != tuple(range(1, EXPECTED_FOLD_COUNT + 1)):
        raise AnchorReproductionError("manifest must contain folds 1 through 5 in order")
    thresholds = payload["thresholds"]
    return AnchorManifest(
        manifest_path=manifest_path,
        candidate_id=str(payload["candidate_id"]),
        expected_sha256=str(payload["expected"]["sha256"]),
        expected_bytes=int(payload["expected"]["bytes"]),
        expected_rows=int(payload["expected"]["rows"]),
        expected_ids=int(payload["expected"]["ids"]),
        expected_predicted_quadruples=int(payload["expected"]["predicted_quadruples"]),
        expected_empty_ids=int(payload["expected"]["empty_ids"]),
        local_oof_f1=float(payload["scores"]["local_oof_f1"]),
        online_f1=float(payload["scores"]["online_f1"]),
        explicit_threshold=float(thresholds["explicit"]),
        implicit_threshold=float(thresholds["implicit"]),
        aggregation_bonus=float(payload["aggregation"]["bonus"]),
        average_present=bool(payload["aggregation"]["average_present"]),
        repository_root=str(payload["repository_root"]),
        expected_source_csv=str(payload["expected_source_csv"]),
        merged_candidates_path=str(payload["merged_candidates_path"]),
        folds=folds,
        config=dict(payload["config"]),
    )


def _verify_checkpoint(path: Path, manifest: AnchorManifest, fold: int) -> None:
    if not path.is_file():
        raise AnchorReproductionError(f"missing checkpoint for fold {fold}: {path}")
    # Meta tensors validate the serialized checkpoint schema without allocating
    # the 157 MB parameter tensors on CPU or MPS.
    import torch

    payload = torch.load(path, map_location="meta", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise AnchorReproductionError(f"invalid checkpoint payload for fold {fold}: {path}")
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise AnchorReproductionError(f"checkpoint config missing for fold {fold}: {path}")
    for key, expected in manifest.config.items():
        expected_value = int(expected) + fold * 1009 if key == "seed" else expected
        if saved_config.get(key) != expected_value:
            raise AnchorReproductionError(
                f"checkpoint config mismatch for fold {fold}, key={key!r}: "
                f"expected {expected_value!r}, got {saved_config.get(key)!r}"
            )


def _write_submission_rows(path: Path, predictions: dict[int, set[Quadruple]]) -> ReproductionReport:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    empty_ids = 0
    predicted = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for rid in sorted(predictions):
            quads = sorted(predictions[rid])
            if not quads:
                quads = [EMPTY]
                empty_ids += 1
            for quad in quads:
                writer.writerow([rid, quad.aspect, quad.opinion, quad.category, quad.polarity])
                rows += 1
                if quad != EMPTY:
                    predicted += 1
    raw = path.read_bytes()
    return ReproductionReport(
        fold_count=0,
        ids=len(predictions),
        rows=rows,
        predicted_quadruples=predicted,
        empty_ids=empty_ids,
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _first_difference(expected: bytes, actual: bytes) -> str:
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for line_no, (left, right) in enumerate(zip(expected_lines, actual_lines), start=1):
        if left != right:
            return f"line {line_no}: expected={left!r} actual={right!r}"
    if len(expected_lines) != len(actual_lines):
        return f"line count differs: expected={len(expected_lines)} actual={len(actual_lines)}"
    return "byte content differs despite equal line content"


def reproduce_anchor(
    manifest: AnchorManifest,
    output_path: str | Path,
    *,
    merged_candidates_path: str | Path | None = None,
) -> ReproductionReport:
    """Replay exactly the archived five-fold candidate maps and thresholds."""

    candidate_maps = []
    for fold in manifest.folds:
        _verify_checkpoint(manifest.resolve(fold.checkpoint), manifest, fold.fold)
        candidate_maps.append(load_candidates(manifest.resolve(fold.test_candidates)))
    if not candidate_maps:
        raise AnchorReproductionError("manifest contains no candidate maps")
    ids = set(candidate_maps[0])
    if any(set(candidate_map) != ids for candidate_map in candidate_maps[1:]):
        raise AnchorReproductionError("fold test candidate maps do not cover identical ids")

    merged = merge_candidate_maps(
        candidate_maps,
        average_present=manifest.average_present,
        bonus=manifest.aggregation_bonus,
    )
    if merged_candidates_path is not None:
        save_candidates(merged_candidates_path, merged)
    predictions = threshold_predictions(
        merged,
        manifest.explicit_threshold,
        manifest.implicit_threshold,
    )
    report = _write_submission_rows(Path(output_path), predictions)
    report = ReproductionReport(
        fold_count=len(candidate_maps),
        ids=report.ids,
        rows=report.rows,
        predicted_quadruples=report.predicted_quadruples,
        empty_ids=report.empty_ids,
        bytes=report.bytes,
        sha256=report.sha256,
    )

    expected_values = {
        "ids": manifest.expected_ids,
        "rows": manifest.expected_rows,
        "predicted_quadruples": manifest.expected_predicted_quadruples,
        "empty_ids": manifest.expected_empty_ids,
        "bytes": manifest.expected_bytes,
        "sha256": manifest.expected_sha256,
    }
    actual_values = asdict(report)
    mismatches = {
        key: {"expected": expected, "actual": actual_values[key]}
        for key, expected in expected_values.items()
        if actual_values[key] != expected
    }
    if mismatches:
        expected_path = manifest.resolve(manifest.expected_source_csv)
        detail = "expected source is unavailable"
        if expected_path.is_file() and report.sha256 != manifest.expected_sha256:
            detail = _first_difference(expected_path.read_bytes(), Path(output_path).read_bytes())
        raise AnchorReproductionError(f"anchor reproduction mismatch: {mismatches}; {detail}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-candidates", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    merged_path = args.merged_candidates
    if merged_path is None:
        merged_path = manifest.resolve(manifest.merged_candidates_path)
    report = reproduce_anchor(manifest, args.output, merged_candidates_path=merged_path)
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
