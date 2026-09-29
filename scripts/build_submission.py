#!/usr/bin/env python3
"""Merge one or more test candidate maps and write a validated submission.

The thresholds must come from a fold-safe OOF report, which the caller passes
explicitly; this script never fits thresholds on the test set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from opinion_mining.analysis import load_candidates, merge_candidate_maps
from opinion_mining.data import load_test_reviews
from opinion_mining.pipeline import threshold_predictions
from opinion_mining.submission import validate_submission, write_submission


ROOT = Path(__file__).resolve().parents[1]
TEST_REVIEWS = ROOT / "artifacts/data/test/TEST/Test_reviews.csv"


def build_submission(
    sources: list[tuple[str, Path]],
    *,
    output_path: Path,
    explicit_threshold: float,
    implicit_threshold: float,
    state_thresholds: dict[str, float] | None = None,
    weights: list[float] | None = None,
    bonus: float = 0.0,
) -> dict[str, object]:
    test_rows = load_test_reviews(TEST_REVIEWS)
    maps = [load_candidates(path) for _, path in sources]
    merged = merge_candidate_maps(maps, weights=weights, average_present=True, bonus=bonus)
    predictions = threshold_predictions(
        merged,
        explicit_threshold,
        implicit_threshold,
        state_thresholds=state_thresholds,
    )
    write_submission(output_path, test_rows, predictions)
    report = validate_submission(output_path, test_rows)
    raw = output_path.read_bytes()
    return {
        "output": str(output_path),
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "weights": weights,
        "bonus": bonus,
        "thresholds": {"explicit": explicit_threshold, "implicit": implicit_threshold, "state": state_thresholds},
        "candidate_ids_covered": len(merged),
        "predicted_ids": sum(1 for values in predictions.values() if values),
        "predicted_quadruples": sum(len(values) for values in predictions.values()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "validation": asdict(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-source", action="append", required=True, help="name=path/test_candidates.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bonus", type=float, default=0.0)
    parser.add_argument("--thresholds", type=Path, required=True, help="OOF report/threshold json with fold-safe thresholds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    sources = []
    for item in args.test_source:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--test-source must be name=path, got {item!r}")
        sources.append((name, Path(path)))
    payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    explicit = payload.get("threshold") or payload["crossfit"]["folds"][0]["threshold"]
    implicit = payload.get("implicit_threshold") or payload["crossfit"]["folds"][0]["implicit_threshold"]
    state_thresholds = payload.get("state_thresholds")
    weights = [float(value) for value in args.weights.split(",")] if args.weights else None
    report = build_submission(
        sources,
        output_path=args.output,
        explicit_threshold=float(explicit),
        implicit_threshold=float(implicit),
        state_thresholds=state_thresholds,
        weights=weights,
        bonus=args.bonus,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
