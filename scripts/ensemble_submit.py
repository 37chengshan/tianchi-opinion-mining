#!/usr/bin/env python3
"""Pick a fold-safe ensemble of strong runs and emit a validated submission.

Steps:
1. Cross-fit several weight configurations on the OOF candidate maps and keep
   the best by leak-free F1.
2. Re-fit the decision thresholds on the full OOF map of that configuration
   (thresholds only; the reported score stays the cross-fit one).
3. Merge the matching test candidate maps, write + validate Result.csv, and
   register the candidate in the local online queue.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from opinion_mining.analysis import load_candidates, merge_candidate_maps
from opinion_mining.data import load_test_reviews, load_train_data
from opinion_mining.pipeline import select_threshold, threshold_predictions
from opinion_mining.submission import validate_submission, write_submission


ROOT = Path(__file__).resolve().parents[1]


def _gold() -> dict[int, set]:
    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    return {row.id: set(row.labels) for row in rows}


def _load_sibling(name: str):
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"local_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest_helper():
    import importlib.util

    path = Path(__file__).resolve().parent / "submission_manifest.py"
    spec = importlib.util.spec_from_file_location("submission_manifest_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="experiment directory with oof/test candidates")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weight-configs", default="1,1|1,0.7|0.7,1|1,0.5|0.5,1")
    args = parser.parse_args()

    evaluate_ensemble = _load_sibling("ensemble_offline").evaluate_ensemble
    run_dirs = [Path(item) for item in args.run]
    sources = [(path.name, path / "oof_candidates.json") for path in run_dirs]
    configs = [[float(value) for value in chunk.split(",")] for chunk in args.weight_configs.split("|")]
    candidates = []
    for weights in configs:
        if len(weights) != len(sources):
            continue
        report = evaluate_ensemble(sources, weights=weights, bonus=0.0, n_splits=args.n_splits)
        candidates.append({"weights": weights, "crossfit_f1": report["crossfit"]["f1"], "report": report})
    if not candidates:
        raise SystemExit("no valid weight configuration")
    best = max(candidates, key=lambda item: item["crossfit_f1"])

    gold = _gold()
    oof_maps = [load_candidates(path) for _, path in sources]
    merged_oof = merge_candidate_maps(oof_maps, weights=best["weights"], average_present=True, bonus=0.0)
    selected = select_threshold(merged_oof, gold)

    test_sources = [(path.name, path / "test_candidates.json") for path in run_dirs]
    test_maps = [load_candidates(path) for _, path in test_sources]
    merged_test = merge_candidate_maps(test_maps, weights=best["weights"], average_present=True, bonus=0.0)
    predictions = threshold_predictions(
        merged_test,
        float(selected["threshold"]),
        float(selected["implicit_threshold"]),
        state_thresholds=selected.get("state_thresholds"),
    )

    output_dir = args.output_dir or (ROOT / "artifacts/submissions/candidates" / args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Result.csv"
    test_rows = load_test_reviews(ROOT / "artifacts/data/test/TEST/Test_reviews.csv")
    write_submission(csv_path, test_rows, predictions)
    validation = validate_submission(csv_path, test_rows)
    raw = csv_path.read_bytes()
    summary = {
        "name": args.name,
        "runs": [str(path) for path in run_dirs],
        "n_splits": args.n_splits,
        "weight_candidates": [{"weights": item["weights"], "crossfit_f1": item["crossfit_f1"]} for item in candidates],
        "chosen_weights": best["weights"],
        "crossfit_f1": best["crossfit_f1"],
        "crossfit_detail": best["report"]["crossfit"],
        "in_sample_thresholds": {
            "explicit": selected["threshold"],
            "implicit": selected["implicit_threshold"],
            "state": selected.get("state_thresholds"),
            "in_sample_f1_not_strict": selected["score"].f1,
        },
        "submission": {
            "path": str(csv_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "validation": asdict(validation),
        },
    }
    (output_dir / "submission_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    register = _load_manifest_helper()
    info = register.create_candidate_manifest(
        csv_path,
        output_dir / "submission_report.json",
        run_dirs[0] / "config.json",
        ROOT / "artifacts/submissions/online_loop",
        candidate_dir=ROOT / "artifacts/submissions/candidates",
        local_score=float(best["crossfit_f1"]),
        expected_test_ids=[row.id for row in test_rows],
    )
    summary["candidate_id"] = info["candidate_id"]
    (output_dir / "submission_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"name": args.name, "candidate_id": info["candidate_id"], "chosen_weights": best["weights"], "crossfit_f1": best["crossfit_f1"], "sha256": summary["submission"]["sha256"], "rows": validation.rows, "predicted": validation.predicted_quadruples}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
