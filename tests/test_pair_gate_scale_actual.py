import json
from pathlib import Path

import pytest

from opinion_mining.analysis import load_candidates
from opinion_mining.data import load_train_data
from scripts.evaluate_pair_gate_scale import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_actual_pair_gate_scale_nested_diagnostic():
    base_path = ROOT / "artifacts/experiments/grid_wwm_3fold_20260928/oof_candidates.json"
    verifier_path = ROOT / "artifacts/experiments/grid_wwm_v21_grid_redecode_3fold/oof_candidates.json"
    folds_path = ROOT / "artifacts/experiments/grid_wwm_v2code_3fold/fold_assignments.json"
    if not (base_path.exists() and verifier_path.exists() and folds_path.exists()):
        pytest.skip("local competition artifacts are not present")

    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    gold = {row.id: set(row.labels) for row in rows}
    base = load_candidates(base_path)
    verifier = load_candidates(verifier_path)
    fold_payload = json.loads(folds_path.read_text(encoding="utf-8"))
    folds = [item["valid_ids"] for item in fold_payload]

    report = evaluate(base, verifier, gold, folds)
    fine_report = evaluate(base, verifier, gold, folds, scales=(0.15, 0.20, 0.25, 0.30, 0.35))

    print("PAIR_GATE_NESTED=" + json.dumps(report, ensure_ascii=False))
    print("PAIR_GATE_FINE=" + json.dumps(fine_report, ensure_ascii=False))
    assert report["crossfit_score"]["f1"] > 0.0
    assert fine_report["crossfit_score"]["f1"] > 0.0
