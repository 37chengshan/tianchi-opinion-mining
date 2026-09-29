import json
from pathlib import Path

import pytest

from opinion_mining.analysis import load_candidates
from opinion_mining.anchor_eval import evaluate_anchor_variant
from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple, ReviewExample, load_train_data
from opinion_mining.folds import fixed_splits, load_assignment_records


EXPLICIT = Quadruple("价格", "便宜", "价格", "正面")
IMPLICIT_A = Quadruple("_", "便宜", "价格", "正面")
IMPLICIT_O = Quadruple("价格", "_", "价格", "正面")
ROOT = Path(__file__).resolve().parents[1]


def _folds():
    return [
        {
            "fold": 1,
            "train_ids": [3, 4],
            "valid_ids": [1, 2],
            "seed": 42,
            "n_splits": 2,
        },
        {
            "fold": 2,
            "train_ids": [1, 2],
            "valid_ids": [3, 4],
            "seed": 42,
            "n_splits": 2,
        },
    ]


def _candidates():
    return {
        1: [Candidate(EXPLICIT, 0.9), Candidate(IMPLICIT_A, 0.2)],
        2: [Candidate(IMPLICIT_O, 0.9)],
        3: [Candidate(IMPLICIT_A, 0.9), Candidate(EXPLICIT, 0.1)],
        4: [Candidate(EXPLICIT, 0.9)],
    }


def _gold():
    return {
        1: {EXPLICIT},
        2: {IMPLICIT_O},
        3: {IMPLICIT_A},
        4: {EXPLICIT},
    }


def _config(tmp_path: Path, *, name: str = "anchor_same_fold_baseline"):
    candidates = _candidates()
    return {
        "name": name,
        "candidates": candidates,
        "gold": _gold(),
        "threshold": 0.5,
        "implicit_threshold": 0.5,
        "output_dir": tmp_path,
        "fold_candidates": {
            1: {1: candidates[1], 2: candidates[2]},
            2: {3: candidates[3], 4: candidates[4]},
        },
    }


def test_evaluator_emits_stable_metrics_and_saves_oof_candidates(tmp_path):
    report = evaluate_anchor_variant(_config(tmp_path), _folds())

    assert report.f1 == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert (tmp_path / "oof_candidates.json").is_file()
    assert report.candidate_count == 6
    assert report.state_metrics["explicit"].tp == 2
    assert report.state_metrics["implicit-A"].tp == 1
    assert report.state_metrics["implicit-O"].tp == 1
    assert report.category_metrics["价格"].f1 == 1.0
    assert report.config_hash


def test_evaluator_reports_delta_against_anchor_baseline(tmp_path):
    baseline = evaluate_anchor_variant(_config(tmp_path), _folds())
    config = _config(tmp_path / "variant", name="anchor_variant")
    config["candidates"] = {rid: items[:1] for rid, items in config["candidates"].items()}
    config["fold_candidates"] = {
        fold: {rid: items[:1] for rid, items in rows.items()}
        for fold, rows in config["fold_candidates"].items()
    }

    report = evaluate_anchor_variant({**config, "baseline": baseline}, _folds())

    assert report.delta_vs_anchor_same_fold_baseline["f1"] == 0.0
    assert report.delta_vs_anchor_same_fold_baseline["tp"] == 0


def test_evaluator_rejects_candidate_scored_by_its_training_fold(tmp_path):
    config = _config(tmp_path)
    config["fold_candidates"] = {
        1: {1: config["candidates"][1], 3: config["candidates"][3]},
        2: {2: config["candidates"][2], 4: config["candidates"][4]},
    }

    with pytest.raises(ValueError, match="train_ids"):
        evaluate_anchor_variant(config, _folds())


@pytest.mark.skipif(
    not (ROOT / "artifacts/experiments/neural_5fold_confirm/oof_candidates.json").is_file(),
    reason="historical anchor OOF candidate map is not part of the repository",
)
def test_historical_anchor_score_uses_canonical_five_fold_assignments(tmp_path):
    train_rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    candidates = load_candidates(ROOT / "artifacts/experiments/neural_5fold_confirm/oof_candidates.json")
    assignments = load_assignment_records(
        ROOT / "artifacts/reports/fold_assignments_seed42.json",
        n_splits=5,
        seed=42,
    )
    fold_candidates = {
        int(item["fold"]): {int(rid): candidates[int(rid)] for rid in item["valid_ids"]}
        for item in assignments
    }
    threshold = json.loads(
        (ROOT / "artifacts/experiments/neural_5fold_confirm/threshold.json").read_text(encoding="utf-8")
    )

    report = evaluate_anchor_variant(
        {
            "name": "anchor_same_fold_baseline",
            "candidates": candidates,
            "gold": {row.id: row.labels for row in train_rows},
            "threshold": threshold["threshold"],
            "implicit_threshold": threshold["implicit_threshold"],
            "output_dir": tmp_path,
            "fold_candidates": fold_candidates,
        },
        assignments,
    )

    assert report.f1 == pytest.approx(0.7208976157082748, abs=1e-12)
    assert report.tp == 4626
    assert report.fp == 1576
    assert report.fn == 2006
    assert report.fold_isolation_verified is True
    assert set(report.state_metrics) == {"explicit", "implicit-A", "implicit-O", "dual-implicit"}


def test_fixed_splits_can_replay_assignment_file_without_regenerating(tmp_path):
    assignment_path = tmp_path / "folds.json"
    assignment_path.write_text(json.dumps({"2": _folds()}), encoding="utf-8")
    rows = [ReviewExample(rid, "") for rid in (1, 2, 3, 4)]

    splits = fixed_splits(rows, n_splits=2, seed=42, assignment_path=assignment_path)

    assert [[rows[index].id for index in valid] for _, valid in splits] == [[1, 2], [3, 4]]
