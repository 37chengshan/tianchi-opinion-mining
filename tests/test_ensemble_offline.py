import csv
import json
from pathlib import Path

from opinion_mining.analysis import save_candidates
from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple
from scripts.ensemble_offline import evaluate_ensemble, evaluate_fold_subset


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    reviews = root / "reviews.csv"
    labels = root / "labels.csv"
    rows = {
        1: "价格便宜",  # fold 1
        2: "价格便宜",
        3: "包装精致",  # folds 2/3
        4: "包装精致",
    }
    with reviews.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "Reviews"])
        for rid, text in rows.items():
            writer.writerow([rid, text])
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "AspectTerms", "A_start", "A_end", "OpinionTerms", "O_start", "O_end", "Categories", "Polarities"])
        order = {1: (0, 2, 2, 4), 2: (0, 2, 2, 4), 3: (0, 2, 2, 4), 4: (0, 2, 2, 4)}
        for rid, text in rows.items():
            aspect = text[:2]
            opinion = text[2:]
            a_start, a_end, o_start, o_end = order[rid]
            writer.writerow([rid, aspect, a_start, a_end, opinion, o_start, o_end, "整体", "正面"])

    assignments = root / "folds.json"
    records = {
        "3": [
            {"fold": 1, "train_ids": [3, 4], "valid_ids": [1, 2], "seed": 42, "n_splits": 3},
            {"fold": 2, "train_ids": [1, 2, 4], "valid_ids": [3], "seed": 42, "n_splits": 3},
            {"fold": 3, "train_ids": [1, 2, 3], "valid_ids": [4], "seed": 42, "n_splits": 3},
        ]
    }
    assignments.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    candidates = root / "candidates.json"
    gold_1 = Quadruple("价格", "便宜", "整体", "正面")
    gold_3 = Quadruple("包装", "精致", "整体", "正面")
    wrong = Quadruple("价格", "便宜", "整体", "负面")
    save_candidates(
        candidates,
        {
            1: [Candidate(gold_1, 0.60), Candidate(wrong, 0.55)],
            2: [Candidate(gold_1, 0.60), Candidate(wrong, 0.55)],
            3: [Candidate(gold_3, 0.90)],
            4: [Candidate(gold_3, 0.90)],
        },
    )
    return reviews, labels, assignments, candidates


def test_fold_subset_uses_other_folds_thresholds(tmp_path):
    reviews, labels, assignments, candidates = _write_fixture(tmp_path)

    report = evaluate_fold_subset(
        [("model", candidates)],
        weights=None,
        bonus=0.0,
        fold=1,
        assignments_path=assignments,
        n_splits=3,
        reviews_path=reviews,
        labels_path=labels,
    )

    assert report["coverage"] == {"target_ids": 2, "with_candidates": 2}
    assert report["thresholds"]["explicit"] >= 0.90
    assert report["score"]["f1"] == 0.0


def test_ensemble_thresholds_are_fitted_without_the_scored_fold(tmp_path):
    reviews, labels, assignments, candidates = _write_fixture(tmp_path)

    report = evaluate_ensemble(
        [("model", candidates)],
        weights=None,
        bonus=0.0,
        assignments_path=assignments,
        n_splits=3,
        reviews_path=reviews,
        labels_path=labels,
    )

    # Fold 1 must be scored with thresholds learned on folds 2+3 (0.90), which
    # drops its 0.60 candidates; fitting on its own rows would keep them.
    fold_one = next(item for item in report["crossfit"]["folds"] if item["fold"] == 1)
    assert fold_one["threshold"] >= 0.90
    assert report["crossfit"]["f1"] < report["in_sample_reference_not_strict"]["f1"]
    assert report["per_source"]["model"]["crossfit_f1"] == report["crossfit"]["f1"]
