from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from sklearn.model_selection import KFold

from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple, ReviewExample
from opinion_mining.metrics import Score


def load_module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_improved_champion_promotion_writes_new_result_instead_of_old_file(tmp_path, monkeypatch):
    module = load_module("tianchi_autopilot_p0", "scripts/autopilot.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    old_result = tmp_path / "Result.csv"
    old_result.write_text("1,旧方面,旧观点,价格,负面\n", encoding="utf-8")

    new_quad = Quadruple("价格", "很好", "价格", "正面")
    new_candidates = {1: [Candidate(new_quad, 0.91, ("neural_pointer_beam",))]}
    selected = {
        "threshold": 0.1,
        "implicit_threshold": 0.1,
        "score": Score(precision=1.0, recall=1.0, f1=1.0, correct=1, predicted=1, gold=1),
    }

    class Dashboard:
        def event(self, message):
            pass

        def update(self, **kwargs):
            pass

    autopilot = object.__new__(module.Autopilot)
    autopilot.submissions = tmp_path / "submissions"
    autopilot.reports = tmp_path / "reports"
    autopilot.test_rows = [ReviewExample(1, "价格很好", frozenset())]
    autopilot.args = SimpleNamespace(train_zip="train.zip", test_zip="test.zip", budget_hours=1.0, evolution_epochs=1)
    autopilot.dashboard = Dashboard()
    autopilot.champion_f1 = 0.5
    autopilot.guard = SimpleNamespace(release_cache=lambda: None)
    stop_calls = iter([False, True])
    autopilot.stopped = lambda: next(stop_calls)
    autopilot.run_neural_cv = lambda **kwargs: {
        "record": {"f1": 0.6, "seconds": 0.0},
        "selected": selected,
        "test_maps": [new_candidates],
        "oof": {},
    }
    autopilot._write_submission = module.Autopilot._write_submission.__get__(autopilot)

    autopilot.evolution_loop(None, {"name": "old"}, module.NeuralConfig(epochs=1))

    expected = "1,价格,很好,价格,正面\n"
    assert (tmp_path / "submissions" / "Result.csv").read_text(encoding="utf-8") == expected
    assert old_result.read_text(encoding="utf-8") == expected
    assert autopilot.last_submission["sha256"]


def _calibration_module():
    return load_module("tianchi_calibration_p0", "scripts/calibrate_group_thresholds.py")


def test_lofo_fits_calibration_and_shrink_on_other_folds_only(monkeypatch):
    module = _calibration_module()
    rows = [ReviewExample(index, f"文本{index}", frozenset()) for index in range(1, 6)]
    quad = Quadruple("方面", "观点", "价格", "正面")
    wrong = Quadruple("方面", "观点", "价格", "负面")
    oof = {row.id: [Candidate(quad, 0.8), Candidate(wrong, 0.2)] for row in rows}
    gold = {row.id: {quad} for row in rows}
    initial = {"threshold": 0.7, "implicit_threshold": 0.3}

    calibrate_calls = []
    shrink_calls = []
    original_calibrate = module.calibrate
    original_shrink = module.shrink_thresholds

    def traced_calibrate(fit_oof, fit_gold, fit_initial):
        calibrate_calls.append((set(fit_oof), set(fit_gold)))
        return original_calibrate(fit_oof, fit_gold, fit_initial)

    def traced_shrink(thresholds, fit_oof, fit_gold, fit_initial, **kwargs):
        shrink_calls.append((set(fit_oof), set(fit_gold)))
        return original_shrink(thresholds, fit_oof, fit_gold, fit_initial, **kwargs)

    monkeypatch.setattr(module, "calibrate", traced_calibrate)
    monkeypatch.setattr(module, "shrink_thresholds", traced_shrink)

    score, fold_scores = module.lofo_score(oof, gold, rows, initial)

    all_ids = set(oof)
    expected_fit_sets = []
    for _, valid_indices in KFold(n_splits=5, shuffle=True, random_state=42).split(rows):
        valid_ids = {rows[int(index)].id for index in valid_indices}
        expected_fit_sets.append(all_ids - valid_ids)

    assert score.gold == len(rows)
    assert len(fold_scores) == 5
    assert [fit_ids for fit_ids, _ in calibrate_calls] == expected_fit_sets
    assert [gold_ids for _, gold_ids in calibrate_calls] == expected_fit_sets
    assert [fit_ids for fit_ids, _ in shrink_calls] == expected_fit_sets
    assert [gold_ids for _, gold_ids in shrink_calls] == expected_fit_sets


def test_lofo_uses_supplied_initial_thresholds_for_unseen_groups(monkeypatch):
    module = _calibration_module()
    rows = [ReviewExample(index, f"文本{index}", frozenset()) for index in range(1, 6)]
    seen = Quadruple("方面", "观点", "价格", "正面")
    unseen = Quadruple("_", "另一个观点", "整体", "负面")
    oof = {row.id: [Candidate(seen, 0.9), Candidate(unseen, 0.4)] for row in rows}
    gold = {row.id: {seen} for row in rows}
    initial = {"threshold": 0.8, "implicit_threshold": 0.6}

    monkeypatch.setattr(module, "calibrate", lambda *args, **kwargs: ({("explicit", "价格", "正面"): 0.8}, (1.0, 1.0, 1.0), 1, 1, 1))
    monkeypatch.setattr(module, "shrink_thresholds", lambda thresholds, *args, **kwargs: thresholds)

    score, _ = module.lofo_score(oof, gold, rows, initial)

    assert score.precision == 1.0
    assert score.recall == 1.0


def test_sparse_group_thresholds_shrink_toward_type_baseline():
    module = _calibration_module()
    initial = {"threshold": 0.7, "implicit_threshold": 0.3}
    sparse = ("explicit", "价格", "正面")
    dense = ("explicit", "质量", "正面")
    absent = ("implicit", "整体", "负面")
    thresholds = {sparse: 1.0, dense: 0.9, absent: 0.1}
    dense_quad = Quadruple("质量", "稳定", "质量", "正面")
    sparse_quad = Quadruple("价格", "便宜", "价格", "正面")
    gold = {index: {dense_quad} for index in range(60)}
    gold[60] = {sparse_quad}

    shrunk = module.shrink_thresholds(thresholds, {}, gold, initial, min_gold=60)

    assert shrunk[sparse] == round(0.7 + (1 / 60) * (1.0 - 0.7), 3)
    assert shrunk[dense] == 0.9
    assert shrunk[absent] == 0.3
