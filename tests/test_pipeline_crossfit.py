from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple
from opinion_mining import pipeline


def test_crossfit_thresholds_are_fit_without_the_held_out_fold(monkeypatch):
    quad = Quadruple("价格", "便宜", "价格", "正面")
    candidates = {
        1: [Candidate(quad, 0.9)],
        2: [Candidate(quad, 0.8)],
        3: [Candidate(quad, 0.7)],
        4: [Candidate(quad, 0.6)],
    }
    gold = {rid: {quad} for rid in candidates}
    calls = []

    def fake_select(fit_candidates, fit_gold, *, separate_implicit=True):
        calls.append((set(fit_candidates), set(fit_gold)))
        return {
            "threshold": 0.0,
            "implicit_threshold": 0.0,
            "score": pipeline.strict_f1(fit_gold, {rid: {quad} for rid in fit_gold}),
            "predictions": {rid: {quad} for rid in fit_gold},
        }

    monkeypatch.setattr(pipeline, "select_threshold", fake_select)
    result = pipeline.crossfit_threshold_score(candidates, gold, [{1, 2}, {3, 4}])

    assert calls == [({3, 4}, {3, 4}), ({1, 2}, {1, 2})]
    assert result["score"].f1 == 1.0
    assert len(result["folds"]) == 2


def test_threshold_predictions_support_four_explicit_implicit_states():
    explicit = Quadruple("价格", "便宜", "价格", "正面")
    implicit_a = Quadruple("_", "便宜", "价格", "正面")
    implicit_o = Quadruple("价格", "_", "价格", "正面")
    dual = Quadruple("_", "_", "整体", "中性")
    candidates = {1: [
        Candidate(explicit, 0.75),
        Candidate(implicit_a, 0.55),
        Candidate(implicit_o, 0.65),
        Candidate(dual, 0.45),
    ]}

    pred = pipeline.threshold_predictions(
        candidates,
        threshold=0.7,
        implicit_threshold=0.5,
        state_thresholds={
            "explicit": 0.7,
            "implicit-A": 0.5,
            "implicit-O": 0.6,
            "dual-implicit": 0.4,
        },
    )

    assert pred[1] == {explicit, implicit_a, implicit_o, dual}
    assert pipeline.quadruple_state(explicit) == "explicit"
    assert pipeline.quadruple_state(implicit_a) == "implicit-A"
    assert pipeline.quadruple_state(implicit_o) == "implicit-O"
    assert pipeline.quadruple_state(dual) == "dual-implicit"


def test_plain_implicit_threshold_applies_to_implicit_opinion_too():
    implicit_o = Quadruple("价格", "_", "价格", "正面")
    candidates = {1: [Candidate(implicit_o, 0.6)]}
    pred = pipeline.threshold_predictions(candidates, threshold=0.9, implicit_threshold=0.5)
    assert implicit_o in pred[1]


def test_select_threshold_learns_separate_implicit_opinion_state():
    explicit = Quadruple("价格", "便宜", "价格", "正面")
    implicit_o = Quadruple("价格", "_", "价格", "正面")
    wrong_io = Quadruple("物流", "_", "物流", "负面")
    candidates = {
        1: [Candidate(explicit, 0.90)],
        2: [Candidate(implicit_o, 0.40), Candidate(wrong_io, 0.20)],
    }
    gold = {1: {explicit}, 2: {implicit_o}}

    selected = pipeline.select_threshold(candidates, gold, separate_implicit=True)

    assert set(selected["state_thresholds"]) == {"explicit", "implicit-A", "implicit-O", "dual-implicit"}
    assert selected["state_thresholds"]["implicit-O"] <= 0.40
    assert selected["state_thresholds"]["dual-implicit"] == 1.0
    assert implicit_o in selected["predictions"][2]
    assert wrong_io not in selected["predictions"][2]


def test_four_state_calibration_optimizes_global_f1_not_each_state_in_isolation():
    candidates = {}
    gold = {}
    for rid in range(1, 11):
        quad = Quadruple(f"a{rid}", f"o{rid}", "整体", "正面")
        candidates[rid] = [Candidate(quad, 0.9)]
        gold[rid] = {quad}
    implicit_good = Quadruple("_", "good", "整体", "正面")
    implicit_bad_1 = Quadruple("_", "bad1", "整体", "负面")
    implicit_bad_2 = Quadruple("_", "bad2", "整体", "负面")
    candidates[11] = [
        Candidate(implicit_good, 0.5),
        Candidate(implicit_bad_1, 0.5),
        Candidate(implicit_bad_2, 0.5),
    ]
    gold[11] = {implicit_good}

    selected = pipeline.select_threshold(candidates, gold, separate_implicit=True)

    assert selected["state_thresholds"]["implicit-A"] > 0.5
    assert selected["score"].f1 > 0.95


def test_crossfit_requires_complete_oof_fold_coverage():
    quad = Quadruple("价格", "便宜", "价格", "正面")
    candidates = {1: [Candidate(quad, 0.9)], 2: [Candidate(quad, 0.8)]}
    gold = {1: {quad}, 2: {quad}}

    try:
        pipeline.crossfit_threshold_score(candidates, gold, [{1}])
    except ValueError as exc:
        assert "do not cover every OOF id" in str(exc)
    else:
        raise AssertionError("incomplete crossfit fold coverage must fail")
