from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple
from opinion_mining.ensemble import filter_predictions_by_pair_support, soft_pair_support_candidates


def _candidate(aspect: str, opinion: str, category: str, polarity: str, score: float) -> Candidate:
    return Candidate(Quadruple(aspect, opinion, category, polarity), score)


def test_soft_pair_support_keeps_supported_and_penalizes_unsupported_candidates():
    base = {
        1: [
            _candidate("价格", "便宜", "价格", "正面", 0.80),
            _candidate("_", "好用", "整体", "正面", 0.80),
        ]
    }
    verifier = {1: [_candidate("价格", "便宜", "整体", "负面", 0.72)]}
    gates = {"explicit": 0.72, "implicit-A": 0.74, "implicit-O": 0.58, "dual-implicit": 1.0}

    fused = soft_pair_support_candidates(base, verifier, gates, support_floor=0.5)

    scores = {item.quadruple: item.score for item in fused[1]}
    assert scores[base[1][0].quadruple] == 0.80
    assert scores[base[1][1].quadruple] == 0.40


def test_soft_pair_support_interpolates_partial_pair_evidence():
    base = {1: [_candidate("价格", "便宜", "价格", "正面", 0.80)]}
    verifier = {1: [_candidate("价格", "便宜", "整体", "负面", 0.36)]}
    gates = {"explicit": 0.72, "implicit-A": 0.74, "implicit-O": 0.58, "dual-implicit": 1.0}

    fused = soft_pair_support_candidates(base, verifier, gates, support_floor=0.5)

    assert abs(fused[1][0].score - 0.60) < 1e-9


def test_pair_support_filter_uses_state_gate_scale_without_requiring_class_match():
    keep = Quadruple("价格", "便宜", "价格", "正面")
    drop = Quadruple("_", "好用", "整体", "正面")
    base_predictions = {1: {keep, drop}}
    verifier = {
        1: [
            _candidate("价格", "便宜", "整体", "负面", 0.40),
            _candidate("_", "好用", "整体", "负面", 0.20),
        ]
    }
    gates = {"explicit": 0.72, "implicit-A": 0.74, "implicit-O": 0.58, "dual-implicit": 1.0}

    filtered = filter_predictions_by_pair_support(
        base_predictions, verifier, gates, gate_scale=0.5
    )

    assert filtered[1] == {keep}
