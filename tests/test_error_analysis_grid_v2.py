from __future__ import annotations

from scripts.error_analysis_grid_v2 import audit_candidate_map
from opinion_mining.data import Quadruple


def q(a: str, o: str, c: str = "功能", p: str = "正面") -> Quadruple:
    return Quadruple(a, o, c, p)


def test_all_six_fn_fp_buckets_and_recoverability() -> None:
    gold = {1: {q("a", "o")}, 2: {q("_", "good")}, 3: {q("screen", "_")}, 4: {q("x", "y")}, 5: {q("z", "ok", "功能", "正面")}, 6: {q("w", "fine", "功能", "正面")}}
    candidates = {1: [q("a", "other")], 2: [q("_", "bad")], 3: [q("other", "_")], 4: [q("u", "v")], 5: [q("z", "ok", "价格", "正面")], 6: [q("w", "fine", "功能", "负面")]}
    report = audit_candidate_map(candidates, gold)
    buckets = report["buckets"]
    for name in ("explicit_span", "implicit-A", "implicit-O", "pair", "category", "polarity"):
        assert buckets[name]["fn"] == 1
        assert buckets[name]["fp"] == 1
    for name in ("explicit_span", "implicit-A", "implicit-O", "pair"):
        assert buckets[name]["theoretical_recoverable_tp"] == 0
    assert buckets["category"]["theoretical_recoverable_tp"] == 1
    assert buckets["polarity"]["theoretical_recoverable_tp"] == 1
    assert report["category_confusion"] == {"功能->价格": 1}
    assert report["polarity_confusion"] == {"正面->负面": 1}


def test_pair_and_implicit_oracles_are_unique_pair_counts() -> None:
    gold = {1: {q("a", "o", "功能", "正面"), q("a", "o", "功能", "负面"), q("_", "good", "整体", "正面")}}
    candidates = {1: [q("a", "o", "功能", "中性"), q("_", "good", "整体", "负面")]}
    report = audit_candidate_map(candidates, gold)
    assert report["pair_gold_count"] == 2
    assert report["pair_hit_count"] == 2
    assert report["pair_recall"] == 1.0
    assert report["implicit_gold_count"] == 1
    assert report["implicit_pair_hit_count"] == 1
    assert report["implicit_recall"] == 1.0
    assert report["buckets"]["polarity"]["theoretical_recoverable_tp"] == 3


def test_zero_implicit_samples_are_distinguished_from_zero_recall() -> None:
    report = audit_candidate_map({1: []}, {1: {q("a", "o")}})
    assert report["implicit_gold_count"] == 0
    assert report["implicit_pair_hit_count"] == 0
    assert report["implicit_recall"] == 0.0

