from opinion_mining.data import Quadruple
from opinion_mining.metrics import strict_f1


def q(a="价格", o="便宜", c="价格", p="正面"):
    return Quadruple(a, o, c, p)


def test_strict_f1_requires_all_four_fields_to_match():
    gold = {1: {q()}}
    pred = {1: {q(p="负面")}}
    score = strict_f1(gold, pred)
    assert score.correct == 0
    assert score.predicted == 1
    assert score.gold == 1
    assert score.f1 == 0.0


def test_strict_f1_deduplicates_predictions():
    gold = {1: {q()}}
    pred = {1: [q(), q()]}
    score = strict_f1(gold, pred)
    assert score.correct == 1
    assert score.predicted == 1
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_strict_f1_accepts_implicit_aspect_literal_underscore():
    item = q(a="_", o="很好", c="整体", p="正面")
    score = strict_f1({7: {item}}, {7: {item}})
    assert score.f1 == 1.0


def test_strict_f1_handles_empty_predictions_without_zero_division():
    score = strict_f1({1: {q()}}, {})
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
