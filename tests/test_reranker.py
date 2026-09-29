from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple, ReviewExample
from opinion_mining.reranker import build_rows, feature_names, labels_for, matrix, quadruple_state


QUAD = Quadruple("价格", "便宜", "价格", "正面")
IMPLICIT = Quadruple("_", "便宜", "价格", "正面")


def test_build_rows_merges_sources_and_keeps_score_features():
    reviews = {1: ReviewExample(1, "价格便宜", frozenset({QUAD}))}
    maps = [
        {1: [Candidate(QUAD, 0.4), Candidate(IMPLICIT, 0.9)]},
        {1: [Candidate(QUAD, 0.7)]},
    ]

    rows = build_rows(maps, reviews)

    assert len(rows) == 2
    names = feature_names(len(maps))
    assert matrix(rows).shape == (2, len(names))
    offset = len(names) - 2 * len(maps)
    shared = next(row for row in rows if row.quadruple == QUAD)
    assert shared.features[0] == 0.7
    assert shared.features[offset] == 0.4 and shared.features[offset + 1] == 0.7
    assert shared.features[offset + len(maps)] == 1.0 and shared.features[offset + len(maps) + 1] == 1.0
    only_first = next(row for row in rows if row.quadruple == IMPLICIT)
    assert only_first.features[offset] == 0.9 and only_first.features[offset + 1] == 0.0
    assert only_first.features[offset + len(maps)] == 1.0 and only_first.features[offset + len(maps) + 1] == 0.0


def test_labels_and_state_helpers():
    reviews = {1: ReviewExample(1, "价格便宜", frozenset({QUAD}))}
    rows = build_rows([{1: [Candidate(QUAD, 0.4), Candidate(IMPLICIT, 0.9)]}], reviews)

    labels = labels_for(rows, {1: {QUAD}})

    assert sorted(labels.tolist()) == [0, 1]
    assert quadruple_state(QUAD) == "explicit"
    assert quadruple_state(IMPLICIT) == "implicit-A"
