from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from opinion_mining.data import LabelSpan, Quadruple, ReviewExample, load_train_data
from opinion_mining import neural


class DummyTokenizer:
    """One token per character, plus a CLS token and padded offsets."""

    def __call__(self, text, *, max_length, **kwargs):
        offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))]
        offsets = offsets[:max_length]
        offsets.extend([(0, 0)] * (max_length - len(offsets)))
        return {
            "input_ids": [0] * max_length,
            "attention_mask": [1] * min(len(text) + 1, max_length)
            + [0] * max(0, max_length - len(text) - 1),
            "offset_mapping": offsets,
        }


def test_dataset_uses_official_offsets_for_repeated_aspect_and_opinion(tmp_path):
    reviews = tmp_path / "Train_reviews.csv"
    labels = tmp_path / "Train_labels.csv"
    reviews.write_text("id,Reviews\n1,好用便宜好用便宜\n", encoding="utf-8")
    labels.write_text(
        "id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n"
        "1,好用,0,2,便宜,2,4,价格,正面\n"
        "1,好用,4,6,便宜,6,8,价格,正面\n",
        encoding="utf-8",
    )
    row = load_train_data(reviews, labels)[0]
    assert [(item.aspect_start, item.aspect_end, item.opinion_start, item.opinion_end) for item in row.label_spans] == [
        (0, 2, 2, 4),
        (4, 6, 6, 8),
    ]

    feature = neural._ReviewDataset([row], DummyTokenizer(), 12, use_official_offsets=True)[0]
    quad = row.label_spans[0].quadruple
    class_id = neural.CLASS_TO_ID[(quad.category, quad.polarity)]

    assert feature.pairs == [(1, 2, 3, 4, class_id), (5, 6, 7, 8, class_id)]
    assert feature.aspect_start[5:7] == [5, 5]
    assert feature.opinion_start[7:9] == [7, 7]
    assert feature.relation[3][class_id] == 1.0
    assert feature.relation[7][class_id] == 1.0


def test_explicit_aspect_with_unmappable_official_offset_is_not_downgraded_to_implicit():
    quad = Quadruple("好用", "很好", "整体", "正面")
    row = ReviewExample(
        1,
        "好用很好",
        frozenset({quad}),
        (LabelSpan(quad, 99, 101, 2, 4),),
    )

    feature = neural._ReviewDataset([row], DummyTokenizer(), 8, use_official_offsets=True)[0]

    assert feature.pairs == []


def test_implicit_opinion_is_supervised_at_cls_and_explicit_aspect_tokens():
    quad = Quadruple("好用", "_", "整体", "中性")
    row = ReviewExample(1, "好用", frozenset({quad}), (LabelSpan(quad, 0, 2),))

    feature = neural._ReviewDataset([row], DummyTokenizer(), 8, use_implicit_opinion_sentinel=True)[0]
    class_id = neural.CLASS_TO_ID[(quad.category, quad.polarity)]

    for token_index in (0, 1, 2):
        assert feature.opinion_start[token_index] == 0
        assert feature.opinion_end[token_index] == 0
        assert feature.relation[token_index][class_id] == 1.0
    assert feature.aspect_start[1:3] == [1, 1]
    assert feature.aspect_end[1:3] == [2, 2]


def test_dual_implicit_and_shared_span_relations_remain_multi_hot():
    first = Quadruple("_", "_", "整体", "正面")
    second = Quadruple("_", "_", "功效", "负面")
    row = ReviewExample(
        1,
        "很好",
        frozenset({first, second}),
        (LabelSpan(first), LabelSpan(second)),
    )

    feature = neural._ReviewDataset([row], DummyTokenizer(), 8, use_implicit_opinion_sentinel=True)[0]
    first_id = neural.CLASS_TO_ID[(first.category, first.polarity)]
    second_id = neural.CLASS_TO_ID[(second.category, second.polarity)]

    assert feature.pairs == [(-1, -1, 0, 0, first_id), (-1, -1, 0, 0, second_id)]
    assert feature.objectiveness[0] == 1.0
    assert feature.relation[0][first_id] == 1.0
    assert feature.relation[0][second_id] == 1.0
    assert sum(feature.relation[0]) == 2.0


def test_endpoint_beam_keeps_cls_pair_and_enforces_span_boundaries():
    start = torch.tensor([0.99, 0.90, 0.80, 0.70])
    end = torch.tensor([0.99, 0.90, 0.80, 0.70])

    pairs = neural._endpoint_beam(
        start,
        end,
        [0, 1, 2, 3],
        max_span_length=2,
        top_k=4,
    )
    endpoints = {(left, right) for left, right, _ in pairs}

    assert (0, 0) in endpoints
    assert (1, 2) in endpoints
    assert (1, 3) not in endpoints
    assert all((left, right) == (0, 0) or (left >= 1 and right >= left) for left, right in endpoints)


@pytest.mark.parametrize(
    ("short_end", "expected_count"),
    [(4, 1), (3, 2)],
    ids=["exactly-eighty-percent-is-suppressed", "below-eighty-percent-is-kept"],
)
def test_pointer_nms_boundary_is_inclusive_at_point_eight(monkeypatch, short_end, expected_count):
    row = ReviewExample(1, "甲乙丙丁戊己", frozenset())
    feature = SimpleNamespace(
        offsets=[(0, 0)] + [(index, index + 1) for index in range(6)] + [(0, 0)] * 3
    )
    config = neural.NeuralConfig(
        max_span_length=10,
        span_top_k=1,
        beam_top_k=2,
        relation_top_k=1,
    )
    sequence_length = len(feature.offsets)
    class_id = neural.CLASS_TO_ID[("整体", "正面")]
    outputs = {
        name: torch.zeros((1, sequence_length, sequence_length))
        for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end")
    }
    outputs["objectiveness"] = torch.full((1, sequence_length), -10.0)
    outputs["objectiveness"][0, 1] = 10.0
    outputs["category"] = torch.zeros((1, sequence_length, len(neural.CATEGORY_ORDER)))
    outputs["polarity"] = torch.zeros((1, sequence_length, len(neural.POLARITY_ORDER)))
    outputs["relation"] = torch.full((1, sequence_length, len(neural.CLASS_PAIRS)), -10.0)
    outputs["relation"][:, :, class_id] = 10.0

    calls = {"count": 0}

    def fake_endpoint_beam(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] % 2:
            return [(1, 5, 1.0), (1, short_end, 0.8)]
        return [(6, 6, 1.0)]

    monkeypatch.setattr(neural, "_endpoint_beam", fake_endpoint_beam)
    candidates = neural._pointer_candidates_from_output(row, feature, outputs, 0, config)

    assert len(candidates) == expected_count
    if expected_count == 1:
        assert candidates[0].quadruple.aspect == "甲乙丙丁戊"
    else:
        assert {item.quadruple.aspect for item in candidates} == {"甲乙丙丁戊", "甲乙丙"}
