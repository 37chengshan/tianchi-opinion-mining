from types import SimpleNamespace

import torch

from opinion_mining.data import LabelSpan, Quadruple, ReviewExample
from opinion_mining import neural


class DummyTokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))]
        offsets = offsets[:max_length]
        offsets.extend([(0, 0)] * (max_length - len(offsets)))
        return {
            "input_ids": [0] * max_length,
            "attention_mask": [1] * min(max_length, len(text) + 1) + [0] * max(0, max_length - len(text) - 1),
            "offset_mapping": offsets,
        }


def test_b3_shared_span_targets_keep_distinct_relation_classes():
    first = Quadruple("好用", "很好", "整体", "正面")
    second = Quadruple("好用", "很好", "功效", "负面")
    row = ReviewExample(
        1,
        "好用很好",
        frozenset({first, second}),
        (LabelSpan(first, 0, 2, 2, 4), LabelSpan(second, 0, 2, 2, 4)),
    )

    feature = neural._ReviewDataset(
        [row],
        DummyTokenizer(),
        8,
        preserve_multi_relation=True,
    )[0]

    class_ids = {
        neural.CLASS_TO_ID[(first.category, first.polarity)],
        neural.CLASS_TO_ID[(second.category, second.polarity)],
    }
    assert {index for index, value in enumerate(feature.relation[1]) if value} == class_ids
    assert feature.pairs == [
        (1, 2, 3, 4, neural.CLASS_TO_ID[(first.category, first.polarity)]),
        (1, 2, 3, 4, neural.CLASS_TO_ID[(second.category, second.polarity)]),
    ]


def test_b3_decoder_keeps_multiple_relation_labels_for_one_span_pair():
    row = ReviewExample(1, "好用", frozenset())
    feature = SimpleNamespace(offsets=[(0, 0), (0, 1), (1, 2)])
    base = neural.NeuralConfig(
        max_span_length=10,
        span_top_k=1,
        beam_top_k=1,
        relation_top_k=2,
    )
    b3 = neural.NeuralConfig(
        max_span_length=10,
        span_top_k=1,
        beam_top_k=1,
        relation_top_k=2,
        preserve_multi_relation=True,
    )
    sequence_length = len(feature.offsets)
    outputs = {
        name: torch.full((1, sequence_length, sequence_length), -10.0)
        for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end")
    }
    outputs["aspect_start"][0, 1, 1] = 10.0
    outputs["aspect_end"][0, 1, 1] = 10.0
    outputs["opinion_start"][0, 1, 2] = 10.0
    outputs["opinion_end"][0, 1, 2] = 10.0
    outputs["objectiveness"] = torch.full((1, sequence_length), -10.0)
    outputs["objectiveness"][0, 1] = 10.0
    outputs["category"] = torch.zeros((1, sequence_length, len(neural.CATEGORY_ORDER)))
    outputs["polarity"] = torch.zeros((1, sequence_length, len(neural.POLARITY_ORDER)))
    outputs["relation"] = torch.full((1, sequence_length, len(neural.CLASS_PAIRS)), -10.0)
    outputs["relation"][0, 1, :3] = 10.0

    historical = neural._pointer_candidates_from_output(row, feature, outputs, 0, base)
    preserved = neural._pointer_candidates_from_output(row, feature, outputs, 0, b3)

    assert len({item.quadruple.category + item.quadruple.polarity for item in historical}) == 2
    assert len({item.quadruple.category + item.quadruple.polarity for item in preserved}) >= 3
