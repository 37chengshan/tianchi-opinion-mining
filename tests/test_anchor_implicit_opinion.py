from pathlib import Path
from types import SimpleNamespace

import torch

from opinion_mining.data import LabelSpan, Quadruple, ReviewExample, load_train_data
from opinion_mining import neural


ROOT = Path(__file__).resolve().parents[1]


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


def test_explicit_aspect_and_implicit_opinion_round_trip_through_cls_sentinel():
    quad = Quadruple("好用", "_", "整体", "正面")
    row = ReviewExample(1, "好用", frozenset({quad}), (LabelSpan(quad, 0, 2),))

    feature = neural._ReviewDataset(
        [row],
        DummyTokenizer(),
        8,
        use_implicit_opinion_sentinel=True,
    )[0]

    assert feature.pairs == [(1, 2, 0, 0, neural.CLASS_TO_ID[("整体", "正面")])]
    assert feature.opinion_start[1:3] == [0, 0]
    assert feature.opinion_end[1:3] == [0, 0]


def test_all_172_implicit_opinion_gold_spans_enter_b2_targets():
    rows = load_train_data(
        ROOT / "artifacts/data/train/TRAIN/Train_reviews.csv",
        ROOT / "artifacts/data/train/TRAIN/Train_labels.csv",
    )
    implicit_opinion_spans = [span for row in rows for span in row.label_spans if span.quadruple.opinion == "_"]
    implicit_opinion_rows = [row for row in rows if any(span.quadruple.opinion == "_" for span in row.label_spans)]

    assert len(implicit_opinion_spans) == 172
    dataset = neural._ReviewDataset(
        implicit_opinion_rows,
        DummyTokenizer(),
        512,
        use_implicit_opinion_sentinel=True,
    )

    assert sum(
        1
        for feature in dataset.features
        for aspect_start, aspect_end, opinion_start, opinion_end, _ in feature.pairs
        if opinion_start == 0 and opinion_end == 0 and aspect_start >= 0 and aspect_end >= aspect_start
    ) == 172


def test_pointer_decoder_emits_exact_competition_sentinel_for_implicit_opinion():
    row = ReviewExample(1, "好用", frozenset())
    feature = SimpleNamespace(offsets=[(0, 0), (0, 1), (1, 2)])
    config = neural.NeuralConfig(
        max_span_length=10,
        span_top_k=1,
        beam_top_k=1,
        relation_top_k=1,
        use_implicit_opinion_sentinel=True,
    )
    sequence_length = len(feature.offsets)
    class_id = neural.CLASS_TO_ID[("整体", "正面")]
    outputs = {
        name: torch.full((1, sequence_length, sequence_length), -10.0)
        for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end")
    }
    outputs["aspect_start"][0, 1, 1] = 10.0
    outputs["aspect_end"][0, 1, 1] = 10.0
    outputs["opinion_start"][0, 1, 0] = 10.0
    outputs["opinion_end"][0, 1, 0] = 10.0
    outputs["objectiveness"] = torch.full((1, sequence_length), -10.0)
    outputs["objectiveness"][0, 1] = 10.0
    outputs["category"] = torch.zeros((1, sequence_length, len(neural.CATEGORY_ORDER)))
    outputs["polarity"] = torch.zeros((1, sequence_length, len(neural.POLARITY_ORDER)))
    outputs["relation"] = torch.full((1, sequence_length, len(neural.CLASS_PAIRS)), -10.0)
    outputs["relation"][0, 1, class_id] = 10.0

    candidates = neural._pointer_candidates_from_output(row, feature, outputs, 0, config)

    assert any(item.quadruple.opinion == "_" for item in candidates)
    assert all("[CLS]" not in item.quadruple.opinion for item in candidates)
