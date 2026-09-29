from opinion_mining.data import LabelSpan, Quadruple, ReviewExample
import torch
from torch import nn

from opinion_mining.neural import NeuralConfig, PointerQuadrupleModel, _ReviewDataset, _collate, _pointer_batch_loss


class DummyTokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))]
        offsets.extend([(0, 0)] * (max_length - len(offsets)))
        return {
            "input_ids": [0] * max_length,
            "attention_mask": [1] * (len(text) + 1) + [0] * (max_length - len(text) - 1),
            "offset_mapping": offsets,
        }


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.projection = nn.Embedding(16, 8)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return type("Output", (), {"last_hidden_state": self.projection(input_ids)})()


def test_official_offsets_are_isolated_behind_b1_flag():
    quad = Quadruple("好用", "便宜", "价格", "正面")
    row = ReviewExample(
        1,
        "好用便宜好用便宜",
        frozenset({quad}),
        (LabelSpan(quad, 4, 6, 6, 8),),
    )

    historical = _ReviewDataset([row], DummyTokenizer(), 16)[0]
    b1 = _ReviewDataset([row], DummyTokenizer(), 16, use_official_offsets=True)[0]

    from opinion_mining import neural
    class_id = neural.CLASS_TO_ID[(quad.category, quad.polarity)]
    assert historical.pairs == [(1, 2, 3, 4, class_id)]
    assert b1.pairs == [(5, 6, 7, 8, class_id)]


def test_neural_config_keeps_historical_offset_mode_by_default():
    assert NeuralConfig().use_official_offsets is False


def test_official_offsets_and_multiple_relations_are_multi_hot():
    first = Quadruple("好用", "很好", "整体", "正面")
    second = Quadruple("好用", "很好", "功效", "负面")
    row = ReviewExample(
        1,
        "好用很好",
        frozenset({first, second}),
        (
            LabelSpan(first, 0, 2, 2, 4),
            LabelSpan(second, 0, 2, 2, 4),
        ),
    )
    feature = _ReviewDataset([row], DummyTokenizer(), 8)[0]

    # Both explicit relations survive on the shared token instead of one
    # category/polarity pair overwriting the other.
    assert sum(feature.relation[1]) == 2.0
    assert feature.opinion_start[1] == 3
    assert feature.opinion_end[1] == 4


def test_implicit_opinion_uses_cls_pointer_without_dropping_relation():
    quad = Quadruple("好用", "_", "整体", "中性")
    row = ReviewExample(1, "好用", frozenset({quad}), (LabelSpan(quad, 0, 2),))
    feature = _ReviewDataset([row], DummyTokenizer(), 6, use_implicit_opinion_sentinel=True)[0]
    assert feature.opinion_start[1] == 0 and feature.opinion_end[1] == 0
    assert sum(feature.relation[1]) == 1.0


def test_dual_implicit_relation_uses_cls_anchor():
    quad = Quadruple("_", "_", "整体", "正面")
    row = ReviewExample(1, "很好", frozenset({quad}), (LabelSpan(quad),))
    feature = _ReviewDataset([row], DummyTokenizer(), 6, use_implicit_opinion_sentinel=True)[0]
    assert feature.objectiveness[0] == 1.0
    assert sum(feature.relation[0]) == 1.0
    assert feature.aspect_start[0] == 0 and feature.aspect_end[0] == 0
    assert feature.opinion_start[0] == 0 and feature.opinion_end[0] == 0


def test_pointer_loss_accepts_multi_label_relation_targets():
    quad = Quadruple("_", "很好", "整体", "正面")
    row = ReviewExample(1, "很好", frozenset({quad}), (LabelSpan(quad, None, None, 0, 2),))
    feature = _ReviewDataset([row], DummyTokenizer(), 6)[0]
    batch = _collate([feature])
    model = PointerQuadrupleModel(DummyEncoder(), trainable_layers=1)
    outputs = model(batch)
    loss, parts = _pointer_batch_loss(model, outputs, batch)
    assert outputs["relation"].shape[-1] == 39
    assert torch.isfinite(loss)
    assert parts["relation_loss"] >= 0
