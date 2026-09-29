import torch

from opinion_mining.opinionet_head import OpinionetHead


def test_opinionet_head_emits_anchor_pointer_and_relation_shapes():
    head = OpinionetHead(hidden_size=8, relation_classes=39, pointer_hidden=12)
    hidden = torch.randn(2, 5, 8)

    outputs = head(hidden)

    assert outputs["objectiveness"].shape == (2, 5)
    for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end"):
        assert outputs[name].shape == (2, 5, 5)
    assert outputs["relation"].shape == (2, 5, 39)
    assert all(torch.isfinite(value).all() for value in outputs.values())
