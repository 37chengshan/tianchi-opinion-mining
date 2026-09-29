from types import SimpleNamespace

import torch

from opinion_mining import neural
from opinion_mining.data import ReviewExample
from opinion_mining.opinionet_decoder import constrained_endpoint_beam, decode_opinionet_candidates


def test_constrained_beam_rejects_reverse_and_overlong_spans():
    start = torch.tensor([0.0, 0.9, 0.8, 0.7])
    end = torch.tensor([0.0, 0.7, 0.8, 0.9])

    pairs = constrained_endpoint_beam(start, end, [0, 1, 2, 3], max_span_length=2, top_k=8)

    assert all((left, right) == (0, 0) or (left <= right and right - left + 1 <= 2) for left, right, _ in pairs)
    assert (3, 1) not in {(left, right) for left, right, _ in pairs}


def test_decoder_keeps_distinct_relations_for_shared_geometry():
    row = ReviewExample(1, "好用", frozenset())
    feature = SimpleNamespace(offsets=[(0, 0), (0, 1), (1, 2)])
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
    outputs["relation"] = torch.full((1, sequence_length, len(neural.CLASS_PAIRS)), -10.0)
    outputs["relation"][0, 1, :3] = 10.0

    candidates = decode_opinionet_candidates(
        row,
        feature,
        outputs,
        row_index=0,
        max_span_length=4,
        span_top_k=1,
        beam_top_k=1,
        relation_top_k=3,
        nms_overlap=0.8,
    )

    high_confidence = [item for item in candidates if item.score > 0.5]
    assert len(high_confidence) == 3
    assert len(candidates) >= 3
    assert all(0.0 <= item.score <= 1.0 for item in candidates)
    assert len({item.quadruple.category + item.quadruple.polarity for item in candidates}) == 3
