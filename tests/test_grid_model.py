from __future__ import annotations

import pytest
import torch

from opinion_mining import grid_model as grid


def pair(aspect_start, aspect_end, opinion_start, opinion_end, class_id):
    return grid.PairSpan(
        aspect_start,
        aspect_end,
        opinion_start,
        opinion_end,
        class_id=class_id,
    )


def test_sparse_targets_cover_all_four_aspect_opinion_states():
    pairs = [
        pair(1, 1, 2, 2, 0),
        pair(-1, -1, 3, 3, 1),
        pair(4, 4, -1, -1, 2),
        pair(-1, -1, -1, -1, 3),
    ]

    targets = grid.build_sparse_relation_targets([pairs], max_length=6)

    assert targets.shape == (1, 6, 6, grid.NUM_RELATION_CLASSES)
    assert targets.indices.shape == (4, 4)
    assert targets.values.shape == (4,)
    assert targets.spans.shape == (4, 6)
    assert set(map(tuple, targets.indices.t().tolist())) == {
        (0, 0, 0, 3),
        (0, 0, 3, 1),
        (0, 1, 2, 0),
        (0, 4, 0, 2),
    }
    assert set(map(tuple, targets.spans.tolist())) == {
        (0, -1, -1, -1, -1, 3),
        (0, -1, -1, 3, 3, 1),
        (0, 1, 1, 2, 2, 0),
        (0, 4, 4, -1, -1, 2),
    }


def test_same_grid_cell_keeps_multiple_relation_classes():
    shared_span = [
        pair(1, 2, 3, 4, 0),
        pair(1, 2, 3, 4, 1),
    ]
    targets = grid.build_sparse_relation_targets([shared_span], max_length=6)

    assert targets.num_positives == 2
    dense = targets.to_dense()
    assert dense[0, 1, 3, 0] == 1.0
    assert dense[0, 1, 3, 1] == 1.0
    assert dense[0, 1, 3].sum() == 2.0

    pair_targets = grid.pair_target_matrix([shared_span[0]], targets)
    assert pair_targets.shape == (1, grid.NUM_RELATION_CLASSES)
    assert pair_targets[0, 0] == 1.0
    assert pair_targets[0, 1] == 1.0
    assert pair_targets[0].sum() == 2.0


def test_low_rank_head_forward_and_sparse_loss_backward_on_dummy_hidden():
    torch.manual_seed(7)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    head = grid.CompactPairRelationHead(hidden_size=8, rank=3, dropout=0.0)
    target_items = [
        [pair(1, 1, 2, 2, 0), pair(-1, -1, 3, 3, 1)],
        [pair(1, 2, -1, -1, 2), pair(-1, -1, -1, -1, 3)],
    ]
    targets = grid.build_sparse_relation_targets(target_items, max_length=6)

    grid_logits = head(hidden)
    assert grid_logits.shape == (2, 6, 6, grid.NUM_RELATION_CLASSES)
    pair_logits = head(
        hidden,
        torch.tensor(
            [
                [0, 1, 1, 2, 2],
                [0, -1, -1, 3, 3],
            ],
            dtype=torch.long,
        ),
    )
    assert pair_logits.shape == (2, grid.NUM_RELATION_CLASSES)

    loss = grid.sparse_grid_bce_loss(
        grid_logits,
        targets,
        negative_ratio=1.0,
        max_negative=64,
    )
    assert torch.isfinite(loss)
    loss.backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    parameter_grads = [parameter.grad for parameter in head.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in parameter_grads)


def test_topk_span_decode_respects_implicit_sentinel_valid_mask_and_length():
    start_logits = torch.tensor([5.0, 4.0, 2.0, 3.0, -10.0, -10.0])
    end_logits = torch.tensor([5.0, 2.0, 4.0, 3.0, -10.0, -10.0])
    valid_mask = torch.tensor([False, True, True, True, False, False])

    spans = grid.constrained_topk_span_decode(
        start_logits,
        end_logits,
        valid_mask=valid_mask,
        top_k=4,
        max_span_length=2,
        allow_implicit=True,
        min_score=0.5,
    )

    assert len(spans) <= 4
    assert any(span.implicit for span in spans)
    assert any((span.start, span.end) == (1, 2) for span in spans)
    assert any((span.start, span.end) == (3, 3) for span in spans)
    assert all(span.implicit or (span.start > 0 and span.end >= span.start and span.end - span.start + 1 <= 2) for span in spans)
    assert all(span.implicit or span.start in {1, 2, 3} for span in spans)


def test_topk_pair_decode_keeps_multiple_relations_at_one_span_pair():
    aspect = [grid.SpanCandidate(1, 2, 0.9)]
    opinion = [grid.SpanCandidate(4, 4, 0.8)]
    relation_logits = torch.full((6, 6, grid.NUM_RELATION_CLASSES), -8.0)
    relation_logits[1, 4, 0] = 7.0
    relation_logits[1, 4, 1] = 6.0

    candidates = grid.constrained_topk_pair_decode(
        aspect,
        opinion,
        relation_logits,
        relation_top_k=2,
        top_k=8,
        min_relation_score=0.5,
    )

    assert len(candidates) == 2
    assert {candidate.class_id for candidate in candidates} == {0, 1}
    assert all(candidate.aspect == aspect[0] for candidate in candidates)
    assert all(candidate.opinion == opinion[0] for candidate in candidates)


def test_nms_suppresses_exact_iou_boundary_but_keeps_below_boundary_and_other_class():
    long_aspect = grid.SpanCandidate(1, 5, 0.95)
    short_aspect = grid.SpanCandidate(1, 4, 0.80)
    opinion = grid.SpanCandidate(7, 7, 0.9)
    same_class_long = grid.GridCandidate(long_aspect, opinion, 0, 0.95)
    same_class_short = grid.GridCandidate(short_aspect, opinion, 0, 0.80)
    other_class_short = grid.GridCandidate(short_aspect, opinion, 1, 0.70)

    kept_at_boundary = grid.nms_grid_candidates(
        [same_class_short, same_class_long],
        iou_threshold=0.8,
    )
    kept_below_boundary = grid.nms_grid_candidates(
        [same_class_short, same_class_long],
        iou_threshold=0.81,
    )
    kept_class_aware = grid.nms_grid_candidates(
        [same_class_short, same_class_long, other_class_short],
        iou_threshold=0.8,
        class_aware=True,
    )

    assert grid._span_overlap(long_aspect, short_aspect) == 0.8
    assert len(kept_at_boundary) == 1
    assert kept_at_boundary[0] == same_class_long
    assert len(kept_below_boundary) == 2
    assert {item.class_id for item in kept_class_aware} == {0, 1}


def test_grid_candidate_conversion_uses_exact_offsets_for_repeated_terms():
    text = "好用便宜好用便宜"
    offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))]
    candidates = [
        grid.GridCandidate(
            grid.SpanCandidate(5, 6, 0.9),
            grid.SpanCandidate(7, 8, 0.8),
            0,
            0.7,
        ),
        grid.GridCandidate(
            grid.SpanCandidate(0, 0, 0.6, implicit=True),
            grid.SpanCandidate(7, 8, 0.8),
            1,
            0.6,
        ),
    ]

    converted = grid.grid_candidates_to_candidates(candidates, text=text, offsets=offsets)

    assert converted[0].quadruple.aspect == "好用"
    assert converted[0].quadruple.opinion == "便宜"
    assert converted[1].quadruple.aspect == "_"
    assert converted[1].quadruple.opinion == "便宜"
    assert all(item.quadruple.aspect in {"好用", "_"} for item in converted)


def test_implicit_pseudo_tokens_round_trip_to_competition_sentinel():
    assert grid.implicit_token_round_trip("_", "满意") == ("_", "满意")
    assert grid.implicit_token_round_trip("外观", "_") == ("外观", "_")


def test_hard_negative_generator_covers_boundary_and_implicit_confusion():
    positives = [pair(2, 2, 5, 5, 0), pair(-1, -1, 5, 5, 1)]
    negatives = grid.hard_negative_pairs(positives, sequence_length=8, max_count=32)
    assert negatives
    assert any(item.aspect_start == 1 for item in negatives)
    assert any(not item.implicit_aspect for item in negatives)
    assert any(item.opinion_start == 5 and not item.implicit_aspect for item in negatives)
    assert any(item.implicit_opinion and item.aspect_start == 2 for item in negatives)


def test_explicit_gold_generates_false_implicit_state_negatives():
    positives = [pair(2, 3, 5, 6, 0)]
    negatives = grid.hard_negative_pairs(positives, sequence_length=10, max_count=32)

    assert any(item.implicit_aspect and not item.implicit_opinion and item.opinion_start == 5 for item in negatives)
    assert any(item.implicit_opinion and not item.implicit_aspect and item.aspect_start == 2 for item in negatives)
    assert any(item.implicit_aspect and item.implicit_opinion for item in negatives)


def test_hard_negative_budget_balances_state_pair_and_boundary_errors():
    positives = [
        pair(1, 1, 4, 4, 0),
        pair(2, 2, 5, 5, 1),
        pair(3, 3, 6, 6, 2),
    ]
    negatives = grid.hard_negative_pairs(positives, sequence_length=9, max_count=3)

    assert len(negatives) == 3
    assert any(item.implicit_aspect or item.implicit_opinion for item in negatives)
    assert any(not item.implicit_aspect and not item.implicit_opinion and (item.aspect_start, item.opinion_start) in {(1, 5), (1, 6), (2, 4), (2, 6), (3, 4), (3, 5)} for item in negatives)
    assert any(not item.implicit_aspect and not item.implicit_opinion and (item.aspect_end > item.aspect_start or item.opinion_end > item.opinion_start) for item in negatives)


def test_pair_validity_head_scores_full_span_pairs_and_backpropagates():
    torch.manual_seed(11)
    hidden = torch.randn(1, 7, 8, requires_grad=True)
    head = grid.CompactPairRelationHead(hidden_size=8, rank=4, dropout=0.0)
    pairs = torch.tensor([
        [0, 1, 2, 4, 4],
        [0, -1, -1, 4, 4],
        [0, 1, 2, -1, -1],
    ])

    logits = head.pair_validity_logits(hidden, pairs)

    assert logits.shape == (3,)
    logits.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_pair_relation_uses_contextual_implicit_and_interaction_residual():
    torch.manual_seed(13)
    hidden = torch.randn(1, 7, 8, requires_grad=True)
    head = grid.CompactPairRelationHead(hidden_size=8, rank=4, dropout=0.0)
    pairs = torch.tensor([
        [0, -1, -1, 4, 5],
        [0, 1, 2, -1, -1],
    ])

    logits = head.pair_logits(hidden, pairs)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (2, grid.NUM_RELATION_CLASSES)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.pair_interaction.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.implicit_aspect_context.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.implicit_opinion_context.parameters())


def test_exact_pair_target_matrix_does_not_turn_boundary_shift_negative_positive():
    positives = [pair(2, 3, 5, 5, 7)]
    candidates = torch.tensor([
        [0, 2, 3, 5, 5],
        [0, 2, 4, 5, 5],
    ])

    targets = grid.exact_pair_target_matrix(candidates, [positives])

    assert targets.shape == (2, grid.NUM_RELATION_CLASSES)
    assert targets[0, 7] == 1.0
    assert targets[0].sum() == 1.0
    assert targets[1].sum() == 0.0


def test_span_decoder_uses_objectiveness_and_separate_implicit_presence():
    start = torch.tensor([5.0, 4.0, 4.0, -8.0])
    end = torch.tensor([5.0, 4.0, 4.0, -8.0])
    objective = torch.tensor([-8.0, -6.0, 6.0, -8.0])
    spans = grid.constrained_topk_span_decode(
        start,
        end,
        valid_mask=torch.tensor([False, True, True, False]),
        top_k=3,
        max_span_length=1,
        allow_implicit=True,
        min_score=0.01,
        objectiveness_logits=objective,
        implicit_logit=torch.tensor(-8.0),
    )

    assert spans[0].start == 2
    assert not any(item.implicit for item in spans)


def test_implicit_loss_weight_is_screening_only():
    hidden = torch.randn(1, 5, 4)
    head = grid.CompactPairRelationHead(4, rank=2)
    targets = grid.build_sparse_relation_targets([[pair(-1, -1, 2, 2, 0)]], max_length=5)
    logits = head(hidden)
    assert torch.isfinite(grid.sparse_grid_bce_loss(logits, targets, implicit_weight=1.5))
    with pytest.raises(ValueError):
        grid.sparse_grid_bce_loss(logits, targets, implicit_weight=1.25)
