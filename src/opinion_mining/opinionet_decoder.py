from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from .baseline import Candidate
from .data import Quadruple, ReviewExample
from .neural import CLASS_PAIRS


def constrained_endpoint_beam(
    start_prob: Tensor,
    end_prob: Tensor,
    allowed: Iterable[int],
    *,
    max_span_length: int,
    top_k: int,
) -> list[tuple[int, int, float]]:
    """Decode legal endpoint pairs, keeping CLS/implicit as the only 0,0 pair."""
    allowed_values = list(dict.fromkeys(int(value) for value in allowed))
    if not allowed_values:
        return []
    beam = max(1, min(int(top_k), len(allowed_values)))
    start_values, start_indices = torch.topk(start_prob[allowed_values], k=beam)
    end_values, end_indices = torch.topk(end_prob[allowed_values], k=beam)
    pairs: dict[tuple[int, int], float] = {}
    for start_value, start_index in zip(start_values, start_indices):
        left = allowed_values[int(start_index)]
        for end_value, end_index in zip(end_values, end_indices):
            right = allowed_values[int(end_index)]
            if left == 0 or right == 0:
                if left == right == 0:
                    pairs[(0, 0)] = max(pairs.get((0, 0), 0.0), float(torch.sqrt(torch.clamp(start_value * end_value, min=0.0))))
                continue
            if right < left or right - left + 1 > max_span_length:
                continue
            score = float(torch.sqrt(torch.clamp(start_value * end_value, min=0.0)))
            pairs[(left, right)] = max(pairs.get((left, right), 0.0), score)
    return [(left, right, score) for (left, right), score in sorted(pairs.items(), key=lambda item: -item[1])[:top_k]]


def _span_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def decode_opinionet_candidates(
    row: ReviewExample,
    feature: object,
    outputs: dict[str, Tensor],
    *,
    row_index: int,
    max_span_length: int,
    span_top_k: int,
    beam_top_k: int,
    relation_top_k: int,
    nms_overlap: float = 0.8,
) -> list[Candidate]:
    offsets = list(feature.offsets)
    usable = [index for index, (left, right) in enumerate(offsets) if right > left and right <= len(row.text)]
    if not usable:
        return []
    allowed = [0, *usable]
    invalid = torch.ones(outputs["objectiveness"].shape[-1], dtype=torch.bool)
    invalid[allowed] = False
    endpoint_probs: dict[str, Tensor] = {}
    for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end"):
        logits = outputs[name][row_index].clone()
        logits[:, invalid] = -1.0e4
        endpoint_probs[name] = torch.softmax(logits, dim=-1)
    objectiveness = torch.sigmoid(outputs["objectiveness"][row_index])
    relation_probs = torch.sigmoid(outputs["relation"][row_index])
    positions = sorted(allowed, key=lambda index: float(objectiveness[index]), reverse=True)[: max(span_top_k * 3, span_top_k)]

    def text_span(left: int, right: int) -> tuple[str, tuple[int, int]] | None:
        if left <= 0 or right < left or right >= len(offsets):
            return None
        if offsets[left][1] <= offsets[left][0] or offsets[right][1] <= offsets[right][0]:
            return None
        chars = (offsets[left][0], offsets[right][1])
        value = row.text[chars[0] : chars[1]]
        return (value, chars) if value else None

    raw: list[tuple[Quadruple, float, tuple[int, int] | None, tuple[int, int] | None]] = []
    for position in positions:
        aspects = constrained_endpoint_beam(
            endpoint_probs["aspect_start"][position],
            endpoint_probs["aspect_end"][position],
            allowed,
            max_span_length=max_span_length,
            top_k=beam_top_k,
        )
        opinions = constrained_endpoint_beam(
            endpoint_probs["opinion_start"][position],
            endpoint_probs["opinion_end"][position],
            allowed,
            max_span_length=max_span_length,
            top_k=beam_top_k,
        )
        relation_values, relation_ids = torch.topk(relation_probs[position], k=min(relation_top_k, relation_probs.shape[-1]))
        for aspect_start, aspect_end, aspect_score in aspects:
            if aspect_start == aspect_end == 0:
                aspect, aspect_chars = "_", None
            else:
                info = text_span(aspect_start, aspect_end)
                if info is None:
                    continue
                aspect, aspect_chars = info
            for opinion_start, opinion_end, opinion_score in opinions:
                if opinion_start == opinion_end == 0:
                    opinion, opinion_chars = "_", None
                else:
                    info = text_span(opinion_start, opinion_end)
                    if info is None:
                        continue
                    opinion, opinion_chars = info
                if aspect_chars is not None and opinion_chars is not None and _span_overlap(aspect_chars, opinion_chars) > 0:
                    continue
                span_score = float(objectiveness[position]) * max(0.0, aspect_score * opinion_score) ** 0.5
                for relation_value, relation_id in zip(relation_values, relation_ids):
                    category, polarity = CLASS_PAIRS[int(relation_id)]
                    score = min(1.0, max(0.0, span_score * float(relation_value)))
                    raw.append((Quadruple(aspect, opinion, category, polarity), score, aspect_chars, opinion_chars))

    raw.sort(key=lambda item: -item[1])
    kept: list[tuple[Quadruple, float, tuple[int, int] | None, tuple[int, int] | None]] = []
    for candidate in raw:
        quad, score, aspect_chars, opinion_chars = candidate
        if any(
            quad == previous_quad
            or (
                quad.category == previous_quad.category
                and quad.polarity == previous_quad.polarity
                and _span_overlap(aspect_chars, previous_aspect) >= nms_overlap
                and _span_overlap(opinion_chars, previous_opinion) >= nms_overlap
            )
            for previous_quad, _, previous_aspect, previous_opinion in kept
        ):
            continue
        kept.append(candidate)
    best: dict[Quadruple, float] = {}
    for quad, score, _, _ in kept:
        best[quad] = max(best.get(quad, 0.0), score)
    return [Candidate(quad, score, ("opinionet",)) for quad, score in sorted(best.items(), key=lambda item: (-item[1], item[0]))]
