from __future__ import annotations

"""Compact implicit-aware relation grid components.

This module is deliberately independent from the encoder in ``neural.py``.
It consumes the hidden states produced by that encoder and keeps relation
targets sparse until a caller explicitly asks for a dense grid.  The grid
cell is ``(aspect_start, opinion_start, class_id)``; span end points remain in
the sparse metadata so a later decoder can use the boundary heads without
allocating a four-dimensional target for every possible span pair.

The current competition has 13 categories and 3 polarities, hence 39
multi-label relation classes.  Token index zero is reserved for the implicit
term sentinel, matching ``_Feature`` and ``PointerQuadrupleModel`` in
``neural.py``.  The public APIs accept tensors and light dataclasses only;
they do not load a tokenizer or a pretrained model.
"""

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baseline import Candidate
from .data import LabelSpan, Quadruple, ReviewExample, IMPLICIT_A_TOKEN, IMPLICIT_O_TOKEN, decode_implicit_terms, encode_implicit_terms
from .submission import CATEGORIES, POLARITIES


CATEGORY_ORDER = tuple(sorted(CATEGORIES))
POLARITY_ORDER = tuple(sorted(POLARITIES))
CLASS_PAIRS = tuple((category, polarity) for category in CATEGORY_ORDER for polarity in POLARITY_ORDER)
CLASS_TO_ID = {pair: index for index, pair in enumerate(CLASS_PAIRS)}
NUM_RELATION_CLASSES = len(CLASS_PAIRS)
IMPLICIT_TOKEN = 0
IMPLICIT_LOSS_WEIGHTS = (1.0, 1.5, 2.0)


def implicit_token_round_trip(aspect: str, opinion: str) -> tuple[str, str]:
    """Encode and decode implicit sides without losing explicit text."""
    return decode_implicit_terms(*encode_implicit_terms(aspect, opinion))


@dataclass(frozen=True)
class PairSpan:
    """A token span pair used by the grid and pair head.

    End points are inclusive, as they are in ``_Feature.pairs``.  ``-1,-1``
    denotes an implicit side in this module.  ``class_id`` is optional so the
    same record can describe a candidate pair as well as a gold relation.
    """

    aspect_start: int
    aspect_end: int
    opinion_start: int
    opinion_end: int
    implicit_aspect: bool = False
    implicit_opinion: bool = False
    class_id: int | None = None

    def __post_init__(self) -> None:
        implicit_aspect = self.implicit_aspect or self.aspect_start < 0 or self.aspect_end < 0
        implicit_opinion = self.implicit_opinion or self.opinion_start < 0 or self.opinion_end < 0
        object.__setattr__(self, "implicit_aspect", implicit_aspect)
        object.__setattr__(self, "implicit_opinion", implicit_opinion)
        if implicit_aspect:
            object.__setattr__(self, "aspect_start", -1)
            object.__setattr__(self, "aspect_end", -1)
        if implicit_opinion:
            object.__setattr__(self, "opinion_start", -1)
            object.__setattr__(self, "opinion_end", -1)
        if not implicit_aspect and self.aspect_end < self.aspect_start:
            raise ValueError("aspect_end must be >= aspect_start")
        if not implicit_opinion and self.opinion_end < self.opinion_start:
            raise ValueError("opinion_end must be >= opinion_start")
        if self.class_id is not None and not 0 <= self.class_id < NUM_RELATION_CLASSES:
            raise ValueError(f"class_id must be in [0, {NUM_RELATION_CLASSES})")

    @property
    def aspect_anchor(self) -> int:
        return IMPLICIT_TOKEN if self.implicit_aspect else self.aspect_start

    @property
    def opinion_anchor(self) -> int:
        return IMPLICIT_TOKEN if self.implicit_opinion else self.opinion_start

    @classmethod
    def from_feature_pair(cls, pair: Sequence[int]) -> "PairSpan":
        """Convert an ``_Feature.pairs`` row.

        The existing feature builder stores implicit A as ``(-1, -1)`` and
        implicit O as ``(0, 0)`` because index zero is its CLS sentinel.  This
        adapter makes both implicit sides explicit while preserving inclusive
        end points for surface spans.
        """

        if len(pair) != 5:
            raise ValueError("feature pair must be (a_left, a_right, o_left, o_right, class_id)")
        a_left, a_right, o_left, o_right, class_id = (int(value) for value in pair)
        return cls(
            a_left,
            a_right,
            o_left,
            o_right,
            implicit_aspect=a_left < 0 or a_right < 0,
            implicit_opinion=(o_left < 0 or o_right < 0 or (o_left == 0 and o_right == 0)),
            class_id=class_id,
        )


@dataclass(frozen=True)
class SparseRelationTargets:
    """Sparse positives for a grid with shape ``[B, L, L, 39]``.

    ``indices`` has shape ``[4, N]`` and stores ``(batch, a_anchor,
    o_anchor, class_id)``.  ``spans`` has shape ``[N, 6]`` and stores
    ``(batch, a_start, a_end, o_start, o_end, class_id)``.  The span columns
    use ``-1`` for an implicit side.  The two representations intentionally
    coexist: the grid is compact for relation supervision while the span
    table retains enough information for constrained decoding.
    """

    shape: tuple[int, int, int, int]
    indices: Tensor
    values: Tensor
    spans: Tensor

    def __post_init__(self) -> None:
        if len(self.shape) != 4 or self.shape[-1] != NUM_RELATION_CLASSES:
            raise ValueError(f"shape must be [B, L, L, {NUM_RELATION_CLASSES}]")
        if self.indices.ndim != 2 or self.indices.shape[0] != 4:
            raise ValueError("indices must have shape [4, N]")
        if self.values.ndim != 1 or self.values.shape[0] != self.indices.shape[1]:
            raise ValueError("values must have shape [N]")
        if self.spans.ndim != 2 or self.spans.shape != (self.indices.shape[1], 6):
            raise ValueError("spans must have shape [N, 6]")

    @property
    def num_positives(self) -> int:
        return int(self.indices.shape[1])

    def to(self, device: torch.device | str) -> "SparseRelationTargets":
        return SparseRelationTargets(
            self.shape,
            self.indices.to(device),
            self.values.to(device),
            self.spans.to(device),
        )

    def as_coo(self) -> Tensor:
        """Return a torch COO tensor only when a caller explicitly requests it."""

        return torch.sparse_coo_tensor(
            self.indices,
            self.values,
            size=self.shape,
            check_invariants=True,
        ).coalesce()

    def to_dense(self) -> Tensor:
        return self.as_coo().to_dense()


@dataclass(frozen=True)
class SpanCandidate:
    """A constrained token span returned by the standalone decoder."""

    start: int
    end: int
    score: float
    implicit: bool = False

    def __post_init__(self) -> None:
        if self.implicit:
            object.__setattr__(self, "start", IMPLICIT_TOKEN)
            object.__setattr__(self, "end", IMPLICIT_TOKEN)
        elif self.start <= IMPLICIT_TOKEN or self.end < self.start:
            raise ValueError("surface spans must use positive, ordered token indices")

    @property
    def token_range(self) -> tuple[int, int] | None:
        return None if self.implicit else (self.start, self.end)


@dataclass(frozen=True)
class GridCandidate:
    """A scored relation before conversion to the project's string candidate."""

    aspect: SpanCandidate
    opinion: SpanCandidate
    class_id: int
    score: float

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < NUM_RELATION_CLASSES:
            raise ValueError(f"class_id must be in [0, {NUM_RELATION_CLASSES})")

    @property
    def category(self) -> str:
        return CLASS_PAIRS[self.class_id][0]

    @property
    def polarity(self) -> str:
        return CLASS_PAIRS[self.class_id][1]


def official_char_to_token_span(
    offsets: Sequence[tuple[int, int]],
    start: int | None,
    end: int | None,
) -> tuple[int, int] | None:
    """Map an official exclusive character range to inclusive token indices.

    The mapping deliberately follows ``neural._span_token_range``: only
    non-empty tokenizer offsets fully contained in the official range are
    accepted.  No string search fallback is used, so repeated terms keep the
    annotation occurrence selected by the CSV offsets.
    """

    if start is None or end is None or not 0 <= start < end:
        return None
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= start and right <= end
    ]
    if not indices:
        return None
    return min(indices), max(indices)


def pair_span_from_label(
    label: LabelSpan,
    offsets: Sequence[tuple[int, int]],
) -> PairSpan | None:
    """Build a canonical pair from a label and the tokenizer offsets."""

    quad = label.quadruple
    class_id = CLASS_TO_ID.get((quad.category, quad.polarity))
    if class_id is None:
        return None
    aspect = None if quad.aspect == "_" else official_char_to_token_span(offsets, label.aspect_start, label.aspect_end)
    opinion = None if quad.opinion == "_" else official_char_to_token_span(offsets, label.opinion_start, label.opinion_end)
    if quad.aspect != "_" and aspect is None:
        return None
    if quad.opinion != "_" and opinion is None:
        return None
    return PairSpan(
        -1 if aspect is None else aspect[0],
        -1 if aspect is None else aspect[1],
        -1 if opinion is None else opinion[0],
        -1 if opinion is None else opinion[1],
        implicit_aspect=aspect is None,
        implicit_opinion=opinion is None,
        class_id=class_id,
    )


def _coerce_pair_span(value: PairSpan | Sequence[int]) -> PairSpan:
    if isinstance(value, PairSpan):
        return value
    values = tuple(int(item) for item in value)
    if len(values) == 5:
        return PairSpan.from_feature_pair(values)
    if len(values) == 4:
        return PairSpan(*values)
    raise ValueError("pair span must be PairSpan or four/five integers")


def _item_pairs(item: Any, item_index: int, offsets: Sequence[Any] | None) -> tuple[list[PairSpan], int | None]:
    if hasattr(item, "pairs"):
        feature_pairs = [_coerce_pair_span(pair) for pair in item.pairs]
        feature_length = len(item.offsets) if hasattr(item, "offsets") else None
        return feature_pairs, feature_length
    if isinstance(item, ReviewExample):
        if offsets is None:
            raise ValueError("offsets are required when building targets from ReviewExample")
        item_offsets = offsets[item_index]
        pairs = [
            pair
            for label in item.label_spans
            if (pair := pair_span_from_label(label, item_offsets)) is not None
        ]
        return pairs, len(item_offsets)
    try:
        return [_coerce_pair_span(pair) for pair in item], None
    except TypeError as exc:
        raise TypeError("target items must be _Feature-like objects, ReviewExample, or pair iterables") from exc


def build_sparse_relation_targets(
    items: Iterable[Any],
    *,
    max_length: int | None = None,
    offsets: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> SparseRelationTargets:
    """Construct deduplicated sparse relation positives.

    ``items`` may be a list of current ``_Feature`` objects, a list of
    ``ReviewExample`` objects plus per-example tokenizer offsets, or a list of
    pair iterables.  Feature pairs use the project's inclusive token endpoint
    convention.  For raw pair iterables, use ``-1,-1`` for an implicit side;
    ``(0, 0)`` is accepted as the legacy implicit-opinion sentinel only in the
    five-column feature form.
    """

    materialized = list(items)
    if offsets is not None and len(offsets) != len(materialized):
        raise ValueError("offsets must have one row per item")
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive")

    rows: list[tuple[int, PairSpan]] = []
    inferred_length = 1
    for item_index, item in enumerate(materialized):
        pairs, feature_length = _item_pairs(item, item_index, offsets)
        if feature_length is not None:
            inferred_length = max(inferred_length, feature_length)
        for pair in pairs:
            if pair.class_id is None:
                raise ValueError("gold relation pairs must carry class_id")
            inferred_length = max(
                inferred_length,
                pair.aspect_end + 1 if not pair.implicit_aspect else 1,
                pair.opinion_end + 1 if not pair.implicit_opinion else 1,
            )
            rows.append((item_index, pair))
    sequence_length = max_length if max_length is not None else inferred_length

    cell_to_span: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int]] = {}
    for batch_index, pair in rows:
        if pair.class_id is None:
            continue
        for endpoint in (
            pair.aspect_start,
            pair.aspect_end,
            pair.opinion_start,
            pair.opinion_end,
        ):
            if endpoint >= sequence_length:
                raise ValueError(f"pair endpoint {endpoint} exceeds max_length={sequence_length}")
        a_anchor = pair.aspect_anchor
        o_anchor = pair.opinion_anchor
        if a_anchor >= sequence_length or o_anchor >= sequence_length:
            raise ValueError("relation anchor exceeds target sequence length")
        key = (batch_index, a_anchor, o_anchor, pair.class_id)
        cell_to_span.setdefault(
            key,
            (
                batch_index,
                pair.aspect_start,
                pair.aspect_end,
                pair.opinion_start,
                pair.opinion_end,
                pair.class_id,
            ),
        )

    ordered = sorted(cell_to_span.items())
    if ordered:
        indices = torch.tensor([key for key, _ in ordered], dtype=torch.long).transpose(0, 1).contiguous()
        spans = torch.tensor([row for _, row in ordered], dtype=torch.long)
    else:
        indices = torch.empty((4, 0), dtype=torch.long)
        spans = torch.empty((0, 6), dtype=torch.long)
    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    return SparseRelationTargets(
        (len(materialized), sequence_length, sequence_length, NUM_RELATION_CLASSES),
        indices,
        values,
        spans,
    )


def pair_spans_to_tensor(
    pairs: Sequence[PairSpan | Sequence[int]] | Tensor,
    *,
    batch_index: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return pair spans as ``[N, 5] = (batch, a0, a1, o0, o1)``."""

    if isinstance(pairs, Tensor):
        tensor = pairs.to(device=device) if device is not None else pairs
        if tensor.ndim == 3 and tensor.shape[-1] == 4:
            batch = torch.arange(tensor.shape[0], device=tensor.device).view(-1, 1, 1).expand(-1, tensor.shape[1], -1)
            return torch.cat([batch, tensor.long()], dim=-1).reshape(-1, 5)
        if tensor.ndim == 2 and tensor.shape[-1] == 5:
            return tensor.long()
        if tensor.ndim == 2 and tensor.shape[-1] == 4:
            prefix = torch.full((tensor.shape[0], 1), batch_index, dtype=torch.long, device=tensor.device)
            return torch.cat([prefix, tensor.long()], dim=-1)
        raise ValueError("tensor pair spans must have shape [N,4], [N,5], or [B,N,4]")
    rows: list[list[int]] = []
    for pair in pairs:
        item = _coerce_pair_span(pair)
        rows.append([batch_index, item.aspect_start, item.aspect_end, item.opinion_start, item.opinion_end])
    target_device = torch.device(device) if device is not None else None
    return torch.tensor(rows, dtype=torch.long, device=target_device).reshape(-1, 5)


class CompactPairRelationHead(nn.Module):
    """Low-rank multi-label relation head with explicit implicit states.

    The pair score is a class-conditioned diagonal bilinear interaction in a
    configurable rank, plus two small additive projections.  The head can
    score a sparse list of span pairs or all token-anchor cells.  In both
    paths, the four states (explicit/explicit, implicit-A, implicit-O,
    dual-implicit) receive separate learned bias vectors.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = 32,
        num_classes: int = NUM_RELATION_CLASSES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        if num_classes != NUM_RELATION_CLASSES:
            raise ValueError(f"this competition head requires {NUM_RELATION_CLASSES} relation classes")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.num_classes = int(num_classes)
        context_size = hidden_size * 2
        self.aspect_projection = nn.Linear(context_size, rank, bias=False)
        self.opinion_projection = nn.Linear(context_size, rank, bias=False)
        self.aspect_additive = nn.Linear(context_size, num_classes)
        self.opinion_additive = nn.Linear(context_size, num_classes)
        self.class_factor = nn.Parameter(torch.empty(num_classes, rank))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.implicit_aspect = nn.Parameter(torch.empty(hidden_size))
        self.implicit_opinion = nn.Parameter(torch.empty(hidden_size))
        self.implicit_aspect_context = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.implicit_opinion_context = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.implicit_bias = nn.Parameter(torch.zeros(4, num_classes))
        interaction_size = rank * 4
        interaction_hidden = max(rank * 2, 8)
        self.pair_interaction = nn.Sequential(
            nn.LayerNorm(interaction_size),
            nn.Linear(interaction_size, interaction_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden, num_classes),
        )
        self.pair_validity = nn.Sequential(
            nn.Linear(rank * 4, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, 1),
        )
        self.pair_validity_state_bias = nn.Parameter(torch.zeros(4))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.class_factor, std=0.02)
        nn.init.normal_(self.implicit_aspect, std=0.02)
        nn.init.normal_(self.implicit_opinion, std=0.02)
        nn.init.zeros_(self.pair_interaction[-1].weight)
        nn.init.zeros_(self.pair_interaction[-1].bias)

    def _pair_contexts(self, hidden: Tensor, pair_spans: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, H]")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden size mismatch: expected {self.hidden_size}, got {hidden.shape[-1]}")
        if hidden.shape[1] < 1:
            raise ValueError("hidden must contain the index-zero sentinel")
        pairs = pair_spans.to(device=hidden.device, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[-1] != 5:
            raise ValueError("pair_spans must have shape [N, 5]")
        batch_index = pairs[:, 0]
        if (batch_index < 0).any() or (batch_index >= hidden.shape[0]).any():
            raise ValueError("pair batch index is outside hidden")
        length = hidden.shape[1]
        a_start, a_end = pairs[:, 1], pairs[:, 2]
        o_start, o_end = pairs[:, 3], pairs[:, 4]
        implicit_a = (a_start < 0) | (a_end < 0)
        implicit_o = (o_start < 0) | (o_end < 0)
        for name, start, end, implicit in (
            ("aspect", a_start, a_end, implicit_a),
            ("opinion", o_start, o_end, implicit_o),
        ):
            invalid = (~implicit) & ((start <= 0) | (end < start) | (end >= length))
            if invalid.any():
                raise ValueError(f"invalid {name} span in pair_spans")

        cumulative = torch.cat(
            [hidden.new_zeros((hidden.shape[0], 1, hidden.shape[2])), hidden.cumsum(dim=1)],
            dim=1,
        )
        cls = hidden[batch_index, IMPLICIT_TOKEN]
        implicit_aspect = self.implicit_aspect.view(1, -1) + self.implicit_aspect_context(cls)
        implicit_opinion = self.implicit_opinion.view(1, -1) + self.implicit_opinion_context(cls)

        def pool(start: Tensor, end: Tensor, implicit: Tensor, learned: Tensor) -> Tensor:
            safe_start = start.clamp(min=0, max=length - 1)
            safe_end = end.clamp(min=0, max=length - 1)
            total = cumulative[batch_index, safe_end + 1] - cumulative[batch_index, safe_start]
            width = (safe_end - safe_start + 1).clamp_min(1).to(hidden.dtype).unsqueeze(-1)
            pooled = total / width
            return torch.where(implicit.unsqueeze(-1), learned, pooled)

        aspect = pool(a_start, a_end, implicit_a, implicit_aspect)
        opinion = pool(o_start, o_end, implicit_o, implicit_opinion)
        aspect_context = torch.cat([aspect, cls], dim=-1)
        opinion_context = torch.cat([opinion, cls], dim=-1)
        state = implicit_a.long() + 2 * implicit_o.long()
        return aspect_context, opinion_context, state

    def _score_contexts(self, aspect_context: Tensor, opinion_context: Tensor, state: Tensor) -> Tensor:
        left = self.aspect_projection(self.dropout(aspect_context))
        right = self.opinion_projection(self.dropout(opinion_context))
        bilinear = torch.einsum("nr,cr,nr->nc", left, self.class_factor, right)
        fused = torch.cat([left, right, left * right, (left - right).abs()], dim=-1)
        logits = (
            bilinear
            + self.aspect_additive(aspect_context)
            + self.opinion_additive(opinion_context)
            + self.class_bias.view(1, -1)
            + self.implicit_bias[state]
            + self.pair_interaction(fused)
        )
        return logits

    def pair_logits(self, hidden: Tensor, pair_spans: Tensor | Sequence[PairSpan | Sequence[int]]) -> Tensor:
        """Score sparse full-span pairs as ``[N, 39]`` multi-label logits."""

        pairs = pair_spans_to_tensor(pair_spans, device=hidden.device)
        aspect_context, opinion_context, state = self._pair_contexts(hidden, pairs)
        return self._score_contexts(aspect_context, opinion_context, state)

    def pair_validity_logits(self, hidden: Tensor, pair_spans: Tensor | Sequence[PairSpan | Sequence[int]]) -> Tensor:
        """Return a learned no-relation / valid-pair logit for each full span pair."""

        pairs = pair_spans_to_tensor(pair_spans, device=hidden.device)
        aspect_context, opinion_context, state = self._pair_contexts(hidden, pairs)
        left = self.aspect_projection(self.dropout(aspect_context))
        right = self.opinion_projection(self.dropout(opinion_context))
        fused = torch.cat([left, right, left * right, (left - right).abs()], dim=-1)
        return self.pair_validity(fused).squeeze(-1) + self.pair_validity_state_bias[state]

    def grid_logits(self, hidden: Tensor) -> Tensor:
        """Score every anchor cell as ``[B, L, L, 39]``."""

        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or hidden.shape[1] < 1:
            raise ValueError("hidden must have shape [B, L, hidden_size] with L >= 1")
        batch, length, _ = hidden.shape
        cls = hidden[:, IMPLICIT_TOKEN]
        implicit_aspect = self.implicit_aspect.view(1, -1) + self.implicit_aspect_context(cls)
        implicit_opinion = self.implicit_opinion.view(1, -1) + self.implicit_opinion_context(cls)
        if length == 1:
            aspect = implicit_aspect.unsqueeze(1)
            opinion = implicit_opinion.unsqueeze(1)
        else:
            aspect = torch.cat([implicit_aspect.unsqueeze(1), hidden[:, 1:]], dim=1)
            opinion = torch.cat([implicit_opinion.unsqueeze(1), hidden[:, 1:]], dim=1)
        cls_for_aspect = cls.unsqueeze(1).expand(-1, length, -1)
        cls_for_opinion = cls.unsqueeze(1).expand(-1, length, -1)
        aspect_context = torch.cat([aspect, cls_for_aspect], dim=-1)
        opinion_context = torch.cat([opinion, cls_for_opinion], dim=-1)
        left = self.aspect_projection(self.dropout(aspect_context))
        right = self.opinion_projection(self.dropout(opinion_context))
        bilinear = torch.einsum("blr,cr,bmr->blmc", left, self.class_factor, right)
        logits = (
            bilinear
            + self.aspect_additive(aspect_context).unsqueeze(2)
            + self.opinion_additive(opinion_context).unsqueeze(1)
            + self.class_bias.view(1, 1, 1, -1)
        )
        state = torch.zeros((batch, length, length), dtype=torch.long, device=hidden.device)
        state[:, 0, :] += 1
        state[:, :, 0] += 2
        return logits + self.implicit_bias[state]

    def forward(self, hidden: Tensor, pair_spans: Tensor | Sequence[PairSpan | Sequence[int]] | None = None) -> Tensor:
        return self.grid_logits(hidden) if pair_spans is None else self.pair_logits(hidden, pair_spans)


def pair_target_matrix(pair_spans: Tensor | Sequence[PairSpan | Sequence[int]], targets: SparseRelationTargets) -> Tensor:
    """Align sparse grid labels to candidate pair rows as a multi-hot matrix."""

    pairs = pair_spans_to_tensor(pair_spans, device=targets.indices.device)
    matrix = torch.zeros((pairs.shape[0], NUM_RELATION_CLASSES), dtype=targets.values.dtype, device=pairs.device)
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for batch_index, a_anchor, o_anchor, class_id in targets.indices.t().tolist():
        lookup.setdefault((batch_index, a_anchor, o_anchor), []).append(class_id)
    for row_index, row in enumerate(pairs.tolist()):
        batch_index, a_start, a_end, o_start, o_end = row
        a_anchor = IMPLICIT_TOKEN if a_start < 0 or a_end < 0 else a_start
        o_anchor = IMPLICIT_TOKEN if o_start < 0 or o_end < 0 else o_start
        for class_id in lookup.get((batch_index, a_anchor, o_anchor), ()):
            matrix[row_index, class_id] = 1.0
    return matrix


def exact_pair_target_matrix(
    pair_spans: Tensor | Sequence[PairSpan | Sequence[int]],
    positives_by_batch: Sequence[Sequence[PairSpan]],
) -> Tensor:
    """Build pair-class targets by exact full-span match, not only start anchors."""

    pairs = pair_spans_to_tensor(pair_spans)
    matrix = torch.zeros((pairs.shape[0], NUM_RELATION_CLASSES), dtype=torch.float32, device=pairs.device)
    lookup: dict[tuple[int, int, int, int, int], list[int]] = {}
    for batch_index, positives in enumerate(positives_by_batch):
        for positive in positives:
            if positive.class_id is None:
                continue
            key = (
                batch_index,
                -1 if positive.implicit_aspect else positive.aspect_start,
                -1 if positive.implicit_aspect else positive.aspect_end,
                -1 if positive.implicit_opinion else positive.opinion_start,
                -1 if positive.implicit_opinion else positive.opinion_end,
            )
            lookup.setdefault(key, []).append(int(positive.class_id))
    for row_index, row in enumerate(pairs.tolist()):
        batch_index, a_start, a_end, o_start, o_end = row
        key = (batch_index, a_start, a_end, o_start, o_end)
        for class_id in lookup.get(key, ()):
            matrix[row_index, class_id] = 1.0
    return matrix


def hard_negative_pairs(
    positives: Sequence[PairSpan],
    *,
    sequence_length: int,
    max_count: int = 32,
) -> list[PairSpan]:
    """Create deterministic structural negatives for candidate generation.

    The returned set covers wrong same sentence pairings, shared opinion,
    nearby boundaries, and implicit/explicit confusion whenever the input
    positives make those cases possible.
    """
    if sequence_length < 2 or max_count < 1:
        return []
    positive_keys = {(p.aspect_start, p.aspect_end, p.opinion_start, p.opinion_end) for p in positives}
    generated: set[tuple[int, int, int, int]] = set()
    state_bucket: list[PairSpan] = []
    pair_bucket: list[PairSpan] = []
    boundary_bucket: list[PairSpan] = []
    explicit_a = [p for p in positives if not p.implicit_aspect]
    explicit_o = [p for p in positives if not p.implicit_opinion]

    def add(bucket: list[PairSpan], a0: int, a1: int, o0: int, o1: int, ia: bool = False, io: bool = False) -> None:
        key = (a0, a1, o0, o1)
        if key in positive_keys or key in generated:
            return
        if not ia and (a0 <= 0 or a1 < a0 or a1 >= sequence_length):
            return
        if not io and (o0 <= 0 or o1 < o0 or o1 >= sequence_length):
            return
        generated.add(key)
        bucket.append(PairSpan(a0, a1, o0, o1, ia, io))

    # State-confusion negatives are guaranteed a share of the budget.  This is
    # essential for suppressing the dominant V1 implicit-A hallucinations.
    for p in positives:
        if not p.implicit_aspect:
            add(state_bucket, -1, -1, p.opinion_start, p.opinion_end, ia=True, io=p.implicit_opinion)
        if not p.implicit_opinion:
            add(state_bucket, p.aspect_start, p.aspect_end, -1, -1, ia=p.implicit_aspect, io=True)
        if not p.implicit_aspect and not p.implicit_opinion:
            add(state_bucket, -1, -1, -1, -1, ia=True, io=True)
        if p.implicit_aspect and explicit_a:
            q = explicit_a[0]
            add(state_bucket, q.aspect_start, q.aspect_end, p.opinion_start, p.opinion_end, io=p.implicit_opinion)
        if p.implicit_opinion and explicit_o:
            q = explicit_o[0]
            add(state_bucket, p.aspect_start, p.aspect_end, q.opinion_start, q.opinion_end, ia=p.implicit_aspect)

    # Cross product of observed aspect and opinion spans creates hard same-review
    # mismatches, including explicit/implicit combinations when present.
    aspect_specs = list(dict.fromkeys((p.aspect_start, p.aspect_end, p.implicit_aspect) for p in positives))
    opinion_specs = list(dict.fromkeys((p.opinion_start, p.opinion_end, p.implicit_opinion) for p in positives))
    for a0, a1, ia in aspect_specs:
        for o0, o1, io in opinion_specs:
            add(pair_bucket, a0, a1, o0, o1, ia=ia, io=io)

    # Near-boundary shifts teach exact span matching instead of anchor-only
    # matching and target the second-largest V1 FP bucket.
    for p in positives:
        if not p.implicit_aspect:
            add(boundary_bucket, max(1, p.aspect_start - 1), p.aspect_end, p.opinion_start, p.opinion_end, io=p.implicit_opinion)
            add(boundary_bucket, p.aspect_start, min(sequence_length - 1, p.aspect_end + 1), p.opinion_start, p.opinion_end, io=p.implicit_opinion)
        if not p.implicit_opinion:
            add(boundary_bucket, p.aspect_start, p.aspect_end, max(1, p.opinion_start - 1), p.opinion_end, ia=p.implicit_aspect)
            add(boundary_bucket, p.aspect_start, p.aspect_end, p.opinion_start, min(sequence_length - 1, p.opinion_end + 1), ia=p.implicit_aspect)

    # Round-robin buckets so a small max_count cannot starve the state or
    # boundary negatives behind a large Cartesian set of pair mismatches.
    buckets = (state_bucket, pair_bucket, boundary_bucket)
    positions = [0, 0, 0]
    out: list[PairSpan] = []
    while len(out) < max_count:
        progressed = False
        for index, bucket in enumerate(buckets):
            if positions[index] < len(bucket):
                out.append(bucket[positions[index]])
                positions[index] += 1
                progressed = True
                if len(out) >= max_count:
                    break
        if not progressed:
            break
    return out


def sparse_grid_bce_loss(
    logits: Tensor,
    targets: SparseRelationTargets,
    *,
    valid_mask: Tensor | None = None,
    negative_ratio: float = 3.0,
    pos_weight: float | Tensor = 3.0,
    max_negative: int = 4096,
    implicit_weight: float = 1.0,
) -> Tensor:
    """Compute BCE from sparse positives plus hard negative sampling.

    ``logits`` must be ``[B, L, L, 39]``.  The target is never densified.  The
    highest-logit valid negatives are selected deterministically, which keeps
    the loss small while still pushing down the current false positives.
    """

    if logits.ndim != 4 or logits.shape[-1] != NUM_RELATION_CLASSES:
        raise ValueError(f"logits must have shape [B, L, L, {NUM_RELATION_CLASSES}]")
    if tuple(logits.shape) != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not match target shape {targets.shape}")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    if implicit_weight not in IMPLICIT_LOSS_WEIGHTS:
        raise ValueError(f"implicit_weight must be one of {IMPLICIT_LOSS_WEIGHTS}")
    batch, length, _, classes = logits.shape
    device = logits.device
    flat_logits = logits.reshape(-1)
    positive_mask = torch.zeros((flat_logits.numel(),), dtype=torch.bool, device=device)
    target_indices = targets.indices.to(device=device, dtype=torch.long)
    if target_indices.numel():
        linear = (((target_indices[0] * length + target_indices[1]) * length + target_indices[2]) * classes + target_indices[3])
        positive_mask[linear] = True

    valid_cells = torch.ones((batch, length, length), dtype=torch.bool, device=device)
    if valid_mask is not None:
        mask = valid_mask.to(device=device, dtype=torch.bool)
        if mask.shape == (batch, length):
            valid_cells = mask.unsqueeze(2) & mask.unsqueeze(1)
        elif mask.shape == (batch, length, length):
            valid_cells = mask
        else:
            raise ValueError("valid_mask must have shape [B,L] or [B,L,L]")
    valid_flat = valid_cells.unsqueeze(-1).expand(-1, -1, -1, classes).reshape(-1)
    negative_mask = valid_flat & ~positive_mask
    positive_logits = flat_logits[positive_mask]
    negative_logits = flat_logits[negative_mask]

    if positive_logits.numel():
        if isinstance(pos_weight, Tensor):
            weights = pos_weight.to(device=device, dtype=logits.dtype).reshape(-1)
            if weights.numel() != classes:
                raise ValueError(f"pos_weight tensor must have {classes} values")
            positive_weights = weights[target_indices[3]]
        else:
            positive_weights = logits.new_full(positive_logits.shape, float(pos_weight))
        implicit = (target_indices[1] == IMPLICIT_TOKEN) | (target_indices[2] == IMPLICIT_TOKEN)
        positive_weights = positive_weights * torch.where(implicit, logits.new_tensor(implicit_weight), logits.new_tensor(1.0))
        positive_sum = (F.softplus(-positive_logits) * positive_weights).sum()
        positive_denominator = positive_weights.sum().clamp_min(1.0)
    else:
        positive_sum = logits.sum() * 0.0
        positive_denominator = logits.new_zeros(())

    if negative_logits.numel() and negative_ratio > 0:
        budget = max(1, ceil(max(1, int(positive_logits.numel())) * negative_ratio))
        budget = min(budget, int(max_negative), int(negative_logits.numel()))
        if budget < negative_logits.numel():
            selected = torch.topk(negative_logits.detach(), k=budget, sorted=False).indices
            negative_logits = negative_logits[selected]
        negative_sum = F.softplus(negative_logits).sum()
        negative_denominator = negative_logits.new_tensor(float(negative_logits.numel()))
    else:
        negative_sum = logits.sum() * 0.0
        negative_denominator = logits.new_zeros(())
    denominator = (positive_denominator + negative_denominator).clamp_min(1.0)
    return (positive_sum + negative_sum) / denominator


def pair_relation_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    pos_weight: float | Tensor = 3.0,
) -> Tensor:
    """BCE loss for sparse pair logits and a ``[N, 39]`` multi-hot target."""

    if logits.ndim != 2 or logits.shape[-1] != NUM_RELATION_CLASSES:
        raise ValueError(f"logits must have shape [N, {NUM_RELATION_CLASSES}]")
    if targets.ndim == 1:
        if (targets < 0).any() or (targets >= NUM_RELATION_CLASSES).any():
            raise ValueError("class targets are outside the 39-class range")
        expanded = torch.zeros_like(logits)
        expanded.scatter_(1, targets.to(device=logits.device, dtype=torch.long).view(-1, 1), 1.0)
        targets = expanded
    if targets.shape != logits.shape:
        raise ValueError("pair targets must have shape [N, 39] or [N]")
    if isinstance(pos_weight, Tensor):
        weight = pos_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if weight.numel() != NUM_RELATION_CLASSES:
            raise ValueError(f"pos_weight tensor must have {NUM_RELATION_CLASSES} values")
    else:
        weight = logits.new_full((NUM_RELATION_CLASSES,), float(pos_weight))
    return F.binary_cross_entropy_with_logits(logits, targets.to(device=logits.device, dtype=logits.dtype), pos_weight=weight)


def _decode_single_spans(
    start_logits: Tensor,
    end_logits: Tensor,
    valid_mask: Tensor | None,
    *,
    top_k: int,
    max_span_length: int,
    allow_implicit: bool,
    min_score: float,
    objectiveness_logits: Tensor | None = None,
    implicit_logit: Tensor | float | None = None,
) -> list[SpanCandidate]:
    if start_logits.ndim != 1 or end_logits.ndim != 1 or start_logits.shape != end_logits.shape:
        raise ValueError("start_logits and end_logits must be matching [L] tensors")
    length = int(start_logits.shape[0])
    if length < 1:
        return []
    if top_k < 1 or max_span_length < 1:
        raise ValueError("top_k and max_span_length must be positive")
    starts_mask = torch.ones((length,), dtype=torch.bool, device=start_logits.device)
    starts_mask[IMPLICIT_TOKEN] = False
    if valid_mask is not None:
        mask = valid_mask.to(device=start_logits.device, dtype=torch.bool)
        if mask.shape != (length,):
            raise ValueError("valid_mask must have shape [L]")
        starts_mask &= mask
    allowed = torch.nonzero(starts_mask, as_tuple=False).flatten()
    start_prob = torch.sigmoid(start_logits.detach())
    end_prob = torch.sigmoid(end_logits.detach())
    objective_prob = None
    if objectiveness_logits is not None:
        if objectiveness_logits.shape != start_logits.shape:
            raise ValueError("objectiveness_logits must match span logits")
        objective_prob = torch.sigmoid(objectiveness_logits.detach())
    result: dict[tuple[int, int, bool], SpanCandidate] = {}
    if allow_implicit:
        if implicit_logit is None:
            implicit_score = sqrt(max(0.0, float(start_prob[0] * end_prob[0])))
        else:
            implicit_score = float(torch.sigmoid(torch.as_tensor(implicit_logit).detach()).item())
        if implicit_score >= min_score:
            result[(0, 0, True)] = SpanCandidate(0, 0, implicit_score, implicit=True)
    if allowed.numel():
        beam = min(int(allowed.numel()), max(top_k * 2, top_k))
        top_starts = allowed[torch.topk(start_prob[allowed], k=beam).indices].tolist()
        top_ends = allowed[torch.topk(end_prob[allowed], k=beam).indices].tolist()
        for left in top_starts:
            for right in top_ends:
                if right < left or right - left + 1 > max_span_length:
                    continue
                boundary_score = sqrt(max(0.0, float(start_prob[left] * end_prob[right])))
                if objective_prob is None:
                    score = boundary_score
                else:
                    objective_score = float(objective_prob[left : right + 1].mean())
                    score = sqrt(max(0.0, boundary_score * objective_score))
                if score < min_score:
                    continue
                candidate = SpanCandidate(left, right, score)
                key = (left, right, False)
                previous = result.get(key)
                if previous is None or candidate.score > previous.score:
                    result[key] = candidate
    return sorted(result.values(), key=lambda item: (-item.score, item.start, item.end, item.implicit))[:top_k]


def constrained_topk_span_decode(
    start_logits: Tensor,
    end_logits: Tensor,
    *,
    valid_mask: Tensor | None = None,
    top_k: int = 8,
    max_span_length: int = 10,
    allow_implicit: bool = True,
    min_score: float = 0.0,
    objectiveness_logits: Tensor | None = None,
    implicit_logit: Tensor | float | None = None,
) -> list[SpanCandidate] | list[list[SpanCandidate]]:
    """Decode valid top-k spans from boundary logits.

    A ``[L]`` input returns one span list.  A ``[B,L]`` input returns one list
    per batch row.  Index zero is considered the implicit sentinel and is never
    used as a surface span endpoint.
    """

    if start_logits.ndim == 1:
        return _decode_single_spans(
            start_logits,
            end_logits,
            valid_mask,
            top_k=top_k,
            max_span_length=max_span_length,
            allow_implicit=allow_implicit,
            min_score=min_score,
            objectiveness_logits=objectiveness_logits,
            implicit_logit=implicit_logit,
        )
    if start_logits.ndim != 2 or end_logits.shape != start_logits.shape:
        raise ValueError("batched boundary logits must have shape [B, L]")
    if valid_mask is not None and valid_mask.shape != start_logits.shape:
        raise ValueError("batched valid_mask must have shape [B, L]")
    if objectiveness_logits is not None and objectiveness_logits.shape != start_logits.shape:
        raise ValueError("batched objectiveness_logits must have shape [B, L]")
    if isinstance(implicit_logit, Tensor) and implicit_logit.ndim > 0 and implicit_logit.shape[0] != start_logits.shape[0]:
        raise ValueError("batched implicit_logit must have shape [B]")
    return [
        _decode_single_spans(
            start_logits[row],
            end_logits[row],
            None if valid_mask is None else valid_mask[row],
            top_k=top_k,
            max_span_length=max_span_length,
            allow_implicit=allow_implicit,
            min_score=min_score,
            objectiveness_logits=None if objectiveness_logits is None else objectiveness_logits[row],
            implicit_logit=(implicit_logit[row] if isinstance(implicit_logit, Tensor) and implicit_logit.ndim > 0 else implicit_logit),
        )
        for row in range(start_logits.shape[0])
    ]


def _span_overlap(left: SpanCandidate, right: SpanCandidate) -> float:
    if left.implicit or right.implicit:
        return 1.0 if left.implicit and right.implicit else 0.0
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start) + 1)
    union = max(left.end, right.end) - min(left.start, right.start) + 1
    return intersection / union if union else 0.0


def constrained_topk_pair_decode(
    aspect_spans: Sequence[SpanCandidate],
    opinion_spans: Sequence[SpanCandidate],
    relation_logits: Tensor,
    *,
    relation_top_k: int = 2,
    top_k: int = 32,
    min_relation_score: float = 0.05,
    disallow_overlap: bool = True,
) -> list[GridCandidate]:
    """Join top-k spans with a dense ``[L,L,39]`` relation grid.

    The grid is indexed by span starts; implicit spans use index zero.  Each
    class is independently sigmoid-scored, so multiple category/polarity
    labels can survive for the same pair.  The function constrains endpoint
    order through the span decoder and rejects overlapping explicit A/O spans.
    """

    if relation_logits.ndim != 3 or relation_logits.shape[-1] != NUM_RELATION_CLASSES:
        raise ValueError(f"relation_logits must have shape [L, L, {NUM_RELATION_CLASSES}]")
    if relation_top_k < 1 or top_k < 1:
        raise ValueError("relation_top_k and top_k must be positive")
    length = relation_logits.shape[0]
    if relation_logits.shape[1] != length:
        raise ValueError("relation grid must be square")
    raw: list[GridCandidate] = []
    relation_probabilities = torch.sigmoid(relation_logits.detach())
    for aspect in aspect_spans:
        for opinion in opinion_spans:
            if disallow_overlap and not aspect.implicit and not opinion.implicit and _span_overlap(aspect, opinion) > 0.0:
                continue
            a_anchor = IMPLICIT_TOKEN if aspect.implicit else aspect.start
            o_anchor = IMPLICIT_TOKEN if opinion.implicit else opinion.start
            if a_anchor >= length or o_anchor >= length:
                continue
            probabilities = relation_probabilities[a_anchor, o_anchor]
            values, class_ids = torch.topk(probabilities, k=min(relation_top_k, NUM_RELATION_CLASSES))
            span_score = sqrt(max(0.0, aspect.score * opinion.score))
            for value, class_id in zip(values.tolist(), class_ids.tolist()):
                if value < min_relation_score:
                    continue
                raw.append(GridCandidate(aspect, opinion, int(class_id), max(0.0, span_score * float(value))))
    raw.sort(key=lambda item: (-item.score, item.class_id, item.aspect.start, item.opinion.start))
    return raw[:top_k]


def nms_grid_candidates(
    candidates: Iterable[GridCandidate],
    *,
    iou_threshold: float = 0.8,
    class_aware: bool = True,
    max_keep: int | None = None,
) -> list[GridCandidate]:
    """Apply span-pair NMS before strict quadruple de-duplication."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(candidates, key=lambda item: (-item.score, item.class_id, item.aspect.start, item.opinion.start))
    kept: list[GridCandidate] = []
    for candidate in ordered:
        suppressed = False
        for previous in kept:
            same_class = candidate.class_id == previous.class_id
            if (not class_aware or same_class) and _span_overlap(candidate.aspect, previous.aspect) >= iou_threshold and _span_overlap(candidate.opinion, previous.opinion) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
            if max_keep is not None and len(kept) >= max_keep:
                break
    return kept


def _candidate_text(span: SpanCandidate, text: str, offsets: Sequence[tuple[int, int]]) -> str:
    if span.implicit:
        return "_"
    if span.start >= len(offsets) or span.end >= len(offsets):
        raise ValueError("decoded span is outside offsets")
    left, _ = offsets[span.start]
    _, right = offsets[span.end]
    if right <= left or left < 0 or right > len(text):
        raise ValueError("decoded span has invalid character offsets")
    value = text[left:right]
    if not value:
        raise ValueError("decoded surface span is empty")
    return value


def grid_candidates_to_candidates(
    candidates: Iterable[GridCandidate],
    *,
    text: str,
    offsets: Sequence[tuple[int, int]],
    source: str = "relation_grid",
) -> list[Candidate]:
    """Convert decoded spans to the project's ``Candidate`` format."""

    merged: dict[Quadruple, tuple[float, set[str]]] = {}
    for item in candidates:
        quad = Quadruple(
            _candidate_text(item.aspect, text, offsets),
            _candidate_text(item.opinion, text, offsets),
            item.category,
            item.polarity,
        )
        current = merged.get(quad)
        if current is None or item.score > current[0]:
            merged[quad] = (item.score, {source})
        elif item.score == current[0]:
            merged[quad] = (current[0], current[1] | {source})
    result = [Candidate(quadruple=quad, score=score, sources=tuple(sorted(sources))) for quad, (score, sources) in merged.items()]
    return sorted(result, key=lambda item: (-item.score, item.quadruple))


def decode_grid_candidates(
    aspect_start_logits: Tensor,
    aspect_end_logits: Tensor,
    opinion_start_logits: Tensor,
    opinion_end_logits: Tensor,
    relation_logits: Tensor,
    *,
    valid_mask: Tensor | None = None,
    top_k_spans: int = 8,
    top_k_pairs: int = 32,
    relation_top_k: int = 2,
    max_span_length: int = 10,
    nms_iou: float = 0.8,
    text: str | None = None,
    offsets: Sequence[tuple[int, int]] | None = None,
) -> list[GridCandidate] | list[Candidate]:
    """Convenience wrapper for constrained span decode, pair decode, and NMS."""

    aspect_spans = constrained_topk_span_decode(
        aspect_start_logits,
        aspect_end_logits,
        valid_mask=valid_mask,
        top_k=top_k_spans,
        max_span_length=max_span_length,
        allow_implicit=True,
    )
    opinion_spans = constrained_topk_span_decode(
        opinion_start_logits,
        opinion_end_logits,
        valid_mask=valid_mask,
        top_k=top_k_spans,
        max_span_length=max_span_length,
        allow_implicit=True,
    )
    if isinstance(aspect_spans, list) and (not aspect_spans or isinstance(aspect_spans[0], list)):
        raise ValueError("decode_grid_candidates currently accepts one example at a time")
    if isinstance(opinion_spans, list) and (not opinion_spans or isinstance(opinion_spans[0], list)):
        raise ValueError("decode_grid_candidates currently accepts one example at a time")
    if relation_logits.ndim != 3:
        raise ValueError("decode_grid_candidates currently accepts one [L,L,39] relation grid")
    pairs = constrained_topk_pair_decode(
        aspect_spans,
        opinion_spans,
        relation_logits,
        relation_top_k=relation_top_k,
        top_k=top_k_pairs,
    )
    kept = nms_grid_candidates(pairs, iou_threshold=nms_iou)
    if text is None:
        return kept
    if offsets is None:
        raise ValueError("offsets are required when text is supplied")
    return grid_candidates_to_candidates(kept, text=text, offsets=offsets)


__all__ = [
    "CATEGORY_ORDER",
    "POLARITY_ORDER",
    "CLASS_PAIRS",
    "CLASS_TO_ID",
    "NUM_RELATION_CLASSES",
    "IMPLICIT_TOKEN",
    "IMPLICIT_A_TOKEN",
    "IMPLICIT_O_TOKEN",
    "IMPLICIT_LOSS_WEIGHTS",
    "implicit_token_round_trip",
    "PairSpan",
    "SparseRelationTargets",
    "SpanCandidate",
    "GridCandidate",
    "official_char_to_token_span",
    "pair_span_from_label",
    "build_sparse_relation_targets",
    "pair_spans_to_tensor",
    "CompactPairRelationHead",
    "pair_target_matrix",
    "exact_pair_target_matrix",
    "hard_negative_pairs",
    "sparse_grid_bce_loss",
    "pair_relation_loss",
    "constrained_topk_span_decode",
    "constrained_topk_pair_decode",
    "nms_grid_candidates",
    "grid_candidates_to_candidates",
    "decode_grid_candidates",
]
