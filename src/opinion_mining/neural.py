from __future__ import annotations

"""Compact MPS-friendly span and relation model.

The model uses a small Chinese pretrained encoder (RBT3 by default), predicts
aspect/opinion span boundaries, and classifies a complete aspect-opinion pair
into the 13 x 3 category/polarity space.  Keeping the pair as one class makes
the training objective match the competition's strict quadruple metric.
"""

from dataclasses import dataclass, asdict
import copy
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .baseline import Candidate
from .data import LabelSpan, Quadruple, ReviewExample
from .resource_guard import ResourceGuard
from .training_tricks import EMAModel, FGM
from .submission import CATEGORIES, POLARITIES


CATEGORY_ORDER = tuple(sorted(CATEGORIES))
POLARITY_ORDER = tuple(sorted(POLARITIES))
CLASS_PAIRS = tuple((category, polarity) for category in CATEGORY_ORDER for polarity in POLARITY_ORDER)
CLASS_TO_ID = {pair: index for index, pair in enumerate(CLASS_PAIRS)}
CATEGORY_TO_ID = {value: index for index, value in enumerate(CATEGORY_ORDER)}
POLARITY_TO_ID = {value: index for index, value in enumerate(POLARITY_ORDER)}


@dataclass
class NeuralConfig:
    model_name: str = "hfl/rbt3"
    max_length: int = 128
    batch_size: int = 4
    gradient_accumulation: int = 2
    epochs: int = 4
    learning_rate: float = 2.0e-5
    head_learning_rate: float = 8.0e-4
    weight_decay: float = 0.01
    trainable_layers: int = 2
    max_span_length: int = 10
    span_top_k: int = 10
    beam_top_k: int = 3
    relation_top_k: int = 2
    span_loss_weight: float = 0.65
    relation_loss_weight: float = 1.0
    seed: int = 42
    device: str = "auto"
    use_official_offsets: bool = False
    use_implicit_opinion_sentinel: bool = False
    preserve_multi_relation: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    fgm_epsilon: float = 0.0


@dataclass
class NeuralFoldResult:
    candidates: dict[int, list[Candidate]]
    test_candidates: dict[int, list[Candidate]]
    history: list[dict[str, Any]]
    checkpoint: str | None
    config: dict[str, Any]


@dataclass
class _Feature:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    offsets: list[tuple[int, int]]
    aspect_start: list[int]
    aspect_end: list[int]
    opinion_start: list[int]
    opinion_end: list[int]
    objectiveness: list[float]
    category: list[int]
    polarity: list[int]
    relation: list[list[float]]
    pairs: list[tuple[int, int, int, int, int]]


def choose_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _span_token_range(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    indices = [index for index, (left, right) in enumerate(offsets) if right > left and left >= start and right <= end]
    if not indices:
        return None
    return min(indices), max(indices)


def _official_char_range(text: str, start: int | None, end: int | None) -> tuple[int, int] | None:
    """Return only an explicitly supplied, valid CSV character range."""
    if start is not None and end is not None and 0 <= start < end <= len(text):
        return start, end
    return None


def _legacy_char_range(text: str, term: str) -> tuple[int, int] | None:
    """Reproduce the historical first-occurrence mapping used by Anchor v0."""
    if not term or term == "_":
        return None
    start = text.find(term)
    return (start, start + len(term)) if start >= 0 else None


class _ReviewDataset(Dataset[_Feature]):
    def __init__(
        self,
        examples: Iterable[ReviewExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        *,
        use_official_offsets: bool = False,
        use_implicit_opinion_sentinel: bool = False,
        preserve_multi_relation: bool = False,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_official_offsets = use_official_offsets
        self.use_implicit_opinion_sentinel = use_implicit_opinion_sentinel
        self.preserve_multi_relation = preserve_multi_relation
        self.mapping_issues: list[dict[str, Any]] = []
        self.features = [self._make_feature(example) for example in self.examples]

    def _make_feature(self, example: ReviewExample) -> _Feature:
        encoded = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        aspect_start = [-1] * self.max_length
        aspect_end = [-1] * self.max_length
        opinion_start = [-1] * self.max_length
        opinion_end = [-1] * self.max_length
        objectiveness = [0.0] * self.max_length
        category = [-1] * self.max_length
        polarity = [-1] * self.max_length
        relation = [[0.0] * len(CLASS_PAIRS) for _ in range(self.max_length)]
        pairs: list[tuple[int, int, int, int, int]] = []
        span_labels = example.label_spans or tuple(LabelSpan(quadruple=quad) for quad in sorted(example.labels))
        for label_index, label in enumerate(span_labels):
            quad = label.quadruple
            # B2 uses the CLS position as an explicit Opinion="_" sentinel.
            # The disabled path preserves the historical dropped implicit-O target.
            if quad.opinion == "_":
                opinion_range = (0, 0) if self.use_implicit_opinion_sentinel else None
            else:
                opinion_chars = (
                    _official_char_range(example.text, label.opinion_start, label.opinion_end)
                    if self.use_official_offsets
                    else _legacy_char_range(example.text, quad.opinion)
                )
                opinion_range = _span_token_range(offsets, *opinion_chars) if opinion_chars else None
            aspect_range = None
            if quad.aspect != "_":
                aspect_chars = (
                    _official_char_range(example.text, label.aspect_start, label.aspect_end)
                    if self.use_official_offsets
                    else _legacy_char_range(example.text, quad.aspect)
                )
                aspect_range = _span_token_range(offsets, *aspect_chars) if aspect_chars else None
                # Explicit terms must not silently become implicit targets when
                # their official offsets cannot be mapped by the tokenizer.
                if aspect_range is None:
                    self.mapping_issues.append(
                        {
                            "id": example.id,
                            "label_index": label_index,
                            "field": "aspect",
                            "term": quad.aspect,
                            "start": label.aspect_start,
                            "end": label.aspect_end,
                            "reason": "official_offset_not_mapped_to_token_span",
                        }
                    )
                    continue
            if opinion_range is None:
                self.mapping_issues.append(
                    {
                        "id": example.id,
                        "label_index": label_index,
                        "field": "opinion",
                        "term": quad.opinion,
                        "start": label.opinion_start,
                        "end": label.opinion_end,
                        "reason": "official_offset_not_mapped_to_token_span",
                    }
                )
                continue
            op_left, op_right = opinion_range
            if aspect_range is None:
                a_left = a_right = -1
            else:
                a_left, a_right = aspect_range
            class_id = CLASS_TO_ID.get((quad.category, quad.polarity))
            if class_id is not None:
                category_id = CATEGORY_TO_ID[CLASS_PAIRS[class_id][0]]
                polarity_id = POLARITY_TO_ID[CLASS_PAIRS[class_id][1]]
                # A token can participate in several quadruples.  Boundary
                # pointers are shared when the span is shared, while relation
                # classes are multi-hot so one label cannot overwrite another.
                active_tokens = set(range(op_left, op_right + 1))
                if a_left >= 0:
                    active_tokens.update(range(a_left, a_right + 1))
                if not active_tokens:
                    active_tokens.add(0)
                for token_index in active_tokens:
                    opinion_start[token_index] = op_left
                    opinion_end[token_index] = op_right
                    aspect_start[token_index] = 0 if a_left < 0 else a_left
                    aspect_end[token_index] = 0 if a_left < 0 else a_right
                    objectiveness[token_index] = 1.0
                    relation[token_index][class_id] = 1.0
                    if category[token_index] < 0:
                        category[token_index] = category_id
                    if polarity[token_index] < 0:
                        polarity[token_index] = polarity_id
                pairs.append((a_left, a_right, op_left, op_right, class_id))
        item = _Feature(
            input_ids=list(encoded["input_ids"]),
            attention_mask=list(encoded["attention_mask"]),
            token_type_ids=list(encoded["token_type_ids"]) if "token_type_ids" in encoded else None,
            offsets=offsets,
            aspect_start=aspect_start,
            aspect_end=aspect_end,
            opinion_start=opinion_start,
        opinion_end=opinion_end,
        objectiveness=objectiveness,
        category=category,
        polarity=polarity,
        relation=relation,
        pairs=pairs,
        )
        return item

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> _Feature:
        return self.features[index]


def _collate(features: list[_Feature]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "input_ids": torch.tensor([x.input_ids for x in features], dtype=torch.long),
        "attention_mask": torch.tensor([x.attention_mask for x in features], dtype=torch.long),
        "aspect_start": torch.tensor([x.aspect_start for x in features], dtype=torch.long),
        "aspect_end": torch.tensor([x.aspect_end for x in features], dtype=torch.long),
        "opinion_start": torch.tensor([x.opinion_start for x in features], dtype=torch.long),
        "opinion_end": torch.tensor([x.opinion_end for x in features], dtype=torch.long),
        "objectiveness": torch.tensor([x.objectiveness for x in features], dtype=torch.float32),
        "category": torch.tensor([x.category for x in features], dtype=torch.long),
        "polarity": torch.tensor([x.polarity for x in features], dtype=torch.long),
        "relation": torch.tensor([x.relation for x in features], dtype=torch.float32),
        "pairs": [x.pairs for x in features],
    }
    if features[0].token_type_ids is not None:
        batch["token_type_ids"] = torch.tensor([x.token_type_ids for x in features], dtype=torch.long)
    return batch


class QuadrupleModel(nn.Module):
    def __init__(self, encoder: PreTrainedModel, trainable_layers: int):
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.aspect_start_head = nn.Linear(hidden_size, 1)
        self.aspect_end_head = nn.Linear(hidden_size, 1)
        self.opinion_start_head = nn.Linear(hidden_size, 1)
        self.opinion_end_head = nn.Linear(hidden_size, 1)
        self.implicit_vector = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.implicit_vector, std=0.02)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size, len(CLASS_PAIRS)),
        )
        self._configure_trainable_layers(trainable_layers)

    def _configure_trainable_layers(self, trainable_layers: int) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        else:
            for layer in list(layers)[-max(0, trainable_layers) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        for module in [self.aspect_start_head, self.aspect_end_head, self.opinion_start_head, self.opinion_end_head, self.relation_head]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.implicit_vector.requires_grad = True

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        return self.encoder(**kwargs).last_hidden_state

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden = self.encode(batch)
        return {
            "hidden": hidden,
            "aspect_start": self.aspect_start_head(hidden).squeeze(-1),
            "aspect_end": self.aspect_end_head(hidden).squeeze(-1),
            "opinion_start": self.opinion_start_head(hidden).squeeze(-1),
            "opinion_end": self.opinion_end_head(hidden).squeeze(-1),
        }

    def pair_logits(self, hidden: Tensor, pair: tuple[int, int, int, int]) -> Tensor:
        return self.relation_head(self.pair_features(hidden, pair))

    def pair_features(self, hidden: Tensor, pair: tuple[int, int, int, int]) -> Tensor:
        aspect_left, aspect_right, opinion_left, opinion_right = pair
        cls = hidden[0]
        if aspect_left < 0:
            aspect = self.implicit_vector
            implicit = self.implicit_vector
        else:
            aspect = hidden[aspect_left : aspect_right + 1].mean(dim=0)
            implicit = torch.zeros_like(cls)
        opinion = hidden[opinion_left : opinion_right + 1].mean(dim=0)
        features = torch.cat([cls, aspect, opinion, implicit], dim=-1)
        return features


class PointerQuadrupleModel(nn.Module):
    """One-stage pointer model: every span token predicts both entities."""

    def __init__(self, encoder: PreTrainedModel, trainable_layers: int, pointer_hidden: int = 128):
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.pointer_hidden = pointer_hidden
        self.aspect_start_left = nn.Linear(hidden_size, pointer_hidden)
        self.aspect_start_right = nn.Linear(hidden_size, pointer_hidden)
        self.aspect_start_out = nn.Linear(pointer_hidden, 1)
        self.aspect_end_left = nn.Linear(hidden_size, pointer_hidden)
        self.aspect_end_right = nn.Linear(hidden_size, pointer_hidden)
        self.aspect_end_out = nn.Linear(pointer_hidden, 1)
        self.opinion_start_left = nn.Linear(hidden_size, pointer_hidden)
        self.opinion_start_right = nn.Linear(hidden_size, pointer_hidden)
        self.opinion_start_out = nn.Linear(pointer_hidden, 1)
        self.opinion_end_left = nn.Linear(hidden_size, pointer_hidden)
        self.opinion_end_right = nn.Linear(hidden_size, pointer_hidden)
        self.opinion_end_out = nn.Linear(pointer_hidden, 1)
        self.objectiveness = nn.Linear(hidden_size, 1)
        self.category = nn.Linear(hidden_size, len(CATEGORY_ORDER))
        self.polarity = nn.Linear(hidden_size, len(POLARITY_ORDER))
        # Relation classes are multi-label: one shared span may map to more
        # than one category/polarity pair.  Keeping the pair intact avoids
        # the false cross-product produced by independent category/polarity
        # heads.
        self.relation = nn.Linear(hidden_size, len(CLASS_PAIRS))
        self._configure_trainable_layers(trainable_layers)

    def _configure_trainable_layers(self, trainable_layers: int) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        else:
            for layer in list(layers)[-max(0, trainable_layers) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        for module in [
            self.aspect_start_left, self.aspect_start_right, self.aspect_start_out,
            self.aspect_end_left, self.aspect_end_right, self.aspect_end_out,
            self.opinion_start_left, self.opinion_start_right, self.opinion_start_out,
            self.opinion_end_left, self.opinion_end_right, self.opinion_end_out,
            self.objectiveness, self.category, self.polarity,
            self.relation,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        return self.encoder(**kwargs).last_hidden_state

    @staticmethod
    def _pointer(left: nn.Linear, right: nn.Linear, out: nn.Linear, hidden: Tensor) -> Tensor:
        pair = left(hidden).unsqueeze(2) + right(hidden).unsqueeze(1)
        return out(F.leaky_relu(pair)).squeeze(-1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden = self.encode(batch)
        return {
            "hidden": hidden,
            "aspect_start": self._pointer(self.aspect_start_left, self.aspect_start_right, self.aspect_start_out, hidden),
            "aspect_end": self._pointer(self.aspect_end_left, self.aspect_end_right, self.aspect_end_out, hidden),
            "opinion_start": self._pointer(self.opinion_start_left, self.opinion_start_right, self.opinion_start_out, hidden),
            "opinion_end": self._pointer(self.opinion_end_left, self.opinion_end_right, self.opinion_end_out, hidden),
            "objectiveness": self.objectiveness(hidden).squeeze(-1),
            "category": self.category(hidden),
            "polarity": self.polarity(hidden),
            "relation": self.relation(hidden),
        }


def _pair_features(hidden: Tensor, pair: tuple[int, int, int, int], implicit_vector: Tensor) -> Tensor:
    aspect_left, aspect_right, opinion_left, opinion_right = pair
    cls = hidden[0]
    if aspect_left < 0:
        aspect = implicit_vector
        implicit = implicit_vector
    else:
        aspect = hidden[aspect_left : aspect_right + 1].mean(dim=0)
        implicit = torch.zeros_like(cls)
    opinion = hidden[opinion_left : opinion_right + 1].mean(dim=0)
    return torch.cat([cls, aspect, opinion, implicit], dim=-1)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _batch_loss(model: QuadrupleModel, outputs: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
    # Positive boundary labels are sparse.  A moderate positive weight keeps
    # the model from learning the all-zero solution on the small dataset.
    pos_weight = torch.tensor(5.0, device=outputs["hidden"].device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    span_loss = sum(
        bce(outputs[name], batch[name])
        for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end")
    ) / 4.0
    relation_losses: list[Tensor] = []
    for row_index, pairs in enumerate(batch["pairs"]):
        hidden = outputs["hidden"][row_index]
        for aspect_left, aspect_right, opinion_left, opinion_right, class_id in pairs:
            logits = model.pair_logits(hidden, (aspect_left, aspect_right, opinion_left, opinion_right)).unsqueeze(0)
            target = torch.tensor([class_id], dtype=torch.long, device=logits.device)
            relation_losses.append(F.cross_entropy(logits, target))
    if relation_losses:
        relation_loss = torch.stack(relation_losses).mean()
    else:
        relation_loss = span_loss.new_zeros(())
    loss = 0.65 * span_loss + relation_loss
    return loss, {"loss": float(loss.detach().cpu()), "span_loss": float(span_loss.detach().cpu()), "relation_loss": float(relation_loss.detach().cpu())}


def _pointer_batch_loss(model: PointerQuadrupleModel, outputs: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
    endpoint_losses = []
    for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end"):
        endpoint_losses.append(F.cross_entropy(outputs[name].transpose(1, 2), batch[name], ignore_index=-1))
    span_loss = torch.stack(endpoint_losses).mean()
    objectiveness = F.binary_cross_entropy_with_logits(outputs["objectiveness"], batch["objectiveness"], pos_weight=torch.tensor(2.0, device=outputs["objectiveness"].device))
    if "relation" in outputs and "relation" in batch:
        relation_raw = F.binary_cross_entropy_with_logits(
            outputs["relation"],
            batch["relation"],
            reduction="none",
            pos_weight=torch.full(
                (len(CLASS_PAIRS),), 3.0, device=outputs["relation"].device
            ),
        )
        active = batch["objectiveness"].unsqueeze(-1)
        relation_loss = (relation_raw * active).sum() / (active.sum().clamp_min(1.0) * relation_raw.shape[-1])
    else:
        # Compatibility path for old checkpoints/features created before the
        # multi-label relation target was introduced.
        category_loss = F.cross_entropy(outputs["category"].transpose(1, 2), batch["category"], ignore_index=-1)
        polarity_loss = F.cross_entropy(outputs["polarity"].transpose(1, 2), batch["polarity"], ignore_index=-1)
        relation_loss = category_loss + polarity_loss
    loss = span_loss + 0.35 * objectiveness + 0.85 * relation_loss
    return loss, {"loss": float(loss.detach().cpu()), "span_loss": float(span_loss.detach().cpu()), "relation_loss": float(relation_loss.detach().cpu())}


def _span_candidates(
    text: str,
    offsets: list[tuple[int, int]],
    start_prob: Tensor,
    end_prob: Tensor,
    *,
    max_span_length: int,
    top_k: int,
) -> list[tuple[int, int, str, float]]:
    usable = [index for index, (left, right) in enumerate(offsets) if right > left and right <= len(text)]
    if not usable:
        return []
    starts = sorted(usable, key=lambda index: float(start_prob[index]), reverse=True)[:top_k]
    ends = sorted(usable, key=lambda index: float(end_prob[index]), reverse=True)[:top_k]
    output: dict[tuple[int, int], tuple[str, float]] = {}
    for left in starts:
        for right in ends:
            if right < left or right - left + 1 > max_span_length:
                continue
            value = text[offsets[left][0] : offsets[right][1]]
            if not value:
                continue
            score = float(torch.sqrt(start_prob[left] * end_prob[right]).detach().cpu())
            key = (left, right)
            if key not in output or score > output[key][1]:
                output[key] = (value, score)
    result = [(left, right, value, score) for (left, right), (value, score) in output.items()]
    result.sort(key=lambda item: (-item[3], item[0], item[1]))
    return result[:top_k]


def _candidates_from_output(
    model: QuadrupleModel,
    row: ReviewExample,
    feature: _Feature,
    outputs: dict[str, Tensor],
    row_index: int,
    config: NeuralConfig,
    relation_head: nn.Module | None = None,
    implicit_vector: Tensor | None = None,
) -> list[Candidate]:
    hidden = outputs["hidden"][row_index]
    aspect_start = torch.sigmoid(outputs["aspect_start"][row_index])
    aspect_end = torch.sigmoid(outputs["aspect_end"][row_index])
    opinion_start = torch.sigmoid(outputs["opinion_start"][row_index])
    opinion_end = torch.sigmoid(outputs["opinion_end"][row_index])
    aspects = _span_candidates(row.text, feature.offsets, aspect_start, aspect_end, max_span_length=config.max_span_length, top_k=config.span_top_k)
    opinions = _span_candidates(row.text, feature.offsets, opinion_start, opinion_end, max_span_length=config.max_span_length, top_k=config.span_top_k)
    store: dict[Quadruple, tuple[float, set[str]]] = {}

    def add(quad: Quadruple, score: float) -> None:
        if quad.aspect != "_" and quad.aspect not in row.text:
            return
        if quad.opinion != "_" and quad.opinion not in row.text:
            return
        current = store.get(quad)
        if current is None or score > current[0]:
            store[quad] = (score, {"neural"})
        elif score == current[0]:
            store[quad] = (score, current[1] | {"neural"})

    specs: list[tuple[Quadruple, float, tuple[int, int, int, int]]] = []
    for op_left, op_right, opinion, opinion_score in opinions:
        specs.append((Quadruple("_", opinion, "", ""), opinion_score, (-1, -1, op_left, op_right)))
        for a_left, a_right, aspect, aspect_score in aspects:
            specs.append((Quadruple(aspect, opinion, "", ""), aspect_score * opinion_score, (a_left, a_right, op_left, op_right)))
    if specs:
        relation_head = relation_head or model.relation_head
        implicit_vector = implicit_vector if implicit_vector is not None else model.implicit_vector
        features = torch.stack([_pair_features(hidden, pair, implicit_vector) for _, _, pair in specs], dim=0)
        probabilities = torch.softmax(relation_head(features), dim=-1)
        top_values, top_indices = torch.topk(probabilities, k=min(config.relation_top_k, probabilities.shape[-1]), dim=-1)
        for spec_index, (base_quad, span_score, _) in enumerate(specs):
            for class_id_tensor, relation_tensor in zip(top_indices[spec_index], top_values[spec_index]):
                class_id = int(class_id_tensor)
                relation_score = float(relation_tensor)
                category, polarity = CLASS_PAIRS[class_id]
                add(Quadruple(base_quad.aspect, base_quad.opinion, category, polarity), max(0.001, span_score * relation_score))
    result = [Candidate(quadruple=quad, score=score, sources=tuple(sorted(sources))) for quad, (score, sources) in store.items()]
    result.sort(key=lambda item: (-item.score, item.quadruple))
    return result


def _model_candidates(
    model: QuadrupleModel,
    tokenizer: PreTrainedTokenizerBase,
    row: ReviewExample,
    config: NeuralConfig,
    device: torch.device,
) -> list[Candidate]:
    dataset = _ReviewDataset([row], tokenizer, config.max_length, use_official_offsets=config.use_official_offsets, use_implicit_opinion_sentinel=config.use_implicit_opinion_sentinel, preserve_multi_relation=config.preserve_multi_relation)
    feature = dataset[0]
    batch = _move_batch(_collate([feature]), device)
    with torch.no_grad():
        return _candidates_from_output(model, row, feature, model(batch), 0, config)


def _endpoint_beam(
    start_prob: Tensor,
    end_prob: Tensor,
    allowed: list[int],
    *,
    max_span_length: int,
    top_k: int,
) -> list[tuple[int, int, float]]:
    """Return constrained top-k endpoint pairs instead of one argmax pair."""
    beam = max(2, min(int(top_k), len(allowed)))
    start_values, start_indices = torch.topk(start_prob[allowed], k=beam)
    end_values, end_indices = torch.topk(end_prob[allowed], k=beam)
    pairs: list[tuple[int, int, float]] = []
    for start_value, start_index in zip(start_values, start_indices):
        left = allowed[int(start_index)]
        for end_value, end_index in zip(end_values, end_indices):
            right = allowed[int(end_index)]
            if right == 0 and left == 0:
                pairs.append((0, 0, float(torch.sqrt(start_value * end_value))))
                continue
            if left <= 0 or right <= 0 or right < left or right - left + 1 > max_span_length:
                continue
            pairs.append((left, right, float(torch.sqrt(start_value * end_value))))
    unique: dict[tuple[int, int], float] = {}
    for left, right, score in pairs:
        unique[(left, right)] = max(unique.get((left, right), 0.0), score)
    return sorted([(left, right, score) for (left, right), score in unique.items()], key=lambda x: -x[2])[:top_k]


def _span_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def _pointer_candidates_from_output(
    row: ReviewExample,
    feature: _Feature,
    outputs: dict[str, Tensor],
    row_index: int,
    config: NeuralConfig,
) -> list[Candidate]:
    offsets = feature.offsets
    usable = [index for index, (left, right) in enumerate(offsets) if right > left and right <= len(row.text)]
    if not usable:
        return []
    hidden_length = outputs["objectiveness"].shape[-1]
    allowed_endpoints = [0, *usable]
    invalid = torch.ones(hidden_length, dtype=torch.bool)
    invalid[allowed_endpoints] = False
    endpoint_probs = {}
    for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end"):
        logits = outputs[name][row_index].clone()
        logits[:, invalid] = -1.0e4
        endpoint_probs[name] = torch.softmax(logits, dim=-1)
    objectiveness = torch.sigmoid(outputs["objectiveness"][row_index])
    category_probs = torch.softmax(outputs["category"][row_index], dim=-1)
    polarity_probs = torch.softmax(outputs["polarity"][row_index], dim=-1)
    positions = sorted([0, *usable], key=lambda index: float(objectiveness[index]), reverse=True)[: max(config.span_top_k * 3, config.span_top_k)]
    raw: list[tuple[Quadruple, float, tuple[int, int] | None, tuple[int, int] | None]] = []

    def text_span(left: int, right: int) -> tuple[str, tuple[int, int]] | None:
        if left <= 0 or right < left or left >= len(offsets) or right >= len(offsets):
            return None
        if offsets[left][1] <= offsets[left][0] or offsets[right][1] <= offsets[right][0]:
            return None
        char_range = (offsets[left][0], offsets[right][1])
        value = row.text[char_range[0] : char_range[1]]
        return (value, char_range) if value else None

    for position in positions:
        aspect_beam = _endpoint_beam(endpoint_probs["aspect_start"][position], endpoint_probs["aspect_end"][position], allowed_endpoints, max_span_length=config.max_span_length, top_k=config.beam_top_k)
        opinion_beam = _endpoint_beam(endpoint_probs["opinion_start"][position], endpoint_probs["opinion_end"][position], allowed_endpoints, max_span_length=config.max_span_length, top_k=config.beam_top_k)
        relation_probs = torch.sigmoid(outputs["relation"][row_index, position]) if "relation" in outputs else None
        for a_start, a_end, a_score in aspect_beam:
            if a_start == 0 and a_end == 0:
                aspect, aspect_chars = "_", None
            else:
                aspect_info = text_span(a_start, a_end)
                if aspect_info is None:
                    continue
                aspect, aspect_chars = aspect_info
            for o_start, o_end, o_score in opinion_beam:
                if o_start == 0 and o_end == 0:
                    opinion, opinion_chars = "_", None
                else:
                    opinion_info = text_span(o_start, o_end)
                    if opinion_info is None:
                        continue
                    opinion, opinion_chars = opinion_info
                if aspect_chars is not None and opinion_chars is not None and _span_overlap(aspect_chars, opinion_chars) > 0:
                    continue
                span_score = float(objectiveness[position]) * max(0.0, a_score * o_score) ** 0.5
                if relation_probs is not None:
                    relation_top_k = max(config.relation_top_k, 4) if config.preserve_multi_relation else config.relation_top_k
                    relation_values, relation_ids = torch.topk(relation_probs, k=min(relation_top_k, relation_probs.numel()))
                    relation_items = [(CLASS_PAIRS[int(relation_id)], float(relation_value)) for relation_value, relation_id in zip(relation_values, relation_ids)]
                else:
                    category_value, category_id = torch.max(category_probs[position], dim=-1)
                    polarity_value, polarity_id = torch.max(polarity_probs[position], dim=-1)
                    relation_items = [((CATEGORY_ORDER[int(category_id)], POLARITY_ORDER[int(polarity_id)]), float(category_value * polarity_value))]
                for (category, polarity), relation_score in relation_items:
                    raw.append((Quadruple(aspect, opinion, category, polarity), max(0.001, span_score * relation_score), aspect_chars, opinion_chars))

    # NMS is applied before strict quadruple de-duplication so nearby beams
    # cannot flood the threshold calibrator with the same relation.
    raw.sort(key=lambda item: -item[1])
    kept: list[tuple[Quadruple, float, tuple[int, int] | None, tuple[int, int] | None]] = []
    for candidate in raw:
        quad, score, aspect_chars, opinion_chars = candidate
        suppressed = False
        for previous_quad, _, previous_a, previous_o in kept:
            if quad == previous_quad or (
                quad.category == previous_quad.category
                and quad.polarity == previous_quad.polarity
                and _span_overlap(aspect_chars, previous_a) >= 0.8
                and _span_overlap(opinion_chars, previous_o) >= 0.8
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    result: dict[Quadruple, float] = {}
    for quad, score, _, _ in kept:
        result[quad] = max(result.get(quad, 0.0), score)
    return [Candidate(quadruple=quad, score=score, sources=("neural_pointer_beam",)) for quad, score in sorted(result.items(), key=lambda item: (-item[1], item[0]))]


def predict_pointer_candidates(
    model: PointerQuadrupleModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Iterable[ReviewExample],
    config: NeuralConfig,
    device: torch.device,
) -> dict[int, list[Candidate]]:
    model.eval()
    rows = list(examples)
    if not rows:
        return {}
    dataset = _ReviewDataset(rows, tokenizer, config.max_length, use_official_offsets=config.use_official_offsets, use_implicit_opinion_sentinel=config.use_implicit_opinion_sentinel, preserve_multi_relation=config.preserve_multi_relation)
    loader = DataLoader(dataset, batch_size=max(1, config.batch_size * 2), shuffle=False, num_workers=0, collate_fn=_default_collate_wrapper)
    result: dict[int, list[Candidate]] = {}
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch)
            cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
            size = len(raw_batch["pairs"])
            for local_index, feature in enumerate(dataset.features[offset : offset + size]):
                row = rows[offset + local_index]
                result[row.id] = _pointer_candidates_from_output(row, feature, cpu_outputs, local_index, config)
            offset += size
    return result


def _top_relation_logits(model: QuadrupleModel, hidden: Tensor, pair: tuple[int, int, int, int], top_k: int) -> list[tuple[int, float]]:
    probabilities = torch.softmax(model.pair_logits(hidden, pair), dim=-1)
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    return [(int(index), float(value)) for index, value in zip(indices.detach().cpu(), values.detach().cpu())]


def predict_candidates(
    model: QuadrupleModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Iterable[ReviewExample],
    config: NeuralConfig,
    device: torch.device,
) -> dict[int, list[Candidate]]:
    if isinstance(model, PointerQuadrupleModel):
        return predict_pointer_candidates(model, tokenizer, examples, config, device)
    model.eval()
    rows = list(examples)
    if not rows:
        return {}
    dataset = _ReviewDataset(rows, tokenizer, config.max_length, use_official_offsets=config.use_official_offsets, use_implicit_opinion_sentinel=config.use_implicit_opinion_sentinel, preserve_multi_relation=config.preserve_multi_relation)
    loader = DataLoader(dataset, batch_size=max(1, config.batch_size * 2), shuffle=False, num_workers=0, collate_fn=_default_collate_wrapper)
    result: dict[int, list[Candidate]] = {}
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch)
            # Encoder inference stays on MPS, while the small relation head and
            # span enumeration run on CPU.  This avoids one device sync for
            # every candidate pair and is substantially faster on Apple GPUs.
            cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
            relation_head_cpu = copy.deepcopy(model.relation_head).cpu()
            implicit_vector_cpu = model.implicit_vector.detach().cpu()
            for local_index, feature in enumerate(dataset.features[offset : offset + len(raw_batch["pairs"])]):
                row = rows[offset + local_index]
                result[row.id] = _candidates_from_output(model, row, feature, cpu_outputs, local_index, config, relation_head_cpu, implicit_vector_cpu)
            offset += len(raw_batch["pairs"])
    return result


def _default_collate_wrapper(features: list[_Feature]) -> dict[str, Any]:
    return _collate(features)


class NeuralTrainer:
    def __init__(self, config: NeuralConfig, *, guard: ResourceGuard | None = None):
        self.config = config
        self.guard = guard or ResourceGuard()
        self.device = choose_device(config.device)
        set_seed(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def _build_model(self) -> QuadrupleModel:
        encoder = AutoModel.from_pretrained(self.config.model_name)
        return PointerQuadrupleModel(encoder, self.config.trainable_layers).to(self.device)

    def train_fold(
        self,
        train_examples: Iterable[ReviewExample],
        valid_examples: Iterable[ReviewExample],
        test_examples: Iterable[ReviewExample] | None = None,
        *,
        output_dir: str | Path | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> NeuralFoldResult:
        train_rows = list(train_examples)
        valid_rows = list(valid_examples)
        test_rows = list(test_examples or [])
        model = self._build_model()
        train_dataset = _ReviewDataset(train_rows, self.tokenizer, self.config.max_length, use_official_offsets=self.config.use_official_offsets, use_implicit_opinion_sentinel=self.config.use_implicit_opinion_sentinel, preserve_multi_relation=self.config.preserve_multi_relation)
        parameters = [p for p in model.parameters() if p.requires_grad]
        encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
        head_parameters = [p for p in parameters if all(p is not candidate for candidate in encoder_parameters)]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.config.learning_rate},
                {"params": head_parameters, "lr": self.config.head_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, Any]] = []
        best_f1 = -1.0
        best_state: dict[str, Tensor] | None = None
        batch_size = self.config.batch_size
        for epoch in range(1, self.config.epochs + 1):
            snapshot = self.guard.snapshot()
            recommendation = self.guard.recommend(self.config)
            batch_size = int(recommendation.get("batch_size", batch_size))
            if "gradient_accumulation" in recommendation:
                self.config.gradient_accumulation = int(recommendation["gradient_accumulation"])
            recommended_layers = int(recommendation.get("trainable_layers", self.config.trainable_layers))
            if recommended_layers < self.config.trainable_layers:
                self.config.trainable_layers = recommended_layers
                model._configure_trainable_layers(recommended_layers)
            if recommendation.get("max_length") != self.config.max_length:
                self.config.max_length = int(recommendation["max_length"])
                train_dataset = _ReviewDataset(train_rows, self.tokenizer, self.config.max_length, use_official_offsets=self.config.use_official_offsets, use_implicit_opinion_sentinel=self.config.use_implicit_opinion_sentinel, preserve_multi_relation=self.config.preserve_multi_relation)
            loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True, num_workers=0, collate_fn=_default_collate_wrapper)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running: dict[str, float] = {"loss": 0.0, "span_loss": 0.0, "relation_loss": 0.0}
            steps = 0
            accumulation = max(1, self.config.gradient_accumulation)
            fgm = FGM(model, epsilon=self.config.fgm_epsilon) if self.config.fgm_epsilon > 0 else None
            ema = EMAModel(parameters, decay=self.config.ema_decay) if self.config.use_ema else None
            for step, raw_batch in enumerate(loader, start=1):
                batch = _move_batch(raw_batch, self.device)
                outputs = model(batch)
                if isinstance(model, PointerQuadrupleModel):
                    loss, parts = _pointer_batch_loss(model, outputs, batch)
                else:
                    loss, parts = _batch_loss(model, outputs, batch)
                (loss / accumulation).backward()
                if step % accumulation == 0 or step == len(loader):
                    if fgm is not None and fgm.attack():
                        adversarial_outputs = model(batch)
                        if isinstance(model, PointerQuadrupleModel):
                            adversarial_loss, _ = _pointer_batch_loss(model, adversarial_outputs, batch)
                        else:
                            adversarial_loss, _ = _batch_loss(model, adversarial_outputs, batch)
                        (adversarial_loss / accumulation).backward()
                        fgm.restore()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update()
                for key, value in parts.items():
                    running[key] += value
                steps += 1
                if on_update and step % 20 == 0:
                    on_update({"epoch": epoch, "step": step, "steps": len(loader), **{key: value / steps for key, value in running.items()}, "resource": asdict(self.guard.snapshot())})
            model.eval()
            # Fixed monitoring threshold prevents selecting an epoch solely by
            # overfitting a validation threshold.  Final calibration is OOF.
            from .pipeline import threshold_predictions
            from .metrics import strict_f1
            valid_gold = {row.id: row.labels for row in valid_rows}

            def monitor_score() -> tuple[Any, Any]:
                candidates = predict_candidates(model, self.tokenizer, valid_rows, self.config, self.device)
                predictions = threshold_predictions(candidates, 0.12, 0.10)
                return strict_f1(valid_gold, predictions), candidates

            score, valid_candidates = monitor_score()
            ema_state: dict[str, Tensor] | None = None
            if ema is not None:
                ema.apply()
                ema_score, ema_candidates = monitor_score()
                if ema_score.f1 > score.f1:
                    score = ema_score
                    valid_candidates = ema_candidates
                    ema_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                ema.restore()
            record = {"epoch": epoch, "step": steps, **{key: value / max(1, steps) for key, value in running.items()}, "precision": score.precision, "recall": score.recall, "f1": score.f1, "used_ema": ema_state is not None, "resource": asdict(self.guard.snapshot())}
            history.append(record)
            if on_update:
                on_update(record)
            if score.f1 > best_f1:
                best_f1 = score.f1
                best_state = ema_state if ema_state is not None else {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            self.guard.release_cache()
        if best_state is not None:
            model.load_state_dict(best_state)
        valid_candidates = predict_candidates(model, self.tokenizer, valid_rows, self.config, self.device)
        test_candidates = predict_candidates(model, self.tokenizer, test_rows, self.config, self.device) if test_rows else {}
        checkpoint_path: str | None = None
        if output_dir:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / "model.pt"
            torch.save({"state_dict": model.state_dict(), "config": asdict(self.config)}, checkpoint)
            checkpoint_path = str(checkpoint)
            (directory / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        self.guard.release_cache()
        return NeuralFoldResult(valid_candidates, test_candidates, history, checkpoint_path, asdict(self.config))
