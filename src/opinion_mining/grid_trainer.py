from __future__ import annotations

"""Small adapter and screening-plan skeleton for ``grid_model``.

This module wires the existing ``_ReviewDataset``/``_collate`` path to
``CompactPairRelationHead`` and exposes both the loss step and a fold-local
training function.  The companion CLI can run bounded CPU smoke paths or emit
a fixed 3-fold plan; it never downloads a model and never claims an official
evaluation score.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import LabelSpan, Quadruple, ReviewExample
from .folds import fixed_splits
from .grid_model import (
    CompactPairRelationHead,
    NUM_RELATION_CLASSES,
    GridCandidate,
    PairSpan,
    build_sparse_relation_targets,
    constrained_topk_span_decode,
    exact_pair_target_matrix,
    grid_candidates_to_candidates,
    hard_negative_pairs,
    nms_grid_candidates,
    pair_relation_loss,
    pair_spans_to_tensor,
    sparse_grid_bce_loss,
)
from .neural import _ReviewDataset, _collate, choose_device, set_seed
from .resource_guard import ResourceGuard
from .baseline import Candidate
from .metrics import strict_f1
from .pipeline import select_threshold, threshold_predictions


MODEL_ALIASES = {
    "macbert": "hfl/chinese-macbert-base",
    "wwm": "hfl/chinese-roberta-wwm-ext",
    "rbt3": "hfl/rbt3",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name.strip().lower(), model_name)


@dataclass
class GridTrainConfig:
    model_name: str = "rbt3"
    max_length: int = 128
    batch_size: int = 2
    gradient_accumulation: int = 2
    epochs: int = 1
    learning_rate: float = 8.0e-4
    relation_rank: int = 32
    trainable_layers: int = 0
    seed: int = 42
    device: str = "cpu"
    n_splits: int = 3
    head_learning_rate: float = 8.0e-4
    weight_decay: float = 0.01
    max_span_length: int = 12
    span_top_k: int = 16
    relation_top_k: int = 1
    nms_iou: float = 0.8
    implicit_loss_weight: float = 1.0
    pair_relation_loss_weight: float = 0.5
    pair_validity_loss_weight: float = 0.5
    pair_ranking_loss_weight: float = 0.25
    pair_ranking_margin: float = 0.5
    implicit_presence_loss_weight: float = 0.25
    hard_negative_ratio: int = 3


class ResourcePressureRestart(RuntimeError):
    """Signal that a fold must restart with a safer runtime configuration."""

    def __init__(self, recommendation: dict[str, Any], snapshot: Any):
        super().__init__("resource pressure requires fold restart")
        self.recommendation = dict(recommendation)
        self.snapshot = snapshot


class DummyTokenizer:
    """Character tokenizer with the same fields consumed by ``_ReviewDataset``."""

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        return_offsets_mapping: bool = True,
    ) -> dict[str, list[Any]]:
        if max_length < 3:
            raise ValueError("dummy tokenizer needs max_length >= 3")
        usable = min(len(text), max_length - 2) if truncation else len(text)
        ids = [1] + [3 + (ord(char) % 251) for char in text[:usable]] + [2]
        offsets = [(0, 0)] + [(index, index + 1) for index in range(usable)] + [(0, 0)]
        attention = [1] * len(ids)
        pad = max_length - len(ids)
        ids += [0] * max(0, pad)
        offsets += [(0, 0)] * max(0, pad)
        attention += [0] * max(0, pad)
        return {
            "input_ids": ids[:max_length],
            "attention_mask": attention[:max_length],
            "token_type_ids": [0] * max_length,
            "offset_mapping": offsets[:max_length],
        }


class DummyEncoder(nn.Module):
    """Tiny CPU encoder used only by the smoke path."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 512):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, token_type_ids: Tensor | None = None) -> Any:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class CompactGridEncoderAdapter(nn.Module):
    """Wrap an encoder and expose the grid head without touching ``neural.py``.

    ``forward(batch)`` returns ``hidden`` and dense relation logits with shape
    ``[B,L,L,39]``.  A later trainer should call ``grid_training_step`` with
    the same raw collated batch to construct sparse targets and loss.  The
    adapter deliberately does not own an optimizer, fold loop, threshold
    calibration, or F1 computation.
    """

    def __init__(self, encoder: nn.Module, *, relation_rank: int = 32, trainable_layers: int = 0):
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.relation_head = CompactPairRelationHead(hidden_size, rank=relation_rank)
        self.aspect_start = nn.Linear(hidden_size, 1)
        self.aspect_end = nn.Linear(hidden_size, 1)
        self.opinion_start = nn.Linear(hidden_size, 1)
        self.opinion_end = nn.Linear(hidden_size, 1)
        self.objectiveness = nn.Linear(hidden_size, 1)
        self.implicit_aspect_presence = nn.Linear(hidden_size, 1)
        self.implicit_opinion_presence = nn.Linear(hidden_size, 1)
        self.category = nn.Linear(hidden_size, 13)
        self.polarity = nn.Linear(hidden_size, 3)
        self._configure_trainable_layers(trainable_layers)

    def _configure_trainable_layers(self, trainable_layers: int) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            if trainable_layers > 0:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = True
        else:
            for layer in list(layers)[-max(0, trainable_layers) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        for parameter in self.relation_head.parameters():
            parameter.requires_grad = True
        for module in (self.aspect_start, self.aspect_end, self.opinion_start, self.opinion_end, self.objectiveness):
            for parameter in module.parameters():
                parameter.requires_grad = True

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        relation_rank: int = 32,
        trainable_layers: int = 0,
    ) -> "CompactGridEncoderAdapter":
        """Load an already cached encoder; ``local_files_only`` forbids downloads."""

        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(resolve_model_name(model_name), local_files_only=True)
        return cls(encoder, relation_rank=relation_rank, trainable_layers=trainable_layers)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        kwargs: dict[str, Tensor] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        hidden = self.encoder(**kwargs).last_hidden_state
        return {
            "hidden": hidden,
            "relation": self.relation_head.grid_logits(hidden),
            "aspect_start": self.aspect_start(hidden).squeeze(-1),
            "aspect_end": self.aspect_end(hidden).squeeze(-1),
            "opinion_start": self.opinion_start(hidden).squeeze(-1),
            "opinion_end": self.opinion_end(hidden).squeeze(-1),
            "objectiveness": self.objectiveness(hidden).squeeze(-1),
            "implicit_aspect_presence": self.implicit_aspect_presence(hidden[:, 0]).squeeze(-1),
            "implicit_opinion_presence": self.implicit_opinion_presence(hidden[:, 0]).squeeze(-1),
            "category": self.category(hidden),
            "polarity": self.polarity(hidden),
        }


def load_cached_tokenizer(model_name: str) -> Any:
    """Load the existing fast-tokenizer path without network fallback."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_model_name(model_name), use_fast=True, local_files_only=True)


def prepare_features(rows: Iterable[ReviewExample], tokenizer: Any, max_length: int) -> list[Any]:
    """Use the project's private dataset implementation without editing it."""

    return list(_ReviewDataset(list(rows), tokenizer, max_length).features)


def collate_features(features: Sequence[Any]) -> dict[str, Any]:
    return _collate(list(features))


def dynamic_runtime(config: GridTrainConfig, guard: ResourceGuard) -> dict[str, Any]:
    """Expose ResourceGuard's batch/accumulation decision for a future loop."""

    recommendation = guard.recommend(config)
    return {
        "batch_size": int(recommendation.get("batch_size", config.batch_size)),
        "gradient_accumulation": int(recommendation.get("gradient_accumulation", config.gradient_accumulation)),
        "max_length": int(recommendation.get("max_length", config.max_length)),
        "device": config.device,
    }


def _deduplicate_feature_pairs(
    pairs_by_row: Sequence[Sequence[Sequence[int]]],
) -> list[list[tuple[int, int, int, int, int]]]:
    """Remove only exact token-span/class duplicates while preserving row order."""

    result: list[list[tuple[int, int, int, int, int]]] = []
    for row in pairs_by_row:
        unique = dict.fromkeys(tuple(int(value) for value in pair) for pair in row)
        result.append([tuple(item) for item in unique])
    return result


def grid_training_step(
    adapter: CompactGridEncoderAdapter,
    raw_batch: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    implicit_loss_weight: float = 1.0,
    pair_relation_loss_weight: float = 0.5,
    pair_validity_loss_weight: float = 0.5,
    pair_ranking_loss_weight: float = 0.25,
    pair_ranking_margin: float = 0.5,
    implicit_presence_loss_weight: float = 0.25,
    hard_negative_ratio: int = 3,
) -> tuple[Tensor, dict[str, float]]:
    """One loss step for a future optimizer loop; no optimizer is created here."""

    target_device = torch.device(device)
    batch = {
        key: value.to(target_device) if isinstance(value, Tensor) else value
        for key, value in raw_batch.items()
    }
    outputs = adapter(batch)
    unique_pairs = _deduplicate_feature_pairs(raw_batch["pairs"])
    targets = build_sparse_relation_targets(unique_pairs, max_length=outputs["hidden"].shape[1]).to(target_device)
    grid_loss = sparse_grid_bce_loss(
        outputs["relation"],
        targets,
        valid_mask=batch.get("attention_mask"),
        implicit_weight=implicit_loss_weight,
    )

    positives_by_batch = [
        [PairSpan.from_feature_pair(pair) for pair in row]
        for row in unique_pairs
    ]
    pair_rows: list[list[int]] = []
    validity_values: list[float] = []
    hard_negative_count = 0
    sequence_length = int(outputs["hidden"].shape[1])
    for batch_index, positives in enumerate(positives_by_batch):
        negatives = hard_negative_pairs(
            positives,
            sequence_length=sequence_length,
            max_count=max(1, len(positives) * max(1, int(hard_negative_ratio))),
        )
        hard_negative_count += len(negatives)
        for item, target in [*((positive, 1.0) for positive in positives), *((negative, 0.0) for negative in negatives)]:
            pair_rows.append([
                batch_index,
                -1 if item.implicit_aspect else item.aspect_start,
                -1 if item.implicit_aspect else item.aspect_end,
                -1 if item.implicit_opinion else item.opinion_start,
                -1 if item.implicit_opinion else item.opinion_end,
            ])
            validity_values.append(target)
    if pair_rows:
        pair_tensor = torch.tensor(pair_rows, dtype=torch.long, device=target_device)
        pair_logits = adapter.relation_head.pair_logits(outputs["hidden"], pair_tensor)
        pair_targets = exact_pair_target_matrix(pair_tensor, positives_by_batch).to(target_device)
        pair_class_loss = pair_relation_loss(pair_logits, pair_targets)
        pair_validity_logits = adapter.relation_head.pair_validity_logits(outputs["hidden"], pair_tensor)
        pair_validity_targets = torch.tensor(validity_values, dtype=pair_validity_logits.dtype, device=target_device)
        pair_validity_loss = F.binary_cross_entropy_with_logits(pair_validity_logits, pair_validity_targets)
        ranking_terms = []
        for batch_index in pair_tensor[:, 0].unique(sorted=True).tolist():
            same_batch = pair_tensor[:, 0] == int(batch_index)
            positive_logits = pair_validity_logits[same_batch & (pair_validity_targets > 0.5)]
            negative_logits = pair_validity_logits[same_batch & (pair_validity_targets <= 0.5)]
            if positive_logits.numel() and negative_logits.numel():
                pairwise_margin = float(pair_ranking_margin) - positive_logits.unsqueeze(1) + negative_logits.unsqueeze(0)
                ranking_terms.append(F.relu(pairwise_margin).mean())
        pair_ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else grid_loss.new_zeros(())
    else:
        pair_class_loss = grid_loss.new_zeros(())
        pair_validity_loss = grid_loss.new_zeros(())
        pair_ranking_loss = grid_loss.new_zeros(())

    implicit_aspect_target = torch.tensor(
        [float(any(item.implicit_aspect for item in positives)) for positives in positives_by_batch],
        dtype=outputs["hidden"].dtype,
        device=target_device,
    )
    implicit_opinion_target = torch.tensor(
        [float(any(item.implicit_opinion for item in positives)) for positives in positives_by_batch],
        dtype=outputs["hidden"].dtype,
        device=target_device,
    )
    implicit_presence_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(outputs["implicit_aspect_presence"], implicit_aspect_target)
        + F.binary_cross_entropy_with_logits(outputs["implicit_opinion_presence"], implicit_opinion_target)
    )

    boundary_targets = _boundary_targets(raw_batch["pairs"], outputs["hidden"].shape[1], target_device)
    attention = batch.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(boundary_targets["objectiveness"])
    boundary_mask = attention.to(dtype=outputs["hidden"].dtype)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=outputs["hidden"].new_tensor(4.0))
    boundary_losses = []
    for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end", "objectiveness"):
        raw = bce(outputs[name], boundary_targets[name])
        boundary_losses.append((raw * boundary_mask).sum() / boundary_mask.sum().clamp_min(1.0))
    boundary_loss = torch.stack(boundary_losses).mean()
    auxiliary_losses = []
    for name, classes in (("category", 13), ("polarity", 3)):
        target = batch.get(name)
        if target is not None:
            flat_target = target.reshape(-1).long()
            valid = flat_target >= 0
            if valid.any():
                auxiliary_losses.append(F.cross_entropy(outputs[name].reshape(-1, classes)[valid], flat_target[valid]))
    auxiliary_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else grid_loss.new_zeros(())
    loss = (
        grid_loss
        + 0.25 * boundary_loss
        + 0.10 * auxiliary_loss
        + float(pair_relation_loss_weight) * pair_class_loss
        + float(pair_validity_loss_weight) * pair_validity_loss
        + float(pair_ranking_loss_weight) * pair_ranking_loss
        + float(implicit_presence_loss_weight) * implicit_presence_loss
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("grid loss is not finite")
    return loss, {
        "loss": float(loss.detach().cpu()),
        "grid_loss": float(grid_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "auxiliary_loss": float(auxiliary_loss.detach().cpu()),
        "pair_relation_loss": float(pair_class_loss.detach().cpu()),
        "pair_validity_loss": float(pair_validity_loss.detach().cpu()),
        "pair_ranking_loss": float(pair_ranking_loss.detach().cpu()),
        "implicit_presence_loss": float(implicit_presence_loss.detach().cpu()),
        "hard_negatives": float(hard_negative_count),
        "positives": float(targets.num_positives),
    }


def _boundary_targets(pairs_by_row: Sequence[Sequence[Sequence[int]]], length: int, device: torch.device) -> dict[str, Tensor]:
    """Create multi-hot endpoint/objectiveness targets without dense grid storage."""
    batch_size = len(pairs_by_row)
    targets = {
        name: torch.zeros((batch_size, length), dtype=torch.float32, device=device)
        for name in ("aspect_start", "aspect_end", "opinion_start", "opinion_end", "objectiveness")
    }
    for row_index, pairs in enumerate(pairs_by_row):
        for pair in pairs:
            a_start, a_end, o_start, o_end, _ = (int(value) for value in pair)
            a_start = 0 if a_start < 0 else a_start
            a_end = 0 if a_end < 0 else a_end
            o_start = 0 if o_start <= 0 else o_start
            o_end = 0 if o_end <= 0 else o_end
            for name, index in (("aspect_start", a_start), ("aspect_end", a_end), ("opinion_start", o_start), ("opinion_end", o_end)):
                if 0 <= index < length:
                    targets[name][row_index, index] = 1.0
            active: set[int] = set()
            if a_start > 0 and a_end >= a_start:
                active.update(range(a_start, min(length - 1, a_end) + 1))
            if o_start > 0 and o_end >= o_start:
                active.update(range(o_start, min(length - 1, o_end) + 1))
            if not active:
                active.add(0)
            for index in active:
                if 0 <= index < length:
                    targets["objectiveness"][row_index, index] = 1.0
    return targets


def _valid_surface_mask(feature: Any, text: str, attention: Tensor | None = None) -> Tensor:
    values = [right > left and right <= len(text) for left, right in feature.offsets]
    mask = torch.tensor(values, dtype=torch.bool)
    if attention is not None:
        mask &= attention.to(dtype=torch.bool).cpu()
    return mask


def predict_grid_candidates(
    adapter: CompactGridEncoderAdapter,
    tokenizer: Any,
    rows: Iterable[ReviewExample],
    config: GridTrainConfig,
    device: torch.device | str,
) -> dict[int, list[Candidate]]:
    """Run full-span pair refinement and convert token spans back to strict candidates."""

    values = list(rows)
    if not values:
        return {}
    adapter.eval()
    dataset = _ReviewDataset(values, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=max(1, config.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    target_device = torch.device(device)
    result: dict[int, list[Candidate]] = {}
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {key: value.to(target_device) if isinstance(value, Tensor) else value for key, value in raw_batch.items()}
            outputs = adapter(batch)
            cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
            batch_size = len(raw_batch["pairs"])
            for local_index in range(batch_size):
                row = values[offset + local_index]
                feature = dataset.features[offset + local_index]
                valid = _valid_surface_mask(feature, row.text, raw_batch["attention_mask"][local_index])
                aspect_spans = constrained_topk_span_decode(
                    cpu_outputs["aspect_start"][local_index],
                    cpu_outputs["aspect_end"][local_index],
                    valid_mask=valid,
                    top_k=config.span_top_k,
                    max_span_length=config.max_span_length,
                    allow_implicit=True,
                    objectiveness_logits=cpu_outputs["objectiveness"][local_index],
                    implicit_logit=cpu_outputs["implicit_aspect_presence"][local_index],
                )
                opinion_spans = constrained_topk_span_decode(
                    cpu_outputs["opinion_start"][local_index],
                    cpu_outputs["opinion_end"][local_index],
                    valid_mask=valid,
                    top_k=config.span_top_k,
                    max_span_length=config.max_span_length,
                    allow_implicit=True,
                    objectiveness_logits=cpu_outputs["objectiveness"][local_index],
                    implicit_logit=cpu_outputs["implicit_opinion_presence"][local_index],
                )
                pair_items = []
                pair_rows = []
                for aspect in aspect_spans:
                    for opinion in opinion_spans:
                        if (
                            not aspect.implicit
                            and not opinion.implicit
                            and max(aspect.start, opinion.start) <= min(aspect.end, opinion.end)
                        ):
                            continue
                        pair_items.append((aspect, opinion))
                        pair_rows.append([
                            0,
                            -1 if aspect.implicit else aspect.start,
                            -1 if aspect.implicit else aspect.end,
                            -1 if opinion.implicit else opinion.start,
                            -1 if opinion.implicit else opinion.end,
                        ])
                refined: list[GridCandidate] = []
                if pair_rows:
                    pair_tensor = torch.tensor(pair_rows, dtype=torch.long, device=target_device)
                    hidden_row = outputs["hidden"][local_index : local_index + 1]
                    class_logits = adapter.relation_head.pair_logits(hidden_row, pair_tensor)
                    validity_logits = adapter.relation_head.pair_validity_logits(hidden_row, pair_tensor)
                    class_probs = torch.sigmoid(class_logits).detach().cpu()
                    validity_probs = torch.sigmoid(validity_logits).detach().cpu()
                    for pair_index, (aspect, opinion) in enumerate(pair_items):
                        values_, class_ids = torch.topk(
                            class_probs[pair_index],
                            k=min(max(1, config.relation_top_k), NUM_RELATION_CLASSES),
                        )
                        span_score = max(0.0, float(aspect.score * opinion.score)) ** 0.5
                        pair_score = float(validity_probs[pair_index])
                        for relation_score, class_id in zip(values_.tolist(), class_ids.tolist()):
                            refined.append(
                                GridCandidate(
                                    aspect,
                                    opinion,
                                    int(class_id),
                                    span_score * pair_score * float(relation_score),
                                )
                            )
                kept = nms_grid_candidates(
                    refined,
                    iou_threshold=config.nms_iou,
                    max_keep=max(config.span_top_k * config.span_top_k, 16),
                )
                result[row.id] = grid_candidates_to_candidates(
                    kept,
                    text=row.text,
                    offsets=feature.offsets,
                    source="relation_grid_v2",
                )
            offset += batch_size
    return result


def _model_parameters(adapter: CompactGridEncoderAdapter) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    encoder_parameters = [parameter for parameter in adapter.encoder.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad and id(parameter) not in encoder_ids]
    return encoder_parameters, head_parameters


def train_grid_fold(
    train_rows: Sequence[ReviewExample],
    valid_rows: Sequence[ReviewExample],
    config: GridTrainConfig,
    *,
    guard: ResourceGuard | None = None,
    test_rows: Sequence[ReviewExample] | None = None,
    output_dir: str | Path | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    tokenizer: Any | None = None,
    encoder: nn.Module | None = None,
    resource_check_interval: int = 20,
    select_best_epoch: bool = False,
) -> dict[str, Any]:
    """Train one real grid fold and return OOF/test candidates.

    The caller owns the outer fold loop and threshold calibration.  This
    function keeps every tokenizer/model instance fold-local and uses
    ``local_files_only`` so an accidentally missing backbone fails before
    consuming GPU memory.
    """
    runtime_guard = guard or ResourceGuard()
    runtime_config = replace(config)
    set_seed(runtime_config.seed)
    # Apply the first resource recommendation before constructing the encoder.
    # This matters for a base model: reducing trainable layers after the model
    # has been built would leave the optimizer holding the wrong parameter set.
    initial_recommendation = runtime_guard.recommend(runtime_config)
    runtime_config.batch_size = int(initial_recommendation.get("batch_size", runtime_config.batch_size))
    runtime_config.gradient_accumulation = int(
        initial_recommendation.get("gradient_accumulation", runtime_config.gradient_accumulation)
    )
    runtime_config.max_length = int(initial_recommendation.get("max_length", runtime_config.max_length))
    runtime_config.trainable_layers = int(
        initial_recommendation.get("trainable_layers", runtime_config.trainable_layers)
    )
    device = choose_device(runtime_config.device)
    active_tokenizer = tokenizer if tokenizer is not None else load_cached_tokenizer(runtime_config.model_name)
    if encoder is None:
        adapter = CompactGridEncoderAdapter.from_pretrained(
            runtime_config.model_name,
            relation_rank=runtime_config.relation_rank,
            trainable_layers=runtime_config.trainable_layers,
        ).to(device)
    else:
        adapter = CompactGridEncoderAdapter(
            encoder,
            relation_rank=runtime_config.relation_rank,
            trainable_layers=runtime_config.trainable_layers,
        ).to(device)
    train_dataset = _ReviewDataset(list(train_rows), active_tokenizer, runtime_config.max_length)
    encoder_parameters, head_parameters = _model_parameters(adapter)
    parameter_groups = [{"params": head_parameters, "lr": runtime_config.head_learning_rate, "weight_decay": runtime_config.weight_decay}]
    if encoder_parameters:
        parameter_groups.insert(0, {"params": encoder_parameters, "lr": runtime_config.learning_rate, "weight_decay": runtime_config.weight_decay})
    optimizer = torch.optim.AdamW(parameter_groups)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    batch_size = runtime_config.batch_size
    for epoch in range(1, runtime_config.epochs + 1):
        recommendation = runtime_guard.recommend(runtime_config)
        batch_size = int(recommendation.get("batch_size", runtime_config.batch_size))
        runtime_config.batch_size = batch_size
        runtime_config.gradient_accumulation = int(recommendation.get("gradient_accumulation", runtime_config.gradient_accumulation))
        recommended_layers = int(recommendation.get("trainable_layers", runtime_config.trainable_layers))
        if recommended_layers < runtime_config.trainable_layers:
            # The optimizer may retain frozen parameters, but they no longer
            # receive gradients.  This makes a step-level layer downshift real
            # instead of merely changing the label shown by the Dashboard.
            adapter._configure_trainable_layers(recommended_layers)
            runtime_config.trainable_layers = recommended_layers
        if int(recommendation.get("max_length", runtime_config.max_length)) != runtime_config.max_length:
            runtime_config.max_length = int(recommendation["max_length"])
            train_dataset = _ReviewDataset(list(train_rows), active_tokenizer, runtime_config.max_length)
        loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {"loss": 0.0, "grid_loss": 0.0, "boundary_loss": 0.0}
        steps = 0
        for step, raw_batch in enumerate(loader, start=1):
            if resource_check_interval > 0 and step % resource_check_interval == 0:
                if hasattr(runtime_guard, "runtime_decision"):
                    decision = runtime_guard.runtime_decision(runtime_config)
                else:
                    snapshot = runtime_guard.snapshot()
                    severe = (
                        getattr(snapshot, "pressure", "unknown") == "critical"
                        or float(getattr(snapshot, "available_gb", 999.0)) < 3.0
                        or float(getattr(snapshot, "swap_growth_gb", 0.0) or 0.0) >= 0.5
                    )
                    decision = {
                        "restart_required": severe,
                        "reason": "red_memory_pressure" if severe else "continue",
                        "recommendation": runtime_guard.recommend(runtime_config),
                        "snapshot": snapshot,
                    }
                snapshot = decision["snapshot"]
                if on_update:
                    on_update({
                        "phase": "resource_check",
                        "epoch": epoch,
                        "step": step,
                        "steps": len(loader),
                        "runtime_config": asdict(runtime_config),
                        "resource": getattr(snapshot, "__dict__", snapshot),
                    })
                if decision.get("restart_required"):
                    if output_dir is not None:
                        directory = Path(output_dir)
                        directory.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"state_dict": adapter.state_dict(), "config": asdict(runtime_config), "recommendation": decision["recommendation"]},
                            directory / "resource_pressure_checkpoint.pt",
                        )
                    runtime_guard.release_cache()
                    raise ResourcePressureRestart(decision["recommendation"], snapshot)
            loss, parts = grid_training_step(
                adapter,
                raw_batch,
                device=device,
                implicit_loss_weight=runtime_config.implicit_loss_weight,
                pair_relation_loss_weight=runtime_config.pair_relation_loss_weight,
                pair_validity_loss_weight=runtime_config.pair_validity_loss_weight,
                pair_ranking_loss_weight=runtime_config.pair_ranking_loss_weight,
                pair_ranking_margin=runtime_config.pair_ranking_margin,
                implicit_presence_loss_weight=runtime_config.implicit_presence_loss_weight,
                hard_negative_ratio=runtime_config.hard_negative_ratio,
            )
            (loss / max(1, runtime_config.gradient_accumulation)).backward()
            if step % max(1, runtime_config.gradient_accumulation) == 0 or step == len(loader):
                trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key in running:
                running[key] += float(parts.get(key, 0.0))
            steps += 1
            if on_update and step % 20 == 0:
                payload = {"phase": "step", "epoch": epoch, "step": step, "steps": len(loader), "runtime_config": asdict(runtime_config), **{key: value / max(1, steps) for key, value in running.items()}, "resource": runtime_guard.snapshot().__dict__}
                on_update(payload)
        valid_candidates = predict_grid_candidates(adapter, active_tokenizer, valid_rows, runtime_config, device)
        valid_gold = {row.id: row.labels for row in valid_rows}
        monitor_selected = select_threshold(valid_candidates, valid_gold, separate_implicit=True)
        monitor = monitor_selected["score"]
        record = {
            "phase": "epoch_complete",
            "epoch": epoch,
            "step": steps,
            "steps": steps,
            "runtime_config": asdict(runtime_config),
            **{key: value / max(1, steps) for key, value in running.items()},
            "precision": monitor.precision,
            "recall": monitor.recall,
            "f1": monitor.f1,
            "monitor_calibration": "fold_local_four_state_display_only",
            "monitor_state_thresholds": monitor_selected.get("state_thresholds", {}),
            "resource": runtime_guard.snapshot().__dict__,
        }
        history.append(record)
        if on_update:
            on_update(record)
        if select_best_epoch and monitor.f1 > best_f1:
            best_f1 = monitor.f1
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
        runtime_guard.release_cache()
    if select_best_epoch and best_state is not None:
        adapter.load_state_dict(best_state)
    valid_candidates = predict_grid_candidates(adapter, active_tokenizer, valid_rows, runtime_config, device)
    test_candidates = predict_grid_candidates(adapter, active_tokenizer, test_rows or [], runtime_config, device) if test_rows else {}
    checkpoint_path = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(directory / "model.pt")
        torch.save({"state_dict": adapter.state_dict(), "config": asdict(runtime_config), "model_name": resolve_model_name(runtime_config.model_name)}, checkpoint_path)
        write_json(directory / "history.json", history)
    del adapter
    runtime_guard.release_cache()
    return {"candidates": valid_candidates, "test_candidates": test_candidates, "history": history, "checkpoint": checkpoint_path, "config": asdict(runtime_config)}


def fixed_screen_plan(rows: Sequence[ReviewExample], config: GridTrainConfig, guard: ResourceGuard) -> dict[str, Any]:
    """Build a serialisable fixed-fold plan without fitting or scoring."""

    if config.n_splits not in (3, 5):
        raise ValueError("n_splits must be 3 or 5")
    splits = fixed_splits(rows, n_splits=config.n_splits, seed=config.seed)
    return {
        "status": "plan_only",
        "model_name": config.model_name,
        "resolved_model_name": resolve_model_name(config.model_name),
        "config": asdict(config),
        "runtime_recommendation": dynamic_runtime(config, guard),
        "folds": [
            {
                "fold": fold,
                "train_indices": train_indices,
                "valid_indices": valid_indices,
                "train_ids": [rows[index].id for index in train_indices],
                "valid_ids": [rows[index].id for index in valid_indices],
            }
            for fold, (train_indices, valid_indices) in enumerate(splits, start=1)
        ],
        "train_loop_next": [
            "instantiate CompactGridEncoderAdapter per fold",
            "prepare _ReviewDataset features and _collate batches",
            "call grid_training_step, divide loss by gradient_accumulation, then optimizer.step",
            "save state_dict checkpoint and serialize validation candidates as OOF JSON",
        ],
        "metrics": "not_computed",
    }


def dummy_smoke(*, max_length: int = 12, hidden_size: int = 16) -> dict[str, Any]:
    """Run one CPU batch through the real dataset/collate/grid loss path."""

    rows = [
        ReviewExample(
            id=1,
            text="很好用",
            label_spans=(
                LabelSpan(
                    quadruple=Quadruple("_", "很好", "整体", "正面"),
                    opinion_start=0,
                    opinion_end=2,
                ),
            ),
        )
    ]
    tokenizer = DummyTokenizer()
    features = prepare_features(rows, tokenizer, max_length)
    raw_batch = collate_features(features)
    adapter = CompactGridEncoderAdapter(DummyEncoder(hidden_size=hidden_size), relation_rank=4)
    loss, parts = grid_training_step(adapter, raw_batch, device="cpu")
    loss.backward()
    return {
        "status": "dummy_smoke_passed",
        "hidden_shape": list(adapter(raw_batch)["hidden"].shape),
        "relation_shape": list(adapter(raw_batch)["relation"].shape),
        "relation_classes": NUM_RELATION_CLASSES,
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "loss": float(loss.detach()),
        "positives": int(parts["positives"]),
        "hard_negatives": int(parts["hard_negatives"]),
        "pair_relation_loss": float(parts["pair_relation_loss"]),
        "pair_validity_loss": float(parts["pair_validity_loss"]),
        "pair_ranking_loss": float(parts["pair_ranking_loss"]),
        "implicit_presence_loss": float(parts["implicit_presence_loss"]),
        "trainable_parameters_with_grad": sum(
            int(parameter.grad is not None) for parameter in adapter.parameters() if parameter.requires_grad
        ),
        "metrics": "not_computed",
    }


class _FixedSmokeGuard:
    def recommend(self, config: GridTrainConfig) -> dict[str, int]:
        return {"batch_size": 1, "gradient_accumulation": 1, "max_length": config.max_length}

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(mode="fixed_smoke")

    @staticmethod
    def release_cache() -> None:
        return None


def dummy_train_smoke(
    *, max_length: int = 12, hidden_size: int = 8, relation_rank: int = 4, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run one synthetic fold through optimizer, decode, and checkpoint code."""
    rows = [
        ReviewExample(1, "很好用", label_spans=(LabelSpan(Quadruple("_", "很好", "整体", "正面"), opinion_start=0, opinion_end=2),)),
        ReviewExample(2, "质量不错", label_spans=(LabelSpan(Quadruple("质量", "不错", "质量", "正面"), aspect_start=0, aspect_end=2, opinion_start=2, opinion_end=4),)),
        ReviewExample(3, "一般般", label_spans=(LabelSpan(Quadruple("_", "一般", "整体", "中性"), opinion_start=0, opinion_end=2),)),
    ]
    config = GridTrainConfig(max_length=max_length, batch_size=1, gradient_accumulation=1, epochs=1, relation_rank=relation_rank, device="cpu")
    result = train_grid_fold(
        rows[:2], rows[2:], config, guard=_FixedSmokeGuard(), output_dir=output_dir,
        tokenizer=DummyTokenizer(), encoder=DummyEncoder(hidden_size=hidden_size),
    )
    return {
        "status": "dummy_train_passed",
        "history_length": len(result["history"]),
        "valid_ids": sorted(result["candidates"]),
        "checkpoint": result["checkpoint"],
        "config_unchanged": config.max_length == max_length and config.gradient_accumulation == 1,
        "metrics": "not_computed",
    }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "MODEL_ALIASES",
    "GridTrainConfig",
    "ResourcePressureRestart",
    "DummyTokenizer",
    "DummyEncoder",
    "CompactGridEncoderAdapter",
    "resolve_model_name",
    "load_cached_tokenizer",
    "prepare_features",
    "collate_features",
    "dynamic_runtime",
    "grid_training_step",
    "predict_grid_candidates",
    "train_grid_fold",
    "fixed_screen_plan",
    "dummy_smoke",
    "dummy_train_smoke",
    "write_json",
]
