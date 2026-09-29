from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import ClassVar

from .neural import NeuralConfig


@dataclass(frozen=True)
class AnchorV1Config:
    """Only the isolated fixes that passed their own RBT3 gates."""

    model_name: str = "hfl/rbt3"
    max_length: int = 96
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
    use_ema: bool = False
    ema_decay: float = 0.999
    fgm_epsilon: float = 0.0
    enabled_fixes: tuple[str, ...] = (
        "official_offsets",
        "implicit_opinion_sentinel",
        "multi_relation",
    )

    SUPPORTED_FIXES: ClassVar[tuple[str, ...]] = (
        "official_offsets",
        "implicit_opinion_sentinel",
        "multi_relation",
    )

    def __post_init__(self) -> None:
        if len(set(self.enabled_fixes)) != len(self.enabled_fixes):
            raise ValueError("enabled anchor fixes must be unique")
        unsupported = sorted(set(self.enabled_fixes) - set(self.SUPPORTED_FIXES))
        if unsupported:
            raise ValueError(f"unsupported anchor fix: {unsupported[0]}")
        canonical = tuple(fix for fix in self.SUPPORTED_FIXES if fix in self.enabled_fixes)
        object.__setattr__(self, "enabled_fixes", canonical)

    @property
    def use_official_offsets(self) -> bool:
        return "official_offsets" in self.enabled_fixes

    @property
    def use_implicit_opinion_sentinel(self) -> bool:
        return "implicit_opinion_sentinel" in self.enabled_fixes

    @property
    def preserve_multi_relation(self) -> bool:
        return "multi_relation" in self.enabled_fixes

    def to_neural_config(self) -> NeuralConfig:
        values = asdict(self)
        values.pop("enabled_fixes")
        return NeuralConfig(
            **values,
            use_official_offsets=self.use_official_offsets,
            use_implicit_opinion_sentinel=self.use_implicit_opinion_sentinel,
            preserve_multi_relation=self.preserve_multi_relation,
        )

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["enabled_fixes"] = list(self.enabled_fixes)
        values["use_official_offsets"] = self.use_official_offsets
        values["use_implicit_opinion_sentinel"] = self.use_implicit_opinion_sentinel
        values["preserve_multi_relation"] = self.preserve_multi_relation
        return values

    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
