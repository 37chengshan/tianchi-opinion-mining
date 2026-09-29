from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class EMAModel:
    """Exponential moving average of the trainable parameters."""

    def __init__(self, parameters: Iterable[nn.Parameter], decay: float = 0.999):
        self.decay = float(decay)
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.shadow = [parameter.detach().clone() for parameter in self.parameters]
        self.backup: list[Tensor] | None = None

    @torch.no_grad()
    def update(self, decay: float | None = None) -> None:
        rate = self.decay if decay is None else float(decay)
        for shadow, parameter in zip(self.shadow, self.parameters):
            shadow.mul_(rate).add_(parameter.detach(), alpha=1.0 - rate)

    @torch.no_grad()
    def apply(self) -> None:
        """Swap the model parameters for the averaged ones."""
        if self.backup is not None:
            raise RuntimeError("EMA parameters already applied")
        self.backup = [parameter.detach().clone() for parameter in self.parameters]
        for shadow, parameter in zip(self.shadow, self.parameters):
            parameter.copy_(shadow)

    @torch.no_grad()
    def restore(self) -> None:
        if self.backup is None:
            raise RuntimeError("EMA parameters were not applied")
        for saved, parameter in zip(self.backup, self.parameters):
            parameter.copy_(saved)
        self.backup = None


class FGM:
    """Fast gradient method: perturb embeddings with the sign of their gradient."""

    def __init__(self, model: nn.Module, epsilon: float = 0.5, name: str = "word_embeddings"):
        self.model = model
        self.epsilon = float(epsilon)
        self.name = name
        self._backup: Tensor | None = None
        self._embedding: nn.Parameter | None = None

    def _find_embedding(self) -> nn.Parameter | None:
        encoder = getattr(self.model, "encoder", self.model)
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, self.name):
            candidate = getattr(embeddings, self.name)
            if isinstance(candidate, nn.Parameter):
                return candidate
        for module in self.model.modules():
            if isinstance(module, nn.Embedding) and module.weight.requires_grad:
                return module.weight
        return None

    def attack(self) -> bool:
        embedding = self._embedding or self._find_embedding()
        self._embedding = embedding
        if embedding is None or embedding.grad is None:
            return False
        norm = torch.norm(embedding.grad)
        if not torch.isfinite(norm) or float(norm) == 0.0:
            return False
        self._backup = embedding.detach().clone()
        perturbation = self.epsilon * embedding.grad / norm
        embedding.detach().add_(perturbation)
        return True

    def restore(self) -> bool:
        if self._backup is None or self._embedding is None:
            return False
        self._embedding.detach().copy_(self._backup)
        self._backup = None
        return True
