from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F


class OpinionetHead(nn.Module):
    """One-stage anchor head with pointer matrices and joint relation classes."""

    def __init__(self, hidden_size: int, relation_classes: int = 39, pointer_hidden: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.pointer_hidden = int(pointer_hidden)
        self.objectiveness = nn.Linear(hidden_size, 1)
        self.relation = nn.Linear(hidden_size, relation_classes)
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

    @staticmethod
    def _pointer(left: nn.Linear, right: nn.Linear, output: nn.Linear, hidden: Tensor) -> Tensor:
        pair = left(hidden).unsqueeze(2) + right(hidden).unsqueeze(1)
        return output(F.gelu(pair)).squeeze(-1)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        return {
            "objectiveness": self.objectiveness(hidden).squeeze(-1),
            "aspect_start": self._pointer(self.aspect_start_left, self.aspect_start_right, self.aspect_start_out, hidden),
            "aspect_end": self._pointer(self.aspect_end_left, self.aspect_end_right, self.aspect_end_out, hidden),
            "opinion_start": self._pointer(self.opinion_start_left, self.opinion_start_right, self.opinion_start_out, hidden),
            "opinion_end": self._pointer(self.opinion_end_left, self.opinion_end_right, self.opinion_end_out, hidden),
            "relation": self.relation(hidden),
        }
