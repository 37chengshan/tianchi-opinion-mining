from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .data import Quadruple


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    f1: float
    correct: int
    predicted: int
    gold: int


def _sets(items: Mapping[int, Iterable[Quadruple]]) -> dict[int, set[Quadruple]]:
    return {int(k): set(v) for k, v in items.items()}


def strict_f1(
    gold: Mapping[int, Iterable[Quadruple]],
    pred: Mapping[int, Iterable[Quadruple]],
) -> Score:
    gold_sets = _sets(gold)
    pred_sets = _sets(pred)
    gold_n = sum(len(v) for v in gold_sets.values())
    pred_n = sum(len(v) for v in pred_sets.values())
    correct = sum(len(gold_sets.get(rid, set()) & values) for rid, values in pred_sets.items())
    precision = correct / pred_n if pred_n else 0.0
    recall = correct / gold_n if gold_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Score(precision, recall, f1, correct, pred_n, gold_n)
