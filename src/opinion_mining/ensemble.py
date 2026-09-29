from __future__ import annotations

from typing import Iterable, Mapping

from .baseline import Candidate
from .pipeline import quadruple_state


def soft_pair_support_candidates(
    base_candidates: Mapping[int, Iterable[Candidate]],
    verifier_candidates: Mapping[int, Iterable[Candidate]],
    verifier_state_thresholds: Mapping[str, float],
    *,
    support_floor: float,
) -> dict[int, list[Candidate]]:
    """Re-score base quadruples with continuous same-pair verifier evidence.

    A verifier score at or above its state threshold leaves the base score
    unchanged. Missing verifier evidence retains ``support_floor`` of the base
    score; partial evidence interpolates linearly. Category/polarity are not
    required to agree because this verifier is intentionally a pair-quality
    signal rather than a second class predictor.
    """

    floor = float(support_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("support_floor must be between 0 and 1")

    support_by_row: dict[int, dict[tuple[str, str], float]] = {}
    for rid, items in verifier_candidates.items():
        pairs: dict[tuple[str, str], float] = {}
        for item in items:
            key = (item.quadruple.aspect, item.quadruple.opinion)
            pairs[key] = max(pairs.get(key, 0.0), float(item.score))
        support_by_row[int(rid)] = pairs

    output: dict[int, list[Candidate]] = {}
    for rid, items in base_candidates.items():
        row_support = support_by_row.get(int(rid), {})
        rescored: list[Candidate] = []
        for item in items:
            state = quadruple_state(item.quadruple)
            gate = float(verifier_state_thresholds[state])
            support = row_support.get((item.quadruple.aspect, item.quadruple.opinion), 0.0)
            ratio = 1.0 if gate <= 0.0 else min(1.0, max(0.0, support / gate))
            factor = floor + (1.0 - floor) * ratio
            rescored.append(
                Candidate(
                    item.quadruple,
                    float(item.score) * factor,
                    tuple(sorted(set(item.sources) | {"soft_pair_support"})),
                )
            )
        output[int(rid)] = sorted(rescored, key=lambda item: (-item.score, item.quadruple))
    return output


def filter_predictions_by_pair_support(
    base_predictions: Mapping[int, Iterable],
    verifier_candidates: Mapping[int, Iterable[Candidate]],
    verifier_state_thresholds: Mapping[str, float],
    *,
    gate_scale: float,
) -> dict[int, set]:
    """Keep base quadruples whose same-pair verifier evidence clears a scaled state gate."""

    scale = float(gate_scale)
    if scale < 0.0:
        raise ValueError("gate_scale must be non-negative")
    support_by_row: dict[int, dict[tuple[str, str], float]] = {}
    for rid, items in verifier_candidates.items():
        row: dict[tuple[str, str], float] = {}
        for item in items:
            key = (item.quadruple.aspect, item.quadruple.opinion)
            row[key] = max(row.get(key, 0.0), float(item.score))
        support_by_row[int(rid)] = row

    output: dict[int, set] = {}
    for rid, quads in base_predictions.items():
        row_support = support_by_row.get(int(rid), {})
        kept = set()
        for quad in quads:
            state = quadruple_state(quad)
            required = float(verifier_state_thresholds[state]) * scale
            if row_support.get((quad.aspect, quad.opinion), 0.0) >= required:
                kept.add(quad)
        output[int(rid)] = kept
    return output
