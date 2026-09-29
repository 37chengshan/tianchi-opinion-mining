from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Mapping

from .baseline import Candidate
from .data import Quadruple
from .metrics import strict_f1


def candidate_to_dict(item: Candidate) -> dict[str, object]:
    return {
        "aspect": item.quadruple.aspect,
        "opinion": item.quadruple.opinion,
        "category": item.quadruple.category,
        "polarity": item.quadruple.polarity,
        "score": float(item.score),
        "sources": list(item.sources),
    }


def candidate_from_dict(item: Mapping[str, object]) -> Candidate:
    return Candidate(
        Quadruple(str(item["aspect"]), str(item["opinion"]), str(item["category"]), str(item["polarity"])),
        float(item["score"]),
        tuple(str(x) for x in item.get("sources", [])),
    )


def save_candidates(path: str | Path, candidates: Mapping[int, Iterable[Candidate]]) -> None:
    payload = {str(rid): [candidate_to_dict(item) for item in items] for rid, items in candidates.items()}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidates(path: str | Path) -> dict[int, list[Candidate]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(rid): [candidate_from_dict(item) for item in items] for rid, items in payload.items()}


def merge_candidate_maps(
    maps: Iterable[Mapping[int, Iterable[Candidate]]],
    *,
    weights: Iterable[float] | None = None,
    average_present: bool = True,
    bonus: float = 0.0,
) -> dict[int, list[Candidate]]:
    maps_list = list(maps)
    weights_list = list(weights or [1.0] * len(maps_list))
    if len(weights_list) != len(maps_list):
        raise ValueError("weights and maps must have equal length")
    store: dict[int, dict[Quadruple, tuple[float, float, set[str]]]] = {}
    for map_index, (mapping, weight) in enumerate(zip(maps_list, weights_list)):
        for rid, items in mapping.items():
            row = store.setdefault(int(rid), {})
            for item in items:
                existing = row.get(item.quadruple)
                if existing is None:
                    row[item.quadruple] = (float(item.score) * weight, weight, set(item.sources) | {f"model_{map_index}"})
                else:
                    row[item.quadruple] = (existing[0] + float(item.score) * weight, existing[1] + weight, existing[2] | set(item.sources) | {f"model_{map_index}"})
    output: dict[int, list[Candidate]] = {}
    for rid, row in store.items():
        values = []
        for quad, (total, present_weight, sources) in row.items():
            denominator = present_weight if average_present else sum(weights_list)
            score = total / denominator if denominator else 0.0
            if len(sources) > 1:
                score = min(0.999, score + bonus)
            values.append(Candidate(quad, score, tuple(sorted(sources))))
        output[rid] = sorted(values, key=lambda item: (-item.score, item.quadruple))
    return output


def error_analysis(
    gold: Mapping[int, Iterable[Quadruple]],
    predictions: Mapping[int, Iterable[Quadruple]],
) -> dict[str, object]:
    gold_sets = {int(k): set(v) for k, v in gold.items()}
    pred_sets = {int(k): set(v) for k, v in predictions.items()}
    true_positive = sum(len(gold_sets.get(rid, set()) & values) for rid, values in pred_sets.items())
    false_positive = sum(len(values - gold_sets.get(rid, set())) for rid, values in pred_sets.items())
    false_negative = sum(len(values - pred_sets.get(rid, set())) for rid, values in gold_sets.items())
    categories = Counter()
    polarities = Counter()
    implicit = Counter()
    for rid, values in gold_sets.items():
        for quad in values - pred_sets.get(rid, set()):
            categories[quad.category] += 1
            polarities[quad.polarity] += 1
            implicit["implicit" if quad.aspect == "_" else "explicit"] += 1
    partial = Counter()
    for rid, values in pred_sets.items():
        for pred in values - gold_sets.get(rid, set()):
            gold_row = gold_sets.get(rid, set())
            if any(pred.aspect == item.aspect and pred.opinion == item.opinion for item in gold_row):
                partial["wrong_category_or_polarity"] += 1
            elif any(pred.opinion == item.opinion for item in gold_row):
                partial["wrong_aspect_or_pair"] += 1
            else:
                partial["span_or_unseen_opinion"] += 1
    score = strict_f1(gold_sets, pred_sets)
    return {
        "score": {"precision": score.precision, "recall": score.recall, "f1": score.f1, "correct": score.correct, "predicted": score.predicted, "gold": score.gold},
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "missed_by_category": dict(categories.most_common()),
        "missed_by_polarity": dict(polarities.most_common()),
        "missed_by_aspect_type": dict(implicit),
        "false_positive_type": dict(partial),
    }
