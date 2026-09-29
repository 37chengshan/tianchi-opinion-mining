from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .baseline import Candidate
from .data import Quadruple, ReviewExample


STATE_ORDER = ("explicit", "implicit-A", "implicit-O", "dual-implicit")
CATEGORY_ORDER = ("包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他")


def quadruple_state(quad: Quadruple) -> str:
    if quad.aspect == "_" and quad.opinion == "_":
        return "dual-implicit"
    if quad.aspect == "_":
        return "implicit-A"
    if quad.opinion == "_":
        return "implicit-O"
    return "explicit"


@dataclass(frozen=True)
class CandidateRow:
    review_id: int
    quadruple: Quadruple
    features: tuple[float, ...]


def feature_names(source_count: int) -> list[str]:
    names = [
        "score",
        "score_rank_normalized",
        "score_gap_to_top",
        "candidate_count_log",
        "aspect_len",
        "opinion_len",
        "aspect_implicit",
        "opinion_implicit",
        "opinion_position_normalized",
        "comment_count",
        "period_count",
        "text_length_log",
    ]
    names += [f"state_{state}" for state in STATE_ORDER]
    names += [f"category_{category}" for category in CATEGORY_ORDER]
    names += [f"source_{index}_score" for index in range(source_count)]
    names += [f"source_{index}_present" for index in range(source_count)]
    return names


def build_rows(
    sources: Sequence[Mapping[int, Iterable[Candidate]]],
    reviews: Mapping[int, ReviewExample],
) -> list[CandidateRow]:
    """Extract one feature row per unique (review, quadruple) across sources."""
    merged: dict[tuple[int, Quadruple], list[float]] = {}
    for source_index, mapping in enumerate(sources):
        for review_id, items in mapping.items():
            for item in items:
                key = (int(review_id), item.quadruple)
                vector = merged.get(key)
                if vector is None:
                    vector = [0.0] * (2 * len(sources))
                    merged[key] = vector
                vector[source_index] = max(vector[source_index], float(item.score))
                vector[len(sources) + source_index] = 1.0

    rows: list[CandidateRow] = []
    by_review: dict[int, list[tuple[Quadruple, list[float]]]] = {}
    for (review_id, quad), vector in merged.items():
        by_review.setdefault(review_id, []).append((quad, vector))

    for review_id, entries in by_review.items():
        row = reviews.get(review_id)
        if row is None:
            continue
        text = row.text
        scores = [max(entry[1][index] for entry in entries) for index in range(len(sources))]
        ranked = sorted(entries, key=lambda entry: (-max(entry[1][: len(sources)]), entry[0]))
        candidate_count = len(entries)
        top_score = max(entry[1][0] for entry in entries) if entries else 0.0
        for rank, (quad, vector) in enumerate(ranked):
            best = max(vector[: len(sources)])
            state = quadruple_state(quad)
            opinion_position = text.find(quad.opinion) if quad.opinion != "_" else len(text)
            features = [
                best,
                rank / max(1, candidate_count - 1),
                top_score - best,
                float(np.log1p(candidate_count)),
                float(len(quad.aspect)) if quad.aspect != "_" else 0.0,
                float(len(quad.opinion)) if quad.opinion != "_" else 0.0,
                1.0 if quad.aspect == "_" else 0.0,
                1.0 if quad.opinion == "_" else 0.0,
                (opinion_position if opinion_position >= 0 else len(text)) / max(1, len(text)),
                float(text.count("，") + text.count("。") + text.count(",")),
                float(text.count("！") + text.count("!")),
                float(np.log1p(len(text))),
            ]
            features += [1.0 if state == value else 0.0 for value in STATE_ORDER]
            features += [1.0 if quad.category == value else 0.0 for value in CATEGORY_ORDER]
            features += list(vector)
            rows.append(CandidateRow(review_id, quad, tuple(float(value) for value in features)))
        del scores
    return rows


def labels_for(rows: Iterable[CandidateRow], gold: Mapping[int, set[Quadruple]]) -> np.ndarray:
    return np.asarray([1 if row.quadruple in gold.get(row.review_id, set()) else 0 for row in rows], dtype=np.int32)


def matrix(rows: Iterable[CandidateRow]) -> np.ndarray:
    values = [row.features for row in rows]
    return np.asarray(values, dtype=np.float32) if values else np.zeros((0, 0), dtype=np.float32)
