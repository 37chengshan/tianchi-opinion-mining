from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .data import Quadruple
from .reranker import CandidateRow


@dataclass
class LabelPriors:
    """Fold-safe label statistics used as extra candidate features.

    The counts come exclusively from the folds the caller excludes, so a row's
    own fold never contributes to its features.
    """

    quadruple_counts: Counter
    triple_counts: Counter
    opinion_category_counts: Counter
    opinion_counts: Counter
    category_counts: Counter
    aspect_counts: Counter
    total: int

    @classmethod
    def from_labels(cls, labels_by_id: Mapping[int, Iterable[Quadruple]]) -> "LabelPriors":
        quadruple_counts: Counter = Counter()
        triple_counts: Counter = Counter()
        opinion_category_counts: Counter = Counter()
        opinion_counts: Counter = Counter()
        category_counts: Counter = Counter()
        aspect_counts: Counter = Counter()
        total = 0
        for values in labels_by_id.values():
            for quad in values:
                quadruple_counts[quad] += 1
                triple_counts[(quad.aspect, quad.opinion, quad.category)] += 1
                opinion_category_counts[(quad.opinion, quad.category)] += 1
                opinion_counts[quad.opinion] += 1
                category_counts[quad.category] += 1
                aspect_counts[quad.aspect] += 1
                total += 1
        return cls(quadruple_counts, triple_counts, opinion_category_counts, opinion_counts, category_counts, aspect_counts, total)

    def features(self, quad: Quadruple) -> list[float]:
        quadruple = self.quadruple_counts.get(quad, 0)
        triple = self.triple_counts.get((quad.aspect, quad.opinion, quad.category), 0)
        opinion_category = self.opinion_category_counts.get((quad.opinion, quad.category), 0)
        opinion = self.opinion_counts.get(quad.opinion, 0)
        category = self.category_counts.get(quad.category, 0)
        aspect = self.aspect_counts.get(quad.aspect, 0)
        return [
            math.log1p(quadruple),
            math.log1p(triple),
            math.log1p(opinion_category),
            math.log1p(opinion),
            math.log1p(category),
            math.log1p(aspect),
            1.0 if quadruple > 0 else 0.0,
            (opinion_category / opinion) if opinion else 0.0,
            (triple / opinion) if opinion else 0.0,
            (quadruple / self.total) if self.total else 0.0,
        ]


PRIOR_FEATURE_NAMES = [
    "prior_quadruple_log",
    "prior_triple_log",
    "prior_opinion_category_log",
    "prior_opinion_log",
    "prior_category_log",
    "prior_aspect_log",
    "prior_quadruple_seen",
    "prior_opinion_category_purity",
    "prior_triple_purity",
    "prior_quadruple_share",
]


def append_prior_features(rows: Sequence[CandidateRow], priors: LabelPriors) -> "list[list[float]]":
    return [list(row.features) + priors.features(row.quadruple) for row in rows]


def prior_feature_names() -> list[str]:
    return list(PRIOR_FEATURE_NAMES)
