from __future__ import annotations

"""Leakage-safe statistical and retrieval candidate generator.

The competition scores complete quadruples.  This module therefore keeps the
quadruple intact throughout fitting and prediction instead of optimising four
independent fields and hoping that they recombine correctly.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Quadruple, ReviewExample


@dataclass(frozen=True)
class Candidate:
    quadruple: Quadruple
    score: float
    sources: tuple[str, ...] = ()

    @property
    def quad(self) -> Quadruple:
        return self.quadruple


def _find_all(text: str, term: str) -> list[int]:
    if not term or term == "_":
        return []
    starts: list[int] = []
    at = 0
    while True:
        pos = text.find(term, at)
        if pos < 0:
            return starts
        starts.append(pos)
        at = pos + max(1, len(term))


class StatisticalOpinionMiner:
    """Phrase purity + nearest-review retrieval miner.

    ``fit`` only consumes the examples passed to it.  ``run_oof`` creates a
    fresh miner per fold, which makes the dictionary and retriever validation
    safe by construction.
    """

    def __init__(self, *, max_opinion_variants: int = 3, retrieval_k: int = 8):
        self.max_opinion_variants = max_opinion_variants
        self.retrieval_k = retrieval_k
        self.examples: list[ReviewExample] = []
        self.opinion_stats: dict[str, Counter[Quadruple]] = {}
        self.aspect_stats: dict[str, Counter[Quadruple]] = {}
        self.pair_stats: dict[tuple[str, str], Counter[Quadruple]] = {}
        self.global_stats: Counter[Quadruple] = Counter()
        self.opinion_terms: tuple[str, ...] = ()
        self.aspect_terms: tuple[str, ...] = ()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def fit(self, examples: Iterable[ReviewExample]) -> "StatisticalOpinionMiner":
        self.examples = list(examples)
        opinion_stats: defaultdict[str, Counter[Quadruple]] = defaultdict(Counter)
        aspect_stats: defaultdict[str, Counter[Quadruple]] = defaultdict(Counter)
        pair_stats: defaultdict[tuple[str, str], Counter[Quadruple]] = defaultdict(Counter)
        global_stats: Counter[Quadruple] = Counter()
        for example in self.examples:
            for quad in example.labels:
                global_stats[quad] += 1
                opinion_stats[quad.opinion][quad] += 1
                aspect_stats[quad.aspect][quad] += 1
                pair_stats[(quad.aspect, quad.opinion)][quad] += 1
        self.opinion_stats = dict(opinion_stats)
        self.aspect_stats = dict(aspect_stats)
        self.pair_stats = dict(pair_stats)
        self.global_stats = global_stats
        self.opinion_terms = tuple(
            sorted((term for term in opinion_stats if term != "_"), key=lambda x: (-len(x), x))
        )
        self.aspect_terms = tuple(
            sorted((term for term in aspect_stats if term != "_"), key=lambda x: (-len(x), x))
        )
        texts = [x.text for x in self.examples]
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None
        return self

    @staticmethod
    def _purity(counter: Counter[Quadruple], item: Quadruple) -> float:
        total = sum(counter.values())
        return counter[item] / total if total else 0.0

    @staticmethod
    def _ranked(counter: Counter[Quadruple], limit: int) -> list[tuple[Quadruple, int]]:
        return counter.most_common(limit)

    def _score(self, quad: Quadruple, count: int, purity: float, source: str) -> float:
        # The score is deliberately monotonic and interpretable.  Thresholds
        # are selected on OOF predictions, so absolute calibration is less
        # important than stable ordering between candidate sources.
        score = 0.34 + 0.42 * purity + 0.075 * min(1.0, math.log1p(count) / math.log(80))
        if source == "exact_pair":
            score += 0.13
        elif source == "retrieval":
            score += 0.04
        elif source == "aspect_memory":
            score += 0.05
        if quad.aspect == "_":
            score += 0.015
        return min(0.999, max(0.001, score))

    def _add(
        self,
        store: dict[Quadruple, tuple[float, set[str]]],
        quad: Quadruple,
        score: float,
        source: str,
        text: str,
    ) -> None:
        if quad.aspect != "_" and quad.aspect not in text:
            return
        if quad.opinion != "_" and quad.opinion not in text:
            return
        current = store.get(quad)
        if current is None:
            store[quad] = (score, {source})
        else:
            store[quad] = (max(current[0], score), current[1] | {source})

    def predict_candidates(self, text: str) -> list[Candidate]:
        if not self.examples:
            return []
        store: dict[Quadruple, tuple[float, set[str]]] = {}

        # Opinion anchored candidates cover most implicit-aspect labels.  Long
        # phrases are considered first and the resulting quadruples are merged,
        # so common short phrases do not create duplicate rows.
        for opinion in self.opinion_terms:
            if opinion not in text:
                continue
            stats = self.opinion_stats[opinion]
            for quad, count in self._ranked(stats, self.max_opinion_variants):
                purity = self._purity(stats, quad)
                self._add(store, quad, self._score(quad, count, purity, "opinion_memory"), "opinion_memory", text)

        # Explicit aspect + opinion pairs are more reliable than an opinion's
        # marginal mapping, so they get a separate high-confidence source.
        present_aspects = [a for a in self.aspect_terms if a in text]
        for aspect in present_aspects:
            for opinion in self.opinion_terms:
                if opinion not in text:
                    continue
                stats = self.pair_stats.get((aspect, opinion))
                if not stats:
                    continue
                for quad, count in self._ranked(stats, self.max_opinion_variants):
                    purity = self._purity(stats, quad)
                    self._add(store, quad, self._score(quad, count, purity, "exact_pair"), "exact_pair", text)

            # Some labelled rows have an explicit aspect and an implicit
            # opinion.  Preserve that pattern when it was observed in train.
            stats = self.aspect_stats.get(aspect, Counter())
            for quad, count in self._ranked(stats, self.max_opinion_variants):
                if quad.opinion == "_":
                    purity = self._purity(stats, quad)
                    self._add(store, quad, self._score(quad, count, purity, "aspect_memory"), "aspect_memory", text)

        # Nearest labelled reviews transfer a complete quadruple only when its
        # source terms remain exact substrings of the new review.  This helps
        # with paraphrases and sparse long-tail mappings without hallucinating
        # terms outside the source text.
        if self.vectorizer is not None and self.matrix is not None:
            query = self.vectorizer.transform([text])
            similarities = cosine_similarity(query, self.matrix).ravel()
            k = min(self.retrieval_k, len(similarities))
            if k:
                nearest = np.argpartition(-similarities, k - 1)[:k]
                for index in nearest:
                    sim = float(similarities[index])
                    if sim <= 0.10:
                        continue
                    example = self.examples[int(index)]
                    for quad in example.labels:
                        self._add(store, quad, 0.26 + 0.42 * sim, "retrieval", text)

        # Return highest confidence first.  A small deterministic cap protects
        # precision when a very generic opinion appears many times.
        result = [Candidate(q, score, tuple(sorted(sources))) for q, (score, sources) in store.items()]
        result.sort(key=lambda item: (-item.score, item.quadruple))
        return result

    def predict(self, text: str, *, threshold: float = 0.5, implicit_threshold: float | None = None) -> set[Quadruple]:
        implicit_threshold = threshold if implicit_threshold is None else implicit_threshold
        output: set[Quadruple] = set()
        for candidate in self.predict_candidates(text):
            limit = implicit_threshold if candidate.quadruple.aspect == "_" else threshold
            if candidate.score >= limit:
                output.add(candidate.quadruple)
        return output
