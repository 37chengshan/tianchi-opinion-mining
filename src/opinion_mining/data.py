from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


IMPLICIT_A_TOKEN = "[IMPLICIT_A]"
IMPLICIT_O_TOKEN = "[IMPLICIT_O]"


def encode_implicit_terms(aspect: str, opinion: str) -> tuple[str, str]:
    """Represent implicit terms with stable pseudo tokens for V2 round trips."""
    return (IMPLICIT_A_TOKEN if aspect == "_" else aspect, IMPLICIT_O_TOKEN if opinion == "_" else opinion)


def decode_implicit_terms(aspect: str, opinion: str) -> tuple[str, str]:
    """Decode V2 pseudo tokens back to the competition's ``_`` sentinel."""
    return ("_" if aspect == IMPLICIT_A_TOKEN else aspect, "_" if opinion == IMPLICIT_O_TOKEN else opinion)


@dataclass(frozen=True, order=True)
class Quadruple:
    aspect: str
    opinion: str
    category: str
    polarity: str


@dataclass(frozen=True)
class LabelSpan:
    """A labelled quadruple together with the official character offsets.

    The competition labels include offsets even when a term string occurs
    several times in one review.  Keeping them beside the string quadruple
    prevents a later tokenizer step from silently choosing the first match.
    Ends follow the CSV convention and are exclusive.
    """

    quadruple: Quadruple
    aspect_start: int | None = None
    aspect_end: int | None = None
    opinion_start: int | None = None
    opinion_end: int | None = None


@dataclass(frozen=True)
class ReviewExample:
    id: int
    text: str
    labels: frozenset[Quadruple] = field(default_factory=frozenset)
    label_spans: tuple[LabelSpan, ...] = field(default_factory=tuple)


def _clean(value: str | None) -> str:
    if value is None:
        return "_"
    value = value.strip()
    return value if value else "_"


def _offset(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_test_reviews(path: str | Path) -> list[ReviewExample]:
    rows: list[ReviewExample] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(ReviewExample(id=int(row["id"]), text=row["Reviews"], labels=frozenset()))
    return rows


def load_train_data(reviews_path: str | Path, labels_path: str | Path) -> list[ReviewExample]:
    reviews = load_test_reviews(reviews_path)
    grouped: dict[int, set[Quadruple]] = {r.id: set() for r in reviews}
    grouped_spans: dict[int, list[LabelSpan]] = {r.id: [] for r in reviews}
    with Path(labels_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["id"])
            quadruple = Quadruple(
                _clean(row.get("AspectTerms")),
                _clean(row.get("OpinionTerms")),
                _clean(row.get("Categories")),
                _clean(row.get("Polarities")),
            )
            grouped.setdefault(rid, set()).add(quadruple)
            grouped_spans.setdefault(rid, []).append(
                LabelSpan(
                    quadruple=quadruple,
                    aspect_start=_offset(row.get("A_start")),
                    aspect_end=_offset(row.get("A_end")),
                    opinion_start=_offset(row.get("O_start")),
                    opinion_end=_offset(row.get("O_end")),
                )
            )
    return [
        ReviewExample(
            r.id,
            r.text,
            frozenset(grouped.get(r.id, set())),
            tuple(grouped_spans.get(r.id, [])),
        )
        for r in reviews
    ]
