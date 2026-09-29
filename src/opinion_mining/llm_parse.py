from __future__ import annotations

import json
import re

from .data import Quadruple
from .submission import CATEGORIES, POLARITIES


class LLMParseError(ValueError):
    """Raised when the model output cannot be parsed into legal quadruples."""


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise LLMParseError("no JSON object found")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise LLMParseError("unbalanced JSON object")


def _position(review_text: str, term: str) -> int:
    if term == "_":
        return 10**9
    index = review_text.find(term)
    return index if index >= 0 else 10**9 - 1


def parse_llm_output(text: str, review_text: str) -> list[Quadruple]:
    """Parse one model answer into deduplicated, ordered legal quadruples.

    Illegal categories/polarities and terms not present in the review are
    dropped instead of raising, because a single bad item must not discard an
    otherwise usable answer.
    """
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, dict):
        raise LLMParseError("top-level JSON must be an object")
    items = payload.get("quadruples")
    if items is None:
        raise LLMParseError("missing 'quadruples' key")
    if not isinstance(items, list):
        raise LLMParseError("'quadruples' must be a list")

    cleaned: set[Quadruple] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect") or "_").strip() or "_"
        raw_opinion = str(item.get("opinion") or "").strip()
        opinion = raw_opinion if raw_opinion else "_"
        category = str(item.get("category") or "").strip()
        polarity = str(item.get("polarity") or "").strip()
        if category not in CATEGORIES or polarity not in POLARITIES:
            continue
        # Both sides implicit has no gold support and is pure false-positive risk.
        if aspect == "_" and opinion == "_":
            continue
        if aspect != "_" and aspect not in review_text:
            continue
        if opinion != "_" and opinion not in review_text:
            continue
        cleaned.add(Quadruple(aspect, opinion, category, polarity))
    return sorted(
        cleaned,
        key=lambda quad: (_position(review_text, quad.opinion), _position(review_text, quad.aspect), quad.category, quad.polarity),
    )
