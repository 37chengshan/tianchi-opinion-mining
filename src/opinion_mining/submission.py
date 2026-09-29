from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .data import Quadruple, ReviewExample

CATEGORIES = frozenset({"包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"})
POLARITIES = frozenset({"正面", "中性", "负面"})
EMPTY = Quadruple("_", "_", "_", "_")


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class SubmissionReport:
    ids: int
    rows: int
    empty_ids: int
    predicted_quadruples: int


def _validate_quad(quad: Quadruple, text: str) -> None:
    if quad == EMPTY:
        return
    if quad.category not in CATEGORIES:
        raise SubmissionError(f"unknown category: {quad.category}")
    if quad.polarity not in POLARITIES:
        raise SubmissionError(f"unknown polarity: {quad.polarity}")
    if quad.aspect != "_" and quad.aspect not in text:
        raise SubmissionError(f"aspect not in review: {quad.aspect!r}")
    if quad.opinion != "_" and quad.opinion not in text:
        raise SubmissionError(f"opinion not in review: {quad.opinion!r}")


def write_submission(
    path: str | Path,
    reviews: Iterable[ReviewExample],
    predictions: Mapping[int, Iterable[Quadruple]],
) -> SubmissionReport:
    review_map = {r.id: r for r in reviews}
    unknown = set(predictions) - set(review_map)
    if unknown:
        raise SubmissionError(f"predictions contain unknown ids: {sorted(unknown)[:5]}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    empty_ids = 0
    predicted = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for rid in sorted(review_map):
            text = review_map[rid].text
            quads = sorted(set(predictions.get(rid, [])))
            if not quads:
                quads = [EMPTY]
                empty_ids += 1
            for quad in quads:
                _validate_quad(quad, text)
                writer.writerow([rid, quad.aspect, quad.opinion, quad.category, quad.polarity])
                rows += 1
                if quad != EMPTY:
                    predicted += 1
    report = validate_submission(output, review_map.values())
    if report.rows != rows or report.empty_ids != empty_ids or report.predicted_quadruples != predicted:
        raise SubmissionError("round-trip validation mismatch")
    return report


def validate_submission(path: str | Path, reviews: Iterable[ReviewExample]) -> SubmissionReport:
    file_path = Path(path)
    raw = file_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SubmissionError("UTF-8 BOM is forbidden")
    review_map = {r.id: r for r in reviews}
    expected_ids = sorted(review_map)
    seen_rows: dict[int, list[Quadruple]] = {}
    ordered_ids: list[int] = []
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise SubmissionError(f"line {line_no}: expected 5 fields, got {len(row)}")
            if line_no == 1 and row[0].strip().lower() == "id":
                raise SubmissionError("header is forbidden")
            try:
                rid = int(row[0])
            except ValueError as exc:
                raise SubmissionError(f"line {line_no}: invalid id {row[0]!r}") from exc
            if rid not in review_map:
                raise SubmissionError(f"line {line_no}: unknown id {rid}")
            quad = Quadruple(row[1], row[2], row[3], row[4])
            _validate_quad(quad, review_map[rid].text)
            seen_rows.setdefault(rid, []).append(quad)
            ordered_ids.append(rid)
    if ordered_ids != sorted(ordered_ids):
        raise SubmissionError("ids must be sorted ascending")
    if sorted(seen_rows) != expected_ids:
        missing = sorted(set(expected_ids) - set(seen_rows))
        extra = sorted(set(seen_rows) - set(expected_ids))
        raise SubmissionError(f"id coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    for rid, items in seen_rows.items():
        if len(items) != len(set(items)):
            raise SubmissionError(f"id {rid}: duplicate quadruple rows")
        if EMPTY in items and len(items) != 1:
            raise SubmissionError(f"id {rid}: empty row cannot coexist with predictions")
    empty_ids = sum(items == [EMPTY] for items in seen_rows.values())
    rows_n = sum(len(values) for values in seen_rows.values())
    predicted = rows_n - empty_ids
    return SubmissionReport(len(seen_rows), rows_n, empty_ids, predicted)
