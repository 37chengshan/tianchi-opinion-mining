from __future__ import annotations

"""Deterministic, serialisable fold construction shared by every model."""

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from sklearn.model_selection import KFold

from .data import ReviewExample


def fixed_splits(
    rows: Sequence[ReviewExample] | Iterable[ReviewExample],
    *,
    n_splits: int,
    seed: int,
    assignments: Sequence[Mapping[str, object]] | None = None,
    assignment_path: str | Path | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return canonical row-index splits, optionally from frozen assignments."""
    if assignments is not None and assignment_path is not None:
        raise ValueError("pass assignments or assignment_path, not both")
    values = list(rows)
    if assignment_path is not None:
        assignments = load_assignment_records(assignment_path, n_splits=n_splits, seed=seed)
    if assignments is not None:
        return splits_from_assignments(values, assignments, n_splits=n_splits, seed=seed)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        ([int(index) for index in train_indices], [int(index) for index in valid_indices])
        for train_indices, valid_indices in splitter.split(values)
    ]


class FoldAssignmentError(ValueError):
    """Raised when a frozen fold assignment is incomplete or inconsistent."""


def validate_assignment_records(
    assignments: Sequence[Mapping[str, object]],
    *,
    row_ids: Iterable[int] | None = None,
    n_splits: int | None = None,
    seed: int | None = None,
) -> list[dict[str, object]]:
    records = [dict(item) for item in assignments]
    expected_splits = int(n_splits if n_splits is not None else len(records))
    if len(records) != expected_splits:
        raise FoldAssignmentError(f"expected {expected_splits} folds, got {len(records)}")
    if [int(item.get("fold", -1)) for item in records] != list(range(1, expected_splits + 1)):
        raise FoldAssignmentError("fold numbers must be consecutive starting at 1")
    expected_ids = {int(value) for value in row_ids} if row_ids is not None else None
    valid_seen: set[int] = set()
    normalized: list[dict[str, object]] = []
    for item in records:
        fold = int(item["fold"])
        train_ids = [int(value) for value in item.get("train_ids", [])]
        valid_ids = [int(value) for value in item.get("valid_ids", [])]
        train_set = set(train_ids)
        valid_set = set(valid_ids)
        if len(train_set) != len(train_ids) or len(valid_set) != len(valid_ids):
            raise FoldAssignmentError(f"fold {fold} contains duplicate ids")
        overlap = train_set & valid_set
        if overlap:
            raise FoldAssignmentError(f"fold {fold} train_ids and valid_ids overlap: {sorted(overlap)[:5]}")
        if seed is not None and int(item.get("seed", seed)) != int(seed):
            raise FoldAssignmentError(f"fold {fold} seed does not match {seed}")
        if int(item.get("n_splits", expected_splits)) != expected_splits:
            raise FoldAssignmentError(f"fold {fold} n_splits does not match {expected_splits}")
        if expected_ids is not None and train_set | valid_set != expected_ids:
            raise FoldAssignmentError(f"fold {fold} does not partition all row ids")
        duplicate_valid = valid_seen & valid_set
        if duplicate_valid:
            raise FoldAssignmentError(f"valid ids appear in multiple folds: {sorted(duplicate_valid)[:5]}")
        valid_seen.update(valid_set)
        normalized.append(
            {
                "fold": fold,
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "seed": int(item.get("seed", seed if seed is not None else 0)),
                "n_splits": expected_splits,
            }
        )
    if expected_ids is not None and valid_seen != expected_ids:
        missing = sorted(expected_ids - valid_seen)
        extra = sorted(valid_seen - expected_ids)
        raise FoldAssignmentError(f"valid ids do not cover rows; missing={missing[:5]} extra={extra[:5]}")
    return normalized


def load_assignment_records(
    path: str | Path,
    *,
    n_splits: int | None = None,
    seed: int | None = None,
) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("folds"), list):
            records = payload["folds"]
        else:
            keys = [str(n_splits)] if n_splits is not None else []
            if not keys:
                numeric_keys = sorted(key for key in payload if str(key).isdigit())
                keys = ["5"] if "5" in payload else numeric_keys[:1]
            if not keys or not isinstance(payload.get(keys[0]), list):
                raise FoldAssignmentError(f"no fold assignment list found in {path}")
            records = payload[keys[0]]
    else:
        raise FoldAssignmentError(f"unsupported fold assignment payload: {path}")
    return validate_assignment_records(records, n_splits=n_splits, seed=seed)


def splits_from_assignments(
    rows: Sequence[ReviewExample] | Iterable[ReviewExample],
    assignments: Sequence[Mapping[str, object]],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    values = list(rows)
    row_index = {row.id: index for index, row in enumerate(values)}
    if len(row_index) != len(values):
        raise FoldAssignmentError("row ids must be unique")
    records = validate_assignment_records(
        assignments,
        row_ids=row_index,
        n_splits=n_splits,
        seed=seed,
    )
    try:
        return [
            ([row_index[int(value)] for value in item["train_ids"]], [row_index[int(value)] for value in item["valid_ids"]])
            for item in records
        ]
    except KeyError as error:
        raise FoldAssignmentError(f"assignment references unknown row id: {error.args[0]}") from error


def assignment_records(rows: Sequence[ReviewExample] | Iterable[ReviewExample], *, n_splits: int, seed: int) -> list[dict[str, object]]:
    values = list(rows)
    return [
        {
            "fold": fold,
            "train_ids": [values[index].id for index in train_indices],
            "valid_ids": [values[index].id for index in valid_indices],
            "seed": seed,
            "n_splits": n_splits,
        }
        for fold, (train_indices, valid_indices) in enumerate(fixed_splits(values, n_splits=n_splits, seed=seed), start=1)
    ]
