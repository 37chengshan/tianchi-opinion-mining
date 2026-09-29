from __future__ import annotations

"""OOF evaluation, threshold calibration, and fold-safe candidate plumbing."""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from .baseline import Candidate, StatisticalOpinionMiner
from .data import Quadruple, ReviewExample
from .folds import fixed_splits
from .metrics import Score, strict_f1


@dataclass
class OOFResult:
    candidates: dict[int, list[Candidate]]
    gold: dict[int, set[Quadruple]]
    folds: list[dict[str, object]] = field(default_factory=list)


def quadruple_state(quadruple: Quadruple) -> str:
    """Return the explicit/implicit state used by calibration and decoding."""
    implicit_a = quadruple.aspect == "_"
    implicit_o = quadruple.opinion == "_"
    if implicit_a and implicit_o:
        return "dual-implicit"
    if implicit_a:
        return "implicit-A"
    if implicit_o:
        return "implicit-O"
    return "explicit"


def threshold_predictions(
    candidates: Mapping[int, Iterable[Candidate]],
    threshold: float,
    implicit_threshold: float | None = None,
    *,
    state_thresholds: Mapping[str, float] | None = None,
) -> dict[int, set[Quadruple]]:
    implicit_threshold = threshold if implicit_threshold is None else implicit_threshold
    defaults = {
        "explicit": float(threshold),
        "implicit-A": float(implicit_threshold),
        "implicit-O": float(implicit_threshold),
        "dual-implicit": float(implicit_threshold),
    }
    if state_thresholds:
        defaults.update({str(key): float(value) for key, value in state_thresholds.items()})
    result: dict[int, set[Quadruple]] = {}
    for rid, items in candidates.items():
        result[int(rid)] = {
            item.quadruple
            for item in items
            if item.score >= defaults[quadruple_state(item.quadruple)]
        }
    return result


def run_oof(
    examples: Iterable[ReviewExample],
    *,
    n_splits: int = 3,
    seed: int = 42,
    miner_factory: Callable[[], StatisticalOpinionMiner] = StatisticalOpinionMiner,
    on_fold: Callable[[dict[str, object]], None] | None = None,
) -> OOFResult:
    rows = list(examples)
    if n_splits < 2 or len(rows) < n_splits:
        raise ValueError("n_splits must be at least 2 and no larger than the number of examples")
    candidates: dict[int, list[Candidate]] = {}
    gold = {row.id: set(row.labels) for row in rows}
    folds: list[dict[str, object]] = []
    for fold_index, (train_indices, valid_indices) in enumerate(fixed_splits(rows, n_splits=n_splits, seed=seed), start=1):
        miner = miner_factory()
        miner.fit(rows[index] for index in train_indices)
        for index in valid_indices:
            row = rows[int(index)]
            candidates[row.id] = miner.predict_candidates(row.text)
        fold = {
            "fold": fold_index,
            "train": len(train_indices),
            "valid": len(valid_indices),
            "candidate_rows": sum(len(candidates[rows[int(i)].id]) for i in valid_indices),
        }
        folds.append(fold)
        if on_fold:
            on_fold(fold)
    return OOFResult(candidates=candidates, gold=gold, folds=folds)


def _candidate_thresholds(candidates: Mapping[int, Iterable[Candidate]]) -> list[float]:
    values = sorted({round(float(item.score), 3) for items in candidates.values() for item in items})
    # Neural span-pair scores are continuous and can produce tens of thousands
    # of distinct values.  A bounded quantile grid keeps calibration tractable
    # while preserving the low/middle/high confidence regions.
    if len(values) > 192:
        positions = {round(index * (len(values) - 1) / 191) for index in range(192)}
        values = [values[index] for index in sorted(positions)]
    # Include a strict no-prediction boundary and a relaxed boundary above the
    # largest score.  The latter is useful when all generated candidates are
    # wrong on a fold.
    return [0.0, *values, 1.0]


def select_threshold(
    candidates: Mapping[int, Iterable[Candidate]],
    gold: Mapping[int, Iterable[Quadruple]],
    *,
    separate_implicit: bool = True,
) -> dict[str, object]:
    gold_sets = {int(k): set(v) for k, v in gold.items()}
    values = _candidate_thresholds(candidates)
    states = ("explicit", "implicit-A", "implicit-O", "dual-implicit")
    gold_n = sum(len(row) for row in gold_sets.values())
    state_gold = {
        state: sum(1 for row in gold_sets.values() for quad in row if quadruple_state(quad) == state)
        for state in states
    }

    # Precompute disjoint state curves. Since total F1 can be written as
    # 2*correct/(predicted+gold), Dinkelbach iterations make the four-state
    # threshold problem separable without a 192^4 brute-force grid.
    curves: dict[str, dict[float, tuple[int, int]]] = {state: {} for state in states}
    for state in states:
        state_items = [
            (int(rid), item)
            for rid, items in candidates.items()
            for item in items
            if quadruple_state(item.quadruple) == state
        ]
        for candidate_threshold in values:
            predicted = correct = 0
            for rid, item in state_items:
                if item.score >= candidate_threshold:
                    predicted += 1
                    if item.quadruple in gold_sets.get(rid, set()):
                        correct += 1
            curves[state][candidate_threshold] = (predicted, correct)

    def score_for_thresholds(state_thresholds: Mapping[str, float]) -> tuple[float, int, int]:
        predicted = correct = 0
        for state in states:
            pred_s, correct_s = curves[state][float(state_thresholds[state])]
            predicted += pred_s
            correct += correct_s
        f1 = (2.0 * correct / (predicted + gold_n)) if (predicted + gold_n) else 0.0
        return f1, predicted, correct

    if separate_implicit:
        lam = 0.5
        state_thresholds = {state: 1.0 for state in states}
        for _ in range(32):
            chosen: dict[str, float] = {}
            for state in states:
                if state_gold[state] == 0:
                    chosen[state] = 1.0
                    continue
                best_key: tuple[float, int, int, float] | None = None
                best_threshold = 1.0
                for candidate_threshold in values:
                    predicted, correct = curves[state][candidate_threshold]
                    objective = 2.0 * correct - lam * predicted
                    key = (objective, correct, -predicted, float(candidate_threshold))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_threshold = float(candidate_threshold)
                chosen[state] = best_threshold
            f1, _, _ = score_for_thresholds(chosen)
            state_thresholds = chosen
            if abs(f1 - lam) < 1e-12:
                break
            lam = f1
    else:
        best_key: tuple[float, float, float] | None = None
        shared = 1.0
        for candidate_threshold in values:
            thresholds = {state: float(candidate_threshold) for state in states}
            f1, predicted, correct = score_for_thresholds(thresholds)
            precision = correct / predicted if predicted else 0.0
            key = (f1, precision, float(candidate_threshold))
            if best_key is None or key > best_key:
                best_key = key
                shared = float(candidate_threshold)
        state_thresholds = {state: shared for state in states}
    explicit = state_thresholds["explicit"]
    implicit = state_thresholds["implicit-A"]
    predictions = threshold_predictions(
        candidates,
        explicit,
        implicit,
        state_thresholds=state_thresholds,
    )
    return {
        "threshold": explicit,
        "implicit_threshold": implicit,
        "state_thresholds": state_thresholds,
        "score": strict_f1(gold_sets, predictions),
        "predictions": predictions,
    }


def crossfit_threshold_score(
    candidates: Mapping[int, Iterable[Candidate]],
    gold: Mapping[int, Iterable[Quadruple]],
    folds: Iterable[Iterable[int]],
    *,
    separate_implicit: bool = True,
) -> dict[str, object]:
    """Score thresholds without fitting them on the fold being evaluated."""
    all_candidates = {int(rid): list(items) for rid, items in candidates.items()}
    gold_sets = {int(rid): set(items) for rid, items in gold.items()}
    all_ids = set(all_candidates)
    predictions: dict[int, set[Quadruple]] = {}
    fold_reports: list[dict[str, object]] = []
    for fold_index, valid_ids_raw in enumerate(folds, start=1):
        valid_ids = {int(rid) for rid in valid_ids_raw}
        fit_ids = all_ids - valid_ids
        if not valid_ids:
            continue
        fit_candidates = {rid: all_candidates[rid] for rid in fit_ids}
        fit_gold = {rid: gold_sets[rid] for rid in fit_ids}
        selected = select_threshold(fit_candidates, fit_gold, separate_implicit=separate_implicit)
        fold_candidates = {rid: all_candidates[rid] for rid in valid_ids}
        fold_predictions = threshold_predictions(
            fold_candidates,
            float(selected["threshold"]),
            float(selected["implicit_threshold"]),
            state_thresholds=selected.get("state_thresholds"),
        )
        predictions.update(fold_predictions)
        fold_score = strict_f1({rid: gold_sets[rid] for rid in valid_ids}, fold_predictions)
        fold_reports.append({
            "fold": fold_index,
            "threshold": float(selected["threshold"]),
            "implicit_threshold": float(selected["implicit_threshold"]),
            "state_thresholds": {key: float(value) for key, value in selected.get("state_thresholds", {}).items()},
            "score": fold_score,
        })
    if set(predictions) != all_ids:
        missing = sorted(all_ids - set(predictions))
        raise ValueError(f"crossfit folds do not cover every OOF id; missing={missing[:5]}")
    return {
        "score": strict_f1(gold_sets, predictions),
        "predictions": predictions,
        "folds": fold_reports,
    }


def fit_predict(
    train: Iterable[ReviewExample],
    test: Iterable[ReviewExample],
    *,
    threshold: float,
    implicit_threshold: float | None = None,
    miner: StatisticalOpinionMiner | None = None,
) -> dict[int, set[Quadruple]]:
    model = miner or StatisticalOpinionMiner()
    model.fit(train)
    return {
        row.id: model.predict(row.text, threshold=threshold, implicit_threshold=implicit_threshold)
        for row in test
    }
