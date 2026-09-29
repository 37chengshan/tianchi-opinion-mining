from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .analysis import load_candidates, save_candidates
from .baseline import Candidate
from .data import Quadruple
from .folds import load_assignment_records, validate_assignment_records
from .metrics import Score, strict_f1
from .pipeline import quadruple_state, threshold_predictions
from .submission import CATEGORIES


MetricSelector = Callable[[Quadruple], bool]


@dataclass(frozen=True)
class MetricSummary:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    predicted: int
    gold: int

    @classmethod
    def from_score(cls, score: Score) -> "MetricSummary":
        return cls(
            precision=float(score.precision),
            recall=float(score.recall),
            f1=float(score.f1),
            tp=int(score.correct),
            fp=int(score.predicted - score.correct),
            fn=int(score.gold - score.correct),
            predicted=int(score.predicted),
            gold=int(score.gold),
        )


@dataclass(frozen=True)
class AnchorEvalReport:
    name: str
    config_hash: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    predicted: int
    gold: int
    candidate_count: int
    candidate_id_count: int
    thresholds: dict[str, float]
    state_metrics: dict[str, MetricSummary]
    category_metrics: dict[str, MetricSummary]
    fold_reports: list[dict[str, Any]]
    fold_isolation_verified: bool
    delta_vs_anchor_same_fold_baseline: dict[str, float | int] | None
    oof_candidates_path: str | None
    report_path: str | None

    @classmethod
    def from_score(
        cls,
        *,
        name: str,
        config_hash: str,
        score: Score,
        candidate_count: int,
        candidate_id_count: int,
        thresholds: dict[str, float],
        state_metrics: dict[str, MetricSummary],
        category_metrics: dict[str, MetricSummary],
        fold_reports: list[dict[str, Any]],
        fold_isolation_verified: bool,
        delta_vs_anchor_same_fold_baseline: dict[str, float | int] | None,
        oof_candidates_path: str | None,
        report_path: str | None,
    ) -> "AnchorEvalReport":
        summary = MetricSummary.from_score(score)
        return cls(
            name=name,
            config_hash=config_hash,
            precision=summary.precision,
            recall=summary.recall,
            f1=summary.f1,
            tp=summary.tp,
            fp=summary.fp,
            fn=summary.fn,
            predicted=summary.predicted,
            gold=summary.gold,
            candidate_count=candidate_count,
            candidate_id_count=candidate_id_count,
            thresholds=thresholds,
            state_metrics=state_metrics,
            category_metrics=category_metrics,
            fold_reports=fold_reports,
            fold_isolation_verified=fold_isolation_verified,
            delta_vs_anchor_same_fold_baseline=delta_vs_anchor_same_fold_baseline,
            oof_candidates_path=oof_candidates_path,
            report_path=report_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _mapping_candidates(value: Mapping[int | str, Iterable[Candidate]] | str | Path) -> dict[int, list[Candidate]]:
    if isinstance(value, (str, Path)):
        return load_candidates(value)
    return {int(rid): list(items) for rid, items in value.items()}



def _mapping_gold(value: Mapping[int | str, Iterable[Quadruple]]) -> dict[int, set[Quadruple]]:
    return {int(rid): set(items) for rid, items in value.items()}



def _assignment_records(
    folds: Sequence[Mapping[str, object]] | str | Path,
    *,
    n_splits: int | None,
    seed: int | None,
) -> list[dict[str, object]]:
    if isinstance(folds, (str, Path)):
        return load_assignment_records(folds, n_splits=n_splits, seed=seed)
    return validate_assignment_records(folds, n_splits=n_splits, seed=seed)



def _canonical_config(config: Mapping[str, Any]) -> str:
    ignored = {"candidates", "gold", "fold_candidates", "baseline", "output_dir"}
    payload = {str(key): value for key, value in config.items() if key not in ignored}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)



def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_config(config).encode("utf-8")).hexdigest()



def _subset_maps(
    gold: Mapping[int, set[Quadruple]],
    predictions: Mapping[int, set[Quadruple]],
    selector: MetricSelector,
) -> tuple[dict[int, set[Quadruple]], dict[int, set[Quadruple]]]:
    ids = set(gold) | set(predictions)
    return (
        {rid: {quad for quad in gold.get(rid, set()) if selector(quad)} for rid in ids},
        {rid: {quad for quad in predictions.get(rid, set()) if selector(quad)} for rid in ids},
    )



def _summary_for(
    gold: Mapping[int, set[Quadruple]],
    predictions: Mapping[int, set[Quadruple]],
    selector: MetricSelector,
) -> MetricSummary:
    selected_gold, selected_predictions = _subset_maps(gold, predictions, selector)
    return MetricSummary.from_score(strict_f1(selected_gold, selected_predictions))



def _normalise_fold_candidates(
    fold_candidates: Mapping[int | str, Mapping[int | str, Iterable[Candidate]]],
) -> dict[int, dict[int, list[Candidate]]]:
    return {
        int(fold): {int(rid): list(items) for rid, items in candidates.items()}
        for fold, candidates in fold_candidates.items()
    }



def _verify_fold_isolation(
    assignments: Sequence[Mapping[str, object]],
    candidates: Mapping[int, list[Candidate]],
    fold_candidates: Mapping[int, Mapping[int, list[Candidate]]],
) -> None:
    all_valid: set[int] = set()
    for assignment in assignments:
        fold = int(assignment["fold"])
        train_ids = {int(value) for value in assignment["train_ids"]}
        valid_ids = {int(value) for value in assignment["valid_ids"]}
        fold_map = fold_candidates.get(fold)
        if fold_map is None:
            raise ValueError(f"missing candidate map for fold {fold}")
        fold_ids = set(fold_map)
        leaked = fold_ids & train_ids
        if leaked:
            raise ValueError(f"fold {fold} candidate ids overlap train_ids: {sorted(leaked)[:5]}")
        if fold_ids != valid_ids:
            raise ValueError(
                f"fold {fold} candidate ids must equal valid_ids; "
                f"missing={sorted(valid_ids - fold_ids)[:5]} extra={sorted(fold_ids - valid_ids)[:5]}"
            )
        all_valid.update(valid_ids)
    if set(candidates) != all_valid:
        raise ValueError(
            f"OOF candidate ids do not match valid ids; "
            f"missing={sorted(all_valid - set(candidates))[:5]} extra={sorted(set(candidates) - all_valid)[:5]}"
        )



def _baseline_delta(
    report: AnchorEvalReport,
    baseline: AnchorEvalReport | Mapping[str, Any] | None,
) -> dict[str, float | int] | None:
    if baseline is None:
        return None
    if isinstance(baseline, AnchorEvalReport):
        values = baseline.to_dict()
    else:
        values = dict(baseline)
    fields = ("precision", "recall", "f1", "tp", "fp", "fn", "predicted", "gold")
    return {field: getattr(report, field) - float(values[field]) if field in ("precision", "recall", "f1") else getattr(report, field) - int(values[field]) for field in fields}



def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")



def evaluate_anchor_variant(
    config: Mapping[str, Any],
    folds: Sequence[Mapping[str, object]] | str | Path,
) -> AnchorEvalReport:
    """Evaluate one candidate variant against one frozen OOF fold assignment.

    ``fold_candidates`` is the provenance boundary: each fold map must contain
    only its valid ids.  This makes it impossible for a training-fold sample to
    enter the scored OOF result unnoticed.
    """
    name = str(config.get("name", "anchor_variant"))
    candidates = _mapping_candidates(config["candidates"])
    gold = _mapping_gold(config["gold"])
    n_splits = int(config["n_splits"]) if config.get("n_splits") is not None else None
    seed = int(config["seed"]) if config.get("seed") is not None else None
    assignments = _assignment_records(folds, n_splits=n_splits, seed=seed)
    candidate_ids = set(candidates)
    gold_ids = set(gold)
    if candidate_ids != gold_ids:
        raise ValueError(
            f"candidate and gold ids differ; missing={sorted(gold_ids - candidate_ids)[:5]} "
            f"extra={sorted(candidate_ids - gold_ids)[:5]}"
        )

    fold_candidates = config.get("fold_candidates")
    if fold_candidates is None:
        fold_candidates = {
            int(item["fold"]): {int(rid): candidates[int(rid)] for rid in item["valid_ids"]}
            for item in assignments
        }
    fold_candidates = _normalise_fold_candidates(fold_candidates)
    _verify_fold_isolation(assignments, candidates, fold_candidates)

    state_thresholds = config.get("state_thresholds")
    predictions = threshold_predictions(
        candidates,
        float(config.get("threshold", 0.727)),
        float(config.get("implicit_threshold", 0.503)),
        state_thresholds=state_thresholds,
    )
    score = strict_f1(gold, predictions)
    state_metrics = {
        state: _summary_for(gold, predictions, lambda quad, state=state: quadruple_state(quad) == state)
        for state in ("explicit", "implicit-A", "implicit-O", "dual-implicit")
    }
    category_metrics = {
        category: _summary_for(gold, predictions, lambda quad, category=category: quad.category == category)
        for category in sorted(CATEGORIES)
    }
    fold_reports = []
    for assignment in assignments:
        valid_ids = {int(value) for value in assignment["valid_ids"]}
        fold_score = strict_f1(
            {rid: gold[rid] for rid in valid_ids},
            {rid: predictions[rid] for rid in valid_ids},
        )
        fold_reports.append(
            {
                "fold": int(assignment["fold"]),
                "valid_ids": len(valid_ids),
                "score": asdict(MetricSummary.from_score(fold_score)),
            }
        )

    output_dir = Path(config["output_dir"]) if config.get("output_dir") is not None else None
    oof_path = None
    report_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        oof_file = output_dir / "oof_candidates.json"
        save_candidates(oof_file, candidates)
        oof_path = str(oof_file)
    report = AnchorEvalReport.from_score(
        name=name,
        config_hash=_config_hash(config),
        score=score,
        candidate_count=sum(len(items) for items in candidates.values()),
        candidate_id_count=len(candidates),
        thresholds={
            "explicit": float(config.get("threshold", 0.727)),
            "implicit": float(config.get("implicit_threshold", 0.503)),
        },
        state_metrics=state_metrics,
        category_metrics=category_metrics,
        fold_reports=fold_reports,
        fold_isolation_verified=True,
        delta_vs_anchor_same_fold_baseline=None,
        oof_candidates_path=oof_path,
        report_path=None,
    )
    delta = _baseline_delta(report, config.get("baseline"))
    if delta is not None:
        report = AnchorEvalReport.from_score(
            name=report.name,
            config_hash=report.config_hash,
            score=score,
            candidate_count=report.candidate_count,
            candidate_id_count=report.candidate_id_count,
            thresholds=report.thresholds,
            state_metrics=report.state_metrics,
            category_metrics=report.category_metrics,
            fold_reports=report.fold_reports,
            fold_isolation_verified=report.fold_isolation_verified,
            delta_vs_anchor_same_fold_baseline=delta,
            oof_candidates_path=report.oof_candidates_path,
            report_path=None,
        )
    if output_dir is not None:
        report_file = output_dir / "report.json"
        _write_json(report_file, report.to_dict())
        report = AnchorEvalReport.from_score(
            name=report.name,
            config_hash=report.config_hash,
            score=score,
            candidate_count=report.candidate_count,
            candidate_id_count=report.candidate_id_count,
            thresholds=report.thresholds,
            state_metrics=report.state_metrics,
            category_metrics=report.category_metrics,
            fold_reports=report.fold_reports,
            fold_isolation_verified=report.fold_isolation_verified,
            delta_vs_anchor_same_fold_baseline=report.delta_vs_anchor_same_fold_baseline,
            oof_candidates_path=report.oof_candidates_path,
            report_path=str(report_file),
        )
        _write_json(report_file, report.to_dict())
    return report
