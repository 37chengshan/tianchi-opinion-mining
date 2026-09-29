#!/usr/bin/env python3
from __future__ import annotations

"""Long-running, resumable-ish local Autopilot for the Tianchi task.

The process is deliberately staged.  Every stage writes artifacts before the
next one starts, so a failed neural fold still leaves a validated statistical
candidate and a useful dashboard state.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import error_analysis, merge_candidate_maps, save_candidates
from opinion_mining.baseline import Candidate, StatisticalOpinionMiner
from opinion_mining.data import Quadruple, ReviewExample, load_test_reviews, load_train_data
from opinion_mining.dashboard_runtime import DashboardWriter
from opinion_mining.metrics import Score, strict_f1
from opinion_mining.neural import NeuralConfig, NeuralTrainer
from opinion_mining.folds import assignment_records, fixed_splits
from opinion_mining.pipeline import run_oof, select_threshold, threshold_predictions
from opinion_mining.resource_guard import ResourceGuard
from opinion_mining.submission import validate_submission, write_submission


def _score_dict(score: Score) -> dict[str, Any]:
    return {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "correct": score.correct,
        "predicted": score.predicted,
        "gold": score.gold,
    }


def _json_dump(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_text(config: Any) -> str:
    if hasattr(config, "__dict__"):
        return json.dumps(config.__dict__, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(config)


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "mps" in text and "memory" in text


def is_strict_improvement(candidate_f1: float, champion_f1: float, *, epsilon: float = 1.0e-12) -> bool:
    """Return whether a candidate is strong enough to replace the champion."""
    return float(candidate_f1) > float(champion_f1) + epsilon


class Autopilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.artifacts = ROOT / "artifacts"
        self.extracted = self.artifacts / "data"
        self.experiments = self.artifacts / "experiments"
        self.reports = self.artifacts / "reports"
        self.submissions = self.artifacts / "submissions"
        for directory in [self.artifacts, self.extracted, self.experiments, self.reports, self.submissions]:
            directory.mkdir(parents=True, exist_ok=True)
        self.budget_seconds = float(args.budget_hours) * 3600.0
        self.deadline = time.time() + self.budget_seconds
        self.dashboard = DashboardWriter(ROOT / "dashboard", budget_seconds=self.budget_seconds)
        self.guard = ResourceGuard(memory_budget_gb=12.0, soft_rss_gb=10.0, hard_rss_gb=12.0)
        self.train_rows: list[ReviewExample] = []
        self.test_rows: list[ReviewExample] = []
        self.champion_f1 = -1.0
        self.trial_index = 0
        self.summary: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": [], "trials": []}
        self.last_submission: dict[str, Any] | None = None
        self.active_selected: dict[str, Any] | None = None
        self.active_fusion: dict[str, Any] | None = None

    def remaining(self) -> float:
        return self.deadline - time.time()

    def stopped(self) -> bool:
        return self.remaining() <= 0 or (self.artifacts / "STOP").exists()

    def stage(self, name: str, message: str) -> None:
        self.dashboard.update(status="running", stage=name, message=message)
        self.dashboard.event(message)
        self.summary["stages"].append({"stage": name, "time": time.time(), "message": message})
        _json_dump(self.reports / "autopilot_progress.json", self.summary)

    def update_resource(self) -> None:
        self.dashboard.resource(self.guard.snapshot())

    def count(self, key: str, amount: int = 1) -> None:
        current = dict(self.dashboard.state.get("counts", {}))
        current[key] = int(current.get(key, 0)) + amount
        self.dashboard.update(counts=current)

    def precheck(self) -> None:
        self.stage("PRECHECK", "检查数据、MPS、内存和 Dashboard 状态")
        train_zip = Path(self.args.train_zip).expanduser()
        test_zip = Path(self.args.test_zip).expanduser()
        if not train_zip.exists() or not test_zip.exists():
            raise FileNotFoundError(f"missing data zip: {train_zip} / {test_zip}")
        _extract(train_zip, self.extracted / "train")
        _extract(test_zip, self.extracted / "test")
        train_root = self.extracted / "train" / "TRAIN"
        test_root = self.extracted / "test" / "TEST"
        self.train_rows = load_train_data(train_root / "Train_reviews.csv", train_root / "Train_labels.csv")
        self.test_rows = load_test_reviews(test_root / "Test_reviews.csv")
        snapshot = self.guard.snapshot()
        self.update_resource()
        self.dashboard.event(f"数据通过预检：train={len(self.train_rows)} reviews, labels={sum(len(x.labels) for x in self.train_rows)}, test={len(self.test_rows)}")
        self.dashboard.event(f"device preference: MPS available={snapshot.mps_available}")
        _json_dump(self.reports / "precheck.json", {"train_reviews": len(self.train_rows), "train_quadruples": sum(len(x.labels) for x in self.train_rows), "test_reviews": len(self.test_rows), "resource": snapshot.__dict__, "train_zip_sha256": _hash(train_zip), "test_zip_sha256": _hash(test_zip)})

    def _record_trial(self, record: dict[str, Any]) -> None:
        self.trial_index += 1
        record = dict(record)
        record.setdefault("trial", self.trial_index)
        record.setdefault("seconds", 0.0)
        self.summary["trials"].append(record)
        self.dashboard.trial(record)
        self.count("completed")
        self.dashboard.metric(precision=record.get("precision"), recall=record.get("recall"), f1=record.get("f1"), champion_f1=self.champion_f1 if self.champion_f1 >= 0 else None)
        if record.get("f1") is not None and is_strict_improvement(float(record["f1"]), self.champion_f1):
            self.champion_f1 = float(record["f1"])
            self.dashboard.update(metrics={"champion_f1": self.champion_f1})
        _json_dump(self.reports / "autopilot_progress.json", self.summary)

    def run_baseline_cv(self, *, folds: int, name: str, max_opinion_variants: int, retrieval_k: int) -> dict[str, Any]:
        started = time.time()
        self.dashboard.update(current={"trial": name, "fold": None, "epoch": None, "config": f"statistical variants={max_opinion_variants} retrieval_k={retrieval_k}"})

        def on_fold(info: dict[str, object]) -> None:
            self.dashboard.update(current={"trial": name, "fold": info["fold"], "epoch": "OOF", "config": f"variants={max_opinion_variants},retrieval={retrieval_k}"})
            self.dashboard.event(f"{name} fold {info['fold']} complete; candidates={info['candidate_rows']}")
            self.update_resource()

        oof = run_oof(
            self.train_rows,
            n_splits=folds,
            seed=self.args.seed,
            miner_factory=lambda: StatisticalOpinionMiner(max_opinion_variants=max_opinion_variants, retrieval_k=retrieval_k),
            on_fold=on_fold,
        )
        selected = select_threshold(oof.candidates, oof.gold)
        score = selected["score"]
        trial_dir = self.experiments / name
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_candidates(trial_dir / "oof_candidates.json", oof.candidates)
        _json_dump(trial_dir / "threshold.json", {"threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "score": _score_dict(score), "folds": oof.folds})
        record = {"name": name, "model": "statistical_retrieval", "stage": f"{folds}-fold", "precision": score.precision, "recall": score.recall, "f1": score.f1, "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "seconds": time.time() - started, "config": f"variants={max_opinion_variants},retrieval_k={retrieval_k}"}
        self._record_trial(record)
        return {"name": name, "oof": oof, "selected": selected, "record": record, "trial_dir": trial_dir}

    def _neural_update(self, name: str, payload: dict[str, Any]) -> None:
        resource = payload.get("resource")
        if resource:
            self.dashboard.resource(resource)
        current = {"trial": name, "fold": payload.get("fold"), "epoch": payload.get("epoch"), "config": payload.get("config", name)}
        self.dashboard.update(current=current)
        if payload.get("f1") is not None:
            self.dashboard.metric(precision=payload.get("precision"), recall=payload.get("recall"), f1=payload.get("f1"), champion_f1=self.champion_f1 if self.champion_f1 >= 0 else None)

    def run_neural_cv(self, *, folds: int, name: str, config: NeuralConfig, include_test: bool) -> dict[str, Any]:
        started = time.time()
        oof: dict[int, list[Candidate]] = {}
        test_maps: list[dict[int, list[Candidate]]] = []
        fold_records: list[dict[str, Any]] = []
        trial_dir = self.experiments / name
        trial_dir.mkdir(parents=True, exist_ok=True)
        splits = fixed_splits(self.train_rows, n_splits=folds, seed=self.args.seed)
        _json_dump(trial_dir / "fold_assignments.json", assignment_records(self.train_rows, n_splits=folds, seed=self.args.seed))
        for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
            if self.stopped():
                break
            train_rows = [self.train_rows[int(i)] for i in train_indices]
            valid_rows = [self.train_rows[int(i)] for i in valid_indices]
            fold_config = copy.deepcopy(config)
            fold_config.seed = config.seed + fold * 1009
            self.dashboard.update(current={"trial": name, "fold": fold, "epoch": 0, "config": _config_text(fold_config)})
            fold_dir = trial_dir / f"fold_{fold}"
            try:
                trainer = NeuralTrainer(fold_config, guard=self.guard)
                result = trainer.train_fold(
                    train_rows,
                    valid_rows,
                    self.test_rows if include_test else [],
                    output_dir=fold_dir,
                    on_update=lambda payload, fold=fold: self._neural_update(name, {**payload, "fold": fold, "config": _config_text(fold_config)}),
                )
            except Exception as error:
                self.dashboard.event(f"{name} fold {fold} failed: {type(error).__name__}: {str(error)[:180]}")
                self.count("oom" if _is_oom(error) else "failed")
                self.guard.release_cache()
                if _is_oom(error) and fold_config.batch_size > 1:
                    retry = copy.deepcopy(fold_config)
                    retry.batch_size = 1
                    retry.max_length = min(retry.max_length, 80)
                    retry.trainable_layers = 1
                    retry.epochs = max(1, min(retry.epochs, 2))
                    try:
                        self.dashboard.event(f"{name} fold {fold}: resource retry with batch=1 length={retry.max_length} layers=1")
                        trainer = NeuralTrainer(retry, guard=self.guard)
                        result = trainer.train_fold(train_rows, valid_rows, self.test_rows if include_test else [], output_dir=fold_dir / "retry", on_update=lambda payload, fold=fold: self._neural_update(name, {**payload, "fold": fold, "config": _config_text(retry)}))
                    except Exception as retry_error:
                        self.dashboard.event(f"{name} fold {fold} retry failed; continuing: {type(retry_error).__name__}: {str(retry_error)[:140]}")
                        self.count("failed")
                        continue
                else:
                    continue
            oof.update(result.candidates)
            if include_test:
                test_maps.append(result.test_candidates)
                save_candidates(fold_dir / "test_candidates.json", result.test_candidates)
            valid_gold = {row.id: row.labels for row in valid_rows}
            provisional = threshold_predictions(result.candidates, 0.12, 0.10)
            fold_score = strict_f1(valid_gold, provisional)
            fold_record = {"fold": fold, "train": len(train_rows), "valid": len(valid_rows), "f1_at_monitor_threshold": fold_score.f1, "checkpoint": result.checkpoint, "history": result.history}
            fold_records.append(fold_record)
            self.dashboard.event(f"{name} fold {fold} complete; monitor F1={fold_score.f1:.4f}")
            self.update_resource()
            del trainer
            self.guard.release_cache()
        gold = {row.id: set(row.labels) for row in self.train_rows}
        selected = select_threshold(oof, gold)
        score = selected["score"]
        save_candidates(trial_dir / "oof_candidates.json", oof)
        _json_dump(trial_dir / "threshold.json", {"threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "score": _score_dict(score), "folds": fold_records, "config": config.__dict__})
        record = {"name": name, "model": "mps_rbt3_span_relation", "stage": f"{folds}-fold", "precision": score.precision, "recall": score.recall, "f1": score.f1, "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "seconds": time.time() - started, "config": _config_text(config), "folds_completed": len(fold_records)}
        self._record_trial(record)
        return {"name": name, "oof": oof, "test_maps": test_maps, "selected": selected, "record": record, "trial_dir": trial_dir, "folds": fold_records}

    def fit_statistical_test(self, baseline: dict[str, Any]) -> dict[int, list[Candidate]]:
        config_text = baseline["record"]["config"]
        variants = int(config_text.split("variants=")[1].split(",")[0])
        retrieval = int(config_text.split("retrieval_k=")[1])
        miner = StatisticalOpinionMiner(max_opinion_variants=variants, retrieval_k=retrieval).fit(self.train_rows)
        return {row.id: miner.predict_candidates(row.text) for row in self.test_rows}

    def run(self) -> None:
        self.precheck()
        self.stage("SCREENING_3FOLD", "开始 3-fold 快速筛选：统计检索配置")
        baseline_trials = []
        for index, (variants, retrieval) in enumerate([(1, 4), (2, 8), (3, 12)], start=1):
            if self.stopped():
                break
            baseline_trials.append(self.run_baseline_cv(folds=3, name=f"baseline_3fold_v{index}", max_opinion_variants=variants, retrieval_k=retrieval))
        baseline_3 = max(baseline_trials, key=lambda item: float(item["record"]["f1"])) if baseline_trials else None
        neural_screen_config = NeuralConfig(epochs=self.args.screen_epochs, max_length=96, batch_size=4, gradient_accumulation=2, trainable_layers=2, seed=self.args.seed)
        neural_3 = None
        if not self.stopped():
            self.stage("SCREENING_3FOLD", "开始 3-fold 快速筛选：MPS RBT3 span/relation 模型")
            try:
                neural_3 = self.run_neural_cv(folds=3, name="neural_3fold_rbt3", config=neural_screen_config, include_test=False)
            except Exception as error:
                self.dashboard.event(f"3-fold neural stage failed; statistical path remains active: {type(error).__name__}: {str(error)[:180]}")
                self.count("failed")
        candidates_for_strict = [item for item in [baseline_3, neural_3] if item is not None]
        if not candidates_for_strict:
            raise RuntimeError("no screening trial completed")
        best_screen = max(candidates_for_strict, key=lambda item: float(item["record"]["f1"]))
        self.stage("STRICT_5FOLD", f"进入 5-fold 严格确认；当前筛选 champion={best_screen['record']['name']} F1={best_screen['record']['f1']:.4f}")
        baseline_5 = None
        if baseline_3 is not None:
            baseline_5 = self.run_baseline_cv(folds=5, name="baseline_5fold_confirm", max_opinion_variants=int(baseline_3["record"]["config"].split("variants=")[1].split(",")[0]), retrieval_k=int(baseline_3["record"]["config"].split("retrieval_k=")[1]))
        neural_5 = None
        if neural_3 is not None and not self.stopped():
            final_config = copy.deepcopy(neural_screen_config)
            final_config.epochs = self.args.final_epochs
            self.stage("STRICT_5FOLD", "运行 MPS neural 5-fold 严格确认并保存每折测试预测")
            try:
                neural_5 = self.run_neural_cv(folds=5, name="neural_5fold_confirm", config=final_config, include_test=True)
            except Exception as error:
                self.dashboard.event(f"5-fold neural confirmation failed; continuing with available candidates: {type(error).__name__}: {str(error)[:180]}")
                self.count("failed")
        self.stage("ERROR_ANALYSIS", "对 OOF 误差做四元组级分解，定位漏召回和过生成")
        analysis_payload: dict[str, Any] = {}
        for label, item in [("baseline_3fold", baseline_3), ("neural_3fold", neural_3), ("baseline_5fold", baseline_5), ("neural_5fold", neural_5)]:
            if item is None:
                continue
            selected = item["selected"]
            analysis_payload[label] = error_analysis({row.id: row.labels for row in self.train_rows}, selected["predictions"])
        _json_dump(self.reports / "error_analysis.json", analysis_payload)
        self.stage("FUSION", "在无泄漏 OOF 上搜索规则 / neural 权重与隐式、显式阈值")
        base_for_fusion = baseline_5 or baseline_3
        neural_for_fusion = neural_5 or neural_3
        fusion_options: list[dict[str, Any]] = []
        if base_for_fusion is not None:
            fusion_options.append({"name": "rule_only", "weights": [1.0], "candidates": base_for_fusion["oof"]})
        if neural_for_fusion is not None:
            fusion_options.append({"name": "neural_only", "weights": [1.0], "candidates": neural_for_fusion["oof"]})
        if base_for_fusion is not None and neural_for_fusion is not None:
            for rule_weight in [0.2, 0.4, 0.6, 0.8]:
                neural_weight = 1.0 - rule_weight
                merged = merge_candidate_maps([base_for_fusion["oof"], neural_for_fusion["oof"]], weights=[rule_weight, neural_weight], average_present=True, bonus=0.015)
                selected = select_threshold(merged, base_for_fusion["oof"].gold)
                fusion_options.append({"name": f"fusion_rule_{rule_weight:.1f}", "weights": [rule_weight, neural_weight], "candidates": merged, "selected": selected})
        for option in fusion_options:
            if "selected" not in option:
                option["selected"] = select_threshold(option["candidates"], base_for_fusion["oof"].gold if base_for_fusion is not None else neural_for_fusion["oof"].gold)
            selected = option["selected"]
            record = {"name": option["name"], "model": "fusion", "stage": "OOF calibration", **_score_dict(selected["score"]), "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "seconds": 0.0, "config": str(option["weights"])}
            self._record_trial(record)
        best_fusion = max(fusion_options, key=lambda item: float(item["selected"]["score"].f1))
        self.stage("ENSEMBLE", f"锁定融合 ensemble={best_fusion['name']} OOF F1={best_fusion['selected']['score'].f1:.4f}")
        baseline_test = self.fit_statistical_test(base_for_fusion) if base_for_fusion is not None else {}
        neural_test = {}
        if neural_for_fusion is not None and neural_for_fusion.get("test_maps"):
            neural_test = merge_candidate_maps(neural_for_fusion["test_maps"], average_present=True, bonus=0.015)
        if best_fusion.get("name") == "neural_only" and neural_test:
            final_candidates = neural_test
        elif best_fusion.get("name") == "rule_only" and baseline_test:
            final_candidates = baseline_test
        elif base_for_fusion is not None and neural_for_fusion is not None:
            weights = best_fusion.get("weights", [0.5, 0.5])
            if len(weights) != 2:
                weights = [0.5, 0.5]
            final_candidates = merge_candidate_maps([baseline_test, neural_test], weights=weights, average_present=True, bonus=0.015)
        elif baseline_test:
            final_candidates = baseline_test
        else:
            final_candidates = neural_test
        selected = best_fusion["selected"]
        final_predictions = threshold_predictions(final_candidates, float(selected["threshold"]), float(selected["implicit_threshold"]))
        self.stage("POSTPROCESS", "完成 OOF / LOFO 后处理；默认关闭低收益 seed-only evolution")
        self._write_submission(final_predictions, selected, best_fusion, base_for_fusion, neural_for_fusion)
        if getattr(self.args, "allow_seed_evolution", False):
            self.stage("EVOLUTION", "显式开启 seed challenger；只有 strict OOF 超过 champion 的候选才会晋级")
            self.evolution_loop(baseline_3, best_fusion, neural_screen_config)
        else:
            self.dashboard.event("seed-only evolution disabled by default; waiting for strong encoder/grid route")
        self.stage("FINAL", "Autopilot 结束，冻结最后一次实际晋级的本地候选")
        # The latest written Result.csv is authoritative.  A challenger may
        # have promoted after the original fusion variable was computed.
        active = self.active_selected or selected
        final_submission = dict(self.last_submission or {})
        final_fusion = self.active_fusion or best_fusion
        self.dashboard.final(
            metrics={"precision": float(active["score"].precision), "recall": float(active["score"].recall), "f1": float(active["score"].f1), "champion_f1": self.champion_f1},
            submission=final_submission,
            message=f"Final Result.csv 已生成并通过严格校验；candidate={final_submission.get('candidate_id', 'local-only')} OOF strict F1={active['score'].f1:.4f}; LB 未提交",
        )

    def _write_submission(self, predictions: Mapping[int, Iterable[Quadruple]], selected: Mapping[str, Any], fusion: Mapping[str, Any], baseline: Mapping[str, Any] | None, neural: Mapping[str, Any] | None) -> None:
        output = self.submissions / "Result.csv"
        report = write_submission(output, self.test_rows, predictions)
        root_output = ROOT / "Result.csv"
        shutil.copy2(output, root_output)
        validation = validate_submission(output, self.test_rows)
        raw = output.read_bytes()
        if len(raw) >= 100_000_000:
            raise RuntimeError("Result.csv exceeds 100MB")
        submission_info = {
            "path": str(output),
            "root_path": str(root_output),
            "rows": validation.rows,
            "predicted_quadruples": validation.predicted_quadruples,
            "empty_ids": validation.empty_ids,
            "bytes": len(raw),
            "sha256": _hash(output),
            "validator": "passed",
            "status": "validated_local_pending",
            "local_score": float(selected["score"].f1),
            "local_score_metric": "strict_oof_f1",
            "leaderboard_score": None,
            "command": f"PYTHONPATH=src python3 scripts/autopilot.py --train-zip '{self.args.train_zip}' --test-zip '{self.args.test_zip}' --budget-hours {self.args.budget_hours}",
        }
        metrics = {"precision": float(selected["score"].precision), "recall": float(selected["score"].recall), "f1": float(selected["score"].f1), "champion_f1": self.champion_f1}
        fusion_payload = {"name": fusion.get("name"), "weights": fusion.get("weights"), "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"]}
        # Register every validated candidate locally.  This never uploads or
        # contacts Tianchi; a browser-capable executor may consume the queue
        # later after independently checking the same SHA256.
        try:
            try:
                from submission_manifest import create_candidate_manifest
            except ModuleNotFoundError:
                from scripts.submission_manifest import create_candidate_manifest
            online_dir = self.submissions / "online_loop"
            config_path = online_dir / "configs" / f"{submission_info['sha256']}.json"
            oof_path = online_dir / "oof" / f"{submission_info['sha256']}.json"
            _json_dump(config_path, {"fusion": fusion_payload, "config": neural.get("config") if neural else None, "command": submission_info["command"]})
            _json_dump(oof_path, {"metrics": {"f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"]}, "source": "validated local OOF"})
            manifest_info = create_candidate_manifest(
                output,
                oof_path,
                config_path,
                online_dir,
                candidate_dir=self.submissions / "candidates",
                expected_test_ids=[row.id for row in self.test_rows],
            )
            submission_info["candidate_id"] = manifest_info["candidate_id"]
            submission_info["manifest_path"] = manifest_info["manifest_path"]
            submission_info["queue_path"] = manifest_info["queue_path"]
            record = manifest_info["record"]
            submission_info.update(
                {
                    "candidate_csv": record["candidate_csv"],
                    "oof_json": record["oof_json"],
                    "config_json": record["config_json"],
                    "config_hash": record["config_hash"],
                    "oof_sha256": record["oof_sha256"],
                    "config_sha256": record["config_sha256"],
                    "test_id_count": int(record["test_id_count"]),
                    "test_ids_sha256": record["test_ids_sha256"],
                    "local_score": float(record["local_score"]),
                    "leaderboard_score": None,
                    "status": record["status"],
                }
            )
            self.dashboard.event(f"local submission queue updated: {manifest_info['candidate_id']}")
        except Exception as error:
            self.dashboard.event(f"local submission manifest skipped: {type(error).__name__}: {str(error)[:180]}")
        # Persist and publish only after manifest enrichment.  Dashboard and
        # reports therefore describe the same bytes and evidence as Result.csv.
        self.last_submission = dict(submission_info)
        self.active_selected = dict(selected)
        self.active_fusion = dict(fusion)
        self.dashboard.update(status="running", submission=self.last_submission, metrics=metrics, message=f"当前候选 Result.csv 已通过严格校验；candidate={submission_info.get('candidate_id', 'local-only')} OOF strict F1={selected['score'].f1:.4f}; LB 未提交")
        report_payload = {"metrics": metrics, "submission": self.last_submission, "fusion": fusion_payload, "baseline_trial": baseline.get("record") if baseline else None, "neural_trial": neural.get("record") if neural else None, "champion_f1": self.champion_f1}
        _json_dump(self.reports / "final_report.json", report_payload)
        _json_dump(self.reports / "champion.json", {"metrics": metrics, "submission": self.last_submission, "fusion": fusion_payload})

    def evolution_loop(self, baseline_3: Mapping[str, Any] | None, best_fusion: Mapping[str, Any], base_config: NeuralConfig) -> None:
        challenger_index = 0
        seeds = [17, 29, 53, 71, 97, 131, 173, 211]
        while not self.stopped():
            seed = seeds[challenger_index % len(seeds)]
            challenger_index += 1
            config = copy.deepcopy(base_config)
            config.seed = seed
            config.epochs = max(1, min(self.args.evolution_epochs, 3))
            config.max_length = 80 if challenger_index % 2 else 96
            config.trainable_layers = 1 if challenger_index % 3 == 0 else 2
            name = f"evolution_3fold_{challenger_index:03d}_seed{seed}"
            try:
                champion_before = self.champion_f1
                result = self.run_neural_cv(folds=3, name=name, config=config, include_test=True)
                # A neural-only challenger can be a valid champion even when
                # its rule fusion is weaker.  Promote it immediately while
                # its fold checkpoints and test candidates are available.
                if result.get("test_maps") and is_strict_improvement(float(result["record"]["f1"]), champion_before):
                    neural_test = merge_candidate_maps(result["test_maps"], average_present=True, bonus=0.015)
                    self._write_submission(
                        threshold_predictions(neural_test, result["selected"]["threshold"], result["selected"]["implicit_threshold"]),
                        result["selected"],
                        {"name": "neural_only", "weights": [1.0]},
                        None,
                        result,
                    )
                    self.dashboard.event(f"champion promoted: {name} OOF F1={float(result['record']['f1']):.4f}")
                if baseline_3 is not None:
                    merged = merge_candidate_maps([baseline_3["oof"], result["oof"]], weights=[0.6, 0.4], average_present=True, bonus=0.015)
                    base_gold = baseline_3["oof"].gold if hasattr(baseline_3["oof"], "gold") else {row.id: set(row.labels) for row in self.train_rows}
                    selected = select_threshold(merged, base_gold)
                    fusion_improves = is_strict_improvement(float(selected["score"].f1), self.champion_f1)
                    self._record_trial({"name": f"{name}_fusion", "model": "evolution_fusion", "stage": "evolution", **_score_dict(selected["score"]), "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "seconds": result["record"]["seconds"], "config": "rule=0.6,challenger=0.4"})
                    if fusion_improves and result.get("test_maps"):
                        neural_test = merge_candidate_maps(result["test_maps"], average_present=True, bonus=0.015)
                        final_candidates = merge_candidate_maps(
                            [self.fit_statistical_test(baseline_3), neural_test],
                            weights=[0.6, 0.4],
                            average_present=True,
                            bonus=0.015,
                        )
                        self._write_submission(
                            threshold_predictions(final_candidates, selected["threshold"], selected["implicit_threshold"]),
                            selected,
                            {"name": f"{name}_fusion", "weights": [0.6, 0.4]},
                            baseline_3,
                            result,
                        )
                        self.dashboard.event(f"champion promoted: {name}_fusion OOF F1={float(selected['score'].f1):.4f}")
                self.dashboard.event(f"evolution challenger {challenger_index} finished; continuing queue")
            except Exception as error:
                self.dashboard.event(f"evolution challenger {challenger_index} failed; continuing queue: {type(error).__name__}: {str(error)[:160]}")
                self.count("failed")
                self.guard.release_cache()
                time.sleep(min(30.0, max(1.0, self.remaining())))

    def finalize_static_dashboard(self) -> None:
        state = json.loads((ROOT / "dashboard/state.json").read_text(encoding="utf-8"))
        history = json.loads((ROOT / "dashboard/history.json").read_text(encoding="utf-8"))
        html = (ROOT / "dashboard/index.html").read_text(encoding="utf-8")
        start = html.find("async function tick()")
        end = html.find("</script>", start)
        if start >= 0 and end >= 0:
            embedded = "const FINAL_STATE=" + json.dumps(state, ensure_ascii=False) + ";const FINAL_HISTORY=" + json.dumps(history, ensure_ascii=False) + ";function tick(){render(FINAL_STATE,FINAL_HISTORY)} tick();"
            html = html[:start] + embedded + html[end:]
        (self.reports / "final_dashboard.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", default="/Users/cc/Downloads/初赛训练数据 2019-08-01.zip")
    parser.add_argument("--test-zip", default="/Users/cc/Downloads/初赛测试数据 2019-08-15.zip")
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--screen-epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=4)
    parser.add_argument("--evolution-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-seed-evolution", action="store_true", help="legacy opt-in; disabled for the V2 plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    autopilot = Autopilot(args)
    try:
        autopilot.run()
    except KeyboardInterrupt:
        autopilot.dashboard.event("收到手动中断；保留已完成 artifacts 和当前 dashboard")
        autopilot.dashboard.update(status="paused", stage="INTERRUPTED", message="Autopilot interrupted; artifacts preserved.")
        raise
    except Exception as error:
        autopilot.dashboard.event(f"Autopilot top-level failure: {type(error).__name__}: {str(error)[:220]}")
        autopilot.dashboard.update(status="failed", stage="FAILED", message=f"主流程失败，但已保留中间 artifacts：{type(error).__name__}")
        (autopilot.reports / "autopilot_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            autopilot.finalize_static_dashboard()
        except Exception:
            pass


if __name__ == "__main__":
    main()
