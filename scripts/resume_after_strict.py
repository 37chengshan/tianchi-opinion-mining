#!/usr/bin/env python3
"""Resume Autopilot after a completed strict CV stage.

Used when post-processing fails after expensive folds.  It reads the saved OOF
artifacts, reconstructs test predictions from saved checkpoints, and continues
with error analysis, fusion, submission validation, and evolution.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opinion_mining.analysis import error_analysis, load_candidates, merge_candidate_maps, save_candidates
from opinion_mining.baseline import Candidate
from opinion_mining.metrics import Score
from opinion_mining.neural import NeuralConfig, NeuralTrainer, predict_candidates
from opinion_mining.pipeline import select_threshold, threshold_predictions
from opinion_mining.resource_guard import ResourceGuard
from opinion_mining.dashboard_runtime import DashboardWriter
from opinion_mining.data import Quadruple
from opinion_mining.submission import write_submission
from autopilot import Autopilot, parse_args as parse_main_args


def _load_trial(auto: Autopilot, name: str) -> dict[str, object]:
    trial_dir = auto.experiments / name
    candidates = load_candidates(trial_dir / "oof_candidates.json")
    threshold = json.loads((trial_dir / "threshold.json").read_text(encoding="utf-8"))
    score = Score(**threshold["score"])
    selected = {
        "threshold": threshold["threshold"],
        "implicit_threshold": threshold["implicit_threshold"],
        "score": score,
        "predictions": threshold_predictions(candidates, threshold["threshold"], threshold["implicit_threshold"]),
    }
    if name.endswith("v1") or name == "baseline_5fold_confirm":
        config_text = "variants=1,retrieval_k=4"
    elif name.endswith("v2"):
        config_text = "variants=2,retrieval_k=8"
    elif name.endswith("v3"):
        config_text = "variants=3,retrieval_k=12"
    else:
        config_text = json.dumps(threshold.get("config", {}), ensure_ascii=False, sort_keys=True)
    return {
        "name": name,
        "oof": candidates,
        "selected": selected,
        "record": {
            "name": name,
            "model": "mps_rbt3_span_relation" if name.startswith("neural") else "statistical_retrieval",
            "stage": "5-fold" if "5fold" in name else "3-fold",
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "threshold": threshold["threshold"],
            "implicit_threshold": threshold["implicit_threshold"],
            "seconds": 0.0,
            "config": config_text,
        },
        "trial_dir": trial_dir,
        "test_maps": [],
    }


def _recover_neural_test(auto: Autopilot, trial: dict[str, object]) -> None:
    trial_dir = Path(trial["trial_dir"])
    test_maps: list[dict[int, list[Candidate]]] = []
    for fold_dir in sorted(trial_dir.glob("fold_*")):
        checkpoint = fold_dir / "model.pt"
        saved_candidates = fold_dir / "test_candidates.json"
        if saved_candidates.exists():
            test_maps.append(load_candidates(saved_candidates))
            continue
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu")
        config = NeuralConfig(**payload["config"])
        trainer = NeuralTrainer(config, guard=auto.guard)
        model = trainer._build_model()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        auto.dashboard.update(current={"trial": trial["name"], "fold": fold_dir.name, "epoch": "test inference", "config": "recover checkpoint"})
        candidates = predict_candidates(model, trainer.tokenizer, auto.test_rows, config, trainer.device)
        save_candidates(saved_candidates, candidates)
        test_maps.append(candidates)
        del model
        auto.guard.release_cache()
        auto.update_resource()
    trial["test_maps"] = test_maps


def _fusion(auto: Autopilot, baseline: dict[str, object], neural: dict[str, object]) -> tuple[dict[str, object], dict[str, object], dict[int, list[Candidate]]]:
    gold = {row.id: set(row.labels) for row in auto.train_rows}
    options: list[dict[str, object]] = []
    options.append({"name": "rule_only", "weights": [1.0], "candidates": baseline["oof"]})
    options.append({"name": "neural_only", "weights": [1.0], "candidates": neural["oof"]})
    for rule_weight in [0.2, 0.4, 0.6, 0.8]:
        merged = merge_candidate_maps([baseline["oof"], neural["oof"]], weights=[rule_weight, 1.0 - rule_weight], average_present=True, bonus=0.015)
        options.append({"name": f"fusion_rule_{rule_weight:.1f}", "weights": [rule_weight, 1.0 - rule_weight], "candidates": merged})
    for option in options:
        option["selected"] = select_threshold(option["candidates"], gold)
        selected = option["selected"]
        auto._record_trial({"name": option["name"], "model": "fusion", "stage": "OOF calibration", "precision": selected["score"].precision, "recall": selected["score"].recall, "f1": selected["score"].f1, "threshold": selected["threshold"], "implicit_threshold": selected["implicit_threshold"], "seconds": 0.0, "config": str(option["weights"])})
    best = max(options, key=lambda item: float(item["selected"]["score"].f1))
    baseline_test = auto.fit_statistical_test(baseline)
    neural_test = merge_candidate_maps(neural.get("test_maps", []), average_present=True, bonus=0.015)
    if best["name"] == "rule_only":
        final_candidates = baseline_test
    elif best["name"] == "neural_only":
        final_candidates = neural_test
    else:
        final_candidates = merge_candidate_maps([baseline_test, neural_test], weights=best["weights"], average_present=True, bonus=0.015)
    return best, best["selected"], final_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", default="/Users/cc/Downloads/初赛训练数据 2019-08-01.zip")
    parser.add_argument("--test-zip", default="/Users/cc/Downloads/初赛测试数据 2019-08-15.zip")
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--screen-epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=4)
    parser.add_argument("--evolution-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    auto = Autopilot(args)
    try:
        auto.precheck()
        auto.stage("RESUME_STRICT", "读取已完成的 3/5-fold OOF 和 checkpoint，跳过已完成训练")
        baseline3 = _load_trial(auto, "baseline_3fold_v1")
        baseline5 = _load_trial(auto, "baseline_5fold_confirm")
        neural3 = _load_trial(auto, "neural_3fold_rbt3")
        neural5 = _load_trial(auto, "neural_5fold_confirm")
        auto.champion_f1 = max(float(item["record"]["f1"]) for item in [baseline3, baseline5, neural3, neural5])
        auto.dashboard.update(metrics={"champion_f1": auto.champion_f1})

        auto.stage("ERROR_ANALYSIS", "恢复后执行四元组级错误分解")
        errors = {}
        gold = {row.id: set(row.labels) for row in auto.train_rows}
        for label, item in [("baseline_3fold", baseline3), ("neural_3fold", neural3), ("baseline_5fold", baseline5), ("neural_5fold", neural5)]:
            errors[label] = error_analysis(gold, item["selected"]["predictions"])
        (auto.reports / "error_analysis.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

        auto.stage("RECOVER_TEST_ENSEMBLE", "从 5-fold checkpoint 恢复测试候选并构建 ensemble")
        _recover_neural_test(auto, neural5)
        auto.stage("FUSION", "在已完成的 5-fold OOF 上校准规则 / neural 融合")
        best, selected, final_candidates = _fusion(auto, baseline5, neural5)
        auto.stage("ENSEMBLE", f"恢复后的融合 champion={best['name']} OOF F1={selected['score'].f1:.4f}")
        auto._write_submission(threshold_predictions(final_candidates, selected["threshold"], selected["implicit_threshold"]), selected, best, baseline5, neural5)
        auto.stage("FINALIZE", "恢复流程完成；默认不再运行低收益 seed-only challenger")
        active = auto.active_selected or selected
        auto.dashboard.final(metrics={"precision": active["score"].precision, "recall": active["score"].recall, "f1": active["score"].f1, "champion_f1": auto.champion_f1}, submission=auto.last_submission or {}, message=f"Final Result.csv 已生成并通过严格校验；OOF strict F1={active['score'].f1:.4f}")
    except Exception as error:
        auto.dashboard.event(f"resume top-level failure: {type(error).__name__}: {str(error)[:220]}")
        auto.dashboard.update(status="failed", stage="FAILED", message=f"恢复流程失败，但已保留中间 artifacts：{type(error).__name__}")
        (auto.reports / "resume_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            auto.finalize_static_dashboard()
        except Exception:
            pass


if __name__ == "__main__":
    main()
