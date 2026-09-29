from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple, ReviewExample
from opinion_mining.metrics import Score
from opinion_mining.dashboard_runtime import DashboardWriter
import opinion_mining.resource_guard as resource_module
from opinion_mining.resource_guard import ResourceGuard, read_macos_memory_stats


def load_script(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_macos_telemetry_is_parsed_without_normal_defaults(monkeypatch):
    module = load_script("runtime_resource_monitor", "scripts/resource_monitor.py")

    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.stderr = ""

    outputs = {
        "memory_pressure": "System-wide memory free percentage: 15%\n",
        "sysctl": "vm.swapusage: total = 4.00G used = 512.00M free = 3.50G\n",
        "vm_stat": "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages stored in compressor: 1024\n",
    }

    def fake_run(command, **kwargs):
        return Completed(outputs[command[0]])

    monkeypatch.setattr(resource_module.subprocess, "run", fake_run)
    stats = read_macos_memory_stats()
    assert stats["pressure"] == "warn"
    assert stats["pressure_free_percent"] == 15.0
    assert stats["swap_used_gb"] == pytest.approx(0.5)
    assert stats["swap_total_gb"] == pytest.approx(4.0)
    assert stats["compressed_gb"] == pytest.approx(4096 * 1024 / 2**30)
    assert module.system_stats() == stats


def test_resource_guard_degrades_on_real_pressure_signals(monkeypatch):
    guard = ResourceGuard()
    guard.snapshot = lambda: SimpleNamespace(
        pressure="critical",
        tracked_gb=11.0,
        available_gb=2.0,
        swap_growth_gb=0.8,
    )
    monkeypatch.setattr(guard, "release_cache", lambda: None)
    recommendation = guard.recommend(SimpleNamespace(batch_size=4, max_length=128, trainable_layers=3, gradient_accumulation=2))
    assert recommendation == {"batch_size": 1, "max_length": 80, "trainable_layers": 1, "gradient_accumulation": 4}


def test_resource_guard_exposes_restart_reason_for_red_state(monkeypatch):
    guard = ResourceGuard()
    guard.snapshot = lambda: SimpleNamespace(
        pressure="critical",
        tracked_gb=8.0,
        available_gb=2.5,
        swap_growth_gb=0.6,
    )
    monkeypatch.setattr(guard, "release_cache", lambda: None)

    decision = guard.runtime_decision(
        SimpleNamespace(batch_size=4, max_length=128, trainable_layers=3, gradient_accumulation=2)
    )

    assert decision["restart_required"] is True
    assert decision["reason"] == "red_memory_pressure"
    assert decision["recommendation"]["batch_size"] == 1


def test_grid_v2_cli_defaults_to_one_relation_per_pair():
    module = load_script("runtime_grid_screen_defaults", "scripts/run_grid_screen.py")
    args = module._parser().parse_args(["--output-dir", "/tmp/grid-v2-plan"])
    config = module._config_from_args(args)

    assert args.relation_top_k == 1
    assert args.span_top_k == 16
    assert args.max_span_length == 12
    assert config.relation_top_k == 1
    assert config.span_top_k == 16
    assert config.max_span_length == 12
    assert config.pair_validity_loss_weight > 0
    assert config.pair_ranking_loss_weight > 0


def test_grid_prefetch_resolves_alias_before_snapshot_download(monkeypatch):
    module = load_script("runtime_grid_screen", "scripts/run_grid_screen.py")
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *, repo_id, **kwargs: calls.append(repo_id) or "/tmp/model-cache")

    path = module._prefetch_backbone("macbert")

    assert path == "/tmp/model-cache"
    assert calls == ["hfl/chinese-macbert-base"]


def test_dashboard_writer_preserves_history_across_new_runs(tmp_path):
    root = tmp_path / "dashboard"
    first = DashboardWriter(root, budget_seconds=100.0)
    first.metric(precision=0.5, recall=0.6, f1=0.55, champion_f1=0.72)
    first.trial({"name": "macbert_fold_1", "f1": 0.55})
    first.event("first run complete")

    second = DashboardWriter(root, budget_seconds=200.0)

    assert second.state["metrics"]["champion_f1"] == pytest.approx(0.72)
    assert second.state["metrics"]["f1"] is None
    assert second.state["leaderboard"][0]["name"] == "macbert_fold_1"
    assert any("first run complete" in item for item in second.state["events"])
    assert len(second.history["f1"]) == 1
    assert len(second.history["trials"]) == 1


def test_grid_formats_human_readable_step_event():
    module = load_script("runtime_grid_step_event", "scripts/run_grid_screen.py")
    message = module._format_training_step_event(
        fold=2,
        total_folds=3,
        epoch=1,
        total_epochs=2,
        step=240,
        total_steps=1076,
        loss=0.43812,
        resource={"mps_driver_gb": 2.31, "available_gb": 3.42},
    )
    assert message == "Fold 2/3 · Epoch 1/2 · Step 240/1076 · loss 0.43812 · MPS 2.31GB · available 3.42GB"


def test_grid_progress_payload_exposes_nested_percentages_and_eta():
    module = load_script("runtime_grid_screen_progress", "scripts/run_grid_screen.py")
    progress = module._progress_payload(
        fold=2,
        total_folds=3,
        epoch=1,
        total_epochs=2,
        step=50,
        total_steps=100,
        elapsed_seconds=300.0,
    )

    assert progress["step_percent"] == pytest.approx(50.0)
    assert progress["epoch_percent"] == pytest.approx(50.0)
    assert progress["fold_percent"] == pytest.approx(25.0)
    assert progress["overall_percent"] == pytest.approx(41.666666, rel=1e-5)
    assert progress["eta_seconds"] == pytest.approx(420.0, rel=1e-5)


def test_historical_champion_recovers_previous_best(tmp_path):
    module = load_script("runtime_grid_screen_champion", "scripts/run_grid_screen.py")
    state = tmp_path / "dashboard" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"metrics": {"champion_f1": 0.55}}), encoding="utf-8")
    report = tmp_path / "artifacts" / "reports" / "autopilot_progress.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"trials": [{"f1": 0.7209}, {"f1": 0.68}]}), encoding="utf-8")

    assert module._load_historical_champion(tmp_path) == pytest.approx(0.7209)


def test_manifest_is_headerless_and_contains_complete_id_and_evidence(tmp_path):
    manifest = load_script("runtime_manifest", "scripts/submission_manifest.py")
    result = tmp_path / "Result.csv"
    result.write_text("1,_,不错,整体,正面\n1,_,_,_,_\n2,价格,便宜,价格,正面\n", encoding="utf-8")
    oof = tmp_path / "oof.json"
    oof.write_text(json.dumps({"metrics": {"f1": 0.75}}, ensure_ascii=False), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 42, "threshold": 0.7}, ensure_ascii=False), encoding="utf-8")

    created = manifest.create_candidate_manifest(result, oof, config, tmp_path / "queue", expected_test_ids=[1, 2])
    raw_manifest = (tmp_path / "queue" / "manifest.csv").read_bytes()
    first_row = next(csv.reader(raw_manifest.decode("utf-8").splitlines()))
    record = created["record"]

    assert not raw_manifest.startswith(b"\xef\xbb\xbf")
    assert first_row[0] == record["candidate_id"]
    assert first_row[0] != "candidate_id"
    assert record["test_ids"] == "1,2"
    assert record["test_id_count"] == "2"
    assert record["sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert record["oof_sha256"] == hashlib.sha256(oof.read_bytes()).hexdigest()
    assert record["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert record["leaderboard_score"] == ""

    with pytest.raises(manifest.ManifestError, match="test-id coverage mismatch"):
        manifest.create_candidate_manifest(result, oof, config, tmp_path / "missing", expected_test_ids=[1, 2, 3])


def test_latest_submission_enriches_dashboard_and_report(monkeypatch, tmp_path):
    module = load_script("runtime_autopilot", "scripts/autopilot.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    row = ReviewExample(1, "价格很好", frozenset())
    quad = Quadruple("价格", "很好", "价格", "正面")
    selected = {
        "threshold": 0.1,
        "implicit_threshold": 0.1,
        "score": Score(1.0, 1.0, 1.0, 1, 1, 1),
    }

    class Dashboard:
        def __init__(self):
            self.updates = []
            self.events = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

        def event(self, message):
            self.events.append(message)

    autopilot = object.__new__(module.Autopilot)
    autopilot.submissions = tmp_path / "submissions"
    autopilot.reports = tmp_path / "reports"
    autopilot.test_rows = [row]
    autopilot.args = SimpleNamespace(train_zip="train.zip", test_zip="test.zip", budget_hours=1.0)
    autopilot.dashboard = Dashboard()
    autopilot.champion_f1 = 1.0
    autopilot.last_submission = None
    autopilot.active_selected = None
    autopilot.active_fusion = None

    autopilot._write_submission({1: [quad]}, selected, {"name": "challenger", "weights": [1.0]}, None, {"config": {"seed": 7}})

    submission = autopilot.last_submission
    assert submission["candidate_id"]
    assert submission["config_hash"]
    assert submission["oof_sha256"]
    assert submission["local_score"] == 1.0
    assert submission["leaderboard_score"] is None
    published = [item["submission"] for item in autopilot.dashboard.updates if "submission" in item][-1]
    assert published["candidate_id"] == submission["candidate_id"]
    assert published["manifest_path"] == submission["manifest_path"]
    assert published["leaderboard_score"] is None
    report = json.loads((tmp_path / "reports" / "final_report.json").read_text(encoding="utf-8"))
    assert report["submission"]["sha256"] == submission["sha256"]
    assert report["submission"]["config_hash"] == submission["config_hash"]
