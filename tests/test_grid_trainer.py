from pathlib import Path
from types import SimpleNamespace
import inspect

import pytest

from opinion_mining import grid_trainer as grid


def test_dummy_smoke_uses_sparse_grid_and_backpropagates():
    result = grid.dummy_smoke(max_length=12, hidden_size=8)
    assert result["status"] == "dummy_smoke_passed"
    assert result["hidden_shape"] == [1, 12, 8]
    assert result["relation_shape"] == [1, 12, 12, 39]
    assert result["loss_finite"] is True
    assert result["positives"] == 1
    assert result["hard_negatives"] > 0
    assert result["pair_validity_loss"] > 0
    assert result["pair_ranking_loss"] >= 0
    assert result["pair_relation_loss"] > 0
    assert result["implicit_presence_loss"] > 0
    assert result["trainable_parameters_with_grad"] > 0


def test_dummy_train_smoke_runs_fold_and_preserves_config(tmp_path: Path):
    result = grid.dummy_train_smoke(max_length=10, hidden_size=8, relation_rank=2, output_dir=tmp_path)
    assert result["status"] == "dummy_train_passed"
    assert result["history_length"] == 1
    assert result["valid_ids"] == [3]
    assert result["config_unchanged"] is True
    assert Path(result["checkpoint"]).is_file()
    assert (tmp_path / "history.json").is_file()


def test_final_fold_can_disable_validation_threshold_checkpoint_selection(tmp_path: Path, monkeypatch):
    calls = []
    original = grid.CompactGridEncoderAdapter.load_state_dict

    def traced(self, state_dict, *args, **kwargs):
        calls.append(True)
        return original(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(grid.CompactGridEncoderAdapter, "load_state_dict", traced)
    rows = [
        grid.ReviewExample(
            index,
            "很好用",
            label_spans=(grid.LabelSpan(grid.Quadruple("_", "很好", "整体", "正面"), opinion_start=0, opinion_end=2),),
        )
        for index in range(1, 4)
    ]
    config = grid.GridTrainConfig(max_length=10, batch_size=1, gradient_accumulation=1, epochs=2, trainable_layers=0, device="cpu")

    grid.train_grid_fold(
        rows[:2],
        rows[2:],
        config,
        guard=grid._FixedSmokeGuard(),
        output_dir=tmp_path,
        tokenizer=grid.DummyTokenizer(),
        encoder=grid.DummyEncoder(hidden_size=8),
        select_best_epoch=False,
    )

    assert calls == []


def test_grid_v2_defaults_to_one_relation_per_exact_pair():
    assert grid.GridTrainConfig().relation_top_k == 1


def test_grid_v2_deduplicates_identical_feature_pairs_but_keeps_distinct_offsets():
    rows = [[(-1, -1, 1, 2, 7), (-1, -1, 1, 2, 7), (-1, -1, 6, 7, 7)]]
    assert grid._deduplicate_feature_pairs(rows) == [[(-1, -1, 1, 2, 7), (-1, -1, 6, 7, 7)]]


def test_fold_training_defaults_to_fixed_final_epoch_weights():
    assert inspect.signature(grid.train_grid_fold).parameters["select_best_epoch"].default is False


def test_fixed_screen_plan_is_three_fold():
    rows = [grid.ReviewExample(index, f"样本{index}") for index in range(3)]
    plan = grid.fixed_screen_plan(rows, grid.GridTrainConfig(), grid._FixedSmokeGuard())
    assert plan["status"] == "plan_only"
    assert len(plan["folds"]) == 3
    assert sorted(item_id for fold in plan["folds"] for item_id in fold["valid_ids"]) == [0, 1, 2]


def test_runtime_pressure_requests_fold_restart_and_writes_emergency_checkpoint(tmp_path: Path):
    class Guard:
        def __init__(self):
            self.calls = 0

        def recommend(self, config):
            self.calls += 1
            if self.calls >= 3:
                return {
                    "batch_size": 1,
                    "max_length": 10,
                    "trainable_layers": 0,
                    "gradient_accumulation": 4,
                }
            return {
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "trainable_layers": config.trainable_layers,
                "gradient_accumulation": config.gradient_accumulation,
            }

        def snapshot(self):
            return SimpleNamespace(
                pressure="critical",
                available_gb=2.0,
                tracked_gb=2.0,
                swap_growth_gb=0.5,
            )

        def release_cache(self):
            pass

    rows = [
        grid.ReviewExample(
            index,
            "很好用",
            label_spans=(
                grid.LabelSpan(
                    quadruple=grid.Quadruple("_", "很好", "整体", "正面"),
                    opinion_start=0,
                    opinion_end=2,
                ),
            ),
        )
        for index in range(1, 5)
    ]
    config = grid.GridTrainConfig(
        max_length=12,
        batch_size=2,
        gradient_accumulation=2,
        epochs=1,
        trainable_layers=0,
        device="cpu",
    )

    with pytest.raises(grid.ResourcePressureRestart) as excinfo:
        grid.train_grid_fold(
            rows[:3],
            rows[3:],
            config,
            guard=Guard(),
            output_dir=tmp_path,
            tokenizer=grid.DummyTokenizer(),
            encoder=grid.DummyEncoder(hidden_size=8),
            resource_check_interval=1,
        )

    assert excinfo.value.recommendation["batch_size"] == 1
    assert (tmp_path / "resource_pressure_checkpoint.pt").is_file()
