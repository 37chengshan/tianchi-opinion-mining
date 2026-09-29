import json
from pathlib import Path

from scripts.build_llm_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_llm_dataset_splits_by_canonical_fold(tmp_path):
    result = build(tmp_path, n_splits=3, valid_fold=1)
    meta = result["meta"]

    assert meta["train_rows"] == 2152
    assert meta["valid_rows"] == 1077
    assert meta["train_implicit_opinion"] > 0, "implicit-O supervision must be kept"
    train = _lines(tmp_path / "train.jsonl")
    valid = _lines(tmp_path / "valid.jsonl")
    test = _lines(tmp_path / "test.jsonl")
    assert len(train) == meta["train_rows"]
    assert len(valid) == meta["valid_rows"]
    assert len(test) == meta["valid_rows"]
    assert all(len(row["messages"]) == 3 for row in train + valid)
    assert all(len(row["messages"]) == 2 for row in test)


def test_build_llm_dataset_full_mode_covers_every_labelled_row(tmp_path):
    result = build(tmp_path / "full", full=True)

    assert result["meta"]["mode"] == "full_train"
    assert result["meta"]["train_rows"] == 3229
    assert result["meta"]["train_implicit_opinion"] > 172 - 60, "implicit-O coverage must be near complete"
    assert result["files"]["train.jsonl"] == 3229


def test_build_llm_dataset_is_deterministic_and_keeps_implicit_opinion_targets(tmp_path):
    first = build(tmp_path / "a", n_splits=3, valid_fold=1)
    second = build(tmp_path / "b", n_splits=3, valid_fold=1)

    assert (tmp_path / "a/train.jsonl").read_bytes() == (tmp_path / "b/train.jsonl").read_bytes()
    assert {k: v for k, v in first["meta"].items() if k != "name"} == {k: v for k, v in second["meta"].items() if k != "name"}
    targets = [row["messages"][-1]["content"] for row in _lines(tmp_path / "a/train.jsonl")]
    assert any('"opinion":"_"' in target for target in targets)
    for target in targets[:200]:
        parsed = json.loads(target)
        assert set(parsed) == {"quadruples"}
        for quad in parsed["quadruples"]:
            assert set(quad) == {"aspect", "opinion", "category", "polarity"}
