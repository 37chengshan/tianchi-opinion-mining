"""Contract tests for scripts/fetch_model.py (ModelScope / Hugging Face fetch)."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_model", ROOT / "scripts" / "fetch_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fetch_model():
    return _load_module()


def _write_model_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (path / "vocab.txt").write_text("的\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\x00" * 32)


def test_known_backbones_resolve_to_modelscope_mirrors(fetch_model):
    assert fetch_model.modelscope_id_for("hfl/rbt3") == "dienstag/rbt3"
    assert (
        fetch_model.modelscope_id_for("hfl/chinese-roberta-wwm-ext")
        == "dienstag/chinese-roberta-wwm-ext"
    )
    assert fetch_model.modelscope_id_for("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_unknown_model_without_override_is_rejected(fetch_model):
    with pytest.raises(ValueError):
        fetch_model.modelscope_id_for("some/unknown-checkpoint")
    assert (
        fetch_model.modelscope_id_for("some/unknown-checkpoint", "vendor/mirror")
        == "vendor/mirror"
    )


def test_inspect_dir_flags_incomplete_downloads(fetch_model, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = fetch_model.inspect_dir(empty)
    assert report["problems"]
    assert report["has_config"] is False


def test_inspect_dir_accepts_complete_model(fetch_model, tmp_path):
    complete = tmp_path / "complete"
    _write_model_dir(complete)
    report = fetch_model.inspect_dir(complete)
    assert report["problems"] == []
    assert report["weight_files"] == ["model.safetensors"]
    assert report["has_tokenizer"] is True
    assert report["total_bytes"] > 0


def test_main_downloads_via_modelscope(fetch_model, tmp_path, monkeypatch, capsys):
    fixture = tmp_path / "fixture"
    _write_model_dir(fixture)
    out = tmp_path / "models"
    calls = {}

    def fake_snapshot_download(repo_id, local_dir):
        calls["repo_id"] = repo_id
        pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
        for item in fixture.iterdir():
            (pathlib.Path(local_dir) / item.name).write_bytes(item.read_bytes())
        return str(local_dir)

    monkeypatch.setattr(fetch_model, "fetch_modelscope", fake_snapshot_download)

    code = fetch_model.main(["--model", "hfl/rbt3", "--out", str(out / "rbt3")])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert calls["repo_id"] == "dienstag/rbt3"
    assert payload["status"] == "downloaded"
    assert payload["resolved_id"] == "dienstag/rbt3"
    assert payload["problems"] == []


def test_main_reuses_existing_directory_without_fetching(fetch_model, tmp_path, monkeypatch, capsys):
    out = tmp_path / "cached"
    _write_model_dir(out)

    def explode(*_args, **_kwargs):
        raise AssertionError("must not download when the directory is complete")

    monkeypatch.setattr(fetch_model, "fetch_modelscope", explode)

    code = fetch_model.main(["--model", "hfl/rbt3", "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "cached"
    assert payload["path"] == str(out)


def test_hf_source_does_not_rewrite_the_id(fetch_model, tmp_path, monkeypatch, capsys):
    fixture = tmp_path / "fixture"
    _write_model_dir(fixture)
    seen = {}

    def fake_hf(model, out):
        seen["model"] = model
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        for item in fixture.iterdir():
            (pathlib.Path(out) / item.name).write_bytes(item.read_bytes())
        return str(out)

    monkeypatch.setattr(fetch_model, "fetch_hf", fake_hf)

    code = fetch_model.main(
        ["--model", "hfl/rbt3", "--source", "hf", "--out", str(tmp_path / "hf_rbt3")]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert seen["model"] == "hfl/rbt3"
    assert payload["resolved_id"] == "hfl/rbt3"
