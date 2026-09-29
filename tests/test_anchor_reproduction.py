from pathlib import Path

import pytest

from scripts.reproduce_anchor import load_manifest, reproduce_anchor


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts/champions/anchor_rbt3_v0/manifest.json"
EXPECTED_PATH = ROOT / "artifacts/submissions/candidates/candidate-f9eac4b6d4ea-b384f88e9621/Result.csv"


def test_anchor_manifest_freezes_online_and_local_scores_separately():
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.candidate_id == "candidate-f9eac4b6d4ea-b384f88e9621"
    assert manifest.expected_sha256 == "f9eac4b6d4ea769e13ea62cac9d3fa9364a70ddf9a0297c8332cc7a80d40d43f"
    assert manifest.local_oof_f1 == 0.7208976157082748
    assert manifest.online_f1 == 0.7162020541
    assert manifest.local_oof_f1 != manifest.online_f1


@pytest.mark.skipif(
    not all(
        (ROOT / f"artifacts/experiments/neural_5fold_confirm/fold_{index}/model.pt").is_file()
        and (ROOT / f"artifacts/experiments/neural_5fold_confirm/fold_{index}/test_candidates.json").is_file()
        for index in range(1, 6)
    ),
    reason="archived fold checkpoints/candidate maps are not part of the repository (see docs/experiment-history-2026-09-29.md)",
)
def test_reproduce_anchor_replays_all_five_fold_candidates(tmp_path):
    manifest = load_manifest(MANIFEST_PATH)
    output_path = tmp_path / "Result.csv"

    report = reproduce_anchor(manifest, output_path)

    assert report.sha256 == manifest.expected_sha256
    assert report.sha256 == "f9eac4b6d4ea769e13ea62cac9d3fa9364a70ddf9a0297c8332cc7a80d40d43f"
    assert output_path.read_bytes() == EXPECTED_PATH.read_bytes()
    assert report.fold_count == 5
    assert report.ids == 2237
    assert report.rows == 5043
    assert report.predicted_quadruples == 4930
    assert report.empty_ids == 113
