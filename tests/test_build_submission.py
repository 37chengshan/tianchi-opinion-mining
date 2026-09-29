import csv
import json
from pathlib import Path

from opinion_mining.analysis import save_candidates
from opinion_mining.baseline import Candidate
from opinion_mining.data import Quadruple
from scripts.build_submission import build_submission


def test_build_submission_writes_validated_csv(tmp_path):
    test_reviews = Path(__file__).resolve().parents[1] / "artifacts/data/test/TEST/Test_reviews.csv"
    with test_reviews.open(encoding="utf-8-sig", newline="") as handle:
        first = next(iter(csv.DictReader(handle)))
    review_id = int(first["id"])
    text = first["Reviews"]

    quad = Quadruple("_", text[:2], "整体", "正面")
    candidates = tmp_path / "test_candidates.json"
    save_candidates(candidates, {review_id: [Candidate(quad, 0.9)]})
    output = tmp_path / "Result.csv"

    report = build_submission(
        [("model", candidates)],
        output_path=output,
        explicit_threshold=0.5,
        implicit_threshold=0.5,
        weights=None,
        bonus=0.0,
    )

    rows = [row for row in csv.reader(output.read_text(encoding="utf-8").splitlines())]
    assert len(rows) == report["validation"]["rows"]
    assert report["validation"]["ids"] == 2237
    assert report["predicted_quadruples"] >= 1
    assert report["sha256"]
    assert any(row[0] == str(review_id) for row in rows)
