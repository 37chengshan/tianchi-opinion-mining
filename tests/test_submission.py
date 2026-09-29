from pathlib import Path

import pytest

from opinion_mining.data import Quadruple, ReviewExample
from opinion_mining.submission import SubmissionError, validate_submission, write_submission


def reviews():
    return [ReviewExample(2, "物流神速", frozenset()), ReviewExample(1, "价格很便宜", frozenset())]


def test_writer_sorts_ids_deduplicates_and_fills_empty(tmp_path: Path):
    out = tmp_path / "Result.csv"
    quad = Quadruple("价格", "很便宜", "价格", "正面")
    write_submission(out, reviews(), {1: [quad, quad]})
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").splitlines() == ["1,价格,很便宜,价格,正面", "2,_,_,_,_"]
    report = validate_submission(out, reviews())
    assert report.ids == 2 and report.rows == 2 and report.empty_ids == 1


def test_writer_rejects_unknown_category(tmp_path: Path):
    with pytest.raises(SubmissionError):
        write_submission(tmp_path / "x.csv", reviews(), {1: [Quadruple("价格", "很便宜", "不存在", "正面")]})


def test_writer_rejects_non_substring_aspect_or_opinion(tmp_path: Path):
    with pytest.raises(SubmissionError):
        write_submission(tmp_path / "x.csv", reviews(), {1: [Quadruple("物流", "很便宜", "价格", "正面")]})
    with pytest.raises(SubmissionError):
        write_submission(tmp_path / "x.csv", reviews(), {1: [Quadruple("价格", "超级便宜", "价格", "正面")]})


def test_validator_rejects_header_and_missing_id(tmp_path: Path):
    out = tmp_path / "bad.csv"
    out.write_text("id,AspectTerm,OpinionTerm,Category,Polarity\n1,价格,很便宜,价格,正面\n", encoding="utf-8")
    with pytest.raises(SubmissionError):
        validate_submission(out, reviews())
