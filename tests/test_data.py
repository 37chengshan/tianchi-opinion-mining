from pathlib import Path

from opinion_mining.data import decode_implicit_terms, encode_implicit_terms, load_test_reviews, load_train_data


def test_implicit_pseudo_token_data_round_trip():
    encoded = encode_implicit_terms("_", "很好")
    assert encoded == ("[IMPLICIT_A]", "很好")
    assert decode_implicit_terms(*encoded) == ("_", "很好")


def test_load_train_data_groups_multiple_labels_per_id(tmp_path: Path):
    reviews = tmp_path / "Train_reviews.csv"
    labels = tmp_path / "Train_labels.csv"
    reviews.write_text("id,Reviews\n1,很好，超值\n2,一般\n", encoding="utf-8")
    labels.write_text(
        "id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n"
        "1,_, , ,很好,0,2,整体,正面\n"
        "1,_, , ,超值,3,5,价格,正面\n"
        "2,_, , ,一般,0,2,整体,中性\n",
        encoding="utf-8",
    )
    rows = load_train_data(reviews, labels)
    assert [r.id for r in rows] == [1, 2]
    assert rows[0].text == "很好，超值"
    assert len(rows[0].labels) == 2
    assert {x.opinion for x in rows[0].labels} == {"很好", "超值"}
    assert [(x.opinion_start, x.opinion_end) for x in rows[0].label_spans] == [(0, 2), (3, 5)]


def test_load_train_data_preserves_repeated_term_offsets(tmp_path: Path):
    reviews = tmp_path / "Train_reviews.csv"
    labels = tmp_path / "Train_labels.csv"
    reviews.write_text("id,Reviews\n1,好用不好用好用\n", encoding="utf-8")
    labels.write_text(
        "id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities\n"
        "1,_, , ,好用,0,2,整体,正面\n"
        "1,_, , ,好用,5,7,整体,正面\n",
        encoding="utf-8",
    )
    row = load_train_data(reviews, labels)[0]
    assert len(row.label_spans) == 2
    assert [(x.opinion_start, x.opinion_end) for x in row.label_spans] == [(0, 2), (5, 7)]


def test_load_test_reviews_preserves_text_and_order(tmp_path: Path):
    reviews = tmp_path / "Test_reviews.csv"
    reviews.write_text("id,Reviews\n2,后一个\n1,前一个\n", encoding="utf-8")
    rows = load_test_reviews(reviews)
    assert [(r.id, r.text) for r in rows] == [(2, "后一个"), (1, "前一个")]
