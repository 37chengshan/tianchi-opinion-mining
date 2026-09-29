import pytest

from opinion_mining.data import Quadruple
from opinion_mining.llm_parse import LLMParseError, parse_llm_output


REVIEW = "包装很精致，但是价格太贵了"


def test_parse_plain_json_answer():
    text = '{"quadruples":[{"aspect":"包装","opinion":"很精致","category":"包装","polarity":"正面"},{"aspect":"价格","opinion":"太贵了","category":"价格","polarity":"负面"}]}'

    quads = parse_llm_output(text, REVIEW)

    assert quads == [
        Quadruple("包装", "很精致", "包装", "正面"),
        Quadruple("价格", "太贵了", "价格", "负面"),
    ]


def test_parse_accepts_code_fence_and_surrounding_prose():
    text = '这是结果：\n```json\n{"quadruples":[{"aspect":"_","opinion":"太贵了","category":"价格","polarity":"负面"}]}\n```\n以上。'

    assert parse_llm_output(text, REVIEW) == [Quadruple("_", "太贵了", "价格", "负面")]


def test_parse_drops_illegal_and_hallucinated_items_but_keeps_valid_ones():
    text = (
        '{"quadruples":['
        '{"aspect":"包装","opinion":"很精致","category":"包装","polarity":"正面"},'
        '{"aspect":"包装","opinion":"很精致","category":"尺寸学","polarity":"正面"},'
        '{"aspect":"包装","opinion":"很精致","category":"包装","polarity":"很好"},'
        '{"aspect":"赠品","opinion":"很精致","category":"价格","polarity":"正面"},'
        '{"aspect":"包装","opinion":"不存在的话","category":"包装","polarity":"正面"}'
        "]}"
    )

    assert parse_llm_output(text, REVIEW) == [Quadruple("包装", "很精致", "包装", "正面")]


def test_parse_keeps_implicit_opinion_and_drops_dual_implicit():
    text = (
        '{"quadruples":['
        '{"aspect":"包装","opinion":"","category":"包装","polarity":"正面"},'
        '{"aspect":"_","opinion":"_","category":"整体","polarity":"正面"}'
        "]}"
    )

    assert parse_llm_output(text, REVIEW) == [Quadruple("包装", "_", "包装", "正面")]


def test_parse_raises_on_missing_json():
    with pytest.raises(LLMParseError):
        parse_llm_output("抱歉，我无法完成。", REVIEW)


def test_parse_is_deterministic_and_ordered_by_opinion_position():
    text = (
        '{"quadruples":['
        '{"aspect":"价格","opinion":"太贵了","category":"价格","polarity":"负面"},'
        '{"aspect":"包装","opinion":"很精致","category":"包装","polarity":"正面"}'
        "]}"
    )

    assert parse_llm_output(text, REVIEW) == [
        Quadruple("包装", "很精致", "包装", "正面"),
        Quadruple("价格", "太贵了", "价格", "负面"),
    ]
