from __future__ import annotations

import json

from .data import Quadruple, ReviewExample


SYSTEM_PROMPT = """你是中文电商评论观点四元组抽取器。
任务：给定一条化妆品评论，抽取其中所有观点四元组 (aspect, opinion, category, polarity)。

规则：
1. aspect：原文中出现的属性词，必须与原文完全一致；若评论没有显式属性词，用 "_"。
2. opinion：原文中连续的观点片段，必须与原文完全一致；若评论只表达整体判断而没有显式观点词，用 "_"。
3. category 只能取：包装, 成分, 尺寸, 服务, 功效, 价格, 气味, 使用体验, 物流, 新鲜度, 真伪, 整体, 其他。
4. polarity 只能取：正面, 中性, 负面。
5. 一条评论可以有多个四元组，不要合并，不要生成重复四元组。
6. 不要臆造原文中不存在的词。
7. 只输出一个 JSON 对象，形如 {"quadruples":[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}]}；没有观点时输出 {"quadruples":[]}。
8. 不输出任何解释、前后缀或多余文本。"""


def _position(text: str, term: str) -> int:
    if term == "_":
        return 10**9
    index = text.find(term)
    return index if index >= 0 else 10**9 - 1


def ordered_quadruples(row: ReviewExample) -> list[Quadruple]:
    """Deterministic target order: opinion position, then aspect position."""
    return sorted(
        set(row.labels),
        key=lambda quad: (
            _position(row.text, quad.opinion),
            _position(row.text, quad.aspect),
            quad.category,
            quad.polarity,
        ),
    )


def target_json(row: ReviewExample) -> str:
    payload = {
        "quadruples": [
            {"aspect": quad.aspect, "opinion": quad.opinion, "category": quad.category, "polarity": quad.polarity}
            for quad in ordered_quadruples(row)
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def user_prompt(row: ReviewExample) -> str:
    return f"评论ID: {row.id}\n评论文本: {row.text}\n只输出一个 JSON 对象。"


def chat_record(row: ReviewExample, *, supervised: bool) -> dict[str, object]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(row)},
    ]
    if supervised:
        messages.append({"role": "assistant", "content": target_json(row)})
    return {"messages": messages}
