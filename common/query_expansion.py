"""中医检索 query 低成本扩展。"""

from __future__ import annotations

from common.text_normalization import query_variants_step_02


_TERM_EXPANSIONS = {
    "治什么": ("主治", "功效", "适应证"),
    "治什麼": ("主治", "功效", "適應證"),
    "味如何": ("气味", "氣味", "性味"),
    "主补": ("主补", "主補", "补", "補", "主治"),
    "主補": ("主补", "主補", "补", "補", "主治"),
}


def expand_zhongyi_query_step_01(query: str) -> list[str]:
    """步骤 01：生成中医术语和繁简兼容 query 变体。

    参数：query 为检索问句。
    返回：去重后的扩展问句列表，首项为原始问句。
    异常：无。
    副作用：无。
    """
    variants = query_variants_step_02(query)
    expanded = list(variants)
    for item in variants:
        for needle, replacements in _TERM_EXPANSIONS.items():
            if needle not in item:
                continue
            for repl in replacements:
                expanded.append(item.replace(needle, repl))
    out: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out
