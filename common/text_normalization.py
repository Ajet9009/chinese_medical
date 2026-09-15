"""中医 RAG 文本繁简归一工具。"""

from __future__ import annotations


_S2T_CHARS = {
    "医": "醫",
    "药": "藥",
    "汤": "湯",
    "气": "氣",
    "证": "證",
    "伤": "傷",
    "阳": "陽",
    "阴": "陰",
    "经": "經",
    "脉": "脈",
    "脏": "臟",
    "腑": "腑",
    "风": "風",
    "湿": "濕",
    "热": "熱",
    "寒": "寒",
    "症": "癥",
    "与": "與",
    "为": "為",
    "无": "無",
    "发": "發",
    "补": "補",
    "泻": "瀉",
    "体": "體",
    "疗": "療",
    "参": "參",
    "么": "麼",
    "黄": "黃",
}
_T2S_CHARS = {v: k for k, v in _S2T_CHARS.items()}


def normalize_text_step_01(text: str, target: str = "simplified") -> str:
    """步骤 01：把文本归一为简体或繁体。

    参数：text 为原始文本；target 支持 simplified / traditional。
    返回：转换后的文本。
    异常：target 非法时抛 ValueError。
    副作用：首次可尝试导入 OpenCC；缺依赖时使用内置常见中医字表。
    """
    raw = str(text or "")
    if target not in {"simplified", "traditional"}:
        raise ValueError("target 仅支持 simplified / traditional")
    try:
        from opencc import OpenCC

        config = "t2s" if target == "simplified" else "s2t"
        return OpenCC(config).convert(raw)
    except Exception:
        # OpenCC 是可选依赖；缺失时使用常见中医高频字兜底，保证主路径不失败。
        table = _T2S_CHARS if target == "simplified" else _S2T_CHARS
        return raw.translate(str.maketrans(table))


def query_variants_step_02(query: str) -> list[str]:
    """步骤 02：生成原文、简体、繁体 query 变体并去重。

    参数：query 为用户检索问句。
    返回：按优先级排序的 query 变体。
    异常：无。
    副作用：无。
    """
    raw = str(query or "").strip()
    if not raw:
        return []
    variants = [
        raw,
        normalize_text_step_01(raw, "simplified"),
        normalize_text_step_01(raw, "traditional"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out
