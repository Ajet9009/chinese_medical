#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据脱敏工具：对文本中的 PII 进行自动替换。

手机号、身份证号、邮箱 → [REDACTED]
通过 LANGFUSE_UNMASK=true 可跳过脱敏。
"""

from __future__ import annotations

import os
import re

# 中国手机号：1 开头，第二位 3-9，共 11 位（不用 \b，避免中文与数字之间无边界）
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# 身份证号：18 位（最后一位可能 X）或 15 位
_ID_RE = re.compile(r"(?<!\d)\d{15}(?:\d{2}[0-9Xx])?(?!\d)")

# 邮箱（ASCII，避免 \w 把中文算进本地部分）
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _should_skip() -> bool:
    return os.getenv("LANGFUSE_UNMASK", "").strip().lower() == "true"


def sanitize(text: str) -> str:
    """对文本中的手机号、身份证号、邮箱替换为 [REDACTED]。

    可通过环境变量 LANGFUSE_UNMASK=true 跳过脱敏。
    """
    if not text:
        return text
    if _should_skip():
        return text
    text = _PHONE_RE.sub("[REDACTED]", text)
    text = _ID_RE.sub("[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED]", text)
    return text
