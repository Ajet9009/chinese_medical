#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据脱敏工具：对文本中的 PII 进行自动替换。

手机号、身份证号、邮箱 → [REDACTED]
通过 LANGFUSE_UNMASK=true 可跳过脱敏。
"""

from __future__ import annotations

import os
import re

# 中国手机号：1 开头，3-9 第二位，共 11 位
_PHONE_RE = re.compile(r"\b1[3-9]\d{9}\b")

# 身份证号：18 位（最后一位可能 X）或 15 位
_ID_RE = re.compile(r"\b\d{15}(?:\d{2}[0-9Xx])?\b")

# 邮箱
_EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w\-]+(?:\.[\w\-]+)+\b")


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
