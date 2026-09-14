#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把本地 fallback 提示词同步到已有 Langfuse 项目（创建新 version，标 production）。

沿用 common/.env / 根目录 .env 里的旧密钥，不新建项目。
用法: python common/create_langfuse_prompts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.env_loader import load_app_env

load_app_env()

from _004_langgraph_more_nodes.answer_generation import _ANSWER_FALLBACK
from _004_langgraph_more_nodes.cypher_generation import _CYPHER_FALLBACK
from _004_langgraph_more_nodes.doc_retrieval import _CRAG_REWRITE_FALLBACK
from _004_langgraph_more_nodes.entity_extraction import _ENTITY_EXTRACTION_FALLBACK
from _004_langgraph_more_nodes.general_response import _GENERAL_FALLBACK
from _004_langgraph_more_nodes.intent_recognition import _INTENT_FALLBACK
from _004_langgraph_more_nodes.standalone_query import _STANDALONE_FALLBACK
from common.context_compressor import _SUMMARY_FALLBACK
from common.langfuse_manager import make_langfuse_client

# 与节点文件中的 fallback 保持同一份，避免再抄一遍。
PROMPTS = {
    "intent_recognition": _INTENT_FALLBACK,
    "entity_extraction": _ENTITY_EXTRACTION_FALLBACK,
    "cypher_generation": _CYPHER_FALLBACK,
    "answer_generation": _ANSWER_FALLBACK,
    "general_response": _GENERAL_FALLBACK,
    "context_compress": _SUMMARY_FALLBACK,
    "standalone_query": _STANDALONE_FALLBACK,
    "crag_rewrite": _CRAG_REWRITE_FALLBACK,
}


def main():
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"连接到已有 Langfuse: {host}")
    print()

    client = make_langfuse_client()

    for name, content in PROMPTS.items():
        try:
            created = client.create_prompt(
                name=name,
                type="text",
                prompt=content,
                labels=["production"],
            )
            version = getattr(created, "version", "?")
            print(f"  [OK] {name}  version={version}  production")
        except Exception as exc:
            print(f"  [ERR] {name}: {exc}")

    client.flush()
    print()
    print("完成。在 Langfuse → Prompts 查看；SDK 约 60s 后拉取新 version。")
    print(f"打开 {host}")


if __name__ == "__main__":
    main()
