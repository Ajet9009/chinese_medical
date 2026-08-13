#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中医知识问答 — Streamlit 前端。

启动: streamlit run _000_demo/streamlit_app.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

import streamlit as st

API_URL = "http://localhost:8000"

# ── 页面配置 ──
st.set_page_config(
    page_title="中医知识问答",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 自定义 CSS（仿 DeepSeek 风格）──
st.markdown("""
<style>
    /* 整体 */
    .stApp { background: #fafbfc; }

    /* 标题栏 */
    .main-header {
        text-align: center; padding: 2rem 0 1rem 0;
        font-size: 2rem; font-weight: 700; color: #1a1a2e;
    }
    .main-header span { color: #2d6a4f; }

    /* 聊天气泡 */
    .chat-row { display: flex; margin: 0.5rem 0; }
    .user-bubble {
        background: #e8f5e9; color: #1b4332; border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.25rem; max-width: 75%; margin-left: auto;
        font-size: 0.95rem; line-height: 1.6;
    }
    .assistant-bubble {
        background: #ffffff; color: #1a1a2e; border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1.25rem; max-width: 85%; margin-right: auto;
        font-size: 0.95rem; line-height: 1.6;
        border: 1px solid #e5e7eb;
    }

    /* 思考过程折叠区 */
    .think-section {
        margin-top: 0.5rem; padding: 0.5rem 0.75rem;
        background: #f0fdf4; border-radius: 10px; border-left: 3px solid #2d6a4f;
        font-size: 0.8rem; color: #4a5568;
    }

    /* 输入区 */
    .stChatInput { padding-bottom: 1rem; }

    /* sidebar */
    section[data-testid="stSidebar"] { background: #f8faf8; }

    /* 隐藏默认元素 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── 初始化 session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    api_url = st.text_input("API 地址", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url

    if st.button("🔄 清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 🌿 中医知识问答")
    st.caption("基于知识图谱的中医智能问答系统")
    st.caption("涵盖方剂、药材、症状、疾病、功效、出处等")

    st.divider()
    st.markdown("### 📋 示例问题")
    examples = [
        "四君子汤有什么功效？",
        "人参的性味归经是什么？",
        "咳嗽应该吃什么中药？",
        "感冒了怎么辨证？",
        "黄芪和党参有什么区别？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()


# ── 标题 ──
st.markdown(
    '<div class="main-header">🌿 中医<span>知识问答</span></div>',
    unsafe_allow_html=True,
)
st.caption("基于 Neo4j 知识图谱 + LangGraph 智能推理")


# ── 消息历史 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if "details" in msg and msg["details"]:
            with st.expander("🔍 查看详情", expanded=False):
                d = msg["details"]
                cols = st.columns(2)
                with cols[0]:
                    st.metric("意图", "中医" if d.get("is_zhongyi_intent") else "普通")
                    st.metric("耗时", f"{d.get('elapsed_ms', 0):.0f}ms")
                with cols[1]:
                    st.caption(f"理由: {d.get('intent_reason', '')}")

                # 匹配实体
                me = d.get("matched_entities", {})
                non_empty = {k: v for k, v in me.items() if v}
                if non_empty:
                    st.caption("**匹配实体**")
                    for k, v in non_empty.items():
                        names = ", ".join(f"{e['name']}({e['score']:.2f})" for e in v)
                        st.caption(f"  {k}: {names}")

                # Cypher
                cyphers = d.get("cypher_queries", [])
                if cyphers:
                    st.caption(f"**Cypher 查询** ({len(cyphers)}条)")
                    for i, c in enumerate(cyphers):
                        st.code(c, language="cypher")


# ── 输入区 ──
question = st.chat_input("输入你的问题…")

# 处理侧边栏示例点击
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    # 用户消息
    st.session_state.messages.append({
        "role": "user",
        "avatar": "🧑",
        "content": question,
        "details": None,
    })

    # 调用 API
    with st.spinner("思考中…"):
        try:
            data = json.dumps({"question": question}).encode("utf-8")
            req = urllib.request.Request(
                f"{st.session_state.api_url}/ask",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

            answer = result.get("answer", "抱歉，未能获取回答。")
            is_tcm = result.get("is_zhongyi_intent", False)

            # 构建详情
            details = {
                "is_zhongyi_intent": is_tcm,
                "intent_reason": result.get("intent_reason", ""),
                "elapsed_ms": result.get("elapsed_ms", 0),
                "matched_entities": result.get("matched_entities", {}),
                "cypher_queries": result.get("cypher_queries", []),
            }

        except urllib.error.URLError as exc:
            answer = f"⚠️ 无法连接 API 服务 ({st.session_state.api_url})。请确认服务已启动。"
            details = None
        except Exception as exc:
            answer = f"⚠️ 请求失败: {exc}"
            details = None

    # AI 回复
    st.session_state.messages.append({
        "role": "assistant",
        "avatar": "🌿",
        "content": answer,
        "details": details,
    })

    st.rerun()
