#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中医知识问答 — Streamlit 前端（流式进度 + 浅色主题）。

启动: python _006_streamlit/run.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

API_URL = "http://localhost:8000"


def _send_feedback(api_url: str, trace_id: str, feedback: str) -> None:
    """发送 👍/👎 反馈到 API，写回 Langfuse score。"""
    try:
        requests.post(
            f"{api_url}/feedback",
            json={"trace_id": trace_id, "feedback": feedback},
            timeout=10,
        )
    except Exception:
        pass  # 反馈失败不影响主流程

# ── 页面配置 ──
st.set_page_config(
    page_title="中医知识问答",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 轻量 CSS（仅标题和布局修饰，不动文字颜色）──
st.markdown("""
<style>
    .brand {
        text-align: center; padding: 1.5rem 0 0.5rem 0;
        font-size: 2.2rem; font-weight: 700;
    }
    .brand .accent { color: #2d6a4f; }
    .subtitle { text-align: center; font-size: 0.9rem; margin-bottom: 1.5rem;
                color: #6b7280; }
    section[data-testid="stSidebar"] { background: #f8faf8; border-right: 1px solid #e5e7eb; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "user_id" not in st.session_state:
    # 每次打开页面（新会话）生成独立 user_id，记忆按此隔离
    import uuid
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.session_id = st.session_state.user_id


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    api_url = st.text_input("API 地址", value=st.session_state.api_url, key="api_input")
    st.session_state.api_url = api_url

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        stream_mode = st.checkbox("流式进度", value=True, help="显示思考过程")

    st.divider()
    st.markdown("### 🌿 中医知识问答")
    st.caption("基于知识图谱 + LangGraph 的中医智能问答")
    st.caption("方剂 · 药材 · 症状 · 疾病 · 功效 · 经络 · 典籍")

    st.divider()
    st.markdown("### 📋 试试这些问题")
    examples = [
        "四君子汤有什么功效？",
        "人参的性味归经是什么？",
        "咳嗽应该吃什么中药？",
        "感冒了怎么辨证？",
        "黄芪和党参有什么区别？",
        "肾虚用什么方剂调理？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:10]}"):
            st.session_state.pending_question = ex
            st.rerun()


# ── 标题 ──
st.markdown(
    '<div class="brand">🌿 中医<span class="accent">知识问答</span></div>'
    '<div class="subtitle">基于 Neo4j 知识图谱 · LangGraph 智能推理 · BGE 向量匹配</div>',
    unsafe_allow_html=True,
)


# ── 消息历史（原生 st.chat_message，自动适配主题）──
for msg in st.session_state.messages:
    avatar = msg.get("avatar", "🧑" if msg["role"] == "user" else "🌿")
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg.get("details"):
            with st.expander("🔍 查看推理过程", expanded=False):
                d = msg["details"]
                c1, c2, c3 = st.columns(3)
                c1.metric("意图", "🌿 中医" if d.get("is_zhongyi_intent") else "💬 普通")
                c2.metric("耗时", f"{d.get('elapsed_ms', 0):.0f}ms")
                c3.metric("Cypher", f"{len(d.get('cypher_queries', []))} 条")
                st.caption(f"理由: {d.get('intent_reason', '')}")

                me = d.get("matched_entities", {})
                non_empty = {k: v for k, v in me.items() if v}
                if non_empty:
                    st.markdown("**匹配实体**")
                    for k, v in non_empty.items():
                        names = ", ".join(f"`{e['name']}` ({e['score']:.2f})" for e in v)
                        st.caption(f"  {k}: {names}")

                cyphers = d.get("cypher_queries", [])
                if cyphers:
                    st.markdown("**Cypher 查询**")
                    for i, c in enumerate(cyphers):
                        st.code(c, language="cypher")

        # 反馈按钮（仅 assistant 消息 + 有 trace_id）
        if msg["role"] == "assistant" and msg.get("trace_id"):
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            with fb_col1:
                if st.button("👍", key=f"up_{msg['trace_id']}", help="回答有帮助"):
                    _send_feedback(st.session_state.api_url, msg["trace_id"], "thumbs_up")
                    st.toast("已反馈 👍")
            with fb_col2:
                if st.button("👎", key=f"down_{msg['trace_id']}", help="回答需改进"):
                    _send_feedback(st.session_state.api_url, msg["trace_id"], "thumbs_down")
                    st.toast("已反馈 👎")


# ── 输入区 ──

question = st.chat_input("输入你的中医问题…")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    # 用户消息
    st.session_state.messages.append({
        "role": "user", "avatar": "🧑",
        "content": question, "details": None,
    })

    final_result: dict[str, Any] | None = None

    # ── 流式模式：SSE 实时进度 ──
    if stream_mode:
        t_start = time.time()
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            status_box = st.status("🤔 开始思考… ⏱ 0.0s", expanded=True)

        try:
            resp = requests.post(
                f"{st.session_state.api_url}/ask/stream",
                json={"question": question},
                headers={
                    "X-User-ID": st.session_state.user_id,
                    "X-Session-ID": st.session_state.session_id,
                },
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            current_event = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:") and current_event:
                    data = json.loads(line[5:].strip())
                    elapsed = time.time() - t_start

                    if current_event == "progress" and data.get("status") == "running":
                        status_box.update(label=f"🤔 思考中… ⏱ {elapsed:.1f}s")
                        with status_box:
                            st.write(f"🟡 {data['label']}…")
                    elif current_event == "progress" and data.get("status") == "done":
                        status_box.update(label=f"🤔 思考中… ⏱ {elapsed:.1f}s")
                        with status_box:
                            out = data.get("output", {})
                            summary = ""
                            if out:
                                parts = []
                                for k, v in out.items():
                                    if isinstance(v, list) and v:
                                        parts.append(f"{k}: {v[0]}")
                                summary = " · " + " | ".join(parts[:2]) if parts else ""
                            st.write(f"✅ {data['label']}{summary}")
                    elif current_event == "done":
                        final_result = data
                        total = time.time() - t_start
                        with status_box:
                            status_box.update(label=f"✅ 思考完成 ⏱ {total:.1f}s", state="complete", expanded=False)
                    elif current_event == "error":
                        final_result = data
                        with status_box:
                            status_box.update(label=f"❌ {data.get('error', '')}", state="error")

        except requests.ConnectionError:
            final_result = {"error": f"无法连接 API ({st.session_state.api_url})，请确认服务已启动。"}
        except Exception as exc:
            final_result = {"error": str(exc)}

        progress_placeholder.empty()

    # ── 非流式模式 ──
    else:
        try:
            resp = requests.post(
                f"{st.session_state.api_url}/ask",
                json={"question": question},
                headers={
                    "X-User-ID": st.session_state.user_id,
                    "X-Session-ID": st.session_state.session_id,
                },
                timeout=300,
            )
            resp.raise_for_status()
            final_result = resp.json()
        except requests.ConnectionError:
            final_result = {"error": f"无法连接 API ({st.session_state.api_url})，请确认服务已启动。"}
        except Exception as exc:
            final_result = {"error": str(exc)}

    # ── 构建回复 ──
    if final_result and final_result.get("error"):
        answer = f"⚠️ {final_result['error']}"
        details = None
    elif final_result:
        answer = final_result.get("answer", "抱歉，未能获取回答。")
        details = {
            "is_zhongyi_intent": final_result.get("is_zhongyi_intent", False),
            "intent_reason": final_result.get("intent_reason", ""),
            "elapsed_ms": final_result.get("elapsed_ms", 0),
            "matched_entities": final_result.get("matched_entities", {}),
            "cypher_queries": final_result.get("cypher_queries", []),
        }
    else:
        answer = "无响应"
        details = None

    st.session_state.messages.append({
        "role": "assistant", "avatar": "🌿",
        "content": answer, "details": details,
        "trace_id": final_result.get("trace_id", "") if final_result else "",
    })

    st.rerun()
