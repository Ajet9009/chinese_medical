#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vllm 部署的 Qwen2.5-1.5B LoRA 微调模型客户端。

腾讯云 CloudStudio 部署，OpenAI 兼容 API (vllm)。
用法: python _007_fine_tune/vllm_client.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

BASE_URL = "https://32c156fa14de4132ada97c5e439a68bc--8000.ap-shanghai2.cloudstudio.club"
MODEL_NAME = "export/qwen2.5-merged0809"


def list_models() -> list[str]:
    """列出可用模型。"""
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", [])]


def chat(prompt: str, system: str = "", temperature: float = 0.1, max_tokens: int = 256) -> dict[str, Any]:
    """调用 vllm chat completion。

    Returns: {"content": "...", "model": "...", "tokens": {"prompt": N, "completion": N}, "elapsed_ms": N}
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = (time.time() - t0) * 1000

    choice = body["choices"][0]
    usage = body.get("usage", {})
    return {
        "content": choice["message"]["content"].strip(),
        "model": body.get("model", "unknown"),
        "tokens": {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        },
        "elapsed_ms": round(elapsed, 1),
    }


def stream_chat(prompt: str, system: str = "", temperature: float = 0.1, max_tokens: int = 256):
    """流式调用 vllm chat completion，逐 token 打印。"""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass


# ============================================================
# main() — 直接运行验证
# ============================================================


def main():
    print("=" * 64)
    print("  vllm Qwen2.5 1.5B LoRA 微调模型 — 客户端验证")
    print("=" * 64)
    print()

    # 1) 列出模型
    print("  1) 模型列表...", end=" ", flush=True)
    try:
        models = list_models()
        print(f"{len(models)} 个: {models}")
    except Exception as exc:
        print(f"失败: {exc}")
        return
    print()

    # 2) 中医问答（非流式）
    test_questions = [
        ("四君子汤有什么功效？", "你是专业的中医知识助手，回答简洁准确。"),
        ("人参的性味归经是什么？", "你是专业的中医知识助手，回答简洁准确。"),
        ("肾虚用什么方剂调理？", "你是专业的中医知识助手，回答简洁准确。"),
    ]

    print("  2) 非流式调用...")
    for i, (q, sys_prompt) in enumerate(test_questions, 1):
        try:
            result = chat(q, system=sys_prompt)
            tokens = result["tokens"]
            print(f"  [{i}] Q: {q}")
            print(f"      A: {result['content'][:120]}")
            print(f"      tokens: prompt={tokens['prompt']} completion={tokens['completion']} "
                  f"elapsed={result['elapsed_ms']}ms")
        except Exception as exc:
            print(f"  [{i}] 失败: {exc}")
        print()

    # 3) 流式调用
    print("  3) 流式调用...")
    q = "感冒了怎么辨证？"
    print(f"  Q: {q}")
    print("  A: ", end="", flush=True)
    try:
        for token in stream_chat(q, system="你是专业的中医知识助手，回答简洁准确。"):
            print(token, end="", flush=True)
        print()
    except Exception as exc:
        print(f"\n  流式失败: {exc}")
    print()

    print("=" * 64)
    print("  验证完成。")
    print("=" * 64)


if __name__ == "__main__":
    main()
