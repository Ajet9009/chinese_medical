#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vllm LoRA 多适配器路由器 — 根据领域自动选择 LoRA 模型。

三个 LoRA 适配器: tcm(中医) / law(法律) / cs(客服)
选择方式: 在 /v1/chat/completions 的 model 字段指定 lora 名称。

用法: python _007_fine_tune/lora_router.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Generator

BASE_URL = "https://32c156fa14de4132ada97c5e439a68bc--8000.ap-shanghai2.cloudstudio.club"
BASE_MODEL = "export/qwen2.5-merged0809"

# LoRA 名称 → 领域
LORA_MAP = {
    "tcm": "中医",
    "law": "法律",
    "cs":  "客服",
}


class LoRaClient:
    """vllm LoRA 多适配器客户端。"""

    def __init__(self, base_url: str = BASE_URL, base_model: str = BASE_MODEL):
        self.base_url = base_url.rstrip("/")
        self.base_model = base_model

    # ── 可用 LoRA 列表 ──

    def list_loras(self) -> dict[str, str]:
        """返回可用的 LoRA 名称 → 领域映射。"""
        return dict(LORA_MAP)

    # ── 非流式调用 ──

    def chat(
        self,
        prompt: str,
        system: str = "",
        lora: str = "tcm",
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """使用指定 LoRA 适配器进行对话。

        Args:
            prompt: 用户输入。
            system: 系统提示词。
            lora: LoRA 名称 (tcm/law/cs)，默认 tcm。
            temperature: 温度。
            max_tokens: 最大生成 token 数。

        Returns:
            {"content": "...", "lora": "tcm", "tokens": {...}, "elapsed_ms": N}
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": lora,  # vllm LoRA: 用 lora 名称作为 model
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        t0 = time.time()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
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
            "lora": lora,
            "tokens": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
            "elapsed_ms": round(elapsed, 1),
        }

    # ── 流式调用 ──

    def stream_chat(
        self,
        prompt: str,
        system: str = "",
        lora: str = "tcm",
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> Generator[str, None, None]:
        """流式调用，逐 token yield。"""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": lora,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
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
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    # ── 对比测试 (同一问题, 不同 LoRA) ──

    def compare(
        self, prompt: str, system: str = "", loras: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """同一问题用不同 LoRA 适配器对比输出。"""
        loras = loras or list(LORA_MAP.keys())
        results: list[dict[str, Any]] = []
        for lora_name in loras:
            result = self.chat(prompt, system=system, lora=lora_name, **kwargs)
            results.append(result)
        return results


# ============================================================
# main() — 直接运行验证
# ============================================================


def main():
    print("=" * 64)
    print("  vllm LoRA 多适配器 — 验证测试")
    print("=" * 64)
    print()

    client = LoRaClient()

    # 1) 列出可用 LoRA
    print("  可用 LoRA 适配器:")
    for name, domain in client.list_loras().items():
        print(f"    {name:8s} → {domain}")
    print()

    # 2) 中医 LoRA 测试
    print("─" * 64)
    print("  [中医 LoRA] 非流式测试")
    test_tcm = [
        ("四君子汤有什么功效？", "你是专业的中医知识助手。"),
        ("人参的性味归经是什么？", "你是专业的中医知识助手。"),
    ]
    for q, sys_prompt in test_tcm:
        try:
            r = client.chat(q, system=sys_prompt, lora="tcm")
            print(f"  Q: {q}")
            print(f"  A: {r['content'][:120]}")
            print(f"     tokens: {r['tokens']}  elapsed={r['elapsed_ms']}ms")
        except Exception as exc:
            print(f"  ERR: {exc}")
        print()

    # 3) 法律 LoRA 测试
    print("─" * 64)
    print("  [法律 LoRA] 非流式测试")
    test_law = [
        ("签订合同需要注意什么？", "你是专业的法律顾问。"),
        ("遭遇交通事故如何处理？", "你是专业的法律顾问。"),
    ]
    for q, sys_prompt in test_law:
        try:
            r = client.chat(q, system=sys_prompt, lora="law")
            print(f"  Q: {q}")
            print(f"  A: {r['content'][:120]}")
            print(f"     tokens: {r['tokens']}  elapsed={r['elapsed_ms']}ms")
        except Exception as exc:
            print(f"  ERR: {exc}")
        print()

    # 4) 客服 LoRA 测试
    print("─" * 64)
    print("  [客服 LoRA] 非流式测试")
    test_cs = [
        ("我想咨询一下会员权益", "你是专业的客服助理。"),
        ("怎么退换货？", "你是专业的客服助理。"),
    ]
    for q, sys_prompt in test_cs:
        try:
            r = client.chat(q, system=sys_prompt, lora="cs")
            print(f"  Q: {q}")
            print(f"  A: {r['content'][:120]}")
            print(f"     tokens: {r['tokens']}  elapsed={r['elapsed_ms']}ms")
        except Exception as exc:
            print(f"  ERR: {exc}")
        print()

    # 5) 对比测试 (同一问题 × 3个 LoRA)
    print("─" * 64)
    print("  [对比] 同一问题 × 3个 LoRA")
    same_q = "气滞血瘀怎么调理？"
    print(f"  Q: {same_q}")
    try:
        results = client.compare(same_q, system="请回答用户问题。")
        for r in results:
            domain = LORA_MAP.get(r["lora"], r["lora"])
            print(f"  [{domain:4s}] {r['content'][:80]}")
    except Exception as exc:
        print(f"  ERR: {exc}")
    print()

    # 6) 流式测试 (中医 LoRA)
    print("─" * 64)
    print("  [中医 LoRA] 流式测试")
    q = "感冒了怎么辨证？"
    print(f"  Q: {q}")
    print("  A: ", end="", flush=True)
    try:
        for token in client.stream_chat(q, system="你是专业的中医知识助手。", lora="tcm"):
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
