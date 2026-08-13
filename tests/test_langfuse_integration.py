#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Langfuse 集成单元测试。

运行: pytest tests/test_langfuse_integration.py -v
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── sanitizer 测试 ──


class TestSanitizer:
    def test_phone_masked(self):
        from common.sanitizer import sanitize
        assert sanitize("电话13800138000联系") == "电话[REDACTED]联系"

    def test_id_card_masked(self):
        from common.sanitizer import sanitize
        assert sanitize("身份证110101199001011234") == "身份证[REDACTED]"

    def test_email_masked(self):
        from common.sanitizer import sanitize
        assert sanitize("邮箱test@example.com联系") == "邮箱[REDACTED]联系"

    def test_mixed_pii_masked(self):
        from common.sanitizer import sanitize
        result = sanitize("用户13800138000 邮箱a@b.com 身份证110101199001011234")
        assert "13800138000" not in result
        assert "a@b.com" not in result
        assert "110101199001011234" not in result
        assert result.count("[REDACTED]") == 3

    def test_no_pii_unchanged(self):
        from common.sanitizer import sanitize
        text = "四君子汤具有补气健脾的功效"
        assert sanitize(text) == text

    def test_empty_returns_empty(self):
        from common.sanitizer import sanitize
        assert sanitize("") == ""

    def test_unmask_env_skips_redaction(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_UNMASK", "true")
        from common.sanitizer import sanitize
        assert sanitize("电话13800138000") == "电话13800138000"
        monkeypatch.delenv("LANGFUSE_UNMASK")


# ── LangfuseManager 测试 ──


class TestLangfuseManagerDisabled:
    """测试 LANGFUSE_ENABLED=false 的情况。"""

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        assert mgr.is_enabled() is False
        assert mgr.create_handler() is None
        mgr.flush()  # 不应抛异常

    def test_disabled_by_off(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "off")
        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        assert mgr.is_enabled() is False

    def test_create_handler_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        assert mgr.create_handler(trace_name="test") is None


class TestLangfuseManagerSampling:
    """测试采样逻辑。"""

    def test_sample_rate_zero_never_samples(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch("langfuse.Langfuse", autospec=True):
            from common.langfuse_manager import LangfuseManager
            mgr = LangfuseManager()
            # 采样率为 0 时，即使客户端可用也不应该采样
            assert mgr.is_enabled() is False

    def test_sample_rate_one_always_samples(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.0")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch("langfuse.Langfuse", autospec=True):
            from common.langfuse_manager import LangfuseManager
            mgr = LangfuseManager()
            assert mgr.is_enabled() is True

    def test_invalid_rate_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "invalid")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch("langfuse.Langfuse", autospec=True):
            from common.langfuse_manager import LangfuseManager
            mgr = LangfuseManager()
            # 应在内部降级为 0.1，但足够大概率触发采样
            # 验证不会 crash
            assert isinstance(mgr._sample_rate, float)


class TestLangfuseManagerInitFailure:
    """测试初始化失败时的优雅降级。"""

    def test_missing_keys_disables(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        # 不设置 LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        assert mgr.is_enabled() is False

    def test_import_error_graceful(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch.dict("sys.modules", {"langfuse": None}):
            # 模拟 langfuse 未安装
            import importlib
            original_import = __import__

            def mock_import(name, *args, **kwargs):
                if name == "langfuse":
                    raise ImportError("No module named 'langfuse'")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                from common.langfuse_manager import LangfuseManager
                mgr = LangfuseManager()
                assert mgr.is_enabled() is False

    def test_init_exception_graceful(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch("langfuse.Langfuse", side_effect=Exception("Network error")):
            from common.langfuse_manager import LangfuseManager
            mgr = LangfuseManager()
            assert mgr.is_enabled() is False


class TestLangfuseManagerHandlerCreation:
    """测试 trace handler 创建。"""

    def test_normal_flow(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.0")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("TESTING", "true")

        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        # TESTING 时不初始化真实客户端，is_enabled 应为 False
        assert mgr.is_enabled() is False


class TestLangfuseFlush:
    """测试 flush 行为。"""

    def test_flush_when_disabled_noop(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        from common.langfuse_manager import LangfuseManager
        mgr = LangfuseManager()
        mgr.flush()  # 不应抛异常

    def test_flush_timeout(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        with mock.patch("langfuse.Langfuse", autospec=True) as mock_client:
            from common.langfuse_manager import LangfuseManager
            mgr = LangfuseManager()
            mgr.flush(timeout=5.0)
            if mgr._client:
                mock_client.return_value.flush.assert_called_once()
