"""运行日志：轮转文件 + 问答生命周期字段。"""

from __future__ import annotations

import logging

from common.app_logging import clip_text, log_ask, reset_logging, setup_logging


def test_clip_text_keeps_short_and_truncates_long():
    assert clip_text("  四君子汤  功效  ") == "四君子汤 功效"
    long = "人参" * 50
    clipped = clip_text(long, limit=10)
    assert clipped.endswith("…")
    assert len(clipped) == 10


def test_setup_logging_writes_rotating_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TESTING", raising=False)
    reset_logging()
    try:
        dest = setup_logging(force=True, directory=tmp_path)
        assert dest == tmp_path
        logging.getLogger("obs").info("degraded_tag=demo")
        log_ask(
            "start",
            request_id="rid-1",
            conv_id="cid-1",
            user="admin",
            question="它由哪些药组成？",
            path="ask/stream",
        )
        text = (tmp_path / "app.log").read_text(encoding="utf-8")
        assert "degraded_tag=demo" in text
        assert "start request_id=rid-1 conv=cid-1 user=admin q=它由哪些药组成？ path=ask/stream" in text
    finally:
        reset_logging()


def test_testing_skips_file_unless_directory_given(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "ignored"))
    reset_logging()
    try:
        assert setup_logging(force=True) is None
        assert not (tmp_path / "ignored").exists()
        dest = setup_logging(force=True, directory=tmp_path)
        assert dest == tmp_path
        assert (tmp_path / "app.log").exists()
    finally:
        reset_logging()
