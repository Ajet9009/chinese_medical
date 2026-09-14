"""运行日志：控制台 + data/logs/app.log（按大小轮转）。

对齐电网 grid-qa 的 setup_logging 落盘方式，用标准库，不引入 loguru。
TESTING=1 时默认不写文件，避免测试污染 data/。
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False
_file_handler: RotatingFileHandler | None = None


def _testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes")


def default_log_dir() -> Path:
    raw = (os.getenv("LOG_DIR") or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent.parent / "data" / "logs"


def reset_logging() -> None:
    """测试用：拆掉本模块装上的 handler。"""
    global _configured, _file_handler
    root = logging.getLogger()
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    _configured = False


def setup_logging(*, force: bool = False, directory: Path | None = None) -> Path | None:
    """初始化 root logger。返回文件目录；仅 stderr 时返回 None。"""
    global _configured, _file_handler
    if _configured and not force:
        if _file_handler is not None:
            return Path(_file_handler.baseFilename).parent
        return None
    if force:
        reset_logging()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_stream = any(
        type(h) is logging.StreamHandler for h in root.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)

    dest: Path | None = None
    write_file = directory is not None or not _testing()
    if write_file:
        dest = Path(directory) if directory is not None else default_log_dir()
        dest.mkdir(parents=True, exist_ok=True)
        _file_handler = RotatingFileHandler(
            dest / "app.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        _file_handler.setFormatter(formatter)
        root.addHandler(_file_handler)

    _configured = True
    logging.getLogger("app").info("日志系统初始化完成 dest=%s", dest or "stderr-only")
    return dest


def clip_text(text: str, limit: int = 80) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def log_ask(
    event: str,
    *,
    request_id: str,
    conv_id: str = "",
    user: str = "",
    question: str = "",
    **fields: object,
) -> None:
    extra = " ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, ""))
    logging.getLogger("ask").info(
        "%s request_id=%s conv=%s user=%s q=%s%s",
        event,
        request_id,
        conv_id or "-",
        user or "-",
        clip_text(question),
        f" {extra}" if extra else "",
    )
