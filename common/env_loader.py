"""Load environment files: common/.env first, then repo-root .env (overrides)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

COMMON_DIR = Path(__file__).resolve().parent
ROOT_DIR = COMMON_DIR.parent
COMMON_ENV = COMMON_DIR / ".env"
ROOT_ENV = ROOT_DIR / ".env"


def load_app_env() -> None:
    """步骤：01 合并 common/.env 与根目录 .env（根目录覆盖同名键），但不覆盖已有进程环境。

    pytest monkeypatch / 系统环境变量优先，避免测试把黄金集写进仓库文件。
    """
    merged: dict[str, str] = {}
    if COMMON_ENV.is_file():
        for key, value in dotenv_values(COMMON_ENV).items():
            if key and value is not None:
                merged[str(key)] = str(value)
    if ROOT_ENV.is_file():
        for key, value in dotenv_values(ROOT_ENV).items():
            if key and value is not None:
                merged[str(key)] = str(value)
    for key, value in merged.items():
        os.environ.setdefault(key, value)
