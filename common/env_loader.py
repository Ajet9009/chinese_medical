"""Load environment files: common/.env first, then repo-root .env (overrides)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

COMMON_DIR = Path(__file__).resolve().parent
ROOT_DIR = COMMON_DIR.parent
COMMON_ENV = COMMON_DIR / ".env"
ROOT_ENV = ROOT_DIR / ".env"


def load_app_env() -> None:
    """Load common/.env then repo-root .env so the root file wins."""
    load_dotenv(COMMON_ENV)
    load_dotenv(ROOT_ENV, override=True)
