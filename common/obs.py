"""Lightweight degraded counters. Never raises; never talks to the network."""

from __future__ import annotations

import logging
from collections import Counter
from threading import Lock
from typing import Any

logger = logging.getLogger("obs")

_lock = Lock()
_counts: Counter[str] = Counter()
_last: dict[str, str] = {}


def reset_obs() -> None:
    """Clear in-memory counts (tests)."""
    with _lock:
        _counts.clear()
        _last.clear()


def degraded(tag: str, exc: BaseException | None = None, msg: str = "") -> None:
    """Record a fail-open fallback. Must not raise, even if logging fails."""
    try:
        name = str(tag or "unknown").strip() or "unknown"
        if msg:
            text = str(msg)
        elif exc is not None:
            text = f"{type(exc).__name__}: {exc}"
        else:
            text = ""
        with _lock:
            _counts[name] += 1
            _last[name] = text[:500]
        logger.warning("degraded tag=%s %s", name, text)
    except Exception:
        return


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "total": int(sum(_counts.values())),
            "byTag": dict(_counts),
            "last": dict(_last),
        }
