"""Redis client. Connection failure returns None (fail-open)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("redis_client")

_client: Any = None
_disabled = False


def reset_redis_client() -> None:
    """Clear cached client (tests)."""
    global _client, _disabled
    _client = None
    _disabled = False


def get_redis():
    """Return a decode_responses Redis client, or None if unavailable."""
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    if not url:
        _disabled = True
        return None
    try:
        import redis as redis_lib

        client = redis_lib.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
        client.ping()
        _client = client
        return _client
    except Exception as exc:
        from common.obs import degraded

        degraded("redis", exc)
        logger.warning("Redis 不可用，会话热路径降级为 SQLite: %s", exc)
        _disabled = True
        return None


def redis_status() -> str:
    client = get_redis()
    if client is None:
        return "down"
    try:
        client.ping()
        return "ok"
    except Exception as exc:
        from common.obs import degraded

        degraded("redis", exc)
        return "down"
