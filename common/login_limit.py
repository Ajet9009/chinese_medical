"""Login rate limit (in-memory, per client IP)."""

from __future__ import annotations

import os
from collections import defaultdict, deque
from time import time

_hits: dict[str, deque[float]] = defaultdict(deque)


def reset_login_limiter() -> None:
    _hits.clear()


def login_allowed(ip: str) -> bool:
    raw = (os.getenv("LOGIN_RATE_MAX") or "").strip()
    if raw:
        max_n = int(raw)
    elif os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes"):
        return True
    else:
        max_n = 10
    if max_n <= 0:
        return True
    window = float(os.getenv("LOGIN_RATE_WINDOW") or "60")
    now = time()
    key = ip or "unknown"
    q = _hits[key]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= max_n:
        return False
    q.append(now)
    return True
