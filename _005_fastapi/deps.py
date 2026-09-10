"""JWT dependencies. /health and /auth/login stay public."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request

from common.user_store import User, UserStore, default_user_store


@lru_cache(maxsize=1)
def get_users() -> UserStore:
    return default_user_store()


def get_current_user(request: Request) -> User:
    header = request.headers.get("Authorization") or ""
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = header[7:].strip()
    from common.security import ExpiredSignatureError, InvalidTokenError, decode_token

    try:
        payload = decode_token(token)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效令牌")
    user_id = str(payload.get("sub") or "")
    user = get_users().get(user_id)
    if user is None or user.status != "active":
        raise HTTPException(status_code=401, detail="用户不可用")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员")
    return user


def is_admin(user: User) -> bool:
    return user.role == "admin"
