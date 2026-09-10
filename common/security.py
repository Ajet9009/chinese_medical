"""Password hashing (bcrypt) and JWT. No passlib (bcrypt>=4.1 dropped __about__)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from common.env_loader import load_app_env

load_app_env()

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def jwt_secret() -> str:
    return os.getenv("JWT_SECRET") or "please-change-this-to-a-random-long-string"


def jwt_expire_minutes() -> int:
    return int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=jwt_expire_minutes())
    payload = {"sub": user_id, "username": username, "role": role, "exp": expire}
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, jwt_secret(), algorithms=[JWT_ALGORITHM])


__all__ = [
    "ExpiredSignatureError",
    "InvalidTokenError",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
