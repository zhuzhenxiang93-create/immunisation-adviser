"""
api/jwt_utils.py — JWT access token helpers (PyJWT).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "imac-adviser-change-this-in-production")
_ALGORITHM = "HS256"
_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def verify_token(token: str) -> Optional[int]:
    """Returns user_id on success, None if invalid/expired."""
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        user_id_str = payload.get("sub")
        return int(user_id_str) if user_id_str else None
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
