"""
api/auth_manager.py — SQLite-backed user authentication.

Users table: id, username, email, salt, password_hash, created_at
Password security: SHA-256 + 32-char random hex salt per user
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path
from typing import Optional


class AuthManager:
    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT    UNIQUE NOT NULL,
                    email         TEXT    UNIQUE NOT NULL,
                    salt          TEXT    NOT NULL,
                    password_hash TEXT    NOT NULL,
                    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ── Password helpers ──────────────────────────────────────────────────────

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    def _verify_password(self, password: str, salt: str, stored_hash: str) -> bool:
        return self._hash_password(password, salt) == stored_hash

    # ── Public API ────────────────────────────────────────────────────────────

    def register_user(self, username: str, email: str, password: str) -> dict:
        """
        Register a new user.
        Returns {id, username, email} on success; raises ValueError on failure.
        """
        if not (2 <= len(username) <= 32):
            raise ValueError("Username must be 2–32 characters")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        if "@" not in email:
            raise ValueError("Invalid email address")

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, email, salt, password_hash) VALUES (?, ?, ?, ?)",
                    (username, email, salt, password_hash),
                )
                conn.commit()
                user_id = cursor.lastrowid
            return {"id": user_id, "username": username, "email": email}
        except sqlite3.IntegrityError as e:
            msg = str(e)
            if "username" in msg:
                raise ValueError(f"Username '{username}' is already taken")
            elif "email" in msg:
                raise ValueError(f"Email '{email}' is already registered")
            raise ValueError("Registration failed, please try again")

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """
        Verify credentials.
        Returns {id, username, email} on success; None on failure.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if row and self._verify_password(password, row["salt"], row["password_hash"]):
            return {"id": row["id"], "username": row["username"], "email": row["email"]}
        return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id, username, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row:
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "created_at": row["created_at"],
            }
        return None
