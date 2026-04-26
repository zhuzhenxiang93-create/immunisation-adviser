"""
api/crm_manager.py — Simulated Salesforce-style CRM for IMAC call case management.

Table: crm_cases
  id, case_number, username, query, caller_type, vaccine_type_json,
  urgency, confidence, status, notes, created_at, updated_at

In production this layer would call the Salesforce REST API instead of SQLite.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class CRMManager:
    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_table(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crm_cases (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_number      TEXT    UNIQUE NOT NULL,
                    username         TEXT    NOT NULL DEFAULT 'unknown',
                    query            TEXT    NOT NULL,
                    caller_type      TEXT    NOT NULL DEFAULT 'unknown',
                    vaccine_type_json TEXT   NOT NULL DEFAULT '[]',
                    urgency          TEXT    NOT NULL DEFAULT 'routine',
                    confidence       TEXT    NOT NULL DEFAULT 'not_found',
                    status           TEXT    NOT NULL DEFAULT 'open',
                    notes            TEXT    NOT NULL DEFAULT '',
                    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _row_to_dict(self, row) -> dict:
        return {
            "id":           row["id"],
            "case_number":  row["case_number"],
            "username":     row["username"],
            "query":        row["query"],
            "caller_type":  row["caller_type"],
            "vaccine_type": json.loads(row["vaccine_type_json"]),
            "urgency":      row["urgency"],
            "confidence":   row["confidence"],
            "status":       row["status"],
            "notes":        row["notes"],
            "created_at":   row["created_at"],
            "updated_at":   row["updated_at"],
        }

    def create_case(
        self,
        query: str,
        caller_type: str,
        vaccine_type: list[str],
        urgency: str,
        confidence: str,
        username: str = "unknown",
    ) -> dict:
        """Insert a new case and return it with a generated case number."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        placeholder = f"IMAC-{today}-TEMP"

        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO crm_cases
                   (case_number, username, query, caller_type, vaccine_type_json,
                    urgency, confidence, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'open', '')""",
                (
                    placeholder,
                    username,
                    query,
                    caller_type,
                    json.dumps(vaccine_type),
                    urgency,
                    confidence,
                ),
            )
            case_id: int = cur.lastrowid  # type: ignore[assignment]
            case_number = f"IMAC-{today}-{case_id:04d}"
            conn.execute(
                "UPDATE crm_cases SET case_number = ? WHERE id = ?",
                (case_number, case_id),
            )
            conn.commit()

        return self.get_case(case_id)  # type: ignore[return-value]

    def get_cases(
        self,
        status: str | None = None,
        username: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM crm_cases WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if username:
            sql += " AND username = ?"
            params.append(username)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_case(self, case_id: int) -> dict | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM crm_cases WHERE id = ?", (case_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_case_by_number(self, case_number: str) -> dict | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM crm_cases WHERE case_number = ?", (case_number,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_case(
        self,
        case_id: int,
        status: str | None = None,
        notes: str | None = None,
    ) -> dict | None:
        """Update status and/or notes. Returns updated case or None if not found."""
        fields = []
        params = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if notes is not None:
            fields.append("notes = ?")
            params.append(notes)
        if not fields:
            return self.get_case(case_id)

        fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(case_id)

        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE crm_cases SET {', '.join(fields)} WHERE id = ?", params
            )
            conn.commit()
        return self.get_case(case_id)
