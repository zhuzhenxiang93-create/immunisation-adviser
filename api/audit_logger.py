"""
audit_logger.py — SQLite-backed audit log for query history.

Table: query_log
  id, username, query, confidence, chunks_retrieved,
  sources_retrieved_json,   -- breadcrumb+URL of every chunk fetched
  citations_json,           -- sources actually cited in the final answer
  answer,                   -- generated answer text (for post-hoc review)
  classification_json, timestamp

Auditability rationale (Responsible AI):
  Logging what was retrieved vs what was cited allows reviewers to check
  whether the answer is grounded in the retrieved sources and to flag
  hallucinations or miscitations after the fact.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class AuditLogger:
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
                CREATE TABLE IF NOT EXISTS query_log (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    username                TEXT    NOT NULL DEFAULT 'unknown',
                    query                   TEXT    NOT NULL,
                    confidence              TEXT    NOT NULL DEFAULT 'not_found',
                    chunks_retrieved        INTEGER NOT NULL DEFAULT 0,
                    sources_retrieved_json  TEXT    NOT NULL DEFAULT '[]',
                    citations_json          TEXT    NOT NULL DEFAULT '[]',
                    answer                  TEXT    NOT NULL DEFAULT '',
                    classification_json     TEXT    NOT NULL DEFAULT '{}',
                    timestamp               DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrate existing tables that lack the new columns
            for col, default in [
                ("sources_retrieved_json", "'[]'"),
                ("citations_json",         "'[]'"),
                ("answer",                 "''"),
            ]:
                try:
                    conn.execute(
                        f"ALTER TABLE query_log ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}"
                    )
                except Exception:
                    pass  # column already exists
            conn.commit()

    def log(
        self,
        query: str,
        confidence: str,
        chunks_retrieved: int,
        classification: dict,
        username: str = "unknown",
        sources_retrieved: list[dict] | None = None,
        citations: list[dict] | None = None,
        answer: str = "",
    ) -> None:
        """
        sources_retrieved: list of {breadcrumb, url} for every retrieved chunk.
        citations:         list of citation dicts from the generated answer.
        answer:            the final answer text (post-redaction).
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO query_log
                   (username, query, confidence, chunks_retrieved,
                    sources_retrieved_json, citations_json, answer,
                    classification_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    username,
                    query,
                    confidence,
                    chunks_retrieved,
                    json.dumps([
                        {"breadcrumb": c.get("breadcrumb", ""), "url": c.get("url", "")}
                        for c in (sources_retrieved or [])
                    ]),
                    json.dumps(citations or []),
                    answer,
                    json.dumps(classification),
                ),
            )
            conn.commit()

    def get_recent(self, limit: int = 20, username: str | None = None) -> list[dict]:
        with self._get_conn() as conn:
            if username:
                rows = conn.execute(
                    """SELECT * FROM query_log WHERE username = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (username, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        result = []
        for row in rows:
            result.append({
                "id":               row["id"],
                "username":         row["username"],
                "query":            row["query"],
                "confidence":       row["confidence"],
                "chunks_retrieved": row["chunks_retrieved"],
                "classification":   json.loads(row["classification_json"]),
                "timestamp":        row["timestamp"],
            })
        return result

    def get_summary(self) -> dict:
        """
        Aggregate statistics across all logged queries.
        Returns counts by confidence, vaccine_type, query_type,
        clinical_scenario, urgency, patient_age_group, and daily volume.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT confidence, chunks_retrieved, classification_json, timestamp FROM query_log"
            ).fetchall()

        total = len(rows)
        confidence_counts: dict[str, int] = {}
        vaccine_counts:    dict[str, int] = {}
        query_type_counts: dict[str, int] = {}
        scenario_counts:   dict[str, int] = {}
        urgency_counts:    dict[str, int] = {}
        age_counts:        dict[str, int] = {}
        daily_counts:      dict[str, int] = {}

        for row in rows:
            # confidence
            c = row["confidence"]
            confidence_counts[c] = confidence_counts.get(c, 0) + 1

            # daily volume (date part of timestamp)
            day = str(row["timestamp"])[:10]
            daily_counts[day] = daily_counts.get(day, 0) + 1

            clf = json.loads(row["classification_json"])

            for v in clf.get("vaccine_type", []):
                if v != "unknown":
                    vaccine_counts[v] = vaccine_counts.get(v, 0) + 1

            for qt in clf.get("query_type", []):
                if qt != "general":
                    query_type_counts[qt] = query_type_counts.get(qt, 0) + 1

            for s in clf.get("clinical_scenario", []):
                scenario_counts[s] = scenario_counts.get(s, 0) + 1

            u = clf.get("urgency", "routine")
            urgency_counts[u] = urgency_counts.get(u, 0) + 1

            ag = clf.get("patient_age_group", "unknown")
            if ag != "unknown":
                age_counts[ag] = age_counts.get(ag, 0) + 1

        def _sort(d: dict) -> dict:
            return dict(sorted(d.items(), key=lambda x: -x[1]))

        return {
            "total_queries":       total,
            "confidence":          _sort(confidence_counts),
            "vaccine_type":        _sort(vaccine_counts),
            "query_type":          _sort(query_type_counts),
            "clinical_scenario":   _sort(scenario_counts),
            "urgency":             _sort(urgency_counts),
            "patient_age_group":   _sort(age_counts),
            "daily_volume":        dict(sorted(daily_counts.items())),
        }

    def clear(self, username: str | None = None) -> None:
        with self._get_conn() as conn:
            if username:
                conn.execute("DELETE FROM query_log WHERE username = ?", (username,))
            else:
                conn.execute("DELETE FROM query_log")
            conn.commit()
