"""
database.py — SQLite Analysis History Store
=============================================
Handles all persistent storage for FakeShield AI:
  • analysis_history  – every analysis run with full results
  • statistics        – aggregated daily counts for the dashboard

Usage:
    db = AnalysisDatabase()
    record_id = db.save_analysis({...})
    rows = db.get_history(limit=20)
    stats = db.get_statistics()
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import AppConfig


class AnalysisDatabase:
    """Thread-safe SQLite wrapper for analysis history."""

    def __init__(self):
        AppConfig.ensure_dirs()
        self.db_path = str(AppConfig.DB_PATH)
        self._init_database()

    # ── Connection helper ────────────────────────────────────────────────────
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row        # dict-like row access
        return conn

    # ── Schema creation ──────────────────────────────────────────────────────
    def _init_database(self):
        """Create tables if they do not yet exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT    NOT NULL,
                    analysis_type    TEXT    NOT NULL,   -- text|image|video|combined
                    input_summary    TEXT,               -- truncated input preview
                    input_image_path TEXT,
                    input_video_url  TEXT,
                    overall_score    REAL,               -- 0.0–1.0 (fake probability)
                    overall_label    TEXT,               -- REAL|SUSPICIOUS|FAKE
                    text_score       REAL,
                    image_score      REAL,
                    video_score      REAL,
                    confidence       REAL,
                    explanation_text TEXT,
                    report_path      TEXT,
                    details          TEXT                -- full JSON blob
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    date       TEXT UNIQUE NOT NULL,
                    total      INTEGER DEFAULT 0,
                    fake_cnt   INTEGER DEFAULT 0,
                    real_cnt   INTEGER DEFAULT 0,
                    susp_cnt   INTEGER DEFAULT 0
                );
            """)

    # ── Write ────────────────────────────────────────────────────────────────
    def save_analysis(self, data: dict) -> int:
        """
        Persist one analysis record.

        Parameters
        ----------
        data : dict  Keys match the column names above plus 'details' (any dict).

        Returns
        -------
        int  Row ID of the new record.
        """
        today = datetime.now().date().isoformat()
        label = data.get("overall_label", "UNKNOWN")

        with self._get_conn() as conn:
            cur = conn.execute("""
                INSERT INTO analysis_history
                    (timestamp, analysis_type, input_summary, input_image_path,
                     input_video_url, overall_score, overall_label, text_score,
                     image_score, video_score, confidence, explanation_text,
                     report_path, details)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(),
                data.get("analysis_type", "text"),
                data.get("input_summary", "")[:500],
                data.get("input_image_path", ""),
                data.get("input_video_url", ""),
                float(data.get("overall_score", 0.5)),
                label,
                data.get("text_score"),
                data.get("image_score"),
                data.get("video_score"),
                data.get("confidence"),
                data.get("explanation_text", ""),
                data.get("report_path", ""),
                json.dumps(data.get("details", {})),
            ))
            row_id = cur.lastrowid

            # Update daily aggregates
            conn.execute("""
                INSERT INTO daily_stats (date, total, fake_cnt, real_cnt, susp_cnt)
                VALUES (?, 1,
                    CASE WHEN ? = 'FAKE'       THEN 1 ELSE 0 END,
                    CASE WHEN ? = 'REAL'       THEN 1 ELSE 0 END,
                    CASE WHEN ? = 'SUSPICIOUS' THEN 1 ELSE 0 END)
                ON CONFLICT(date) DO UPDATE SET
                    total    = total    + 1,
                    fake_cnt = fake_cnt + CASE WHEN ? = 'FAKE'       THEN 1 ELSE 0 END,
                    real_cnt = real_cnt + CASE WHEN ? = 'REAL'       THEN 1 ELSE 0 END,
                    susp_cnt = susp_cnt + CASE WHEN ? = 'SUSPICIOUS' THEN 1 ELSE 0 END
            """, (today, label, label, label, label, label, label))

        return row_id

    # ── Read ─────────────────────────────────────────────────────────────────
    def get_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent *limit* analysis records as dicts."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM analysis_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_record(self, record_id: int) -> dict | None:
        """Fetch a single record by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_history WHERE id = ?", (record_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_statistics(self) -> dict:
        """Return overall counts and daily breakdown."""
        with self._get_conn() as conn:
            totals = conn.execute("""
                SELECT
                    COUNT(*)                           AS total,
                    SUM(overall_label = 'FAKE')        AS fake,
                    SUM(overall_label = 'REAL')        AS real,
                    SUM(overall_label = 'SUSPICIOUS')  AS suspicious,
                    AVG(overall_score)                 AS avg_score
                FROM analysis_history
            """).fetchone()

            daily = conn.execute("""
                SELECT date, total, fake_cnt, real_cnt, susp_cnt
                FROM daily_stats
                ORDER BY date DESC
                LIMIT 30
            """).fetchall()

        return {
            "total":      totals["total"]      or 0,
            "fake":       totals["fake"]       or 0,
            "real":       totals["real"]       or 0,
            "suspicious": totals["suspicious"] or 0,
            "avg_score":  round(totals["avg_score"] or 0.5, 3),
            "daily":      [dict(r) for r in daily],
        }

    # ── Delete ───────────────────────────────────────────────────────────────
    def delete_record(self, record_id: int):
        """Remove one record."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))

    def clear_all(self):
        """Wipe the entire history (use with caution)."""
        with self._get_conn() as conn:
            conn.executescript("DELETE FROM analysis_history; DELETE FROM daily_stats;")
