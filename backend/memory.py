"""
memory.py — SQLite-backed conversation memory for NyayaSathi.

Schema:
  history(id, session_id, role, content, ts)
  feedback(id, session_id, message_id, rating, comment, ts)

No external dependencies beyond the stdlib sqlite3.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH, MAX_HISTORY_TURNS


# ─── Init ─────────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS history (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content    TEXT NOT NULL,
                ts         TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_session
                ON history(session_id, ts);

            CREATE TABLE IF NOT EXISTS feedback (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                rating     INTEGER NOT NULL CHECK(rating IN (0,1)),
                comment    TEXT,
                ts         TEXT NOT NULL
            );
        """)


# ─── Read ─────────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> list[dict]:
    """
    Return the last MAX_HISTORY_TURNS pairs for the session,
    ordered oldest → newest (ready for prompt injection).
    """
    limit = MAX_HISTORY_TURNS * 2  # each turn = 1 user + 1 assistant
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, ts FROM history
            WHERE session_id = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    # Reverse so oldest is first
    return [{"role": r["role"], "content": r["content"], "ts": r["ts"]}
            for r in reversed(rows)]


def get_all_sessions() -> list[str]:
    """Return distinct session IDs (for admin / debugging)."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT session_id FROM history").fetchall()
    return [r["session_id"] for r in rows]


# ─── Write ────────────────────────────────────────────────────────────────────

def save_turn(
    session_id: str,
    question: str,
    answer: str,
    message_id: Optional[str] = None,
) -> str:
    """
    Persist one full turn (user + assistant).
    Returns the message_id so the frontend can reference it for feedback.
    """
    ts = datetime.now(timezone.utc).isoformat()
    message_id = message_id or str(uuid.uuid4())

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO history VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), session_id, "user", question, ts),
        )
        conn.execute(
            "INSERT INTO history VALUES (?,?,?,?,?)",
            (message_id, session_id, "assistant", answer, ts),
        )

    return message_id


def save_feedback(
    session_id: str,
    message_id: str,
    rating: int,
    comment: Optional[str] = None,
) -> None:
    """Store user feedback (👍 = 1, 👎 = 0) for a specific AI message."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                session_id,
                message_id,
                rating,
                comment,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


# ─── Delete ───────────────────────────────────────────────────────────────────

def clear_session(session_id: str) -> None:
    """Wipe all history for a session (called when user clicks 'New chat')."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))