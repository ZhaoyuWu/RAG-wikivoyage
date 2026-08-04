"""Server-side conversation store (SQLite, standard library only).

Moves chat history off the browser's localStorage so it survives a
device switch and can be audited or exported. Two tables, no ORM.

Export closes the loop: a conversation can be written back to the vault
as markdown and re-indexed, turning Q&A into new knowledge.
"""

import sqlite3
import time
from pathlib import Path

from .config import PROJECT_ROOT

DB_PATH = Path(PROJECT_ROOT / "data" / "conversations.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                collection TEXT,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
        """)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def create_conversation(user: str, collection: str, title: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO conversations(user, collection, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user, collection, title[:80], _now(), _now()),
        )
        return cur.lastrowid


def add_message(conversation_id: int, role: str, content: str,
                sources: str | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages(conversation_id, role, content, sources, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources, _now()),
        )
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                  (_now(), conversation_id))


def list_conversations(user: str, limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, collection, title, created_at, updated_at FROM conversations"
            " WHERE user = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
            (user, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_messages(conversation_id: int, user: str) -> list[dict] | None:
    """Messages for a conversation the user owns, else None."""
    with _conn() as c:
        owner = c.execute("SELECT user FROM conversations WHERE id = ?",
                          (conversation_id,)).fetchone()
        if owner is None or owner["user"] != user:
            return None
        rows = c.execute(
            "SELECT role, content, sources, created_at FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_conversation(conversation_id: int, user: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM conversations WHERE id = ? AND user = ?",
                        (conversation_id, user))
        return cur.rowcount > 0


def export_markdown(conversation_id: int, user: str) -> str | None:
    """Render a conversation as markdown for saving back into the vault."""
    msgs = get_messages(conversation_id, user)
    if msgs is None:
        return None
    with _conn() as c:
        conv = c.execute("SELECT title, collection, created_at FROM conversations"
                         " WHERE id = ?", (conversation_id,)).fetchone()
    lines = [f"# {conv['title']}", "",
             f"> 来源库: {conv['collection']} · {conv['created_at']}", ""]
    for m in msgs:
        who = "**问**" if m["role"] == "user" else "**答**"
        lines.append(f"{who}: {m['content']}")
        if m["sources"]:
            lines.append(f"\n> 引用: {m['sources']}")
        lines.append("")
    return "\n".join(lines)
