"""Append-only JSONL audit log, one file per day.

Every Q&A leaves a line that can answer "who asked what, when, and which
documents the answer was built from". Answers are stored as a hash, not
text: the audit trail must not become a second copy of private content.
"""

import hashlib
import json
import time

from .config import LOGS_DIR


def answer_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_event(event: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    path = LOGS_DIR / f"audit-{time.strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def recent(limit: int = 50) -> list[dict]:
    """Newest events first, across day files."""
    if not LOGS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(LOGS_DIR.glob("audit-*.jsonl"), reverse=True):
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                return out
    return out
