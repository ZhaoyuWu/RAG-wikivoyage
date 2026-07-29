"""Markdown parsing and chunking.

Strategy:
- Strip YAML frontmatter.
- Split on level-2 headings (##). Text before the first ## becomes an
  "(引言)" (intro) chunk.
- Chunks shorter than MIN_CHUNK_CHARS are merged into the previous chunk.
- Chunks longer than MAX_CHUNK_CHARS are re-split on paragraph boundaries.
- Each chunk gets a stable ID: md5(relative path + heading + chunk index),
  formatted as a UUID so repeated indexing upserts idempotently.
"""

import hashlib
import re
from dataclasses import dataclass

from .config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS

INTRO_HEADING = "(引言)"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str          # UUID string derived from md5, stable across runs
    file: str        # relative path, forward slashes
    heading: str
    text: str
    category: str
    file_hash: str
    chunk_index: int


def stable_chunk_id(rel_path: str, heading: str, chunk_index: int) -> str:
    digest = hashlib.md5(f"{rel_path}|{heading}|{chunk_index}".encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def strip_frontmatter(content: str) -> str:
    return _FRONTMATTER_RE.sub("", content, count=1)


def split_by_headings(content: str) -> list[tuple[str, str]]:
    """Return (heading, text) sections. Intro text gets INTRO_HEADING."""
    matches = list(_H2_RE.finditer(content))
    sections: list[tuple[str, str]] = []

    intro = content[: matches[0].start()] if matches else content
    if intro.strip():
        sections.append((INTRO_HEADING, intro.strip()))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[m.end(): end].strip()
        sections.append((m.group(1), body))
    return sections


def _split_long_text(text: str, limit: int) -> list[str]:
    """Split text on paragraph boundaries into pieces of at most ~limit chars.

    A single paragraph longer than the limit is kept intact rather than cut
    mid-sentence.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for p in paragraphs:
        candidate = f"{current}\n\n{p}" if current else p
        if len(candidate) > limit and current:
            pieces.append(current)
            current = p
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [text]


def chunk_markdown(content: str, rel_path: str, category: str, file_hash: str) -> list[Chunk]:
    """Chunk one markdown file into indexable pieces."""
    body = strip_frontmatter(content)
    sections = split_by_headings(body)

    # Merge tiny sections into their predecessor, split oversized ones.
    merged: list[tuple[str, str]] = []
    for heading, text in sections:
        if not text.strip():
            continue
        if len(text) < MIN_CHUNK_CHARS and merged:
            prev_heading, prev_text = merged[-1]
            merged[-1] = (prev_heading, f"{prev_text}\n\n## {heading}\n{text}")
        else:
            merged.append((heading, text))

    chunks: list[Chunk] = []
    index = 0
    for heading, text in merged:
        parts = _split_long_text(text, MAX_CHUNK_CHARS) if len(text) > MAX_CHUNK_CHARS else [text]
        for part in parts:
            chunks.append(
                Chunk(
                    id=stable_chunk_id(rel_path, heading, index),
                    file=rel_path,
                    heading=heading,
                    text=part,
                    category=category,
                    file_hash=file_hash,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
