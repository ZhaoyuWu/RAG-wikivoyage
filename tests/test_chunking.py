"""Unit tests for the chunking logic."""

from src.chunking import (
    INTRO_HEADING,
    chunk_markdown,
    split_by_headings,
    stable_chunk_id,
    strip_frontmatter,
)
from src.config import MAX_CHUNK_CHARS


def test_strip_frontmatter():
    content = "---\ntags: [a, b]\n---\n# Title\nbody"
    assert strip_frontmatter(content) == "# Title\nbody"


def test_strip_frontmatter_no_frontmatter():
    content = "# Title\nbody"
    assert strip_frontmatter(content) == content


def test_split_by_headings_intro_and_sections():
    content = "intro text here\n\n## First\naaa\n\n## Second\nbbb"
    sections = split_by_headings(content)
    assert sections[0] == (INTRO_HEADING, "intro text here")
    assert sections[1] == ("First", "aaa")
    assert sections[2] == ("Second", "bbb")


def test_small_chunk_merged_into_previous():
    long_text = "x" * 200
    content = f"## Big\n{long_text}\n\n## Tiny\nshort"
    chunks = chunk_markdown(content, "a.md", "工作", "hash")
    assert len(chunks) == 1
    assert "short" in chunks[0].text
    assert chunks[0].heading == "Big"


def test_large_chunk_split_by_paragraphs():
    para = "y" * 600
    content = "## Long\n" + "\n\n".join([para] * 4)  # 2400+ chars
    chunks = chunk_markdown(content, "a.md", "学习", "hash")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    # chunk_index must be unique and sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_stable_ids_are_deterministic_and_uuid_shaped():
    id1 = stable_chunk_id("话题/工作/a.md", "JWT", 3)
    id2 = stable_chunk_id("话题/工作/a.md", "JWT", 3)
    id3 = stable_chunk_id("话题/工作/a.md", "JWT", 4)
    assert id1 == id2
    assert id1 != id3
    parts = id1.split("-")
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


def test_payload_fields():
    content = "## Section\n" + "z" * 100
    chunks = chunk_markdown(content, "话题/日常/b.md", "日常", "abc123")
    c = chunks[0]
    assert c.file == "话题/日常/b.md"
    assert c.category == "日常"
    assert c.file_hash == "abc123"
