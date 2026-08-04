"""Conversation store: persistence, ownership isolation, markdown export."""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    import src.store as s

    monkeypatch.setattr(s, "DB_PATH", tmp_path / "test.db")
    importlib.reload(s)
    monkeypatch.setattr(s, "DB_PATH", tmp_path / "test.db")
    s.init_db()
    return s


def test_create_and_read(store):
    cid = store.create_conversation("alice", "wikivoyage", "柏林之旅")
    store.add_message(cid, "user", "柏林有什么博物馆")
    store.add_message(cid, "assistant", "很多。", sources="Berlin :: Sehenswürdigkeiten")
    msgs = store.get_messages(cid, "alice")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"] == "Berlin :: Sehenswürdigkeiten"


def test_ownership_isolation(store):
    cid = store.create_conversation("alice", "vault", "私密")
    store.add_message(cid, "user", "secret")
    assert store.get_messages(cid, "bob") is None       # not the owner
    assert store.get_messages(cid, "alice") is not None
    assert store.delete_conversation(cid, "bob") is False
    assert store.delete_conversation(cid, "alice") is True


def test_list_orders_by_recency(store):
    a = store.create_conversation("u", "c", "first")
    b = store.create_conversation("u", "c", "second")
    store.add_message(b, "user", "touch b")             # bumps updated_at
    ids = [c["id"] for c in store.list_conversations("u")]
    assert ids[0] == b and a in ids


def test_export_markdown(store):
    cid = store.create_conversation("alice", "wikivoyage", "科隆")
    store.add_message(cid, "user", "科隆大教堂门票")
    store.add_message(cid, "assistant", "进堂免费。", sources="Köln")
    md = store.export_markdown(cid, "alice")
    assert "# 科隆" in md and "科隆大教堂门票" in md and "进堂免费" in md
    assert store.export_markdown(cid, "bob") is None
