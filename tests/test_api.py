"""API-layer integration tests over the FastAPI app.

Uses TestClient WITHOUT its context manager so the startup warmup thread
(which touches Qdrant) never runs. Identity is injected through
dependency_overrides, and every heavy leg (hybrid_search, ask, route,
get_qdrant_client) is monkeypatched, so no vector store, model, or network
is involved. What is exercised for real: auth gating, role-based collection
authorization, deny-category propagation into retrieval, and the
sliding-window rate limiter returning 429.
"""

import pytest
from fastapi.testclient import TestClient

from src import api, ratelimit
from src.api import app, require_user


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # The limiter's counters are module-global; isolate each test.
    ratelimit._hits.clear()
    yield
    ratelimit._hits.clear()


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _as(role, user="tester"):
    return lambda: {"user": user, "role": role}


# --- Authentication gating ------------------------------------------------

def test_unauthenticated_is_rejected(client, monkeypatch):
    # With auth ON and no override, a missing token -> 401.
    monkeypatch.setattr(api, "AUTH_ENABLED", True)
    # require_user reads AUTH_ENABLED from its own module namespace.
    r = client.get("/conversations")
    assert r.status_code == 401


def test_auth_disabled_acts_as_admin(client, monkeypatch):
    monkeypatch.setattr(api, "AUTH_ENABLED", False)
    monkeypatch.setattr(api.store, "list_conversations", lambda user: [])
    r = client.get("/conversations")
    assert r.status_code == 200


# --- Role-based access control -------------------------------------------

def test_admin_only_endpoint_forbids_guest(client):
    app.dependency_overrides[require_user] = _as("guest")
    r = client.get("/audit")
    assert r.status_code == 403


def test_admin_only_endpoint_allows_admin(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    monkeypatch.setattr(api.audit, "recent", lambda n: [{"event": "login"}])
    r = client.get("/audit")
    assert r.status_code == 200
    assert r.json() == [{"event": "login"}]


def test_restricted_role_cannot_query_forbidden_collection(client, monkeypatch):
    # Give a role a whitelist that excludes 'secret', then query it -> 403.
    monkeypatch.setitem(api.ROLE_RULES, "limited",
                        {"collections": ["public"], "deny_categories": []})
    app.dependency_overrides[require_user] = _as("limited")
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    r = client.post("/search", json={"query": "x", "collection": "secret"})
    assert r.status_code == 403
    assert "may not query" in r.json()["detail"]


def test_deny_categories_reach_retrieval(client, monkeypatch):
    # A guest's deny_categories (['工作']) must be handed to hybrid_search so
    # restricted chunks are excluded at the store, not after the fact.
    captured = {}

    def fake_search(query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search", fake_search)
    app.dependency_overrides[require_user] = _as("guest")
    r = client.post("/search", json={"query": "社保"})
    assert r.status_code == 200
    assert captured["deny_categories"] == ["工作"]


def test_admin_sees_no_deny_categories(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search",
                        lambda q, **k: captured.update(k) or [])
    app.dependency_overrides[require_user] = _as("admin")
    client.post("/search", json={"query": "社保"})
    assert captured["deny_categories"] == []


# --- Rate limiting --------------------------------------------------------

def test_search_returns_429_over_limit(client, monkeypatch):
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search", lambda q, **k: [])
    monkeypatch.setattr(api, "SEARCH_RATE_PER_MIN", 3)
    app.dependency_overrides[require_user] = _as("admin", user="rl-user")

    codes = [client.post("/search", json={"query": "x"}).status_code
             for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429
    assert codes[4] == 429


def test_rate_limit_is_per_user(client, monkeypatch):
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search", lambda q, **k: [])
    monkeypatch.setattr(api, "SEARCH_RATE_PER_MIN", 2)

    # user A exhausts the bucket...
    app.dependency_overrides[require_user] = _as("admin", user="A")
    for _ in range(3):
        last_a = client.post("/search", json={"query": "x"}).status_code
    assert last_a == 429
    # ...user B still has a fresh bucket.
    app.dependency_overrides[require_user] = _as("admin", user="B")
    r = client.post("/search", json={"query": "x"})
    assert r.status_code == 200


def test_429_carries_retry_after_header(client, monkeypatch):
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search", lambda q, **k: [])
    monkeypatch.setattr(api, "SEARCH_RATE_PER_MIN", 1)
    app.dependency_overrides[require_user] = _as("admin", user="hdr")
    client.post("/search", json={"query": "x"})
    r = client.post("/search", json={"query": "x"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


# --- Request validation ---------------------------------------------------

def test_empty_query_is_422(client):
    app.dependency_overrides[require_user] = _as("admin")
    r = client.post("/search", json={"query": ""})
    assert r.status_code == 422


def test_top_k_out_of_range_is_422(client):
    app.dependency_overrides[require_user] = _as("admin")
    r = client.post("/search", json={"query": "x", "top_k": 99})
    assert r.status_code == 422


def test_route_mode_pattern_enforced(client):
    app.dependency_overrides[require_user] = _as("admin")
    r = client.post("/route", json={"from_place": "A", "to_place": "B",
                                    "mode": "teleport"})
    assert r.status_code == 422


# --- Route endpoint error mapping ----------------------------------------

def test_route_unknown_place_is_404(client, monkeypatch):
    def boom(frm, to, mode, dep=None):
        raise LookupError("Atlantis")

    monkeypatch.setattr(api, "route", boom)
    app.dependency_overrides[require_user] = _as("admin")
    r = client.post("/route", json={"from_place": "Essen", "to_place": "Atlantis"})
    assert r.status_code == 404


def test_route_backend_down_is_502(client, monkeypatch):
    def boom(frm, to, mode, dep=None):
        raise RuntimeError("OSRM unreachable")

    monkeypatch.setattr(api, "route", boom)
    app.dependency_overrides[require_user] = _as("admin")
    r = client.post("/route", json={"from_place": "Essen", "to_place": "Köln"})
    assert r.status_code == 502


# --- Metrics endpoint -----------------------------------------------------

def test_metrics_is_prometheus_text(client, monkeypatch):
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "hybrid_search", lambda q, **k: [])
    monkeypatch.setattr(api, "SEARCH_RATE_PER_MIN", 1)
    app.dependency_overrides[require_user] = _as("admin", user="m")
    # Trip the limiter so a counter is emitted.
    client.post("/search", json={"query": "x"})
    client.post("/search", json={"query": "x"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_rate_limited_total" in r.text
