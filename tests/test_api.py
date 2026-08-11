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

# The app pulls in fastapi + the retrieval/routing stack at import; skip in
# environments without them (the lightweight CI job installs only
# ruff/pytest/pyjwt).
pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("qdrant_client", reason="vector stack not installed")

from fastapi.testclient import TestClient  # noqa: E402

from src import api, ratelimit  # noqa: E402
from src.api import app, require_user  # noqa: E402


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


# --- Reach search (drive-time filtered retrieval) --------------------------

def test_reach_search_prefilters_by_circle_then_prices(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "geocode",
                        lambda p: {"name": "Essen", "lat": 51.45, "lon": 7.01})

    captured = {}

    def fake_search(q, **k):
        captured.update(k)
        return ["hit1", "hit2"]

    monkeypatch.setattr(api, "hybrid_search", fake_search)
    import src.isochrone as iso
    monkeypatch.setattr(
        iso, "reach_filter",
        lambda center, budget, hits: ([{"file": "a.md", "drive_min": 40.0}], 1))

    r = client.post("/reach/search",
                    json={"query": "Burgen", "place": "Essen", "minutes": 60})
    assert r.status_code == 200
    body = r.json()
    assert body["center"]["name"] == "Essen"
    # The straight-line prefilter circle scales with the budget (60 * 1.9).
    assert body["radius_km"] == 114
    assert captured["geo"]["radius_km"] == 114
    assert body["considered"] == 2
    assert body["dropped"] == 1
    assert body["hits"][0]["drive_min"] == 40.0


def test_reach_search_unknown_place_is_404(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    monkeypatch.setattr(api, "geocode", lambda p: None)
    r = client.post("/reach/search", json={"query": "Burgen", "place": "Nirgendwo"})
    assert r.status_code == 404


def test_reach_search_requires_a_centre(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    monkeypatch.setattr(api, "_check_collection", lambda c: None)
    r = client.post("/reach/search", json={"query": "Burgen"})
    assert r.status_code == 422


# --- Plan follow-up context -------------------------------------------------

def test_plan_stream_passes_history_through(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    import src.planner as planner

    seen = {}

    def fake_plan_stream(query, collection=None, history=None):
        seen["query"], seen["history"] = query, history
        yield {"type": "done", "answer": "ok", "stops": [], "trace": None}

    monkeypatch.setattr(planner, "plan_stream", fake_plan_stream)
    history = [{"role": "user", "content": "essen周边1日游"},
               {"role": "assistant", "content": "…行程…"}]
    r = client.post("/plan/stream",
                    json={"query": "改成开车", "history": history})
    assert r.status_code == 200
    assert seen["query"] == "改成开车"
    assert seen["history"] == history


# --- Intent classification --------------------------------------------------

def test_intent_endpoint_classifies(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    import src.intent as intent
    monkeypatch.setattr(intent, "geocode",
                        lambda p: {"name": p} if p == "Essen" else None)

    r = client.post("/intent", json={"query": "essen周边1日游"})
    assert r.json() == {"kind": "plan"}

    r = client.post("/intent", json={"query": "Essen出发90分钟车程内有什么城堡"})
    assert r.json() == {"kind": "reach", "place": "Essen", "minutes": 90}

    r = client.post("/intent", json={"query": "Essen有什么好吃的"})
    assert r.json() == {"kind": "ask"}


# --- Geo filter reaches ask retrieval ---------------------------------------

def test_ask_stream_forwards_geo_filter(client, monkeypatch):
    app.dependency_overrides[require_user] = _as("admin")
    monkeypatch.setattr(api, "_check_collection", lambda c: None)

    seen = {}

    def fake_ask_stream(q, **k):
        seen.update(k)
        yield {"type": "done", "answer": "ok", "sources": [], "trace": None}

    monkeypatch.setattr(api, "ask_stream", fake_ask_stream)
    r = client.post("/ask/stream", json={
        "query": "Burgen im Harz",
        "geo": {"lat": 51.75, "lon": 10.63, "radius_km": 40},
    })
    assert r.status_code == 200
    assert seen["geo"] == {"lat": 51.75, "lon": 10.63, "radius_km": 40.0}
