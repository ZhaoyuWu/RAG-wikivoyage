"""Trip planner: geometry, constraint parsing, and stop ordering.

Network-dependent steps (retrieval, routing) are exercised in live
integration tests; here we pin the pure logic.
"""

import pytest

from src import planner


def test_haversine_known_distance():
    # Essen (~51.45, 7.01) to Cologne (~50.94, 6.96) is ~57 km.
    essen = {"lat": 51.4556, "lon": 7.0116}
    cologne = {"lat": 50.9375, "lon": 6.9603}
    d = planner._haversine_km(essen, cologne)
    assert 50 < d < 65


def test_haversine_zero():
    p = {"lat": 51.0, "lon": 7.0}
    assert planner._haversine_km(p, p) == pytest.approx(0.0, abs=1e-6)


def test_heading_mapping():
    assert planner._heading_for("城堡") == "Sehenswürdigkeiten"
    assert planner._heading_for("美食") == "Küche"
    assert planner._heading_for("徒步") == "Aktivitäten"
    assert planner._heading_for("夜店") == "Nachtleben"
    assert planner._heading_for("量子物理") is None


def test_order_stops_is_nearest_neighbour_chain():
    origin = {"lat": 51.0, "lon": 7.0}
    cands = [
        {"file": "b", "geo": {"lat": 51.3, "lon": 7.0}, "dist_km": 33},
        {"file": "a", "geo": {"lat": 51.1, "lon": 7.0}, "dist_km": 11},
        {"file": "c", "geo": {"lat": 51.5, "lon": 7.0}, "dist_km": 55},
    ]
    chain = planner.order_stops(origin, cands, max_stops=3, max_hop_km=100)
    assert [s["file"] for s in chain] == ["a", "b", "c"]


def test_order_stops_breaks_on_far_hop():
    origin = {"lat": 51.0, "lon": 7.0}
    cands = [
        {"file": "near", "geo": {"lat": 51.1, "lon": 7.0}, "dist_km": 11},
        {"file": "faraway", "geo": {"lat": 54.0, "lon": 7.0}, "dist_km": 333},
    ]
    chain = planner.order_stops(origin, cands, max_stops=4, max_hop_km=100)
    assert [s["file"] for s in chain] == ["near"]  # far hop is dropped


def test_order_stops_respects_max():
    origin = {"lat": 51.0, "lon": 7.0}
    cands = [{"file": f"c{i}", "geo": {"lat": 51.0 + i / 10, "lon": 7.0}}
             for i in range(6)]
    assert len(planner.order_stops(origin, cands, max_stops=3)) == 3


def test_parse_constraints_extracts_and_defaults():
    # Fake generate() returns a fixed JSON blob; geocode is monkeypatched so
    # the origin resolves without touching the corpus.
    def fake_generate(context, question, history, system):
        yield '{"origin": "Essen", "days": 2, "likes": ["城堡", "徒步"], '
        yield '"excludes": ["夜店"], "mode": "car"}'

    orig_geocode = planner.geocode
    planner.geocode = lambda name: {"name": name, "lat": 51.45, "lon": 7.01}
    try:
        c = planner.parse_constraints("三天两地", fake_generate)
    finally:
        planner.geocode = orig_geocode
    assert c["origin"] == "Essen"
    assert c["days"] == 2
    assert c["mode"] == "car"
    assert "城堡" in c["likes"] and "夜店" in c["excludes"]


def test_parse_constraints_rejects_unresolvable_origin():
    def fake_generate(context, question, history, system):
        yield '{"origin": "Atlantis", "days": 1, "likes": [], "excludes": []}'

    orig_geocode = planner.geocode
    planner.geocode = lambda name: None
    try:
        with pytest.raises(ValueError):
            planner.parse_constraints("去亚特兰蒂斯", fake_generate)
    finally:
        planner.geocode = orig_geocode
