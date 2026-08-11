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
        # No day count in the text, so the LLM's days=2 stands.
        c = planner.parse_constraints("想去玩玩", fake_generate)
    finally:
        planner.geocode = orig_geocode
    assert c["origin"] == "Essen"
    assert c["days"] == 2
    assert c["mode"] == "car"
    assert "城堡" in c["likes"] and "夜店" in c["excludes"]


def test_radius_grows_with_days():
    assert planner._radius_for_days(1) == 150.0
    assert planner._radius_for_days(3) == 390.0
    assert planner._radius_for_days(10) == 500.0  # capped


def test_cluster_single_day_is_passthrough():
    cands = [{"file": f"c{i}", "geo": {"lat": 51.0, "lon": 7.0 + i}, "dist_km": i}
             for i in range(4)]
    groups = planner.cluster_by_day(cands, 1)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_cluster_splits_two_regions():
    # Two tight geographic groups: near (lat 51) and far (lat 54).
    west = [{"file": f"w{i}", "geo": {"lat": 51.0, "lon": 7.0 + i * 0.1}, "dist_km": 10 + i}
            for i in range(3)]
    east = [{"file": f"e{i}", "geo": {"lat": 54.0, "lon": 12.0 + i * 0.1}, "dist_km": 300 + i}
            for i in range(3)]
    groups = planner.cluster_by_day(west + east, 2)
    assert len(groups) == 2
    # Each group should be internally consistent (all one region).
    lats = [{round(c["geo"]["lat"]) for c in g} for g in groups]
    assert {51} in lats and {54} in lats
    # Nearer cluster comes first (day 1).
    assert min(c["dist_km"] for c in groups[0]) < min(c["dist_km"] for c in groups[1])


def test_cluster_deterministic():
    cands = [{"file": f"c{i}", "geo": {"lat": 51.0 + i * 0.5, "lon": 7.0}, "dist_km": i * 30}
             for i in range(6)]
    a = planner.cluster_by_day(cands, 3)
    b = planner.cluster_by_day(cands, 3)
    assert [[c["file"] for c in g] for g in a] == [[c["file"] for c in g] for g in b]


def test_parse_falls_back_to_text_when_llm_flakes():
    # LLM returns useless placeholders; the text fallback must recover the
    # interests, day count, and drive mode the user plainly wrote.
    def flaky_generate(context, question, history, system):
        yield '{"origin": "Essen", "days": 1, "likes": ["?", "?"], '
        yield '"excludes": [], "mode": "transit"}'

    orig_geocode = planner.geocode
    planner.geocode = lambda name: {"name": name, "lat": 51.45, "lon": 7.01}
    try:
        c = planner.parse_constraints("从Essen出发三天,想看城堡和徒步,开车,不要夜店",
                                      flaky_generate)
    finally:
        planner.geocode = orig_geocode
    assert c["days"] == 3            # recovered from "三天"
    assert "城堡" in c["likes"] and "徒步" in c["likes"]  # recovered from text
    assert "?" not in c["likes"]
    assert c["mode"] == "car"        # recovered from "开车"
    assert "夜店" in c["excludes"]


def test_days_from_text():
    assert planner._days_from_text("三天两地") == 3
    assert planner._days_from_text("a 5 days trip") == 5
    assert planner._days_from_text("没有天数") is None


def test_point_to_segment_endpoints_and_middle():
    a = {"lat": 51.0, "lon": 7.0}
    b = {"lat": 51.0, "lon": 8.0}
    # A point on the segment is ~0 km away.
    on = {"lat": 51.0, "lon": 7.5}
    assert planner._point_to_segment_km(on, a, b) < 1
    # A point due north of the midpoint is offset only by latitude.
    off = {"lat": 51.5, "lon": 7.5}
    d = planner._point_to_segment_km(off, a, b)
    assert 50 < d < 60  # ~0.5 deg lat ≈ 55 km


def test_min_dist_to_route_picks_nearest_segment():
    line = [[7.0, 51.0], [8.0, 51.0], [9.0, 51.0]]  # [lon, lat] along 51°N
    near = {"lat": 51.05, "lon": 8.5}
    far = {"lat": 52.0, "lon": 8.5}
    assert planner._min_dist_to_route(near, line) < planner._min_dist_to_route(far, line)
    assert planner._min_dist_to_route(near, line) < 10


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


def test_parse_constraints_forwards_history_to_llm():
    # A follow-up turn must reach the parse LLM so it can merge the new
    # message into the previous constraints (the parse prompt handles the
    # merge; here we pin the plumbing).
    seen = {}

    def fake_generate(context, question, history, system):
        seen["history"] = history
        yield '{"origin": "Essen", "days": 3, "likes": ["城堡"], '
        yield '"excludes": [], "mode": "car"}'

    history = [{"role": "user", "content": "从Essen出发三天看城堡"},
               {"role": "assistant", "content": "…行程…"}]
    orig_geocode = planner.geocode
    planner.geocode = lambda name: {"name": name, "lat": 51.45, "lon": 7.01}
    try:
        c = planner.parse_constraints("改成开车", fake_generate, history)
    finally:
        planner.geocode = orig_geocode
    assert seen["history"] == history
    assert c["days"] == 3          # carried over from the merged parse
    assert c["mode"] == "car"


def test_parse_constraints_empty_history_is_none():
    seen = {}

    def fake_generate(context, question, history, system):
        seen["history"] = history
        yield '{"origin": "Essen", "days": 1, "likes": ["景点"], '
        yield '"excludes": [], "mode": "transit"}'

    orig_geocode = planner.geocode
    planner.geocode = lambda name: {"name": name, "lat": 51.45, "lon": 7.01}
    try:
        planner.parse_constraints("Essen一日游", fake_generate, [])
    finally:
        planner.geocode = orig_geocode
    assert seen["history"] is None
