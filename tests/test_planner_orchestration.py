"""Planner orchestration: candidate gathering, parallel routing, corridor
detour ranking, and itinerary synthesis input.

test_planner.py pins the geometry and parsing helpers; this file covers the
steps that stitch retrieval + routing together. Every external leg
(hybrid_search, route) is stubbed, so nothing loads Qdrant, a model, or the
network.
"""

import types

import pytest

from src import planner


def _hit(file, lat, lon, score=0.9, heading="Sehenswürdigkeiten", text="t"):
    """A stand-in for retrieval.Hit with just the attributes the planner reads."""
    return types.SimpleNamespace(
        file=file, heading=heading, score=score, text=text,
        geo={"lat": lat, "lon": lon},
    )


# --- gather_candidates: geo-first retrieval, dedup, heading scoping -------

def test_gather_dedupes_by_file_keeping_best_score(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    # Same article returned twice with different scores; the higher wins.
    calls = {"n": 0}

    def fake_search(like, **kwargs):
        calls["n"] += 1
        return [_hit("Koeln.md", 50.9, 6.9, score=0.4),
                _hit("Koeln.md", 50.9, 6.9, score=0.8)]

    monkeypatch.setattr(planner, "hybrid_search", fake_search)
    out = planner.gather_candidates(["城堡"], [], origin, max_km=150)
    assert len(out) == 1
    assert out[0]["file"] == "Koeln.md"
    assert out[0]["score"] == 0.8


def test_gather_passes_geo_radius_and_heading(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    seen = {}

    def fake_search(like, **kwargs):
        seen.update(kwargs)
        return [_hit("A.md", 51.1, 7.0)]

    monkeypatch.setattr(planner, "hybrid_search", fake_search)
    planner.gather_candidates(["城堡"], [], origin, max_km=200)
    # Geo radius is a hard pre-constraint (the "波罗的海 bug" fix), not a
    # post-filter, and the interest maps to a heading.
    assert seen["geo"]["radius_km"] == 200
    assert seen["geo"]["lat"] == 51.0
    assert seen["headings"] == ["Sehenswürdigkeiten"]


def test_gather_skips_excluded_interest_heading(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    searched = []

    def fake_search(like, **kwargs):
        searched.append(like)
        return [_hit("X.md", 51.1, 7.0)]

    monkeypatch.setattr(planner, "hybrid_search", fake_search)
    # 夜店 and 城堡 share no heading; excluding 夜店 must not suppress 城堡,
    # but a like whose heading is excluded is skipped entirely.
    planner.gather_candidates(["城堡", "夜店"], ["夜店"], origin)
    assert "城堡" in searched
    assert "夜店" not in searched  # its heading (Nachtleben) is excluded


def test_gather_drops_hits_without_geo(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    ungeo = types.SimpleNamespace(file="NoGeo.md", heading="H", score=0.9,
                                  text="t", geo=None)

    monkeypatch.setattr(planner, "hybrid_search",
                        lambda like, **k: [ungeo, _hit("Ok.md", 51.1, 7.0)])
    out = planner.gather_candidates(["城堡"], [], origin)
    assert [c["file"] for c in out] == ["Ok.md"]


def test_gather_sorts_by_distance(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    monkeypatch.setattr(planner, "hybrid_search", lambda like, **k: [
        _hit("far.md", 52.0, 7.0),   # ~111 km
        _hit("near.md", 51.1, 7.0),  # ~11 km
    ])
    out = planner.gather_candidates(["城堡"], [], origin, max_km=500)
    assert [c["file"] for c in out] == ["near.md", "far.md"]


# --- leg_durations: parallel routing, failure tolerance ------------------

def test_leg_durations_picks_fastest_option(monkeypatch):
    def fake_route(frm, to, mode, departure=None):
        return {"options": [{"duration_min": 90}, {"duration_min": 60}]}

    monkeypatch.setattr(planner, "route", fake_route)
    legs = planner.leg_durations("Essen", [{"file": "Koeln"}], "transit")
    assert legs == [{"from": "Essen", "to": "Koeln", "duration_min": 60}]


def test_leg_durations_marks_unresolvable_hop_unknown(monkeypatch):
    def fake_route(frm, to, mode, departure=None):
        if to == "Atlantis":
            raise LookupError("Atlantis")
        return {"options": [{"duration_min": 30}]}

    monkeypatch.setattr(planner, "route", fake_route)
    legs = planner.leg_durations("Essen",
                                 [{"file": "Koeln"}, {"file": "Atlantis"}],
                                 "car")
    assert legs[0]["duration_min"] == 30
    assert legs[1]["duration_min"] is None  # unknown, not a crash


def test_leg_durations_empty_stops():
    assert planner.leg_durations("Essen", [], "transit") == []


# --- along_route: corridor filtering and detour ranking ------------------

_LINE = {"type": "LineString",
         "coordinates": [[7.0, 51.0], [8.0, 51.0], [9.0, 51.0]]}


def _fake_car_route(frm, to, mode, departure=None):
    return {"from": {"name": frm, "lat": 51.0, "lon": 7.0},
            "to": {"name": to, "lat": 51.0, "lon": 9.0},
            "geometry": _LINE}


def test_along_route_keeps_only_corridor_stops(monkeypatch):
    monkeypatch.setattr(planner, "route", _fake_car_route)
    # on-corridor (~5 km off the line) vs far (~110 km north).
    monkeypatch.setattr(planner, "hybrid_search", lambda like, **k: [
        _hit("OnRoute.md", 51.05, 8.0),
        _hit("FarNorth.md", 52.0, 8.0),
    ])
    out = planner.along_route("Essen", "Kassel", ["城堡"], corridor_km=25)
    files = [s["file"] for s in out["stops"]]
    assert "OnRoute.md" in files
    assert "FarNorth.md" not in files


def test_along_route_ranks_by_detour(monkeypatch):
    monkeypatch.setattr(planner, "route", _fake_car_route)
    monkeypatch.setattr(planner, "hybrid_search", lambda like, **k: [
        _hit("Closer.md", 51.02, 8.0),
        _hit("Farther.md", 51.15, 8.0),
    ])
    out = planner.along_route("Essen", "Kassel", ["城堡"], corridor_km=25)
    assert [s["file"] for s in out["stops"]] == ["Closer.md", "Farther.md"]
    assert out["stops"][0]["detour_km"] <= out["stops"][1]["detour_km"]


def test_along_route_raises_without_geometry(monkeypatch):
    def no_geom(frm, to, mode, departure=None):
        return {"from": {"name": frm, "lat": 51.0, "lon": 7.0},
                "to": {"name": to, "lat": 51.0, "lon": 9.0},
                "geometry": None}

    monkeypatch.setattr(planner, "route", no_geom)
    with pytest.raises(RuntimeError):
        planner.along_route("Essen", "Kassel", ["城堡"])


# --- _build_synth_input: duration formatting edge cases ------------------

def test_synth_input_formats_durations():
    constraints = {"origin": "Essen", "likes": ["城堡"], "mode": "car", "days": 1}
    day_plans = [{"stops": [
        {"file": "A", "heading": "H", "text": "desc"},
        {"file": "B", "heading": "H", "text": "desc"},
    ], "legs": [{"duration_min": 75}, {"duration_min": None}]}]
    text = planner._build_synth_input(constraints, day_plans)
    assert "1h15m from previous" in text   # 75 min -> 1h15m
    assert "? from previous" in text        # None -> ?
    assert "Origin: Essen" in text


def test_gather_keeps_named_pois_capped(monkeypatch):
    origin = {"lat": 51.0, "lon": 7.0}
    pois = [{"name": f"POI{i}", "lat": 51.1, "lon": 7.0} for i in range(9)]
    pois.insert(0, {"lat": 51.1, "lon": 7.0})  # nameless: dropped
    hit = types.SimpleNamespace(file="A.md", heading="H", score=0.9, text="t",
                                geo={"lat": 51.1, "lon": 7.0}, pois=pois)
    monkeypatch.setattr(planner, "hybrid_search", lambda like, **k: [hit])
    out = planner.gather_candidates(["城堡"], [], origin)
    assert out[0]["pois"] == [f"POI{i}" for i in range(6)]  # named only, max 6


def test_synth_input_lists_sights_when_present():
    constraints = {"origin": "Essen", "likes": ["城堡"], "mode": "car", "days": 1}
    day_plans = [{"stops": [
        {"file": "A", "heading": "H", "text": "desc",
         "pois": ["Schloss Broich", "Aquarius Wassermuseum"]},
        {"file": "B", "heading": "H", "text": "desc"},   # no pois key: no line
    ], "legs": [{"duration_min": 30}, {"duration_min": 20}]}]
    text = planner._build_synth_input(constraints, day_plans)
    assert "Sights: Schloss Broich, Aquarius Wassermuseum" in text
    assert "Notes: desc" in text
    assert text.count("Sights:") == 1


def test_synth_input_zero_duration_is_unknown():
    # 0 is falsy, so the current formatter renders it as "?" — pin that.
    constraints = {"origin": "Essen", "likes": ["城堡"], "mode": "car", "days": 1}
    day_plans = [{"stops": [{"file": "A", "heading": "H", "text": "d"}],
                  "legs": [{"duration_min": 0}]}]
    text = planner._build_synth_input(constraints, day_plans)
    assert "? from previous" in text
