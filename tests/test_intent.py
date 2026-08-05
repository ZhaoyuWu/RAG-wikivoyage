"""Route-intent detection: does a question leave the RAG path for routing?

detect_route_intent only pulls a question out of retrieval when BOTH places
resolve in the gazetteer, so false positives degrade into normal RAG. These
tests stub geocode with a small allow-list so no real gazetteer is needed,
and pin the tri-lingual patterns plus the car/transit split.
"""

import src.intent as intent

KNOWN = {"essen", "berlin", "köln", "goslar", "hamburg"}


def _fake_geocode(place):
    return {"name": place} if place.strip().lower() in KNOWN else None


def _detect(question, monkeypatch):
    monkeypatch.setattr(intent, "geocode", _fake_geocode)
    return intent.detect_route_intent(question)


def test_chinese_from_to(monkeypatch):
    r = _detect("从Essen到Berlin怎么去", monkeypatch)
    assert r == {"from_place": "Essen", "to_place": "Berlin", "mode": "transit"}


def test_chinese_a_to_b_duration(monkeypatch):
    r = _detect("Essen到Köln多久", monkeypatch)
    assert r["from_place"] == "Essen" and r["to_place"] == "Köln"


def test_german_von_nach(monkeypatch):
    r = _detect("Wie komme ich von Essen nach Berlin?", monkeypatch)
    assert r["from_place"] == "Essen" and r["to_place"] == "Berlin"


def test_english_from_to(monkeypatch):
    r = _detect("How long from Essen to Hamburg?", monkeypatch)
    assert r["from_place"] == "Essen" and r["to_place"] == "Hamburg"


def test_car_marker_switches_mode(monkeypatch):
    r = _detect("从Essen到Berlin开车多久", monkeypatch)
    assert r["mode"] == "car"


def test_driving_english_marker(monkeypatch):
    r = _detect("from Essen to Berlin by driving, how long", monkeypatch)
    assert r["mode"] == "car"


def test_no_route_hint_returns_none(monkeypatch):
    # No routing verb at all -> stays on the RAG path.
    assert _detect("Essen有什么好吃的", monkeypatch) is None


def test_unknown_place_falls_back_to_rag(monkeypatch):
    # Pattern matches but Atlantis is not in the gazetteer -> None.
    assert _detect("从Essen到Atlantis怎么去", monkeypatch) is None


def test_both_unknown_returns_none(monkeypatch):
    assert _detect("从Narnia到Atlantis怎么走", monkeypatch) is None


def test_geocode_failure_is_swallowed(monkeypatch):
    # If the gazetteer itself errors, we must not crash the request.
    def boom(_place):
        raise RuntimeError("gazetteer offline")

    monkeypatch.setattr(intent, "geocode", boom)
    assert intent.detect_route_intent("从Essen到Berlin怎么去") is None
