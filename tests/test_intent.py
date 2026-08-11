"""Route-intent detection: does a question leave the RAG path for routing?

detect_route_intent only pulls a question out of retrieval when BOTH places
resolve in the gazetteer, so false positives degrade into normal RAG. These
tests stub geocode with a small allow-list so no real gazetteer is needed,
and pin the tri-lingual patterns plus the car/transit split.
"""

import pytest

# intent.py imports from routing, which imports httpx at module load; skip
# where the request stack isn't installed (lightweight CI job).
pytest.importorskip("httpx", reason="routing's http stack not installed")

import src.intent as intent  # noqa: E402

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


# --- Plan intent -----------------------------------------------------------

def test_plan_strong_tokens_fire_alone():
    assert intent.detect_plan_intent("essen周边1日游")
    assert intent.detect_plan_intent("帮我安排一个行程")
    assert intent.detect_plan_intent("weekend day trip ideas")


def test_plan_day_count_needs_trip_flavour():
    assert intent.detect_plan_intent("从Essen出发三天,想看城堡")
    # A bare day count without trip language stays on the RAG path.
    assert not intent.detect_plan_intent("三天前我问过社保的事")


def test_plain_question_is_not_plan():
    assert not intent.detect_plan_intent("Essen有什么好吃的")


# --- Reach intent ----------------------------------------------------------

def test_reach_extracts_place_and_minutes(monkeypatch):
    monkeypatch.setattr(intent, "geocode", _fake_geocode)
    r = intent.detect_reach_intent("Essen出发90分钟车程内有什么城堡")
    assert r == {"place": "Essen", "minutes": 90}


def test_reach_hours_convert_and_clamp(monkeypatch):
    monkeypatch.setattr(intent, "geocode", _fake_geocode)
    r = intent.detect_reach_intent("从Essen出发,1小时以内能到的徒步")
    assert r["minutes"] == 60
    # 3 hours clamps to the API's 120-minute ceiling.
    r = intent.detect_reach_intent("从Essen出发三小时车程内")
    assert r["minutes"] == 120


def test_reach_needs_marker_and_known_place(monkeypatch):
    monkeypatch.setattr(intent, "geocode", _fake_geocode)
    # A duration without reach language ("多久") is not a reach query.
    assert intent.detect_reach_intent("开车去柏林要几小时") is None
    # Unknown centre -> RAG path.
    assert intent.detect_reach_intent("Atlantis出发60分钟车程内") is None


def test_reach_geocode_failure_is_swallowed(monkeypatch):
    def boom(_place):
        raise RuntimeError("gazetteer offline")

    monkeypatch.setattr(intent, "geocode", boom)
    assert intent.detect_reach_intent("Essen出发90分钟车程内") is None
