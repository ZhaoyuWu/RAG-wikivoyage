"""Lightweight intent detection: which pipeline should a free-text query use?

Deterministic rules, validated against the corpus gazetteer where a place is
involved. Detection is deliberately conservative: anything ambiguous falls
back to the RAG path (which itself detects routing questions), so a false
positive degrades into a normal answer, never a broken one.

Detectors:
- detect_route_intent: "A到B怎么去" -> the routing backend (used inside ask).
- detect_plan_intent:  "essen周边1日游" -> the trip planner.
- detect_reach_intent: "Essen出发90分钟车程内..." -> reach search.
- detect_along_intent: "从A到B沿途有什么景点" -> corridor search along the drive.
"""

import re

from .routing import geocode

# "从A到B怎么去/多久", "A到B怎么走" ...
_ZH_PATTERNS = [
    re.compile(r"从\s*(?P<a>[\w()\- ]{2,30}?)\s*(?:到|去)\s*(?P<b>[\w()\- ]{2,30}?)(?:怎么|要|多久|多长|坐|开|走|的|$)"),
    re.compile(r"(?P<a>[\w()\- ]{2,30}?)\s*到\s*(?P<b>[\w()\- ]{2,30}?)\s*(?:怎么去|怎么走|多久|多长时间|要多久)"),
]
# "von A nach B", "from A to B", "A nach B fahren"
_LATIN_PATTERNS = [
    re.compile(r"\bvon\s+(?P<a>[\w()\- ]{2,30}?)\s+nach\s+(?P<b>[\w()\- ]{2,30}?)(?:\?|$|\s+fahren|\s+kommen|\s+mit)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(?P<a>[\w()\- ]{2,30}?)\s+to\s+(?P<b>[\w()\- ]{2,30}?)(?:\?|$|\s+by|\s+how)", re.IGNORECASE),
]

_CAR_MARKERS = ("开车", "自驾", "drive", "driving", "auto", "mit dem auto", "车程")
_ROUTE_HINTS = ("怎么去", "怎么走", "多久", "多长时间", "nach", "how do i get",
                "how long", "wie komme", "到", " to ")


# Trip-planning language. Strong tokens fire alone; a bare day count ("三天")
# needs trip-flavoured company so "三天前问过的" stays on the RAG path.
_STRONG_PLAN = re.compile(
    r"[一二两三四五六七八九十\d]\s*日游|周边游|行程|旅[行游]计划|游玩计划|"
    r"itinerary|day ?trip|tages(?:tour|ausflug)", re.IGNORECASE)
_DAY_COUNT = re.compile(r"[一二两三四五六七八九十\d]\s*(?:天|日|days?|tage)", re.IGNORECASE)
_TRIP_FLAVOUR = re.compile(r"[游玩逛]|出发|计划|安排|plan|trip|reise", re.IGNORECASE)


def detect_plan_intent(question: str) -> bool:
    """True when the question asks for an itinerary, not an answer."""
    if _STRONG_PLAN.search(question):
        return True
    return bool(_DAY_COUNT.search(question) and _TRIP_FLAVOUR.search(question))


# Along-route: an A-to-B pair plus corridor language ("沿途", "on the way").
_ALONG_MARKERS = ("沿途", "路上", "途中", "顺路", "一路",
                  "along the way", "on the way", "unterwegs", "entlang")
_ALONG_PAIRS = [
    re.compile(r"从\s*(?P<a>[\w()\- ]{2,30}?)\s*(?:到|去|至)\s*"
               r"(?P<b>[\w()\- ]{2,30}?)(?:的|,|，|。|\s|沿途|路上|途中|顺路|一路|有|$)"),
    re.compile(r"(?P<a>[\w()\- ]{2,30}?)\s*到\s*"
               r"(?P<b>[\w()\- ]{2,30}?)(?:的|,|，|。|\s|沿途|路上|途中|顺路|一路|有|$)"),
    re.compile(r"\bfrom\s+(?P<a>[\w()\- ]{2,30}?)\s+to\s+(?P<b>[\w()\- ]{2,30}?)(?:\s|,|$)",
               re.IGNORECASE),
    re.compile(r"\bvon\s+(?P<a>[\w()\- ]{2,30}?)\s+nach\s+(?P<b>[\w()\- ]{2,30}?)(?:\s|,|$)",
               re.IGNORECASE),
]


def detect_along_intent(question: str) -> dict | None:
    """Return {from_place, to_place, interests} when the question asks what
    lies ALONG a drivable A-to-B corridor and both places resolve in the
    gazetteer; else None. Interests are literal keywords from the planner's
    vocabulary, defaulting to sights."""
    lowered = question.lower()
    if not any(m in question or m in lowered for m in _ALONG_MARKERS):
        return None
    for pattern in _ALONG_PAIRS:
        m = pattern.search(question)
        if not m:
            continue
        a, b = m.group("a").strip(), m.group("b").strip()
        try:
            if geocode(a) and geocode(b):
                from .planner import INTEREST_TO_HEADING

                interests = [kw for kw in INTEREST_TO_HEADING
                             if kw in question or kw in lowered][:3]
                return {"from_place": a, "to_place": b,
                        "interests": interests or ["景点"]}
        except Exception:
            return None  # gazetteer unavailable: stay on the RAG path
    return None


# Reach: a drive-time budget plus a resolvable centre.
_TIME_BUDGET = re.compile(
    r"(?P<n>\d+|半|[一二两三四五六七八九十]+)\s*(?:个)?\s*"
    r"(?P<u>小时|分钟|min(?:ute[ns]?)?\b|h\b|stunden?)", re.IGNORECASE)
_REACH_MARKERS = ("车程", "以内", "之内", "能到", "可达", "范围", "内",
                  "drive", "erreichbar", "reachable", "within")
_CENTER_PATTERNS = [
    re.compile(r"从?\s*(?P<p>[\w()\- ]{2,20}?)\s*(?:出发|周边|附近|为中心)"),
    re.compile(r"(?:around|von|from)\s+(?P<p>[\w()\- ]{2,20}?)(?:\s|,|$)", re.IGNORECASE),
]
_CN_SMALL = {"半": 0.5, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}


def detect_reach_intent(question: str) -> dict | None:
    """Return {place, minutes} when the question limits itself to a drive-time
    budget around a place that resolves in the gazetteer; else None."""
    m = _TIME_BUDGET.search(question)
    if not m:
        return None
    lowered = question.lower()
    if not any(k in question or k in lowered for k in _REACH_MARKERS):
        return None

    n = m.group("n")
    value = float(n) if n.isdigit() else _CN_SMALL.get(n)
    if value is None:
        return None
    unit = m.group("u").lower()
    minutes = value * 60 if unit.startswith(("小时", "h", "stund")) else value
    minutes = int(max(15, min(minutes, 120)))

    for pattern in _CENTER_PATTERNS:
        mm = pattern.search(question)
        if not mm:
            continue
        place = mm.group("p").strip()
        try:
            if geocode(place):
                return {"place": place, "minutes": minutes}
        except Exception:
            return None  # gazetteer unavailable: stay on the RAG path
    return None


def detect_route_intent(question: str) -> dict | None:
    """Return {from_place, to_place, mode} when the question is a routing
    request whose places both exist in the gazetteer; else None."""
    lowered = question.lower()
    if not any(h in question or h in lowered for h in _ROUTE_HINTS):
        return None

    for pattern in _ZH_PATTERNS + _LATIN_PATTERNS:
        m = pattern.search(question)
        if not m:
            continue
        a, b = m.group("a").strip(), m.group("b").strip()
        try:
            if geocode(a) and geocode(b):
                mode = "car" if any(c in lowered or c in question for c in _CAR_MARKERS) else "transit"
                return {"from_place": a, "to_place": b, "mode": mode}
        except Exception:
            return None  # gazetteer unavailable: stay on the RAG path
    return None
