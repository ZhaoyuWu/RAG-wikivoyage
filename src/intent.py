"""Lightweight intent detection: is this question really a routing request?

Rules extract candidate place pairs; the corpus gazetteer then validates
them. Only when both places resolve does the question leave the RAG path,
so false positives degrade gracefully into normal retrieval.
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
