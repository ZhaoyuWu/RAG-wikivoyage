"""Trip planner: orchestrate retrieval, geo-clustering, and routing into an
itinerary from one natural-language request.

This is the agentic layer that makes the three standalone tools multiply:
retrieval says WHERE to go, the geo coordinates say how to string stops
together, and the routing backend says HOW LONG it takes. The orchestration
is deliberately guided (parse -> retrieve -> cluster -> route -> synthesize)
rather than fully autonomous tool-calling: the skeleton is fixed and
controllable, while each step may use the LLM internally.
"""

import json
import math

from .rag import _pick_provider
from .retrieval import hybrid_search
from .routing import geocode, route

PLANNER_COLLECTION = "wikivoyage"

# Chinese/English interest words -> the Wikivoyage section headings that hold
# them, so a "美食" request can be narrowed to the Küche sections.
INTEREST_TO_HEADING = {
    "美食": "Küche", "吃": "Küche", "餐厅": "Küche", "food": "Küche",
    "cuisine": "Küche", "essen": "Küche",
    "景点": "Sehenswürdigkeiten", "城堡": "Sehenswürdigkeiten",
    "古城": "Sehenswürdigkeiten", "教堂": "Sehenswürdigkeiten",
    "博物馆": "Sehenswürdigkeiten", "sights": "Sehenswürdigkeiten",
    "castle": "Sehenswürdigkeiten", "museum": "Sehenswürdigkeiten",
    "活动": "Aktivitäten", "徒步": "Aktivitäten", "hiking": "Aktivitäten",
    "wandern": "Aktivitäten", "activities": "Aktivitäten",
    "购物": "Einkaufen", "shopping": "Einkaufen",
    "夜生活": "Nachtleben", "夜店": "Nachtleben", "nightlife": "Nachtleben",
}

_PARSE_SYSTEM = (
    "You extract trip-planning constraints from a user's message. Reply with "
    "ONLY a JSON object, no prose:\n"
    '{"origin": "<city name, German spelling if known>", '
    '"days": <int, default 1>, '
    '"likes": ["<interest keyword>", ...], '
    '"excludes": ["<thing to avoid>", ...], '
    '"mode": "car" | "transit"}\n'
    "Keep origin as a plain city name. likes/excludes are short keywords "
    "(e.g. 城堡, 徒步, 美食, 夜店). Default mode to transit unless the user "
    "clearly wants to drive."
)

_SYNTH_SYSTEM = (
    "You are a travel planner. Given an origin, a list of candidate stops "
    "with short descriptions, and the travel time between them, write a "
    "concise day itinerary in the language of the user's request. Give each "
    "stop a rough time slot and one sentence on why it fits. Be practical. "
    "Do not invent stops that are not in the list."
)


def _haversine_km(a: dict, b: dict) -> float:
    """Great-circle distance between two {lat, lon} points, in km."""
    r = 6371.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp = math.radians(b["lat"] - a["lat"])
    dl = math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_constraints(query: str, generate) -> dict:
    """LLM-parse the request into structured constraints. Raises ValueError
    if the origin cannot be resolved against the corpus gazetteer."""
    raw = "".join(generate(None, query, None, _PARSE_SYSTEM)).strip()
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    data = json.loads(raw)
    origin = (data.get("origin") or "").strip()
    if not origin or geocode(origin) is None:
        raise ValueError(f"无法定位出发地: {origin or '(未识别)'}")
    return {
        "origin": origin,
        "days": max(1, int(data.get("days") or 1)),
        "likes": [str(x) for x in (data.get("likes") or []) if x][:5] or ["景点"],
        "excludes": [str(x) for x in (data.get("excludes") or []) if x][:5],
        "mode": "car" if data.get("mode") == "car" else "transit",
    }


def _heading_for(interest: str) -> str | None:
    lowered = interest.lower()
    for key, heading in INTEREST_TO_HEADING.items():
        if key in interest or key in lowered:
            return heading
    return None


def gather_candidates(likes: list[str], excludes: list[str],
                      origin_geo: dict, max_km: float = 150.0) -> list[dict]:
    """Retrieve geotagged candidate stops for each interest, near the origin.

    De-duplicates by article file, keeping the best-scoring chunk, and drops
    anything farther than max_km straight-line from the origin. The default
    radius is tuned for a day trip; multi-day planning (P2) widens it.
    """
    exclude_headings = {h for e in excludes if (h := _heading_for(e))}
    # Constrain retrieval to a geo radius FIRST, then rank by relevance within
    # it. Abstract interest words ("城堡", "徒步") match articles all over
    # Germany, so filtering by distance after a top-k retrieval would let a
    # castle 400 km away crowd out a closer one. The corpus's own geo filter
    # (already used by search mode) makes proximity a hard constraint.
    geo = {"lat": origin_geo["lat"], "lon": origin_geo["lon"], "radius_km": max_km}
    by_file: dict[str, dict] = {}
    for like in likes:
        heading = _heading_for(like)
        if heading and heading in exclude_headings:
            continue
        hits = hybrid_search(like, top_k=10, collection=PLANNER_COLLECTION, geo=geo)
        for h in hits:
            if not h.geo:
                continue
            if exclude_headings and h.heading in exclude_headings:
                continue
            dist = _haversine_km(origin_geo, h.geo)
            prev = by_file.get(h.file)
            if prev is None or h.score > prev["score"]:
                by_file[h.file] = {
                    "file": h.file, "heading": h.heading, "score": h.score,
                    "geo": h.geo, "dist_km": round(dist, 1),
                    "text": h.text[:300], "interest": like,
                }
    return sorted(by_file.values(), key=lambda c: c["dist_km"])


def order_stops(origin_geo: dict, candidates: list[dict],
                max_stops: int = 4, max_hop_km: float = 100.0) -> list[dict]:
    """Greedy nearest-neighbour chain from the origin through the closest
    candidates, so the day forms a route instead of a scatter. Stops the
    chain once the next-nearest hop exceeds max_hop_km, so a day trip does
    not sprawl across the country."""
    remaining = candidates[:12]  # cap the search space
    chain: list[dict] = []
    current = {"lat": origin_geo["lat"], "lon": origin_geo["lon"]}
    while remaining and len(chain) < max_stops:
        nxt = min(remaining, key=lambda c: _haversine_km(current, c["geo"]))
        if chain and _haversine_km(current, nxt["geo"]) > max_hop_km:
            break  # next stop is too far for a coherent day
        chain.append(nxt)
        remaining.remove(nxt)
        current = nxt["geo"]
    return chain


def leg_durations(origin: str, stops: list[dict], mode: str) -> list[dict]:
    """Real travel time for each hop (origin -> s1 -> s2 -> ...). A hop that
    the routing backend cannot resolve is marked unknown, not fatal."""
    legs = []
    prev = origin
    for stop in stops:
        leg = {"from": prev, "to": stop["file"], "duration_min": None}
        try:
            result = route(prev, stop["file"], mode)
            leg["duration_min"] = min(o["duration_min"] for o in result["options"])
        except (LookupError, RuntimeError):
            pass
        legs.append(leg)
        prev = stop["file"]
    return legs


def _build_synth_input(constraints: dict, stops: list[dict],
                       legs: list[dict]) -> str:
    lines = [f"Origin: {constraints['origin']}",
             f"Interests: {', '.join(constraints['likes'])}",
             f"Mode: {constraints['mode']}", "", "Candidate stops in order:"]
    for i, s in enumerate(stops):
        dur = legs[i]["duration_min"]
        travel = f"{dur // 60}h{dur % 60:02d}m" if dur else "?"
        lines.append(f"{i + 1}. {s['file']} ({s['heading']}, {travel} from previous) "
                     f"— {s['text'][:160]}")
    return "\n".join(lines)


def plan_stream(query: str, collection: str | None = None):
    """Yield planning events: parse, candidates, itinerary route, synthesized
    text deltas, and a final done event with the map-ready stops."""
    provider, model, generate = _pick_provider(collection or PLANNER_COLLECTION)

    # 1. Parse the request into structured constraints.
    try:
        constraints = parse_constraints(query, generate)
    except (ValueError, json.JSONDecodeError) as e:
        yield {"type": "error", "detail": f"无法理解行程请求: {e}"}
        return
    yield {"type": "parse", **constraints}

    origin_geo = geocode(constraints["origin"])

    # 2. Retrieve geotagged candidates for each interest.
    candidates = gather_candidates(constraints["likes"], constraints["excludes"],
                                   origin_geo)
    if not candidates:
        yield {"type": "done", "answer": "没有找到符合条件的候选地点,试试放宽兴趣或换个出发地。",
               "stops": [], "trace": None}
        return
    yield {"type": "candidates", "count": len(candidates),
           "sample": [c["file"] for c in candidates[:8]]}

    # 3. Chain the closest ones into a day route.
    stops = order_stops(origin_geo, candidates)
    yield {"type": "cluster", "stops": [s["file"] for s in stops]}

    # 4. Real travel time for each hop.
    legs = leg_durations(constraints["origin"], stops, constraints["mode"])
    total_min = sum(leg["duration_min"] or 0 for leg in legs)
    yield {"type": "routing", "legs": legs, "total_travel_min": total_min}

    # 5. Synthesize an itinerary from the concrete stops and durations.
    synth_input = _build_synth_input(constraints, stops, legs)
    parts: list[str] = []
    try:
        for delta in generate(synth_input, query, None, _SYNTH_SYSTEM):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except RuntimeError as e:
        yield {"type": "error", "detail": str(e)}
        return

    yield {
        "type": "done",
        "answer": "".join(parts),
        "stops": [{"file": s["file"], "heading": s["heading"],
                   "geo": s["geo"], "dist_km": s["dist_km"]} for s in stops],
        "origin": origin_geo,
        "trace": {"provider": provider, "model": model,
                  "candidates": len(candidates), "stops": len(stops),
                  "total_travel_min": total_min},
    }
