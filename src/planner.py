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
import re

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


# Cheap deterministic fallbacks, so a flaky LLM parse cannot wipe out the
# interests or days the user plainly stated in the text.
_DAY_PATTERNS = [
    (re.compile(r"([一二三四五六七八九十两]|\d+)\s*天"), None),
    (re.compile(r"(\d+)\s*days?", re.IGNORECASE), None),
]
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _days_from_text(query: str) -> int | None:
    for pat, _ in _DAY_PATTERNS:
        m = pat.search(query)
        if m:
            tok = m.group(1)
            if tok.isdigit():
                return int(tok)
            return _CN_NUM.get(tok)
    return None


def _interests_from_text(query: str) -> list[str]:
    """Pull any known interest keyword the user literally wrote."""
    return [kw for kw in INTEREST_TO_HEADING if kw in query][:5]


def parse_constraints(query: str, generate) -> dict:
    """LLM-parse the request into structured constraints, with deterministic
    fallbacks from the raw text. Raises ValueError if the origin cannot be
    resolved against the corpus gazetteer."""
    try:
        raw = "".join(generate(None, query, None, _PARSE_SYSTEM)).strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        data = {}

    origin = (data.get("origin") or "").strip()
    if not origin or geocode(origin) is None:
        raise ValueError(f"无法定位出发地: {origin or '(未识别)'}")

    # Keep only clean LLM interests (drop '?' placeholders and unknowns), then
    # union with keywords found literally in the text.
    llm_likes = [str(x).strip() for x in (data.get("likes") or [])
                 if x and "?" not in str(x)]
    text_likes = _interests_from_text(query)
    likes = list(dict.fromkeys(llm_likes + text_likes))[:5] or ["景点"]

    # The user's literal text wins over the LLM for day count: a number
    # written in the request ("三天") is ground truth the model sometimes
    # flattens to the default of 1.
    text_days = _days_from_text(query)
    llm_days = data.get("days")
    llm_days = int(llm_days) if isinstance(llm_days, int) and llm_days >= 1 else None
    days = text_days or llm_days or 1

    text_excludes = [kw for kw in ("夜店", "夜生活") if kw in query]
    excludes = list(dict.fromkeys(
        [str(x).strip() for x in (data.get("excludes") or []) if x and "?" not in str(x)]
        + text_excludes))[:5]

    mode = "car" if (data.get("mode") == "car" or "开车" in query
                     or "自驾" in query or "drive" in query.lower()) else "transit"

    return {"origin": origin, "days": max(1, days), "likes": likes,
            "excludes": excludes, "mode": mode}


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
    exclude_headings = sorted({h for e in excludes if (h := _heading_for(e))})
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
        # When the interest maps to a known section (城堡 -> Sehenswürdigkeiten),
        # pin retrieval to that heading so a "美食" request cannot return a
        # transport paragraph that merely mentions food.
        hits = hybrid_search(
            like, top_k=10, collection=PLANNER_COLLECTION, geo=geo,
            headings=[heading] if heading else None,
            deny_headings=exclude_headings or None,
        )
        for h in hits:
            if not h.geo:
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


def cluster_by_day(candidates: list[dict], days: int) -> list[list[dict]]:
    """Partition candidates into `days` geographic clusters via k-means on
    lat/lon, so each day stays in one area instead of criss-crossing.

    Deterministic: seeds are the candidates farthest apart, not random, so
    the same request yields the same plan (and no RNG, which the workflow
    sandbox forbids anyway)."""
    if days <= 1 or len(candidates) <= days:
        return [candidates]

    pts = [(c["geo"]["lat"], c["geo"]["lon"]) for c in candidates]
    # Seed with the two farthest points, then greedily add the point maximizing
    # distance to existing seeds — a spread-out, deterministic init.
    seeds = [0]
    while len(seeds) < days:
        far = max(range(len(pts)),
                  key=lambda i: min(_haversine_km(
                      {"lat": pts[i][0], "lon": pts[i][1]},
                      {"lat": pts[s][0], "lon": pts[s][1]}) for s in seeds))
        if far in seeds:
            break
        seeds.append(far)
    centers = [pts[s] for s in seeds]

    for _ in range(10):  # Lloyd iterations; converges fast on a few dozen pts
        clusters: list[list[int]] = [[] for _ in centers]
        for i, p in enumerate(pts):
            j = min(range(len(centers)), key=lambda c: _haversine_km(
                {"lat": p[0], "lon": p[1]},
                {"lat": centers[c][0], "lon": centers[c][1]}))
            clusters[j].append(i)
        new_centers = []
        for cl in clusters:
            if cl:
                new_centers.append((sum(pts[i][0] for i in cl) / len(cl),
                                    sum(pts[i][1] for i in cl) / len(cl)))
            else:
                new_centers.append(centers[len(new_centers)])
        if new_centers == centers:
            break
        centers = new_centers

    groups = [[candidates[i] for i in cl] for cl in clusters if cl]
    # Order days by proximity of each cluster's nearest stop to the origin.
    groups.sort(key=lambda g: min(c["dist_km"] for c in g))
    return groups


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


def _one_leg(frm: str, to: str, mode: str) -> dict:
    leg = {"from": frm, "to": to, "duration_min": None}
    try:
        result = route(frm, to, mode)
        leg["duration_min"] = min(o["duration_min"] for o in result["options"])
    except (LookupError, RuntimeError):
        pass
    return leg


def leg_durations(origin: str, stops: list[dict], mode: str) -> list[dict]:
    """Real travel time for each hop (origin -> s1 -> s2 -> ...). Hops are
    independent, so they are routed concurrently — the transit backend is
    the slow part, and a 4-stop day serialized costs 4x a single call. A hop
    the backend cannot resolve is marked unknown, not fatal."""
    from concurrent.futures import ThreadPoolExecutor

    places = [origin] + [s["file"] for s in stops]
    hops = [(places[i], places[i + 1]) for i in range(len(stops))]
    if not hops:
        return []
    with ThreadPoolExecutor(max_workers=min(6, len(hops))) as pool:
        return list(pool.map(lambda h: _one_leg(h[0], h[1], mode), hops))


def _point_to_segment_km(p: dict, a: dict, b: dict) -> float:
    """Approximate distance from point p to segment a-b, in km, using a local
    equirectangular projection (fine at country scale)."""
    import math as _m

    lat0 = _m.radians((a["lat"] + b["lat"]) / 2)
    kx = 111.32 * _m.cos(lat0)  # km per degree lon at this latitude
    ky = 110.57                 # km per degree lat
    ax, ay = a["lon"] * kx, a["lat"] * ky
    bx, by = b["lon"] * kx, b["lat"] * ky
    px, py = p["lon"] * kx, p["lat"] * ky
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0:
        return _m.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    cx, cy = ax + t * dx, ay + t * dy
    return _m.hypot(px - cx, py - cy)


def _min_dist_to_route(geo: dict, line: list[list[float]]) -> float:
    """Shortest distance from a point to a route polyline (list of [lon,lat])."""
    p = {"lat": geo["lat"], "lon": geo["lon"]}
    best = float("inf")
    for i in range(len(line) - 1):
        a = {"lon": line[i][0], "lat": line[i][1]}
        b = {"lon": line[i + 1][0], "lat": line[i + 1][1]}
        best = min(best, _point_to_segment_km(p, a, b))
    return best


def along_route(from_place: str, to_place: str, interests: list[str],
                corridor_km: float = 25.0, top_n: int = 5) -> dict:
    """Stops worth a detour on the drive from A to B.

    Drives the route (OSRM geometry), then for each interest retrieves
    geotagged candidates and keeps those within corridor_km of the line,
    ranked by how little they stray from it. Reuses the same geo+heading
    retrieval as the day planner.
    """
    result = route(from_place, to_place, "car")
    geom = result.get("geometry")
    if not geom or geom.get("type") != "LineString":
        raise RuntimeError("驾车路线没有几何信息,无法做走廊推荐")
    line = geom["coordinates"]

    frm, to = result["from"], result["to"]
    mid = {"lat": (frm["lat"] + to["lat"]) / 2, "lon": (frm["lon"] + to["lon"]) / 2}
    # Radius that comfortably covers the corridor from the route's midpoint.
    span_km = _haversine_km(frm, to)
    geo = {"lat": mid["lat"], "lon": mid["lon"], "radius_km": span_km / 2 + corridor_km}

    by_file: dict[str, dict] = {}
    for like in interests or ["景点"]:
        heading = _heading_for(like)
        hits = hybrid_search(like, top_k=15, collection=PLANNER_COLLECTION, geo=geo,
                             headings=[heading] if heading else None)
        for h in hits:
            if not h.geo:
                continue
            detour = _min_dist_to_route(h.geo, line)
            if detour > corridor_km:
                continue
            prev = by_file.get(h.file)
            if prev is None or detour < prev["detour_km"]:
                by_file[h.file] = {
                    "file": h.file, "heading": h.heading, "geo": h.geo,
                    "detour_km": round(detour, 1), "text": h.text[:200],
                    "interest": like,
                }
    stops = sorted(by_file.values(), key=lambda c: c["detour_km"])[:top_n]
    return {"from": frm, "to": to, "geometry": geom,
            "corridor_km": corridor_km, "stops": stops}


# A multi-day trip can reach farther than a day trip, so the search radius
# grows with the number of days (capped so it stays regional).
def _radius_for_days(days: int) -> float:
    return min(150.0 + (days - 1) * 120.0, 500.0)


def _build_synth_input(constraints: dict, day_plans: list[dict]) -> str:
    lines = [f"Origin: {constraints['origin']}",
             f"Interests: {', '.join(constraints['likes'])}",
             f"Mode: {constraints['mode']}",
             f"Days: {constraints['days']}", ""]
    for d, day in enumerate(day_plans, 1):
        lines.append(f"--- Day {d} ---")
        for i, s in enumerate(day["stops"]):
            dur = day["legs"][i]["duration_min"]
            travel = f"{dur // 60}h{dur % 60:02d}m" if dur else "?"
            lines.append(f"{i + 1}. {s['file']} ({s['heading']}, {travel} from "
                         f"previous) — {s['text'][:150]}")
        lines.append("")
    return "\n".join(lines)


def plan_stream(query: str, collection: str | None = None):
    """Yield planning events: parse, candidates, per-day routes, synthesized
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
    days = constraints["days"]

    # 2. Retrieve geotagged candidates; radius scales with trip length.
    candidates = gather_candidates(constraints["likes"], constraints["excludes"],
                                   origin_geo, max_km=_radius_for_days(days))
    if not candidates:
        yield {"type": "done", "answer": "没有找到符合条件的候选地点,试试放宽兴趣或换个出发地。",
               "stops": [], "trace": None}
        return
    yield {"type": "candidates", "count": len(candidates),
           "sample": [c["file"] for c in candidates[:8]]}

    # 3. Partition candidates into geographic day-clusters, then chain each.
    clusters = cluster_by_day(candidates, days)
    day_plans = []
    all_stops = []
    for d, cluster in enumerate(clusters, 1):
        # Each day starts from the origin (P2 keeps it simple; overnight
        # lodging is a later enhancement).
        stops = order_stops(origin_geo, cluster, max_stops=4)
        if not stops:
            continue
        legs = leg_durations(constraints["origin"], stops, constraints["mode"])
        day_plans.append({"day": d, "stops": stops, "legs": legs})
        all_stops.extend(stops)
    yield {"type": "cluster", "days": len(day_plans),
           "stops_per_day": [[s["file"] for s in dp["stops"]] for dp in day_plans]}

    total_min = sum(leg["duration_min"] or 0
                    for dp in day_plans for leg in dp["legs"])
    yield {"type": "routing",
           "legs": [leg for dp in day_plans for leg in dp["legs"]],
           "total_travel_min": total_min}

    # 4. Synthesize the itinerary from concrete stops and durations.
    synth_input = _build_synth_input(constraints, day_plans)
    parts: list[str] = []
    try:
        for delta in generate(synth_input, query, None, _SYNTH_SYSTEM):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except RuntimeError as e:
        yield {"type": "error", "detail": str(e)}
        return

    stops_out = [{"file": s["file"], "heading": s["heading"], "geo": s["geo"],
                  "dist_km": s["dist_km"], "day": dp["day"]}
                 for dp in day_plans for s in dp["stops"]]
    yield {
        "type": "done",
        "answer": "".join(parts),
        "stops": stops_out,
        "origin": origin_geo,
        "trace": {"provider": provider, "model": model,
                  "candidates": len(candidates), "days": len(day_plans),
                  "stops": len(stops_out), "total_travel_min": total_min},
    }
