"""A-to-B routing: driving via OSRM, rail/transit via the DB REST API.

Place names are resolved against our own corpus first: every geotagged
Wikivoyage article doubles as a gazetteer entry (title -> centroid).
Both backends are free public services; only place names and
coordinates are sent, never any corpus content.
"""

from functools import lru_cache

import httpx

from .config import get_qdrant_client

WIKIVOYAGE_COLLECTION = "wikivoyage"
OSRM_URL = "https://router.project-osrm.org"
DB_REST_URL = "https://v6.db.transport.rest"

# Public API etiquette: identify the client (Transitous rejects blank agents).
_HEADERS = {"User-Agent": "vault-rag/0.1 (github.com/ZhaoyuWu/RAG-wikivoyage)"}


def _get(url: str, **kwargs):
    return httpx.get(url, headers=_HEADERS, **kwargs)


@lru_cache(maxsize=1)
def _gazetteer() -> dict[str, dict]:
    """title -> {lat, lon} for every geotagged article in the corpus."""
    client = get_qdrant_client()
    places: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=WIKIVOYAGE_COLLECTION,
            limit=2048,
            offset=offset,
            with_payload=["file", "geo"],
            with_vectors=False,
        )
        for p in points:
            geo = p.payload.get("geo")
            if geo:
                places.setdefault(p.payload["file"], geo)
        if offset is None:
            break
    return places


def geocode(place: str) -> dict | None:
    """Resolve a place name to coordinates using the corpus gazetteer."""
    places = _gazetteer()
    if place in places:
        return {"name": place, **places[place]}
    lowered = place.lower()
    for title, geo in places.items():
        if title.lower() == lowered:
            return {"name": title, **geo}
    # Prefix match as a fallback (e.g. "Essen" vs "Essen (Ruhr)")
    candidates = [t for t in places if t.lower().startswith(lowered)]
    if candidates:
        best = min(candidates, key=len)
        return {"name": best, **places[best]}
    return None


def route_car(frm: dict, to: dict) -> dict:
    """Driving route via the public OSRM demo server."""
    url = (f"{OSRM_URL}/route/v1/driving/"
           f"{frm['lon']},{frm['lat']};{to['lon']},{to['lat']}")
    resp = _get(url, params={"overview": "full", "geometries": "geojson"},
                     timeout=20.0)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM returned no route: {data.get('code')}")
    r = data["routes"][0]
    return {
        "options": [{
            "summary": f"Drive, {r['distance'] / 1000:.0f} km",
            "duration_min": round(r["duration"] / 60),
            "transfers": 0,
            "legs": [],
        }],
        "geometry": r["geometry"],
    }


def _db_station(query: str) -> dict:
    resp = _get(f"{DB_REST_URL}/locations",
                     params={"query": query, "results": 1, "poi": "false",
                             "addresses": "false"},
                     timeout=20.0)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise RuntimeError(f"No station found for '{query}'")
    return results[0]


TRANSITOUS_URL = "https://api.transitous.org"


def route_transit_transitous(frm: dict, to: dict, departure: str | None = None) -> dict:
    """Transit fallback via Transitous (MOTIS), queried by coordinates."""
    from datetime import datetime

    params = {"fromPlace": f"{frm['lat']},{frm['lon']}",
              "toPlace": f"{to['lat']},{to['lon']}"}
    if departure:
        params["time"] = departure
    resp = _get(f"{TRANSITOUS_URL}/api/v1/plan", params=params, timeout=25.0)
    resp.raise_for_status()
    itineraries = resp.json().get("itineraries", [])

    options = []
    for it in itineraries[:3]:
        legs = [l for l in it.get("legs", []) if l.get("mode") != "WALK"]
        if not legs:
            continue
        t_dep = datetime.fromisoformat(it["startTime"])
        t_arr = datetime.fromisoformat(it["endTime"])
        lines = " → ".join(l.get("routeShortName") or l.get("mode") for l in legs)
        options.append({
            "summary": f"{t_dep.strftime('%H:%M')} → {t_arr.strftime('%H:%M')}  ({lines})",
            "duration_min": round(it["duration"] / 60),
            "transfers": max(len(legs) - 1, 0),
            "legs": [{
                "line": l.get("routeShortName") or l.get("mode"),
                "from": (l.get("from") or {}).get("name") or "",
                "to": (l.get("to") or {}).get("name") or "",
            } for l in legs],
        })
    if not options:
        raise RuntimeError("No transit connection found")
    return {"options": options, "geometry": None}


def route_transit(frm: dict, to: dict, departure: str | None = None) -> dict:
    """Rail/transit journeys via the community DB REST API."""
    s_from = _db_station(frm["name"])
    s_to = _db_station(to["name"])
    params = {"from": s_from["id"], "to": s_to["id"], "results": 3}
    if departure:
        params["departure"] = departure
    resp = _get(f"{DB_REST_URL}/journeys", params=params, timeout=25.0)
    resp.raise_for_status()
    journeys = resp.json().get("journeys", [])

    options = []
    for j in journeys:
        legs = [leg for leg in j.get("legs", []) if not leg.get("walking")]
        if not legs:
            continue
        dep = legs[0].get("departure") or legs[0].get("plannedDeparture")
        arr = legs[-1].get("arrival") or legs[-1].get("plannedArrival")
        if not (dep and arr):
            continue
        from datetime import datetime

        t_dep = datetime.fromisoformat(dep)
        t_arr = datetime.fromisoformat(arr)
        lines = " → ".join(
            (leg.get("line") or {}).get("name") or "?" for leg in legs
        )
        options.append({
            "summary": f"{t_dep.strftime('%H:%M')} → {t_arr.strftime('%H:%M')}  ({lines})",
            "duration_min": round((t_arr - t_dep).total_seconds() / 60),
            "transfers": len(legs) - 1,
            "legs": [{
                "line": (leg.get("line") or {}).get("name") or "walk",
                "from": (leg.get("origin") or {}).get("name") or "",
                "to": (leg.get("destination") or {}).get("name") or "",
            } for leg in legs],
        })
    if not options:
        raise RuntimeError("No transit connection found")
    return {"options": options, "geometry": None}


def route(from_place: str, to_place: str, mode: str,
          departure: str | None = None) -> dict:
    """departure: ISO 8601 local datetime, e.g. 2026-08-02T09:00. None = now.
    Driving durations are traffic-free estimates, so departure only affects
    transit."""
    frm = geocode(from_place)
    to = geocode(to_place)
    if frm is None:
        raise LookupError(f"Unknown place: {from_place}")
    if to is None:
        raise LookupError(f"Unknown place: {to_place}")

    if departure:
        # Naive datetimes are taken as German local time; both backends
        # want a full ISO timestamp with offset.
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            dt = datetime.fromisoformat(departure)
        except ValueError:
            raise LookupError(f"Invalid departure time: {departure}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
        departure = dt.isoformat()

    if mode == "car":
        try:
            result = route_car(frm, to)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Routing service unreachable: {e}")
    else:
        # Two independent free backends; the community DB API has outages.
        try:
            result = route_transit(frm, to, departure)
        except (httpx.HTTPError, RuntimeError):
            try:
                result = route_transit_transitous(frm, to, departure)
            except httpx.HTTPError as e:
                raise RuntimeError(
                    f"Both transit backends unreachable (DB REST and Transitous): {e}"
                )
    return {"from": frm, "to": to, "mode": mode, **result}
