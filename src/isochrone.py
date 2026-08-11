"""Isochrones: colour the map by how far you can drive in N minutes.

Given a centre point, sample a grid of destinations across Germany and ask
OSRM for the driving time from the centre to every grid point in ONE request
(the /table service), then bucket each reachable point into a time band. The
frontend paints the bands, turning "1 hour from Essen" into a coloured region.

One HTTP call, not one-per-point: /table returns the whole centre->grid time
row at once, so a 12x12 grid costs a single request, not 144.
"""

# Default time bands (minutes) and the order they stack, nearest first.
DEFAULT_BANDS = (30, 60, 90, 120)

# How far around the centre to sample, in degrees. ~2.2 deg lat is ~245 km,
# comfortably beyond a 120-minute drive, so the grid stays dense where points
# are actually reachable instead of wasting samples on the far side of Germany.
_SPAN_LAT = 2.2
_SPAN_LON = 3.2  # wider: a degree of longitude is shorter at these latitudes


def build_grid(center: dict, n: int = 12) -> list[dict]:
    """An n x n lat/lon grid centred on `center`. Returns [{lat, lon}, ...].

    Centred on the query point (not a fixed Germany box) so samples stay
    dense in the reachable region. A regular grid rather than the corpus POIs
    keeps the isochrone independent of the vector store."""
    n = max(2, min(n, 25))  # keep the /table matrix a sane size
    lat0, lon0 = center["lat"], center["lon"]
    lats = [lat0 - _SPAN_LAT + 2 * _SPAN_LAT * i / (n - 1) for i in range(n)]
    lons = [lon0 - _SPAN_LON + 2 * _SPAN_LON * j / (n - 1) for j in range(n)]
    return [{"lat": lat, "lon": lon} for lat in lats for lon in lons]


def band_for(minutes: float | None, bands=DEFAULT_BANDS) -> int | None:
    """The index of the first band a duration falls into, or None if the
    point is unreachable or beyond the largest band."""
    if minutes is None:
        return None
    for i, edge in enumerate(bands):
        if minutes <= edge:
            return i
    return None


def _table_durations(center: dict, grid: list[dict]) -> list[float | None]:
    """OSRM /table: driving minutes from center to each grid point, in one
    request. Returns a list aligned with `grid`; None where unreachable."""
    from .routing import OSRM_URL, _get

    # Coordinate string: source first, then every grid point.
    coords = [f"{center['lon']},{center['lat']}"]
    coords += [f"{g['lon']},{g['lat']}" for g in grid]
    url = f"{OSRM_URL}/table/v1/driving/" + ";".join(coords)
    resp = _get(url, params={"sources": "0"}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table returned {data.get('code')}")
    # durations[0] is the row from the single source to all destinations;
    # element 0 of that row is source->source (~0), so skip it. A code:Ok
    # response with a missing/empty/null matrix still reaches here (seen with
    # flaky upstreams and proxies), so validate the shape and fail as a clean
    # RuntimeError -> 502, not a bare KeyError/IndexError -> 500.
    durations = data.get("durations")
    if not isinstance(durations, list) or not durations or durations[0] is None:
        raise RuntimeError("OSRM table returned no duration matrix")
    row = durations[0][1:]
    return [None if d is None else d / 60.0 for d in row]


# OSRM /table handles a 14x14 grid (197 coords) fine; stay under that
# known-good size when batching candidate coordinates.
_MAX_TABLE_POINTS = 150


def reach_filter(center: dict, budget_min: float, hits,
                 table_fn=None) -> tuple[list[dict], int]:
    """Filter retrieval hits by REAL driving time, not straight-line distance.

    Collects each hit's article centroid plus the POIs mentioned in its text,
    asks OSRM for the driving minutes to all of them in ONE /table call, then
    keeps only what is reachable within `budget_min`. The isochrone grid is
    for painting; candidates get exact per-point times because a single table
    call prices them all anyway.

    Returns (kept, dropped): `kept` are hit dicts carrying `drive_min` (the
    centroid time) and per-POI `drive_min`, in the original relevance order;
    `dropped` counts candidates that were retrieved inside the straight-line
    prefilter circle but are NOT drivable within budget — the number that
    makes the circle-vs-isochrone story concrete. Hits without coordinates
    cannot be verified and count as dropped. `table_fn` is injectable for
    tests; it defaults to the OSRM-backed `_table_durations`.
    """
    table_fn = table_fn or _table_durations

    # Dedup coordinates across centroids and POIs: one table column each.
    index_of: dict[tuple, int] = {}
    points: list[dict] = []

    def _slot(lat, lon) -> int | None:
        key = (round(lat, 4), round(lon, 4))
        if key in index_of:
            return index_of[key]
        if len(points) >= _MAX_TABLE_POINTS:
            return None
        index_of[key] = len(points)
        points.append({"lat": lat, "lon": lon})
        return index_of[key]

    slots = []  # per hit: (centroid_slot, [(poi, poi_slot), ...])
    for h in hits:
        geo = h.geo
        c_slot = _slot(geo["lat"], geo["lon"]) if geo else None
        poi_slots = [(p, _slot(p["lat"], p["lon"]))
                     for p in (h.pois or [])
                     if "lat" in p and "lon" in p]
        slots.append((c_slot, poi_slots))

    minutes = table_fn(center, points) if points else []

    kept: list[dict] = []
    dropped = 0
    for h, (c_slot, poi_slots) in zip(hits, slots):
        drive = minutes[c_slot] if c_slot is not None else None
        if drive is None or drive > budget_min:
            dropped += 1
            continue
        pois = []
        for poi, p_slot in poi_slots:
            p_min = minutes[p_slot] if p_slot is not None else None
            if p_min is None or p_min > budget_min:
                continue
            pois.append({**poi, "drive_min": round(p_min, 1)})
        kept.append({**vars(h), "pois": pois, "drive_min": round(drive, 1)})
    return kept, dropped


def isochrone(center: dict, grid_n: int = 12, bands=DEFAULT_BANDS) -> dict:
    """Compute drive-time bands from `center` over a grid.

    Returns {center, bands, cell, points: [{lat, lon, minutes, band}]} where
    band is the DEFAULT_BANDS index (0 = innermost) and cell is the grid
    half-spacing {lat, lon} so the frontend can paint edge-to-edge squares
    that tile into continuous bands. Unreachable / too-far points are
    dropped so only coloured cells are painted."""
    n = max(2, min(grid_n, 25))
    grid = build_grid(center, n)
    durations = _table_durations(center, grid)
    points = []
    for g, minutes in zip(grid, durations):
        b = band_for(minutes, bands)
        if b is None:
            continue
        points.append({"lat": g["lat"], "lon": g["lon"],
                       "minutes": round(minutes, 1), "band": b})
    # Half the spacing between adjacent grid points: a square of this half-size
    # centred on each point meets its neighbours with no gap.
    cell = {"lat": _SPAN_LAT / (n - 1), "lon": _SPAN_LON / (n - 1)}
    return {"center": center, "bands": list(bands), "cell": cell,
            "points": points}
