"""Isochrone logic: grid generation, band bucketing, OSRM table parsing.

The single network call (OSRM /table) is stubbed, so these exercise pure
logic: the centred grid, the duration->band mapping, and dropping
unreachable / too-far points.
"""

import pytest

pytest.importorskip("httpx", reason="isochrone imports routing's http stack")

from src import isochrone as iso  # noqa: E402


def test_grid_is_n_squared_and_centred():
    center = {"lat": 51.0, "lon": 7.0}
    grid = iso.build_grid(center, n=10)
    assert len(grid) == 100
    lats = [g["lat"] for g in grid]
    lons = [g["lon"] for g in grid]
    # Centre sits in the middle of the sampled span.
    assert min(lats) < 51.0 < max(lats)
    assert min(lons) < 7.0 < max(lons)
    assert abs((min(lats) + max(lats)) / 2 - 51.0) < 1e-9


def test_grid_size_is_clamped():
    center = {"lat": 51.0, "lon": 7.0}
    assert len(iso.build_grid(center, n=1)) == 4      # min 2 -> 2x2
    assert len(iso.build_grid(center, n=99)) == 25 * 25  # max 25


def test_band_for_buckets_by_first_edge():
    assert iso.band_for(10) == 0        # <=30
    assert iso.band_for(30) == 0        # inclusive edge
    assert iso.band_for(31) == 1        # <=60
    assert iso.band_for(90) == 2
    assert iso.band_for(120) == 3
    assert iso.band_for(121) is None    # beyond the largest band
    assert iso.band_for(None) is None   # unreachable


def test_isochrone_drops_unreachable_and_far(monkeypatch):
    center = {"name": "Essen", "lat": 51.45, "lon": 7.01}

    # Fake /table: 4-point grid, durations of 20, 200 (too far), None, 80 min.
    def fake_durations(c, grid):
        assert c is center
        return [20.0, 200.0, None, 80.0][:len(grid)]

    monkeypatch.setattr(iso, "build_grid",
                        lambda c, n=12: [{"lat": 51.0 + i, "lon": 7.0} for i in range(4)])
    monkeypatch.setattr(iso, "_table_durations", fake_durations)

    out = iso.isochrone(center)
    # Only the 20-min and 80-min points survive (200 too far, None unreachable).
    assert len(out["points"]) == 2
    bands = sorted(p["band"] for p in out["points"])
    assert bands == [0, 2]                 # 20->band0, 80->band2
    assert out["center"] == center
    assert out["bands"] == list(iso.DEFAULT_BANDS)


class _Hit:
    """Duck-typed stand-in for retrieval.Hit (reach_filter uses vars())."""

    def __init__(self, file, geo, pois=None):
        self.score = 0.5
        self.file = file
        self.heading = "H"
        self.text = "t"
        self.category = "c"
        self.geo = geo
        self.pois = pois


def test_reach_filter_keeps_only_drivable():
    center = {"lat": 51.45, "lon": 7.01}
    hits = [
        _Hit("near.md", {"lat": 51.5, "lon": 7.1}),
        _Hit("far.md", {"lat": 52.9, "lon": 9.9}),      # in circle, 130 min by road
        _Hit("island.md", {"lat": 53.7, "lon": 7.3}),   # OSRM says unreachable
        _Hit("nogeo.md", None),                          # unverifiable
    ]

    def fake_table(c, points):
        assert c is center
        return [40.0, 130.0, None][:len(points)]

    kept, dropped = iso.reach_filter(center, 90, hits, table_fn=fake_table)
    assert [h["file"] for h in kept] == ["near.md"]
    assert kept[0]["drive_min"] == 40.0
    assert dropped == 3


def test_reach_filter_dedups_coords_and_prices_pois():
    center = {"lat": 51.45, "lon": 7.01}
    hit = _Hit("a.md", {"lat": 51.5, "lon": 7.1}, pois=[
        {"name": "Schloss", "lat": 51.5, "lon": 7.1},   # same coord as centroid
        {"name": "FarPoi", "lat": 51.9, "lon": 8.5},
    ])
    calls = []

    def fake_table(c, points):
        calls.append(list(points))
        return [30.0, 95.0]

    kept, dropped = iso.reach_filter(center, 90, [hit], table_fn=fake_table)
    # Centroid and Schloss share one table column; FarPoi gets its own.
    assert len(calls[0]) == 2
    assert dropped == 0
    assert kept[0]["drive_min"] == 30.0
    # The over-budget POI is removed; the reachable one carries its own time.
    assert [(p["name"], p["drive_min"]) for p in kept[0]["pois"]] == [("Schloss", 30.0)]


def test_reach_filter_empty_hits_makes_no_table_call():
    def boom(c, points):
        raise AssertionError("no hits -> no OSRM call")

    assert iso.reach_filter({"lat": 51.0, "lon": 7.0}, 90, [], table_fn=boom) == ([], 0)


def test_table_durations_parses_osrm_row(monkeypatch):
    center = {"lat": 51.0, "lon": 7.0}
    grid = [{"lat": 51.1, "lon": 7.0}, {"lat": 51.2, "lon": 7.0}]

    class _Resp:
        def raise_for_status(self): pass
        # durations in seconds; row[0] is source->source, skipped.
        def json(self):
            return {"code": "Ok", "durations": [[0.0, 1800.0, 3600.0]]}

    monkeypatch.setattr(iso, "_table_durations", iso._table_durations)  # keep real
    import src.routing as routing
    monkeypatch.setattr(routing, "_get", lambda url, **k: _Resp())

    mins = iso._table_durations(center, grid)
    assert mins == [30.0, 60.0]   # 1800s -> 30min, 3600s -> 60min


def test_table_durations_raises_on_osrm_error(monkeypatch):
    center = {"lat": 51.0, "lon": 7.0}

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"code": "NoRoute"}

    import src.routing as routing
    monkeypatch.setattr(routing, "_get", lambda url, **k: _Resp())
    with pytest.raises(RuntimeError):
        iso._table_durations(center, [{"lat": 51.1, "lon": 7.0}])


@pytest.mark.parametrize("payload", [
    {"code": "Ok"},                       # no durations key
    {"code": "Ok", "durations": []},      # empty matrix
    {"code": "Ok", "durations": [None]},  # null row
])
def test_table_durations_malformed_matrix_is_runtime_error(monkeypatch, payload):
    # A code:Ok response with a broken matrix must fail as RuntimeError (which
    # the endpoint maps to 502), not a bare KeyError/IndexError/TypeError (500).
    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return payload

    import src.routing as routing
    monkeypatch.setattr(routing, "_get", lambda url, **k: _Resp())
    with pytest.raises(RuntimeError):
        iso._table_durations({"lat": 51.0, "lon": 7.0}, [{"lat": 51.1, "lon": 7.0}])
