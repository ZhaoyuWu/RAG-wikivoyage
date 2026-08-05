"""Performance-regression guards for the planner's pure-computation hot paths.

These are NOT microbenchmarks chasing a number. They pin the *shape* of the
cost so a future refactor that turns an O(n) loop into O(n^2) fails loudly,
without depending on absolute machine speed:

  - a complexity-ratio check: double the input, assert time grows sub-
    quadratically (this is the durable guard — it is machine-independent).
  - a coarse wall-clock ceiling, set generously (10x headroom over a normal
    laptop run) as a backstop against catastrophic blow-ups.

Marked `perf` so they can be run alone (`-m perf`) or skipped on a slow or
noisy CI runner (`-m 'not perf'`). No new dependency: standard-library timing.
"""

import time

import pytest

from src import planner

pytestmark = pytest.mark.perf


def _time(fn, repeat=3):
    """Best-of-N wall time in seconds; best-of reduces scheduler noise."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _grid_candidates(n):
    """n candidates spread over a lat/lon grid, deterministic."""
    return [
        {"file": f"c{i}",
         "geo": {"lat": 50.0 + (i % 20) * 0.2, "lon": 6.0 + (i // 20) * 0.2},
         "dist_km": float(i)}
        for i in range(n)
    ]


# --- cluster_by_day: k-means over candidate coordinates ------------------

def test_cluster_scales_subquadratically():
    """Doubling the candidate count must not quadruple the runtime. k-means
    here is O(points * days * iterations); days/iterations are bounded, so
    cost should stay ~linear in points."""
    small = _grid_candidates(200)
    large = _grid_candidates(400)

    t_small = _time(lambda: planner.cluster_by_day(small, 5))
    t_large = _time(lambda: planner.cluster_by_day(large, 5))

    # Linear would be ~2x. Allow 3x for constant-factor and cache noise;
    # a genuine O(n^2) regression would be ~4x and trip this.
    assert t_large < t_small * 3 + 0.01, f"{t_small=:.4f} {t_large=:.4f}"


def test_cluster_500_points_under_ceiling():
    cands = _grid_candidates(500)
    assert _time(lambda: planner.cluster_by_day(cands, 5)) < 1.0


# --- _min_dist_to_route: point-to-polyline over a long route -------------

def test_min_dist_to_route_scales_linearly_in_segments():
    """Cost is linear in polyline length; doubling segments ~doubles time."""
    short = [[6.0 + i * 0.01, 51.0] for i in range(250)]
    long = [[6.0 + i * 0.01, 51.0] for i in range(500)]
    pt = {"lat": 51.2, "lon": 8.5}

    t_short = _time(lambda: [planner._min_dist_to_route(pt, short)
                             for _ in range(200)])
    t_long = _time(lambda: [planner._min_dist_to_route(pt, long)
                            for _ in range(200)])

    assert t_long < t_short * 3 + 0.01, f"{t_short=:.4f} {t_long=:.4f}"


# --- haversine: the innermost primitive, called everywhere ---------------

def test_haversine_throughput():
    """A sanity ceiling on the primitive that every geo step calls in a loop.
    100k calls should be well under a second on any modern machine."""
    a = {"lat": 51.45, "lon": 7.01}
    b = {"lat": 50.94, "lon": 6.96}
    assert _time(lambda: [planner._haversine_km(a, b) for _ in range(100_000)]) < 1.0
