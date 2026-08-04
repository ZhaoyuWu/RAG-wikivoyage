"""Metrics rendering: counters and histogram in Prometheus text format."""

from src import metrics


def test_counter_and_labels():
    metrics.inc("t_requests_total", {"route": "ask"})
    metrics.inc("t_requests_total", {"route": "ask"})
    metrics.inc("t_requests_total", {"route": "search"})
    out = metrics.render()
    assert 't_requests_total{route="ask"} 2' in out
    assert 't_requests_total{route="search"} 1' in out


def test_histogram_buckets_are_cumulative():
    # Unique label value isolates this metric from module-global state
    # accumulated by other tests in the same process.
    lbl = {"c": "hist_test"}
    for s in (0.3, 1.5, 3.0, 50.0):
        metrics.observe("t_latency_seconds", s, lbl)
    out = metrics.render()
    # 0.3 -> le=0.5 bucket; 4 observations total in +Inf
    assert 't_latency_seconds_bucket{c="hist_test",le="0.5"} 1' in out
    assert 't_latency_seconds_bucket{c="hist_test",le="+Inf"} 4' in out
    assert 't_latency_seconds_count{c="hist_test"} 4' in out
