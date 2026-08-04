"""Minimal in-process metrics in Prometheus text exposition format.

No external client library: counters and a small latency histogram are
enough to demonstrate SLIs (request rate, error rate, P50/P95 latency,
provider mix) and keep the dependency surface tiny.
"""

import threading

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], int] = {}

# Histogram buckets in seconds for end-to-end /ask latency.
_BUCKETS = [0.5, 1, 2, 4, 8, 16, 32, 64]
_hist_counts: dict[tuple, list[int]] = {}
_hist_sum: dict[tuple, float] = {}


def inc(name: str, labels: dict | None = None, value: int = 1) -> None:
    key = (name, tuple(sorted((labels or {}).items())))
    with _lock:
        _counters[key] = _counters.get(key, 0) + value


def observe(name: str, seconds: float, labels: dict | None = None) -> None:
    key = (name, tuple(sorted((labels or {}).items())))
    with _lock:
        # Non-cumulative per-bucket counts: increment only the first bucket
        # whose edge the observation fits; render() makes them cumulative.
        # One extra slot holds the overflow (> largest edge) for +Inf.
        counts = _hist_counts.setdefault(key, [0] * (len(_BUCKETS) + 1))
        placed = False
        for i, edge in enumerate(_BUCKETS):
            if seconds <= edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
        _hist_sum[key] = _hist_sum.get(key, 0.0) + seconds


def _fmt_labels(labels: tuple, extra: dict | None = None) -> str:
    items = dict(labels)
    if extra:
        items.update(extra)
    if not items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in items.items())
    return "{" + inner + "}"


def render() -> str:
    """Serialize all metrics to the Prometheus text format."""
    lines: list[str] = []
    with _lock:
        for (name, labels), val in sorted(_counters.items()):
            lines.append(f"{name}{_fmt_labels(labels)} {val}")
        for (name, labels), counts in sorted(_hist_counts.items()):
            cumulative = 0
            for edge, c in zip(_BUCKETS, counts):
                cumulative += c
                lines.append(f'{name}_bucket{_fmt_labels(labels, {"le": edge})} {cumulative}')
            cumulative += counts[-1]  # overflow slot (> largest edge)
            lines.append(f'{name}_bucket{_fmt_labels(labels, {"le": "+Inf"})} {cumulative}')
            lines.append(f"{name}_sum{_fmt_labels(labels)} {_hist_sum.get((name, labels), 0.0):.3f}")
            lines.append(f"{name}_count{_fmt_labels(labels)} {cumulative}")
    return "\n".join(lines) + "\n"
