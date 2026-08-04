"""In-process sliding-window rate limiter.

Good for a single uvicorn worker; a multi-worker deployment would move
the counters to Redis, keeping this module's interface.
"""

import time
from collections import defaultdict, deque

_hits: dict[str, deque] = defaultdict(deque)


def check(key: str, limit: int, window_s: float = 60.0) -> float | None:
    """Record one hit for key. Return None when allowed, otherwise the
    number of seconds until the oldest hit leaves the window."""
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        return window_s - (now - q[0])
    q.append(now)
    return None
