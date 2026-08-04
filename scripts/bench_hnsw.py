"""Benchmark retrieval latency: embedded brute-force vs. Qdrant server HNSW.

Embedded local mode does an exact scan; a Qdrant server builds an HNSW
index. This measures the gap on the real 49k-chunk wikivoyage collection.

Run against whichever backend QDRANT_MODE selects:
    QDRANT_MODE=local  python -m scripts.bench_hnsw
    QDRANT_MODE=server python -m scripts.bench_hnsw   # docker compose up first

Prints P50/P95/P99 over a fixed query set so the two runs are comparable.
"""

import time

from src.config import QDRANT_MODE, utf8_stdout
from src.retrieval import hybrid_search

utf8_stdout()

QUERIES = [
    "柏林有什么博物馆", "科隆大教堂", "汉堡港口城", "慕尼黑英国花园",
    "法兰克福机场转机", "德累斯顿老城", "莱比锡景点", "海德堡城堡",
    "黑森林徒步路线", "波罗的海海滩", "巴伐利亚新天鹅堡", "莱茵河谷葡萄酒",
]


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def main(runs: int = 5):
    # Warm the models so the first query does not skew results.
    hybrid_search("warmup", top_k=1, collection="wikivoyage")

    latencies: list[float] = []
    for _ in range(runs):
        for q in QUERIES:
            t0 = time.perf_counter()
            hybrid_search(q, top_k=5, collection="wikivoyage")
            latencies.append((time.perf_counter() - t0) * 1000)

    print(f"== Retrieval latency ({QDRANT_MODE} mode, "
          f"{len(latencies)} queries) ==")
    print(f"  P50: {percentile(latencies, 50):.0f} ms")
    print(f"  P95: {percentile(latencies, 95):.0f} ms")
    print(f"  P99: {percentile(latencies, 99):.0f} ms")
    print(f"  max: {max(latencies):.0f} ms")


if __name__ == "__main__":
    main()
