"""Hybrid retrieval: dense (bge-m3) + sparse (BM25) fused with RRF.

CLI:
    python -m src.retrieval "查询词" [--dense-only] [--category 工作] [--top-k 5]
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import models

from .config import (
    COLLECTION_NAME,
    DENSE_MODEL,
    RERANKER_MODEL,
    SPARSE_MODEL,
    USE_RERANKER,
    get_qdrant_client,
    utf8_stdout,
)


@dataclass
class Hit:
    score: float
    file: str
    heading: str
    text: str
    category: str
    geo: dict | None = None
    pois: list | None = None


@lru_cache(maxsize=1)
def _dense_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DENSE_MODEL)


@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(SPARSE_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def _client():
    return get_qdrant_client()


def _build_filter(category: str | None, geo: dict | None) -> models.Filter | None:
    """Combine optional category and geo-radius conditions."""
    must: list = []
    if category:
        must.append(
            models.FieldCondition(key="category", match=models.MatchValue(value=category))
        )
    if geo:
        must.append(
            models.FieldCondition(
                key="geo",
                geo_radius=models.GeoRadius(
                    center=models.GeoPoint(lat=geo["lat"], lon=geo["lon"]),
                    radius=geo.get("radius_km", 50) * 1000,
                ),
            )
        )
    return models.Filter(must=must) if must else None


def _to_hits(points) -> list[Hit]:
    return [
        Hit(
            score=p.score,
            file=p.payload["file"],
            heading=p.payload["heading"],
            text=p.payload["text"],
            category=p.payload["category"],
            geo=p.payload.get("geo"),
            pois=p.payload.get("pois"),
        )
        for p in points
    ]


def dense_search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    collection: str | None = None,
    geo: dict | None = None,
) -> list[Hit]:
    """Dense-only search, used as the baseline for hybrid comparison."""
    dense_vec = _dense_model().encode(query, normalize_embeddings=True).tolist()
    result = _client().query_points(
        collection_name=collection or COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        query_filter=_build_filter(category, geo),
        limit=top_k,
        with_payload=True,
    )
    return _to_hits(result.points)


def hybrid_search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    rerank: bool | None = None,
    collection: str | None = None,
    geo: dict | None = None,
) -> list[Hit]:
    """Prefetch dense and sparse candidates, fuse with RRF, optionally rerank."""
    if rerank is None:
        rerank = USE_RERANKER

    dense_vec = _dense_model().encode(query, normalize_embeddings=True).tolist()
    sparse_raw = next(iter(_sparse_model().embed([query])))
    sparse_vec = models.SparseVector(
        indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist()
    )

    query_filter = _build_filter(category, geo)
    fetch_k = 20 if rerank else top_k
    result = _client().query_points(
        collection_name=collection or COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=20,
                            filter=query_filter),
            models.Prefetch(query=sparse_vec, using="sparse", limit=20,
                            filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_k,
        with_payload=True,
    )
    hits = _to_hits(result.points)

    if rerank and hits:
        scores = _reranker().predict([(query, f"{h.heading}\n{h.text}") for h in hits])
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        for score, hit in zip(sorted(scores, reverse=True), hits):
            hit.score = float(score)
        hits = hits[:top_k]
    return hits


def _print_hits(hits: list[Hit]) -> None:
    for i, h in enumerate(hits, 1):
        preview = h.text.replace("\n", " ")[:80]
        print(f"{i}. [{h.score:.4f}] {h.file} :: {h.heading}")
        print(f"   {preview}...")


if __name__ == "__main__":
    utf8_stdout()
    parser = argparse.ArgumentParser(description="Query the vault index")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None)
    parser.add_argument("--dense-only", action="store_true")
    args = parser.parse_args()

    if args.dense_only:
        print("== dense-only ==")
        _print_hits(dense_search(args.query, args.top_k, args.category))
    else:
        print("== hybrid (RRF) ==")
        _print_hits(hybrid_search(args.query, args.top_k, args.category))
