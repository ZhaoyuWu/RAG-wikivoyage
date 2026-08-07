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


def _build_filter(category: str | None, geo: dict | None,
                  deny_categories: list[str] | None = None,
                  headings: list[str] | None = None,
                  deny_headings: list[str] | None = None) -> models.Filter | None:
    """Combine optional category, geo-radius, heading, and ACL conditions.

    deny_categories enforces document-level access control at the
    retrieval layer: excluded chunks never reach the LLM, so a restricted
    user's answer cannot leak them. headings/deny_headings narrow retrieval
    to (or away from) specific Wikivoyage sections, used by the planner to
    align results with a travel intent.
    """
    must: list = []
    must_not: list = []
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
    if headings:
        must.append(
            models.FieldCondition(key="heading",
                                  match=models.MatchAny(any=list(headings)))
        )
    for denied in deny_categories or []:
        must_not.append(
            models.FieldCondition(key="category", match=models.MatchValue(value=denied))
        )
    if deny_headings:
        must_not.append(
            models.FieldCondition(key="heading",
                                  match=models.MatchAny(any=list(deny_headings)))
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


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
    deny_categories: list[str] | None = None,
    headings: list[str] | None = None,
    deny_headings: list[str] | None = None,
) -> list[Hit]:
    """Prefetch dense and sparse candidates, fuse with RRF, optionally rerank."""
    if rerank is None:
        rerank = USE_RERANKER

    # Rescue transliterated place names ("杜塞尔多夫" -> +"Düsseldorf") so the
    # sparse leg gets a German token to match. No-op when nothing matches.
    from .aliases import expand_query
    query = expand_query(query)

    dense_vec = _dense_model().encode(query, normalize_embeddings=True).tolist()
    sparse_raw = next(iter(_sparse_model().embed([query])))
    sparse_vec = models.SparseVector(
        indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist()
    )

    query_filter = _build_filter(category, geo, deny_categories,
                                 headings, deny_headings)
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
        import math

        raw = _reranker().predict([(query, f"{h.heading}\n{h.text}") for h in hits])
        # Sigmoid turns the cross-encoder logits into calibrated 0-1
        # relevance, which RRF's rank-only fusion cannot provide.
        scores = [1 / (1 + math.exp(-float(s))) for s in raw]
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        for score, hit in zip(sorted(scores, reverse=True), hits):
            hit.score = score
        hits = hits[:top_k]
    return hits


def _named_search(vec, using: str, query_filter, collection, limit=20):
    """Single-leg search (dense or sparse) returning (file, heading, score)
    rows, used only by the traced view to expose each leg's own ranking."""
    result = _client().query_points(
        collection_name=collection or COLLECTION_NAME,
        query=vec, using=using, query_filter=query_filter,
        limit=limit, with_payload=True,
    )
    return [(p.payload["file"], p.payload["heading"], p.score)
            for p in result.points]


def hybrid_search_traced(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    collection: str | None = None,
    deny_categories: list[str] | None = None,
) -> dict:
    """Like hybrid_search, but exposes the full retrieval pipeline for an
    X-ray view: what the dense leg ranked, what the sparse leg ranked, how RRF
    fused them, and how the reranker re-ordered the survivors.

    This runs the dense and sparse legs SEPARATELY (the production path fuses
    them inside one Qdrant query, so their individual rankings aren't
    observable). It costs two extra queries, so it is only for the debug/trace
    endpoint, never the hot ask path. Returns:
      {query, expanded_query, dense[], sparse[], fused[], stages}
    where each row is {file, heading, score, rank} and fused rows also carry
    dense_rank / sparse_rank / final_rank so the frontend can draw movement.
    """
    from .aliases import expand_query

    expanded = expand_query(query)
    query_filter = _build_filter(category, None, deny_categories)

    dense_vec = _dense_model().encode(expanded, normalize_embeddings=True).tolist()
    sparse_raw = next(iter(_sparse_model().embed([expanded])))
    sparse_vec = models.SparseVector(
        indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist()
    )

    dense_rows = _named_search(dense_vec, "dense", query_filter, collection)
    sparse_rows = _named_search(sparse_vec, "sparse", query_filter, collection)
    dense_rank = {f: i for i, (f, _, _) in enumerate(dense_rows)}
    sparse_rank = {f: i for i, (f, _, _) in enumerate(sparse_rows)}

    # The real fused ranking, straight from the production path (RRF, then the
    # reranker if enabled). Fetch a few extra so the fused list is informative.
    fused_hits = hybrid_search(expanded, top_k=max(top_k, 8), category=category,
                               collection=collection,
                               deny_categories=deny_categories)

    def rows(items):
        return [{"file": f, "heading": h, "score": round(s, 4), "rank": i}
                for i, (f, h, s) in enumerate(items)]

    fused = []
    for i, hit in enumerate(fused_hits):
        fused.append({
            "file": hit.file, "heading": hit.heading,
            "score": round(hit.score, 4), "final_rank": i,
            "dense_rank": dense_rank.get(hit.file),
            "sparse_rank": sparse_rank.get(hit.file),
        })

    # Did anything the reranker promoted come from a weak fused position? Flag
    # a promotion when a fused row's dense and sparse ranks are both worse than
    # its final rank, i.e. fusion/rerank rescued it. (Purely informational.)
    promotions = sum(
        1 for r in fused
        if r["dense_rank"] is not None and r["sparse_rank"] is not None
        and min(r["dense_rank"], r["sparse_rank"]) > r["final_rank"]
    )

    return {
        "query": query,
        "expanded_query": expanded if expanded != query else None,
        "dense": rows(dense_rows)[:top_k],
        "sparse": rows(sparse_rows)[:top_k],
        "fused": fused[:top_k],
        "stages": {
            "dense_candidates": len(dense_rows),
            "sparse_candidates": len(sparse_rows),
            "reranker_on": bool(USE_RERANKER),
            "promotions": promotions,
            "sigmoid_note": "final scores are sigmoid-calibrated 0-1 when the reranker is on"
                            if USE_RERANKER else "final scores are RRF fusion ranks",
        },
    }


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
