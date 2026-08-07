"""Retrieval X-ray (hybrid_search_traced): rank alignment across stages.

The vector store, encoders, and reranker are stubbed so only the trace
assembly logic runs — how the dense leg, sparse leg, and fused result are
lined up into per-file rank movement, and the promotion counter.
"""

import types

import pytest

pytest.importorskip("qdrant_client", reason="vector stack not installed")

import src.retrieval as R  # noqa: E402


def _hit(file, heading="H", score=0.9, text="t"):
    return R.Hit(score=score, file=file, heading=heading, text=text, category="c")


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub the encoders, the two named single-leg searches, and the fused
    hybrid_search so hybrid_search_traced runs on known data."""
    class _V:
        def tolist(self): return [0.0]

    class _Dense:
        def encode(self, q, **k): return _V()

    class _SparseEmb:
        indices = types.SimpleNamespace(tolist=lambda: [1])
        values = types.SimpleNamespace(tolist=lambda: [1.0])

    class _Sparse:
        def embed(self, texts): return iter([_SparseEmb()])

    monkeypatch.setattr(R, "_dense_model", lambda: _Dense())
    monkeypatch.setattr(R, "_sparse_model", lambda: _Sparse())
    monkeypatch.setattr(R, "_build_filter", lambda *a, **k: None)
    return R


def test_traced_aligns_ranks_across_legs(stub_pipeline, monkeypatch):
    # dense ranks A,B,C ; sparse ranks C,A,D ; fusion returns A,C,B.
    def fake_named(vec, using, qf, coll, limit=20):
        if using == "dense":
            return [("A.md", "H", 0.9), ("B.md", "H", 0.8), ("C.md", "H", 0.7)]
        return [("C.md", "H", 0.6), ("A.md", "H", 0.5), ("D.md", "H", 0.4)]

    monkeypatch.setattr(R, "_named_search", fake_named)
    monkeypatch.setattr(R, "hybrid_search",
                        lambda *a, **k: [_hit("A.md"), _hit("C.md"), _hit("B.md")])

    out = R.hybrid_search_traced("q", top_k=5)

    assert [r["file"] for r in out["dense"]] == ["A.md", "B.md", "C.md"]
    assert [r["file"] for r in out["sparse"]] == ["C.md", "A.md", "D.md"]
    fused = {r["file"]: r for r in out["fused"]}
    # A: dense rank 0, sparse rank 1, final rank 0
    assert fused["A.md"]["dense_rank"] == 0 and fused["A.md"]["sparse_rank"] == 1
    assert fused["A.md"]["final_rank"] == 0
    # C: dense rank 2, sparse rank 0, final rank 1
    assert fused["C.md"]["dense_rank"] == 2 and fused["C.md"]["sparse_rank"] == 0


def test_traced_counts_promotions(stub_pipeline, monkeypatch):
    # E is weak in both legs (dense rank 5, sparse rank 6) but fusion/rerank
    # puts it at final rank 0 -> a promotion.
    def fake_named(vec, using, qf, coll, limit=20):
        base = [(f"x{i}.md", "H", 0.9 - i * 0.1) for i in range(5)]
        return base + [("E.md", "H", 0.1)] if using == "dense" \
            else base + [("y.md", "H", 0.2), ("E.md", "H", 0.05)]

    monkeypatch.setattr(R, "_named_search", fake_named)
    monkeypatch.setattr(R, "hybrid_search", lambda *a, **k: [_hit("E.md")])

    out = R.hybrid_search_traced("q", top_k=5)
    assert out["stages"]["promotions"] == 1


def test_traced_reports_expanded_query(stub_pipeline, monkeypatch):
    monkeypatch.setattr(R, "_named_search", lambda *a, **k: [])
    monkeypatch.setattr(R, "hybrid_search", lambda *a, **k: [])
    # A transliteration should expand; a plain query should not.
    monkeypatch.setattr("src.aliases.expand_query",
                        lambda q: q + " Düsseldorf" if "杜塞尔多夫" in q else q)

    out = R.hybrid_search_traced("杜塞尔多夫 old town")
    assert out["expanded_query"] and "Düsseldorf" in out["expanded_query"]

    out2 = R.hybrid_search_traced("Berlin museums")
    assert out2["expanded_query"] is None  # no expansion -> None


def test_traced_handles_no_overlap(stub_pipeline, monkeypatch):
    # A fused file absent from both legs (shouldn't happen, but be robust):
    # its dense_rank/sparse_rank are None and it isn't counted as a promotion.
    monkeypatch.setattr(R, "_named_search", lambda *a, **k: [("A.md", "H", 0.5)])
    monkeypatch.setattr(R, "hybrid_search", lambda *a, **k: [_hit("Z.md")])

    out = R.hybrid_search_traced("q")
    z = out["fused"][0]
    assert z["dense_rank"] is None and z["sparse_rank"] is None
    assert out["stages"]["promotions"] == 0
