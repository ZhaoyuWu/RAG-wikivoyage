"""Retrieval logic: filter construction and rerank calibration.

The vector-store round-trip needs Qdrant + models, so these tests pin the
two pieces that are pure logic:

1. _build_filter — the retrieval-layer access control. "answer leakage =
   document leakage" hinges on deny_categories becoming a Qdrant must_not,
   so a restricted chunk never reaches the LLM in the first place.
2. hybrid_search's rerank stage — the sigmoid calibration and the
   descending re-sort, exercised with a stub reranker so no model loads.
"""

import types

import pytest

# retrieval.py imports qdrant_client at module load; skip where it (and the
# rest of the vector stack) isn't installed, e.g. the lightweight CI job.
pytest.importorskip("qdrant_client", reason="vector stack not installed")

from qdrant_client import models  # noqa: E402

from src.retrieval import Hit, _build_filter  # noqa: E402


# --- _build_filter: retrieval-layer access control -----------------------

def test_no_conditions_returns_none():
    # No category, geo, or ACL -> unfiltered query, not an empty Filter.
    assert _build_filter(None, None) is None


def test_category_becomes_must():
    f = _build_filter("工作", None)
    assert f.must_not is None
    assert len(f.must) == 1
    assert f.must[0].match.value == "工作"


def test_deny_category_becomes_must_not():
    # The core RBAC guarantee: a denied category is excluded at the store,
    # so its chunks can never surface in an answer.
    f = _build_filter(None, None, deny_categories=["私人"])
    assert f.must is None
    assert len(f.must_not) == 1
    assert f.must_not[0].match.value == "私人"


def test_multiple_denies_all_excluded():
    f = _build_filter(None, None, deny_categories=["私人", "财务"])
    denied = {c.match.value for c in f.must_not}
    assert denied == {"私人", "财务"}


def test_geo_radius_uses_metres():
    # radius_km is stored as metres on the GeoRadius.
    f = _build_filter(None, {"lat": 52.52, "lon": 13.40, "radius_km": 30})
    cond = f.must[0]
    assert cond.geo_radius.radius == 30_000
    assert cond.geo_radius.center.lat == 52.52


def test_geo_radius_defaults_to_50km():
    f = _build_filter(None, {"lat": 52.52, "lon": 13.40})
    assert f.must[0].geo_radius.radius == 50_000


def test_headings_and_deny_headings_split_across_must():
    f = _build_filter(None, None, headings=["Sehenswürdigkeiten"],
                      deny_headings=["Nachtleben"])
    assert any(isinstance(c.match, models.MatchAny) for c in f.must)
    assert any(isinstance(c.match, models.MatchAny) for c in f.must_not)


def test_category_and_deny_combine():
    f = _build_filter("工作", None, deny_categories=["私人"])
    assert f.must[0].match.value == "工作"
    assert f.must_not[0].match.value == "私人"


# --- hybrid_search rerank stage: sigmoid + descending re-sort ------------

class _StubReranker:
    """Returns a fixed logit per (query, doc) pair, keyed by a marker in the
    doc text so a test can assert the final order."""

    def __init__(self, logit_by_marker):
        self._logits = logit_by_marker

    def predict(self, pairs):
        out = []
        for _query, doc in pairs:
            out.append(next(v for m, v in self._logits.items() if m in doc))
        return out


@pytest.fixture
def patched_retrieval(monkeypatch):
    """Stub every heavy leg of hybrid_search so only the fusion/rerank
    logic runs: fake dense/sparse encoders, a fake Qdrant client returning
    preset points, and a caller-supplied reranker."""
    import src.retrieval as R

    class _FakeDense:
        def encode(self, _q, **_k):
            class _V:
                def tolist(self_inner):
                    return [0.0, 0.0]
            return _V()

    class _FakeSparseEmbed:
        indices = types.SimpleNamespace(tolist=lambda: [1])
        values = types.SimpleNamespace(tolist=lambda: [1.0])

    class _FakeSparse:
        def embed(self, _texts):
            return iter([_FakeSparseEmbed()])

    def _make_points(markers):
        pts = []
        for m in markers:
            pts.append(types.SimpleNamespace(
                score=0.5,
                payload={"file": f"{m}.md", "heading": "H",
                         "text": f"doc-{m}", "category": "c",
                         "geo": None, "pois": None},
            ))
        return types.SimpleNamespace(points=pts)

    class _FakeClient:
        result = _make_points(["a", "b", "c"])

        def query_points(self, **_k):
            return self.result

    monkeypatch.setattr(R, "_dense_model", lambda: _FakeDense())
    monkeypatch.setattr(R, "_sparse_model", lambda: _FakeSparse())
    monkeypatch.setattr(R, "_client", lambda: _FakeClient())
    # aliases.expand_query is imported inside hybrid_search; leave it real
    # (no-op for ascii markers).
    return R


def test_rerank_sorts_descending_and_calibrates(patched_retrieval, monkeypatch):
    R = patched_retrieval
    # b should rank first (highest logit), then a, then c.
    stub = _StubReranker({"doc-a": 0.0, "doc-b": 5.0, "doc-c": -5.0})
    monkeypatch.setattr(R, "_reranker", lambda: stub)

    hits = R.hybrid_search("q", top_k=3, rerank=True)

    assert [h.file for h in hits] == ["b.md", "a.md", "c.md"]
    # Sigmoid maps logits into (0, 1); 0.0 -> 0.5 exactly.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[1] == pytest.approx(0.5, abs=1e-6)
    assert 0.0 < scores[-1] < scores[0] < 1.0


def test_rerank_truncates_to_top_k(patched_retrieval, monkeypatch):
    R = patched_retrieval
    stub = _StubReranker({"doc-a": 1.0, "doc-b": 2.0, "doc-c": 3.0})
    monkeypatch.setattr(R, "_reranker", lambda: stub)

    hits = R.hybrid_search("q", top_k=2, rerank=True)

    assert len(hits) == 2
    assert [h.file for h in hits] == ["c.md", "b.md"]


def test_hit_is_a_dataclass_with_defaults():
    h = Hit(score=1.0, file="f.md", heading="H", text="t", category="c")
    assert h.geo is None and h.pois is None
