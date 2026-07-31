"""FastAPI service exposing search and RAG endpoints.

Start with:
    uvicorn src.api:app --port 8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import COLLECTION_NAME, get_qdrant_client
from .rag import ask, ask_stream
from .retrieval import hybrid_search
from .routing import route

app = FastAPI(
    title="Vault RAG",
    description="Hybrid retrieval + RAG over a personal Obsidian vault",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _check_collection(collection: str | None) -> None:
    """Reject requests for collections that do not exist."""
    if collection is None:
        return
    if not get_qdrant_client().collection_exists(collection):
        raise HTTPException(status_code=404, detail=f"Unknown collection: {collection}")


class GeoFilter(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=50, gt=0, le=1000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(
        default=None,
        description="Optional filter: first-level folder name, e.g. 工作. Omit to search everything.",
    )
    collection: str | None = Field(
        default=None,
        description="Collection to query. Omit for the configured default.",
    )
    geo: GeoFilter | None = Field(
        default=None,
        description="Optional geo-radius filter, e.g. {lat: 51.45, lon: 7.01, radius_km: 50}.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "社保要注意什么", "top_k": 5}]
        }
    }


class SearchHit(BaseModel):
    score: float
    file: str
    heading: str
    text: str
    category: str
    geo: dict | None = None
    pois: list | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(
        default=None,
        description="Optional filter: first-level folder name, e.g. 工作. Omit to search everything.",
    )
    collection: str | None = Field(
        default=None,
        description="Collection to query. Omit for the configured default.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "换了雇主社保要注意什么", "top_k": 5}]
        }
    }


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace: dict | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/collections")
def collections():
    """List available collections with point counts, default first."""
    client = get_qdrant_client()
    names = sorted(
        (c.name for c in client.get_collections().collections),
        key=lambda n: (n != COLLECTION_NAME, n),
    )
    return [{"name": n, "points": client.count(n).count} for n in names]


@app.get("/health")
def health():
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"status": "empty", "collection": COLLECTION_NAME, "points": 0,
                "hint": "run: python -m src.indexer"}
    count = client.count(COLLECTION_NAME).count
    return {"status": "ok", "collection": COLLECTION_NAME, "points": count}


@app.post("/search", response_model=list[SearchHit])
def search(req: SearchRequest):
    _check_collection(req.collection)
    hits = hybrid_search(
        req.query,
        top_k=req.top_k,
        category=req.category,
        collection=req.collection,
        geo=req.geo.model_dump() if req.geo else None,
    )
    return [SearchHit(**vars(h)) for h in hits]


class RouteRequest(BaseModel):
    from_place: str = Field(min_length=1, description="Start, e.g. Essen")
    to_place: str = Field(min_length=1, description="Destination, e.g. Goslar")
    mode: str = Field(default="transit", pattern="^(car|transit)$")

    model_config = {
        "json_schema_extra": {
            "examples": [{"from_place": "Essen", "to_place": "Goslar", "mode": "transit"}]
        }
    }


@app.post("/route")
def route_endpoint(req: RouteRequest):
    """Resolve A and B against the corpus gazetteer, then query a routing
    backend: OSRM for driving, the DB REST API for rail."""
    try:
        return route(req.from_place, req.to_place, req.mode)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/ask/stream")
def ask_stream_endpoint(req: AskRequest):
    """Server-sent events: retrieval trace, answer deltas, final summary."""
    _check_collection(req.collection)

    def gen():
        try:
            for event in ask_stream(
                req.query, top_k=req.top_k, category=req.category, collection=req.collection
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    _check_collection(req.collection)
    try:
        result = ask(
            req.query, top_k=req.top_k, category=req.category, collection=req.collection
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AskResponse(**result)
