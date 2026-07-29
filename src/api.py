"""FastAPI service exposing search and RAG endpoints.

Start with:
    uvicorn src.api:app --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import COLLECTION_NAME, get_qdrant_client
from .rag import ask
from .retrieval import hybrid_search

app = FastAPI(
    title="Vault RAG",
    description="Hybrid retrieval + RAG over a personal Obsidian vault",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


class SearchHit(BaseModel):
    score: float
    file: str
    heading: str
    text: str
    category: str


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


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
    hits = hybrid_search(req.query, top_k=req.top_k, category=req.category)
    return [SearchHit(**vars(h)) for h in hits]


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        result = ask(req.query, top_k=req.top_k, category=req.category)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AskResponse(**result)
