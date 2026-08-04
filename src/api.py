"""FastAPI service exposing search and RAG endpoints.

Start with:
    uvicorn src.api:app --port 8000
"""

import json
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import audit, ratelimit
from .auth import authenticate, create_token, decode_token
from .config import (
    ASK_RATE_PER_MIN,
    AUTH_ENABLED,
    COLLECTION_NAME,
    ROLE_RULES,
    SEARCH_RATE_PER_MIN,
    get_qdrant_client,
    utf8_stdout,
)
from .rag import ask, ask_stream
from .retrieval import hybrid_search
from .routing import route

utf8_stdout()  # Windows consoles default to cp1252; CJK paths crash prints

app = FastAPI(
    title="Vault RAG",
    description="Hybrid retrieval + RAG over a personal Obsidian vault",
    version="0.1.0",
)

WARM = {"done": False}


@app.on_event("startup")
def warmup():
    """Load embedding models, vector collections, and the local LLM in the
    background so the first real query does not pay the cold start."""
    import threading

    def _warm():
        try:
            client = get_qdrant_client()
            for c in client.get_collections().collections:
                hybrid_search("warmup", top_k=1, collection=c.name)
            from .config import LLM_PROVIDER
            if LLM_PROVIDER == "ollama":
                from .rag import ollama_keepalive
                ollama_keepalive()
        except Exception as e:
            print("warmup skipped: " + repr(e).encode("ascii", "replace").decode(),
                  flush=True)
        WARM["done"] = True
        print("warmup complete", flush=True)

    threading.Thread(target=_warm, daemon=True).start()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --- Identity, roles, and per-user limits ---------------------------------

def require_user(authorization: str | None = Header(default=None)) -> dict:
    """Resolve the caller's identity. With auth disabled (single-user local
    mode) every request acts as admin."""
    if not AUTH_ENABLED:
        return {"user": "local", "role": "admin"}
    if authorization and authorization.startswith("Bearer "):
        ident = decode_token(authorization[len("Bearer "):])
        if ident:
            return ident
    raise HTTPException(status_code=401, detail="Not authenticated")


def require_admin(ident: dict = Depends(require_user)) -> dict:
    if ident["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return ident


def _role_rules(ident: dict) -> dict:
    return ROLE_RULES.get(ident["role"], ROLE_RULES["guest"])


def _authorize_collection(ident: dict, collection: str | None) -> list[str]:
    """403 if the role may not query this collection; otherwise return the
    categories that must be excluded from retrieval for this role."""
    rules = _role_rules(ident)
    resolved = collection or COLLECTION_NAME
    if rules["collections"] is not None and resolved not in rules["collections"]:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{ident['role']}' may not query collection '{resolved}'",
        )
    return rules["deny_categories"]


def _enforce_rate(ident: dict, bucket: str, per_min: int) -> None:
    retry = ratelimit.check(f"{bucket}:{ident['user']}", per_min)
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {per_min}/min for {bucket}. Retry in {int(retry) + 1}s.",
            headers={"Retry-After": str(int(retry) + 1)},
        )


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


@app.post("/login")
def login(req: LoginRequest):
    # Throttled hard: failed guesses against PBKDF2 are cheap for us,
    # brute force should not be.
    retry = ratelimit.check("login", 10)
    if retry is not None:
        raise HTTPException(status_code=429, detail="Too many login attempts")
    ident = authenticate(req.username, req.password)
    if ident is None:
        audit.log_event({"event": "login_failed", "user": req.username})
        raise HTTPException(status_code=401, detail="Invalid credentials")
    audit.log_event({"event": "login", "user": ident["user"], "role": ident["role"]})
    return {"token": create_token(ident["user"], ident["role"]),
            "user": ident["user"], "role": ident["role"]}


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
    history: list[dict] | None = Field(
        default=None,
        description="Prior conversation turns [{role, content}], newest last.",
        max_length=12,
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
def collections(ident: dict = Depends(require_user)):
    """List collections the caller may query, with point counts."""
    rules = _role_rules(ident)
    client = get_qdrant_client()
    names = sorted(
        (c.name for c in client.get_collections().collections),
        key=lambda n: (n != COLLECTION_NAME, n),
    )
    if rules["collections"] is not None:
        names = [n for n in names if n in rules["collections"]]
    return [{"name": n, "points": client.count(n).count} for n in names]


@app.get("/health")
def health():
    if not WARM["done"]:
        # Keep Qdrant untouched while the warmup thread is loading it.
        return {"status": "warming", "collection": COLLECTION_NAME,
                "points": None, "auth_required": AUTH_ENABLED}
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"status": "empty", "collection": COLLECTION_NAME, "points": 0,
                "auth_required": AUTH_ENABLED, "hint": "run: python -m src.indexer"}
    count = client.count(COLLECTION_NAME).count
    return {"status": "ok", "collection": COLLECTION_NAME, "points": count,
            "auth_required": AUTH_ENABLED}


@app.get("/audit")
def audit_log(limit: int = 50, ident: dict = Depends(require_admin)):
    """Most recent audit events, admin only."""
    return audit.recent(min(limit, 500))


@app.post("/search", response_model=list[SearchHit])
def search(req: SearchRequest, ident: dict = Depends(require_user)):
    _check_collection(req.collection)
    deny = _authorize_collection(ident, req.collection)
    _enforce_rate(ident, "search", SEARCH_RATE_PER_MIN)
    hits = hybrid_search(
        req.query,
        top_k=req.top_k,
        category=req.category,
        collection=req.collection,
        geo=req.geo.model_dump() if req.geo else None,
        deny_categories=deny,
    )
    return [SearchHit(**vars(h)) for h in hits]


class RouteRequest(BaseModel):
    from_place: str = Field(min_length=1, description="Start, e.g. Essen")
    to_place: str = Field(min_length=1, description="Destination, e.g. Goslar")
    mode: str = Field(default="transit", pattern="^(car|transit)$")
    departure: str | None = Field(
        default=None,
        description="ISO datetime for transit departure, e.g. 2026-08-02T09:00. Omit for now.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"from_place": "Essen", "to_place": "Goslar", "mode": "transit"}]
        }
    }


@app.post("/route")
def route_endpoint(req: RouteRequest, ident: dict = Depends(require_user)):
    """Resolve A and B against the corpus gazetteer, then query a routing
    backend: OSRM for driving, the DB REST API for rail."""
    _enforce_rate(ident, "route", SEARCH_RATE_PER_MIN)
    try:
        return route(req.from_place, req.to_place, req.mode, req.departure)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _audit_done(ident: dict, req: "AskRequest", event: dict, rewritten: str | None) -> None:
    trace = event.get("trace") or {}
    audit.log_event({
        "event": "ask",
        "user": ident["user"],
        "role": ident["role"],
        "collection": req.collection or COLLECTION_NAME,
        "query": req.query,
        "rewritten": rewritten,
        "provider": trace.get("provider"),
        "model": trace.get("model"),
        "retrieval_ms": trace.get("retrieval_ms"),
        "generation_s": trace.get("generation_s"),
        "files": [{"file": s["file"], "score": s.get("score")}
                  for s in event.get("sources") or []],
        "answer_sha": audit.answer_hash(event.get("answer") or ""),
    })


@app.post("/ask/stream")
def ask_stream_endpoint(req: AskRequest, ident: dict = Depends(require_user)):
    """Server-sent events: retrieval trace, answer deltas, final summary."""
    _check_collection(req.collection)
    deny = _authorize_collection(ident, req.collection)
    _enforce_rate(ident, "ask", ASK_RATE_PER_MIN)

    def gen():
        rewritten = None
        try:
            for event in ask_stream(
                req.query, top_k=req.top_k, category=req.category,
                collection=req.collection, history=req.history,
                deny_categories=deny,
            ):
                if event["type"] == "rewrite":
                    rewritten = event["query"]
                elif event["type"] == "done":
                    _audit_done(ident, req, event, rewritten)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            audit.log_event({"event": "ask_error", "user": ident["user"],
                             "query": req.query, "detail": str(e)})
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, ident: dict = Depends(require_user)):
    _check_collection(req.collection)
    deny = _authorize_collection(ident, req.collection)
    _enforce_rate(ident, "ask", ASK_RATE_PER_MIN)
    try:
        result = ask(
            req.query, top_k=req.top_k, category=req.category,
            collection=req.collection, deny_categories=deny,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    audit.log_event({
        "event": "ask", "user": ident["user"], "role": ident["role"],
        "collection": req.collection or COLLECTION_NAME, "query": req.query,
        "provider": (result.get("trace") or {}).get("provider"),
        "files": [{"file": s["file"], "score": s.get("score")}
                  for s in result.get("sources") or []],
        "answer_sha": audit.answer_hash(result.get("answer") or ""),
    })
    return AskResponse(**result)
