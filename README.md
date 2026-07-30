# Vault RAG

Hybrid retrieval question answering over a personal Obsidian vault. Notes are chunked by heading, embedded twice (dense multilingual vectors and BM25 sparse vectors), stored in Qdrant, fused with Reciprocal Rank Fusion, and answered by Claude with source citations.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │                Indexing                  │
 Obsidian *.md ──▶  chunking.py ──▶ bge-m3 (dense, 1024d) ──┐  │
                    │   by ## heading                        │  │
                    │            └─▶ BM25 (sparse) ──────────┤  │
                    └────────────────────────────────────────┼──┘
                                                             ▼
                                                    ┌────────────────┐
                                                    │     Qdrant     │
                                                    │ named vectors: │
                                                    │ dense + sparse │
                                                    └───────┬────────┘
                    ┌───────────────────────────────────────┼──────┐
                    │                Query                  ▼      │
 question ──▶ embed both ways ──▶ prefetch dense + sparse ──▶ RRF  │
                    │                                        │     │
                    │              top-5 chunks ◀────────────┘     │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                        Claude (claude-sonnet-5)
                                     ▼
                        answer + source citations
```

FastAPI exposes `/search` (retrieval only) and `/ask` (full RAG). A Prefect flow handles incremental re-indexing by comparing file hashes, so editing one note only re-embeds that note.

## Quick Start

Requirements: Python 3.11, about 3 GB disk for the embedding model (CPU inference is fine).

```bash
git clone https://github.com/YOUR_NAME/vault-rag
cd vault-rag
python -m venv venv
venv/Scripts/activate        # Windows; on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# Point at your own markdown folder
copy .env.example .env       # then edit VAULT_PATH and ANTHROPIC_API_KEY

# 1. Full index (first run downloads the bge-m3 model, ~2.3 GB)
python -m src.indexer

# 2. Try retrieval from the CLI
python -m src.retrieval "your query"

# 3. Start the API and open http://localhost:8000/docs
uvicorn src.api:app --port 8000
```

Set the environment variables from `.env` in your shell before running (or use a tool like `python-dotenv` / your IDE's env support). `ANTHROPIC_API_KEY` is only needed for `/ask`.

### Docker

```bash
docker compose up --build
```

This starts a Qdrant server plus the API container. Set `QDRANT_MODE=server` when running the indexer against it.

### Incremental re-index

```bash
python -m flows.reindex_flow
```

Compares md5 hashes stored in the payload against the files on disk, then re-embeds only changed files and deletes chunks of removed files.

## Hybrid vs Dense: why both?

Dense embeddings handle semantic paraphrase: the query "换了雇主社保和养老金要注意什么" (what about social insurance and pension after changing employers) finds the pension notes even though they never use that phrasing. BM25 handles exact rare terms that embedding models dilute: proper nouns, product names, error codes.

Measured example from this vault, query `"Riester 值得买吗"` (a German pension product name):

| Rank | Dense only (cosine) | Hybrid (RRF) |
|---|---|---|
| 1 | pension note, 0.4427 | pension note, **0.8333** |
| 2 | pension note, 0.4408 | pension note, **0.8333** |
| 3 | hiking trip note, 0.4407 | hiking trip note, 0.2500 |

With dense vectors alone, an irrelevant hiking note scores within 0.0001 of a correct hit, so a slightly different query flips the ranking. The BM25 leg matches the literal token "Riester" and the fusion separates the correct chunks decisively.

## Tech

| Component | Choice |
|---|---|
| Vector store | Qdrant (local mode or Docker server) |
| Dense embedding | BAAI/bge-m3 via sentence-transformers (PyTorch) |
| Sparse embedding | BM25 via fastembed |
| Fusion | Qdrant Query API prefetch + RRF |
| Reranker (optional) | BAAI/bge-reranker-v2-m3 cross-encoder |
| API | FastAPI + pydantic v2 |
| Generation | Claude API (claude-sonnet-5) or local Ollama model, switch via LLM_PROVIDER |
| Orchestration | Prefect |

## Privacy

Note contents only ever reach the local Qdrant store and, for `/ask`, the Claude API. Nothing in `qdrant_db/` or `.env` is committed. The repository contains code only.
