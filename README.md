# Vault RAG

Hybrid retrieval question answering over local markdown corpora. Notes are chunked by heading, embedded twice (dense multilingual vectors and BM25 sparse vectors), stored in Qdrant, fused with Reciprocal Rank Fusion, and answered by an LLM with source citations. Everything runs locally; the answering model can be a local Ollama model or the Claude API, switched by one config value.

Built and tested on two corpora: a personal Obsidian vault (371 chunks) and the German Wikivoyage travel guide (49,460 chunks), indexed by the included data pipeline.

![Streaming answer with live retrieval trace](assets/ask-streaming.png)

## What it does

- **Hybrid retrieval**: dense semantic search (bge-m3, 100+ languages) plus BM25 keyword search, fused with RRF. Semantic paraphrase and exact terms both hit.
- **Cross-language**: ask in Chinese or English over a German corpus. One embedding space for all languages.
- **Grounded answers with citations**: the LLM answers only from retrieved chunks and lists its sources. When retrieval finds nothing relevant, it says so instead of guessing.
- **Transparent pipeline**: every answer streams over SSE with a live trace. You watch the retrieved chunks appear, then the answer generate token by token, then the process panel collapses into an expandable "How this answer was made" summary with timings.
- **Incremental indexing**: a Prefect flow compares file hashes and re-embeds only changed files.
- **Two deployment modes**: embedded Qdrant on disk for zero-setup local use, or Docker Compose (Qdrant server + API) when the corpus outgrows brute-force search.

## Web UI

A single static page served by the API. Collection switcher, search and ask modes, streaming output, error surfacing. No build tooling, no framework, no keys in the browser.

| Search mode | Ask mode (answer with sources) |
|---|---|
| ![Search results](assets/search-results.png) | ![Answer with citations](assets/ask-answer.png) |

The second screenshot shows a Chinese question answered from German source articles, with the retrieval trace collapsed above the answer.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │                Indexing                  │
 markdown files ──▶ chunking by ## heading ──▶ bge-m3 (dense) ─┐
                    │                     └──▶ BM25 (sparse) ──┤
                    └──────────────────────────────────────────┼──┐
                                                               ▼  │
                                                      ┌────────────────┐
                                                      │     Qdrant     │
                                                      │ named vectors: │
                                                      │ dense + sparse │
                                                      └───────┬────────┘
                    ┌─────────────────────────────────────────┼─────┐
                    │                 Query                   ▼     │
 question ──▶ embed both ways ──▶ prefetch dense + sparse ──▶ RRF   │
                    │                                          │    │
                    │               top-k chunks ◀─────────────┘    │
                    └────────────────┬──────────────────────────────┘
                                     ▼
                     Ollama (local) or Claude API
                                     ▼
                     streamed answer + citations + trace
```

## Quick Start

Requirements: Python 3.11, about 3 GB disk for the embedding model. CPU is enough.

```bash
git clone https://github.com/ZhaoyuWu/RAG-wikivoyage
cd RAG-wikivoyage
python -m venv venv
venv/Scripts/activate        # Windows; on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# Point at your own markdown folder
copy .env.example .env       # then edit VAULT_PATH

# 1. Full index (first run downloads the bge-m3 model, ~2.3 GB)
python -m src.indexer

# 2. Start the API, open http://localhost:8000 for the web UI
#    (Swagger stays available at /docs)
uvicorn src.api:app --port 8000
```

For the ask mode, either install [Ollama](https://ollama.com), pull a model (`ollama pull qwen2.5:7b-instruct`) and set `LLM_PROVIDER=ollama` in `.env`, or set `ANTHROPIC_API_KEY` to use the Claude API. Search mode needs neither.

### Docker

```bash
docker compose up --build
```

Starts a Qdrant server plus the API container. Set `QDRANT_MODE=server` when running the indexer against it.

### Incremental re-index

```bash
python -m flows.reindex_flow
```

Compares md5 hashes stored in the payload against the files on disk, then re-embeds only changed files and deletes chunks of removed files.

## Hybrid vs Dense: why both?

Dense embeddings handle semantic paraphrase: the query "换了雇主社保和养老金要注意什么" (what about social insurance and pension after changing employers) finds the pension notes even though they never use that phrasing. BM25 handles exact rare terms that embedding models dilute: proper nouns, product names, error codes.

Measured example, query `"Riester 值得买吗"` (a German pension product name):

| Rank | Dense only (cosine) | Hybrid (RRF) |
|---|---|---|
| 1 | pension note, 0.4427 | pension note, **0.8333** |
| 2 | pension note, 0.4408 | pension note, **0.8333** |
| 3 | hiking trip note, 0.4407 | hiking trip note, 0.2500 |

With dense vectors alone, an irrelevant hiking note scores within 0.0001 of a correct hit, so a slightly different query flips the ranking. The BM25 leg matches the literal token "Riester" and the fusion separates the correct chunks decisively.

The flip side, also measured: cross-language retrieval weakens on transliterated proper nouns. "哈茨山区" (the Chinese transliteration of Harz) does not anchor to the region, while keeping the original spelling ("Harz 地区有哪些城堡") retrieves correctly, because the BM25 leg gets an exact token to match. Known limitation, honest fix: keep proper nouns in their original script.

## Scaling case study: 371 to 49,460 chunks

To test the stack beyond a personal vault, a second pipeline (`pipelines/wikivoyage/`) indexes the German Wikivoyage dump: 20,968 articles streamed from the official XML export, cleaned from wikitext to markdown (POI templates rendered as text, section headings preserved), filtered to the 3,873 Germany articles via the IstIn breadcrumb graph, and embedded into a separate collection of 49,460 chunks.

```bash
python -m pipelines.wikivoyage.parse            # dump -> articles.jsonl
python -m pipelines.wikivoyage.build --scope germany
```

Lessons that only showed up at this scale, all reproducible in the git history:

- A 64-text benchmark predicted 3.5 chunks/s; production ran at 0.6. Mixed-length batches pad every text to the longest member, and sustained load throttles a laptop CPU in ways a short benchmark never sees. Length-sorted batching plus a 512-token cap roughly doubled real throughput.
- A 6-hour job needs to survive interruption. Upserts happen per article batch, so a resumed run skips completed articles by comparing stored file hashes; transient disk errors (external USB drive) are retried with backoff instead of killing the run.
- Query latency on 49k chunks is ~1.2s in embedded local mode, which does exact brute-force search. That is the point where switching to the Docker Qdrant server (HNSW index) stops being over-engineering, which is why both modes exist behind one config switch.

## Tech

| Component | Choice |
|---|---|
| Vector store | Qdrant (embedded local mode or Docker server) |
| Dense embedding | BAAI/bge-m3 via sentence-transformers (PyTorch) |
| Sparse embedding | BM25 via fastembed |
| Fusion | Qdrant Query API prefetch + RRF |
| Reranker (optional) | BAAI/bge-reranker-v2-m3 cross-encoder |
| API | FastAPI + pydantic v2, SSE streaming |
| Generation | Ollama (local) or Claude API, one config switch |
| Orchestration | Prefect |
| Frontend | Single static HTML page, no build step |

## Privacy

Corpus text only ever reaches the local Qdrant store and, when the Claude provider is selected, the Claude API. Nothing in `qdrant_db/`, `data/` or `.env` is committed. The repository contains code only.
