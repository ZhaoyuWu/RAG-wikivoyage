"""Build the Wikivoyage collection: filter, chunk, embed, upsert.

The Germany subset is computed from the IstIn breadcrumb graph: an article
belongs to the subset when walking its parent chain reaches "Deutschland".

Run (after parse.py):
    python -m pipelines.wikivoyage.build --scope germany
    python -m pipelines.wikivoyage.build --scope all
    python -m pipelines.wikivoyage.build --scope germany --limit 20   # smoke test
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.chunking import chunk_markdown
from src.config import utf8_stdout, get_qdrant_client
from src.indexer import embed_points, ensure_collection, load_embedders

from .clean import extract_parent, to_markdown
from .parse import ARTICLES_PATH

COLLECTION = "wikivoyage"
GERMANY_ROOTS = {"Deutschland"}
MAX_HOPS = 20


def load_articles() -> list[dict]:
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def germany_subset(articles: list[dict]) -> list[dict]:
    """Keep articles whose IstIn ancestor chain reaches Deutschland."""
    parent = {a["title"]: extract_parent(a["wikitext"]) for a in articles}

    def in_germany(title: str) -> bool:
        seen = set()
        current = title
        for _ in range(MAX_HOPS):
            if current in GERMANY_ROOTS:
                return True
            if current in seen or current not in parent:
                return False
            seen.add(current)
            current = parent[current]
            if current is None:
                return False
        return False

    return [a for a in articles if in_germany(a["title"])]


def indexed_hashes(client) -> dict[str, str]:
    """Return {title: file_hash} already present in the collection."""
    state: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=1024,
            offset=offset,
            with_payload=["file", "file_hash"],
            with_vectors=False,
        )
        for p in points:
            state[p.payload["file"]] = p.payload["file_hash"]
        if offset is None:
            break
    return state


def build(scope: str, limit: int | None) -> None:
    """Index articles one by one so an interrupted run can resume.

    Points for an article are upserted only after the whole article is
    embedded, so every title found in the collection is complete and can
    be skipped on the next run.
    """
    started = time.time()
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    if scope == "germany":
        articles = germany_subset(articles)
        print(f"Germany subset: {len(articles)} articles")
    if limit:
        articles = articles[:limit]

    client = get_qdrant_client()
    ensure_collection(client, collection=COLLECTION)
    done = indexed_hashes(client)

    todo = []
    for a in articles:
        file_hash = hashlib.md5(a["wikitext"].encode("utf-8")).hexdigest()
        if done.get(a["title"]) == file_hash:
            continue
        todo.append((a, file_hash))
    print(f"Already indexed: {len(articles) - len(todo)}, to do: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    dense_model, sparse_model = load_embedders()
    chunk_count = 0
    for i, (a, file_hash) in enumerate(todo, 1):
        md = to_markdown(a["wikitext"])
        if len(md) < 200:  # skip stubs
            continue
        parent = extract_parent(a["wikitext"]) or "(none)"
        chunks = chunk_markdown(md, a["title"], parent, file_hash)
        points = embed_points(chunks, dense_model, sparse_model)
        client.upsert(collection_name=COLLECTION, points=points)
        chunk_count += len(chunks)
        if i % 25 == 0 or i == len(todo):
            elapsed = (time.time() - started) / 60
            rate = i / elapsed if elapsed else 0
            eta = (len(todo) - i) / rate if rate else 0
            print(f"  {i}/{len(todo)} articles, {chunk_count} chunks, "
                  f"{elapsed:.0f} min elapsed, ~{eta:.0f} min left", flush=True)

    count = client.count(COLLECTION).count
    print(f"Done in {(time.time() - started) / 60:.1f} min. "
          f"Collection '{COLLECTION}' now has {count} points.")


if __name__ == "__main__":
    utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["germany", "all"], default="germany")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build(args.scope, args.limit)
