"""Attach geo payload to already-indexed Wikivoyage chunks.

Recomputes chunk IDs deterministically (same hash as build.py), extracts
POI coordinates from the wikitext, and writes them into the existing
points with set_payload. No re-embedding needed.

Per chunk payload added:
    geo:  {lat, lon}      article centroid, used for geo-radius filtering
    pois: [{name, lat, lon}, ...]   POIs whose name appears in the chunk text

Run (after build.py):
    python -m pipelines.wikivoyage.geotag
"""

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.chunking import chunk_markdown
from src.config import get_qdrant_client, utf8_stdout

from .build import COLLECTION, germany_subset, load_articles
from .clean import centroid, extract_parent, extract_pois, to_markdown

MAX_POIS_PER_CHUNK = 40


def run() -> None:
    started = time.time()
    articles = germany_subset(load_articles())
    client = get_qdrant_client()

    tagged_chunks = 0
    tagged_articles = 0
    for i, a in enumerate(articles, 1):
        pois = extract_pois(a["wikitext"])
        center = centroid(pois)
        if center is None:
            continue

        md = to_markdown(a["wikitext"])
        if len(md) < 200:
            continue
        parent = extract_parent(a["wikitext"]) or "(none)"
        file_hash = hashlib.md5(a["wikitext"].encode("utf-8")).hexdigest()
        chunks = chunk_markdown(md, a["title"], parent, file_hash)

        for chunk in chunks:
            chunk_pois = [p for p in pois if p["name"] and p["name"] in chunk.text]
            client.set_payload(
                collection_name=COLLECTION,
                payload={"geo": center, "pois": chunk_pois[:MAX_POIS_PER_CHUNK]},
                points=[chunk.id],
            )
            tagged_chunks += 1
        tagged_articles += 1
        if i % 200 == 0:
            print(f"  {i}/{len(articles)} articles, {tagged_chunks} chunks tagged", flush=True)

    print(f"Done in {(time.time() - started) / 60:.1f} min: "
          f"{tagged_articles} articles, {tagged_chunks} chunks got geo payload.")


if __name__ == "__main__":
    utf8_stdout()
    run()
