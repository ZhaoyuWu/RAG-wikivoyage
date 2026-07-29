"""Indexing pipeline: walk the vault, chunk, embed (dense + sparse), upsert.

Run a full index with:
    python -m src.indexer
"""

import hashlib
import sys
import time
from pathlib import Path

from qdrant_client import models

from .chunking import Chunk, chunk_markdown
from .config import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_MODEL,
    EXCLUDED_DIRS,
    SPARSE_MODEL,
    VAULT_PATH,
    ensure_vault_available,
    get_qdrant_client,
    utf8_stdout,
)


def iter_note_files(root: Path):
    """Yield markdown files under the vault, skipping excluded directories."""
    for path in sorted(root.rglob("*.md")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def load_chunks(root: Path) -> tuple[list[Chunk], int]:
    """Parse and chunk every note. Returns (chunks, file_count)."""
    all_chunks: list[Chunk] = []
    file_count = 0
    for path in iter_note_files(root):
        raw = path.read_bytes()
        content = raw.decode("utf-8", errors="replace")
        rel_path = path.relative_to(root).as_posix()
        # Category is the first directory level (学习/工作/娱乐/日常).
        parts = path.relative_to(root).parts
        category = parts[0] if len(parts) > 1 else "(root)"
        file_hash = hashlib.md5(raw).hexdigest()
        chunks = chunk_markdown(content, rel_path, category, file_hash)
        all_chunks.extend(chunks)
        file_count += 1
    return all_chunks, file_count


def ensure_collection(client) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        client.create_payload_index(
            COLLECTION_NAME, field_name="file", field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            COLLECTION_NAME, field_name="category", field_schema=models.PayloadSchemaType.KEYWORD
        )


def embed_and_upsert(client, chunks: list[Chunk], batch_size: int = 16) -> None:
    """Compute dense + sparse vectors and upsert points in batches."""
    from fastembed import SparseTextEmbedding
    from sentence_transformers import SentenceTransformer

    print(f"Loading dense model {DENSE_MODEL} (first run downloads ~2.3GB)...")
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [f"{c.heading}\n{c.text}" for c in batch]
        dense_vecs = dense_model.encode(texts, normalize_embeddings=True)
        sparse_vecs = list(sparse_model.embed(texts))

        points = [
            models.PointStruct(
                id=chunk.id,
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "file": chunk.file,
                    "heading": chunk.heading,
                    "text": chunk.text,
                    "category": chunk.category,
                    "file_hash": chunk.file_hash,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  indexed {min(start + batch_size, total)}/{total} chunks", flush=True)


def run_full_index() -> None:
    ensure_vault_available()
    started = time.time()

    print(f"Scanning {VAULT_PATH} ...")
    chunks, file_count = load_chunks(VAULT_PATH)
    print(f"Found {file_count} files -> {len(chunks)} chunks")

    client = get_qdrant_client()
    ensure_collection(client)
    embed_and_upsert(client, chunks)

    count = client.count(COLLECTION_NAME).count
    print(f"Done in {time.time() - started:.1f}s. Collection '{COLLECTION_NAME}' now has {count} points.")


if __name__ == "__main__":
    utf8_stdout()
    try:
        run_full_index()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
