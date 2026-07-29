"""Prefect flow: incremental re-index.

Compares each note's current md5 against the file_hash stored in Qdrant
payloads. Only changed or new files are re-embedded; chunks belonging to
deleted files are removed.

Run with:
    python -m flows.reindex_flow
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prefect import flow, task
from qdrant_client import models

from src.chunking import chunk_markdown
from src.config import COLLECTION_NAME, VAULT_PATH, ensure_vault_available, get_qdrant_client
from src.indexer import embed_and_upsert, ensure_collection, iter_note_files


@task
def scan_vault() -> dict[str, str]:
    """Return {relative_path: md5} for every note currently on disk."""
    ensure_vault_available()
    return {
        p.relative_to(VAULT_PATH).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
        for p in iter_note_files(VAULT_PATH)
    }


@task
def indexed_state(client) -> dict[str, str]:
    """Return {relative_path: file_hash} currently stored in Qdrant."""
    state: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["file", "file_hash"],
            with_vectors=False,
        )
        for p in points:
            state[p.payload["file"]] = p.payload["file_hash"]
        if offset is None:
            break
    return state


@task
def delete_file_chunks(client, files: list[str]) -> None:
    if not files:
        return
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="file", match=models.MatchAny(any=files))]
            )
        ),
    )


@task
def reindex_files(client, files: list[str]) -> int:
    """Re-chunk and re-embed the given files. Returns chunk count."""
    all_chunks = []
    for rel_path in files:
        path = VAULT_PATH / rel_path
        raw = path.read_bytes()
        content = raw.decode("utf-8", errors="replace")
        parts = Path(rel_path).parts
        category = parts[0] if len(parts) > 1 else "(root)"
        file_hash = hashlib.md5(raw).hexdigest()
        all_chunks.extend(chunk_markdown(content, rel_path, category, file_hash))
    if all_chunks:
        embed_and_upsert(client, all_chunks)
    return len(all_chunks)


@flow(name="vault-incremental-reindex", log_prints=True)
def incremental_reindex() -> None:
    client = get_qdrant_client()
    ensure_collection(client)

    on_disk = scan_vault()
    in_index = indexed_state(client)

    changed = [f for f, h in on_disk.items() if in_index.get(f) != h]
    removed = [f for f in in_index if f not in on_disk]

    print(f"On disk: {len(on_disk)} files, indexed: {len(in_index)} files")
    print(f"Changed/new: {len(changed)}, removed: {len(removed)}")

    # Delete stale chunks first: removed files entirely, and changed files
    # so that chunks which no longer exist after re-chunking don't linger.
    delete_file_chunks(client, removed + changed)
    n = reindex_files(client, changed)

    print(f"Re-indexed {len(changed)} files ({n} chunks); removed {len(removed)} files.")


if __name__ == "__main__":
    incremental_reindex()
