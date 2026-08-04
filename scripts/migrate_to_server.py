"""Migrate collections from embedded local Qdrant to a Qdrant server.

Scrolls every point (payload + named vectors) out of the on-disk local
store and upserts it into the server, which builds HNSW indexes. Run the
server first (docker compose up qdrant), then:

    python -m scripts.migrate_to_server
"""

import os

from qdrant_client import QdrantClient, models

from src.config import DENSE_DIM, QDRANT_PATH, utf8_stdout

utf8_stdout()

SERVER_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
BATCH = 256


def ensure_collection(server: QdrantClient, name: str) -> None:
    if server.collection_exists(name):
        return
    server.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(
            size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )


def migrate(name: str, local: QdrantClient, server: QdrantClient) -> int:
    ensure_collection(server, name)
    moved, offset = 0, None
    while True:
        points, offset = local.scroll(
            collection_name=name, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        server.upsert(name, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ])
        moved += len(points)
        print(f"  {name}: {moved} points", flush=True)
        if offset is None:
            break
    return moved


def main():
    local = QdrantClient(path=str(QDRANT_PATH))
    server = QdrantClient(url=SERVER_URL)
    for coll in local.get_collections().collections:
        total = migrate(coll.name, local, server)
        print(f"migrated {coll.name}: {total} points", flush=True)


if __name__ == "__main__":
    main()
