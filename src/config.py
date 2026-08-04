"""Central configuration: paths, model names, collection settings.

All values can be overridden via environment variables so the same code
runs in local mode (embedded Qdrant) and server mode (Docker).
"""

import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader: KEY=VALUE lines, no quoting rules, env wins."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Source notes. Only markdown files below this directory are indexed.
# Set VAULT_PATH in the environment or in a .env file at the project root.
VAULT_PATH = Path(os.environ.get("VAULT_PATH", ""))

# Directories that must never be indexed, wherever they appear.
EXCLUDED_DIRS = {".obsidian", ".playwright-mcp", "_claude-memory-backup", "附件"}

# Qdrant. "local" uses an embedded on-disk instance, "server" connects via HTTP.
QDRANT_MODE = os.environ.get("QDRANT_MODE", "local")  # local | server
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", PROJECT_ROOT / "qdrant_db"))

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "vault")

# Embedding models
DENSE_MODEL = "BAAI/bge-m3"            # multilingual (zh/de/en), 1024-dim
DENSE_DIM = 1024
SPARSE_MODEL = "Qdrant/bm25"           # BM25 sparse vectors via fastembed
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = os.environ.get("USE_RERANKER", "0") == "1"

# Chunking
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 1500

# Generation. Provider "anthropic" needs ANTHROPIC_API_KEY; "ollama" runs
# fully offline against a local Ollama server.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")  # anthropic | ollama
CLAUDE_MODEL = "claude-sonnet-5"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

# Optional cloud acceleration via Groq's free tier. Collections listed in
# CLOUD_COLLECTIONS hold public data only and may be answered by the cloud
# model; everything else stays on the local provider regardless of the key.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
CLOUD_COLLECTIONS = {
    c.strip() for c in os.environ.get("CLOUD_COLLECTIONS", "wikivoyage").split(",")
    if c.strip()
}


def utf8_stdout() -> None:
    """Force UTF-8 console output; Windows defaults to cp1252 which cannot
    print CJK paths."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def ensure_vault_available() -> None:
    """Fail loudly if VAULT_PATH is unset or the drive is not mounted."""
    if not os.environ.get("VAULT_PATH"):
        raise FileNotFoundError(
            "VAULT_PATH is not set. Point it at your markdown folder via the "
            "environment or a .env file (see .env.example)."
        )
    if not VAULT_PATH.exists():
        raise FileNotFoundError(
            f"Vault path not found: {VAULT_PATH}. "
            "Is the external drive plugged in? Check VAULT_PATH."
        )


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Return the process-wide QdrantClient.

    Cached because local-mode Qdrant holds an exclusive storage lock; a
    second instance in the same process would fail to acquire it.
    """
    from qdrant_client import QdrantClient

    if QDRANT_MODE == "server":
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))
