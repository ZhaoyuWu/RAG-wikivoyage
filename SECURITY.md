# Security & Privacy

This project handles personal notes and talks to third-party services, so
security is a first-class concern. This document states the model, the
threats considered, and the boundaries.

## Data boundaries

| Data | Where it lives | Leaves the machine? |
|------|----------------|---------------------|
| Personal notes (`vault`) | Local embedded Qdrant, gitignored | **Never.** Answered only by the local model. |
| Public travel corpus (`wikivoyage`) | Local Qdrant | Excerpt context is sent to the cloud model when `GROQ_API_KEY` is set. |
| Map / routing | — | Only place names and coordinates go to OSRM / DB / Transitous. No corpus content. |
| Conversations, audit logs | Local SQLite / JSONL, gitignored | Never. |

The provider split is enforced in code (`CLOUD_COLLECTIONS`): a collection
not on that list is always answered locally, regardless of the API key.

## Authentication & access control

- JWT bearer tokens over PBKDF2-SHA256 password hashes (`src/auth.py`).
  `AUTH_ENABLED=0` keeps single-user local mode open; enable for multi-user.
- Role rules filter **retrieval itself** (`deny_categories` become a Qdrant
  `must_not`), so a restricted user's answer can never quote a document
  they may not read — answer leakage is document leakage.
- Per-user sliding-window rate limits; login attempts are throttled and
  audited.

## Prompt-injection defense (defense in depth)

1. **Input guard** (`src/guard.py`): a regex blacklist for jailbreak /
   prompt-extraction phrasings, plus an optional LLM classifier on the
   cloud path. Blocked requests never reach retrieval.
2. **Excerpt isolation**: retrieved text is wrapped in `<excerpt>` tags the
   system prompt declares as data, **and** scrubbed sentence-by-sentence of
   embedded injection commands. A poisoned-document experiment showed tag
   declaration alone was insufficient (the model copied the injected
   string); sanitizing stops it.

## Secrets

`.env` is gitignored and read only from the environment. The repository is
grep-clean of names, personal paths, and keys.

**This project's `.env` file is a single-machine convenience, not a
production pattern.** In a real deployment, secrets belong in a managed
store (HashiCorp Vault, AWS/GCP KMS) with rotation and immediate
revocation on exposure. `JWT_SECRET` and any API key should be rotated
periodically and whenever they may have been disclosed.

## Reporting

This is a personal learning project with no production users. Security
observations are welcome via a GitHub issue.
