"""RAG answer generation: retrieve chunks, ask an LLM, cite sources.

Two providers, selected via LLM_PROVIDER:
- "anthropic": Claude API, needs ANTHROPIC_API_KEY
- "ollama": local Ollama server, fully offline
"""

import os

from .config import CLAUDE_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL
from .retrieval import Hit, hybrid_search

SYSTEM_PROMPT = (
    "You are a personal knowledge base assistant. Answer questions using the "
    "provided note excerpts. The notes are mostly in Chinese with some German "
    "and English. Answer in the same language as the question.\n"
    "Rules:\n"
    "- Base every claim on the excerpts. Do not invent facts.\n"
    "- If the excerpts are only partially relevant, answer with what they do "
    "contain and briefly note what is missing.\n"
    "- Only if none of the excerpts relate to the question at all, reply with "
    "exactly 笔记库里没有相关内容 and nothing else, no sources.\n"
    "- Otherwise end your answer with a '来源:' section listing each source "
    "you actually used as 'file :: heading', one per line."
)


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h.file} :: {h.heading}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _generate_anthropic(context: str, question: str) -> str:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either set it, or switch to a local "
            "model with LLM_PROVIDER=ollama."
        )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Note excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    if response.stop_reason == "refusal":
        return "请求被安全策略拒绝,请换一种问法。"
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_ollama(context: str, question: str) -> str:
    import httpx

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Note excerpts:\n\n{context}\n\nQuestion: {question}",
                    },
                ],
            },
            timeout=300.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it installed and running? "
            "Install from https://ollama.com, then: ollama pull " + OLLAMA_MODEL
        )
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    return data["message"]["content"]


def _stream_ollama(context: str, question: str):
    """Yield answer text deltas from a local Ollama model."""
    import json as _json

    import httpx

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Note excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    try:
        with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload, timeout=300.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = _json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it installed and running? "
            "Install from https://ollama.com, then: ollama pull " + OLLAMA_MODEL
        )


def _stream_anthropic(context: str, question: str):
    """Yield answer text deltas from the Claude API."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either set it, or switch to a local "
            "model with LLM_PROVIDER=ollama."
        )
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Note excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    ) as stream:
        yield from stream.text_stream


def ask_stream(
    question: str,
    top_k: int = 5,
    category: str | None = None,
    collection: str | None = None,
):
    """Yield pipeline events: retrieval trace, answer deltas, final summary."""
    import time

    t0 = time.perf_counter()
    hits = hybrid_search(question, top_k=top_k, category=category, collection=collection)
    retrieval_ms = round((time.perf_counter() - t0) * 1000)

    chunks = [
        {"score": h.score, "file": h.file, "heading": h.heading, "text": h.text}
        for h in hits
    ]
    yield {"type": "retrieval", "retrieval_ms": retrieval_ms, "chunks": chunks}

    if not hits:
        yield {"type": "done", "answer": "笔记库里没有相关内容。", "sources": [], "trace": None}
        return

    context = build_context(hits)
    model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else CLAUDE_MODEL
    generate = _stream_ollama if LLM_PROVIDER == "ollama" else _stream_anthropic

    t1 = time.perf_counter()
    parts: list[str] = []
    for delta in generate(context, question):
        parts.append(delta)
        yield {"type": "delta", "text": delta}
    generation_s = round(time.perf_counter() - t1, 1)

    yield {
        "type": "done",
        "answer": "".join(parts),
        "sources": [{"file": h.file, "heading": h.heading, "score": h.score} for h in hits],
        "trace": {
            "provider": LLM_PROVIDER,
            "model": model,
            "retrieval_ms": retrieval_ms,
            "generation_s": generation_s,
            "context_chars": len(context),
            "chunks": chunks,
        },
    }


def ask(
    question: str,
    top_k: int = 5,
    category: str | None = None,
    collection: str | None = None,
) -> dict:
    """Retrieve context and generate a cited answer, with a pipeline trace."""
    import time

    t0 = time.perf_counter()
    hits = hybrid_search(question, top_k=top_k, category=category, collection=collection)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not hits:
        return {"answer": "笔记库里没有相关内容。", "sources": [], "trace": None}

    context = build_context(hits)
    t1 = time.perf_counter()
    if LLM_PROVIDER == "ollama":
        answer = _generate_ollama(context, question)
        model = OLLAMA_MODEL
    else:
        answer = _generate_anthropic(context, question)
        model = CLAUDE_MODEL
    generation_s = time.perf_counter() - t1

    sources = [{"file": h.file, "heading": h.heading, "score": h.score} for h in hits]
    trace = {
        "provider": LLM_PROVIDER,
        "model": model,
        "retrieval_ms": round(retrieval_ms),
        "generation_s": round(generation_s, 1),
        "context_chars": len(context),
        "chunks": [
            {"score": h.score, "file": h.file, "heading": h.heading, "text": h.text}
            for h in hits
        ],
    }
    return {"answer": answer, "sources": sources, "trace": trace}
