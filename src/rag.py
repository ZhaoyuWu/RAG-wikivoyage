"""RAG answer generation: retrieve chunks, ask an LLM, cite sources.

Two providers, selected via LLM_PROVIDER:
- "anthropic": Claude API, needs ANTHROPIC_API_KEY
- "ollama": local Ollama server, fully offline
"""

import os

from .config import (
    CLAUDE_MODEL,
    CLOUD_COLLECTIONS,
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_URL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
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
                "keep_alive": "2h",
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


def ollama_keepalive() -> None:
    """Preload the model into RAM and keep it resident for two hours."""
    import httpx

    httpx.post(f"{OLLAMA_URL}/api/generate",
               json={"model": OLLAMA_MODEL, "keep_alive": "2h"}, timeout=300.0)


def _chat_messages(context: str | None, question: str,
                   history: list[dict] | None, system: str) -> list[dict]:
    """System prompt, prior turns, then the current question with excerpts."""
    messages = [{"role": "system", "content": system}]
    for m in (history or [])[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:1500]})
    content = question if context is None else \
        f"Note excerpts:\n\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": content})
    return messages


def _stream_ollama(context: str | None, question: str,
                   history: list[dict] | None = None, system: str = SYSTEM_PROMPT):
    """Yield answer text deltas from a local Ollama model."""
    import json as _json

    import httpx

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "keep_alive": "2h",
        "options": {"temperature": 0.2},
        "messages": _chat_messages(context, question, history, system),
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


def _stream_anthropic(context: str | None, question: str,
                      history: list[dict] | None = None, system: str = SYSTEM_PROMPT):
    """Yield answer text deltas from the Claude API."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either set it, or switch to a local "
            "model with LLM_PROVIDER=ollama."
        )
    client = anthropic.Anthropic()
    messages = _chat_messages(context, question, history, system)[1:]  # system separate
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        yield from stream.text_stream


def _stream_groq(context: str | None, question: str,
                 history: list[dict] | None = None, system: str = SYSTEM_PROMPT):
    """Yield answer text deltas from Groq's OpenAI-compatible API."""
    import json as _json

    import httpx

    payload = {
        "model": GROQ_MODEL,
        "stream": True,
        "temperature": 0.2,
        "messages": _chat_messages(context, question, history, system),
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        with httpx.stream("POST", f"{GROQ_URL}/chat/completions",
                          json=payload, headers=headers, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                delta = (_json.loads(data)["choices"][0]
                         .get("delta", {}).get("content"))
                if delta:
                    yield delta
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Groq API error {e.response.status_code} "
            f"({'rate limited' if e.response.status_code == 429 else 'request failed'})"
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"Groq unreachable: {e}")


def _local_provider():
    """The configured non-cloud provider (ollama or anthropic)."""
    if LLM_PROVIDER == "ollama":
        return "ollama", OLLAMA_MODEL, _stream_ollama
    return "anthropic", CLAUDE_MODEL, _stream_anthropic


def _pick_provider(collection: str | None):
    """Return (name, model, stream_fn) for a collection.

    Public-data collections go to the fast cloud model when a key is set;
    private collections always stay on the local provider.
    """
    resolved = collection or COLLECTION_NAME
    if GROQ_API_KEY and resolved in CLOUD_COLLECTIONS:
        return "groq", GROQ_MODEL, _stream_groq
    return _local_provider()


_ANAPHORA = ("那", "这", "它", "他", "她", "呢", "此", "还有", "继续", "上面",
             "刚才", "之前", "the same", "there", "it ", "that ", "dort", "das")


def _needs_rewrite(question: str, history: list[dict]) -> bool:
    """Cheap gate: only pay an LLM rewrite when the question looks like a
    follow-up that cannot stand alone."""
    if not history:
        return False
    if len(question) <= 12:
        return True
    lowered = question.lower()
    return any(marker in question or marker in lowered for marker in _ANAPHORA)


def _rewrite_query(question: str, history: list[dict], generate=None) -> str:
    """Condense a follow-up into a standalone search query via the LLM."""
    recent = history[-4:]
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)
    prompt = (
        "Conversation so far:\n" + convo +
        f"\n\nFollow-up question: {question}\n\n"
        "Rewrite the follow-up as one standalone search query that keeps all "
        "names and places from the context. Reply with the query only, in the "
        "language of the follow-up."
    )
    if generate is None:
        generate = _local_provider()[2]
    rewrite_system = ("You rewrite follow-up questions into standalone search "
                      "queries. Reply with the query only, nothing else.")
    try:
        try:
            rewritten = "".join(generate(None, prompt, None, rewrite_system)).strip()
        except RuntimeError:
            fallback = _local_provider()[2]
            if generate is fallback:
                raise
            rewritten = "".join(fallback(None, prompt, None, rewrite_system)).strip()
        # Guard against a chatty model: keep it only if it looks like a query.
        rewritten = rewritten.split("来源")[0].split("Source:")[0].strip().strip('"')
        if 0 < len(rewritten) < 200 and "\n" not in rewritten:
            return rewritten
    except RuntimeError:
        pass
    return question


def ask_stream(
    question: str,
    top_k: int = 5,
    category: str | None = None,
    collection: str | None = None,
    history: list[dict] | None = None,
):
    """Yield pipeline events: retrieval trace, answer deltas, final summary."""
    import time

    history = history or []

    # A routing question ("How do I get from A to B?") is answered by the
    # routing backends, not by retrieval.
    from .intent import detect_route_intent
    from .routing import route

    intent = detect_route_intent(question)
    if intent:
        yield {"type": "intent", "intent": "route", **intent}
        try:
            result = route(intent["from_place"], intent["to_place"], intent["mode"])
        except (LookupError, RuntimeError) as e:
            yield {"type": "error", "detail": str(e)}
            return
        best = min(o["duration_min"] for o in result["options"])
        answer = (f"{result['from']['name']} → {result['to']['name']} "
                  f"({'driving' if intent['mode'] == 'car' else 'train'}): "
                  f"fastest option {best // 60}h {best % 60}m, "
                  f"{len(result['options'])} option(s) shown on the map.")
        yield {"type": "route", "data": result}
        yield {"type": "done", "answer": answer, "sources": [], "trace": None}
        return

    provider, model, generate = _pick_provider(collection)

    search_query = question
    if _needs_rewrite(question, history):
        t_r = time.perf_counter()
        search_query = _rewrite_query(question, history, generate)
        if search_query != question:
            yield {"type": "rewrite", "query": search_query,
                   "rewrite_s": round(time.perf_counter() - t_r, 1)}

    t0 = time.perf_counter()
    hits = hybrid_search(search_query, top_k=top_k, category=category, collection=collection)
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

    t1 = time.perf_counter()
    parts: list[str] = []
    try:
        for delta in generate(context, question, history):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except RuntimeError as e:
        # Cloud outage or rate limit: retry locally if nothing streamed yet.
        if provider == "groq" and not parts:
            provider, model, generate = _local_provider()
            yield {"type": "fallback", "detail": str(e), "provider": provider}
            for delta in generate(context, question, history):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
        else:
            raise
    generation_s = round(time.perf_counter() - t1, 1)

    yield {
        "type": "done",
        "answer": "".join(parts),
        "sources": [{"file": h.file, "heading": h.heading, "score": h.score} for h in hits],
        "trace": {
            "provider": provider,
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
