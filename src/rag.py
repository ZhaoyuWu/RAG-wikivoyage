"""RAG answer generation: retrieve chunks, ask an LLM, cite sources.

Two providers, selected via LLM_PROVIDER:
- "anthropic": Claude API, needs ANTHROPIC_API_KEY
- "ollama": local Ollama server, fully offline
"""

import os

from .config import CLAUDE_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL
from .retrieval import Hit, hybrid_search

SYSTEM_PROMPT = (
    "You are a personal knowledge base assistant. Answer questions using ONLY "
    "the provided note excerpts. The notes are mostly in Chinese with some "
    "German and English. Answer in the same language as the question.\n"
    "Rules:\n"
    "- Base every claim on the excerpts. Do not invent facts.\n"
    "- If the excerpts do not contain relevant information, say exactly that "
    "the notes do not cover this topic (笔记库里没有相关内容).\n"
    "- End your answer with a '来源:' section listing each cited source as "
    "'file :: heading', one per line."
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


def ask(question: str, top_k: int = 5, category: str | None = None) -> dict:
    """Retrieve context and generate a cited answer."""
    hits = hybrid_search(question, top_k=top_k, category=category)
    if not hits:
        return {"answer": "笔记库里没有相关内容。", "sources": []}

    context = build_context(hits)
    if LLM_PROVIDER == "ollama":
        answer = _generate_ollama(context, question)
    else:
        answer = _generate_anthropic(context, question)

    sources = [{"file": h.file, "heading": h.heading, "score": h.score} for h in hits]
    return {"answer": answer, "sources": sources}
