"""MCP server exposing the travel corpus to any MCP client (Claude Desktop,
Claude Code, ...).

This turns the project from a standalone web app into a tool other agents can
call: an assistant asked "where can I hike in Germany this weekend?" can invoke
`search_travel_corpus` and get grounded results from the 49k-chunk Wikivoyage
index, or call `route_a_to_b` for real travel times.

Runs over stdio — the client launches this process and talks to it on
stdin/stdout. Because it is its own process, it holds the embedded Qdrant lock
alone; don't run it while the web API is also open on the same qdrant_db.

Register with Claude Desktop by adding to its config
(claude_desktop_config.json):

    {
      "mcpServers": {
        "vault-rag": {
          "command": "d:/.../vault-rag/venv/Scripts/python.exe",
          "args": ["-m", "src.mcp_server"],
          "cwd": "d:/.../vault-rag"
        }
      }
    }
"""

from mcp.server import MCPServer

mcp = MCPServer("vault-rag")

TRAVEL_COLLECTION = "wikivoyage"


@mcp.tool()
def search_travel_corpus(query: str, top_k: int = 5) -> list[dict]:
    """Search a German travel guide (Wikivoyage) with hybrid retrieval
    (dense + BM25, fused with RRF). Ask in any language — the index is
    multilingual. Returns the most relevant passages with their source
    article and heading.

    Args:
        query: What to look for, e.g. "castles near Cologne" or "哈茨徒步".
        top_k: How many passages to return (1-20).
    """
    from .retrieval import hybrid_search

    hits = hybrid_search(query, top_k=max(1, min(top_k, 20)),
                         collection=TRAVEL_COLLECTION)
    return [{"score": round(h.score, 3), "article": h.file,
             "heading": h.heading, "text": h.text[:600],
             "geo": h.geo} for h in hits]


@mcp.tool()
def route_a_to_b(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Find how to travel from one German place to another, with real travel
    time. Place names are resolved against the travel corpus, so use names
    that appear there (German spelling works best, e.g. "Köln", "Goslar").

    Args:
        from_place: Origin, e.g. "Essen".
        to_place: Destination, e.g. "Berlin".
        mode: "transit" for trains/public transport, or "car" for driving.
    """
    from .routing import route

    mode = "car" if mode == "car" else "transit"
    try:
        result = route(from_place, to_place, mode)
    except LookupError as e:
        return {"error": f"Unknown place: {e}"}
    except RuntimeError as e:
        return {"error": f"Routing service unavailable: {e}"}

    best = min((o["duration_min"] for o in result["options"]), default=None)
    return {
        "from": result["from"]["name"],
        "to": result["to"]["name"],
        "mode": mode,
        "fastest_min": best,
        "options": [{"summary": o["summary"],
                     "duration_min": o["duration_min"],
                     "transfers": o.get("transfers", 0)}
                    for o in result["options"][:3]],
    }


if __name__ == "__main__":
    mcp.run()
