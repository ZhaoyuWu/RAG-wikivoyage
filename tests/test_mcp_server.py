"""MCP server tool registration.

Skipped when the mcp package isn't installed (e.g. CI, which only installs
ruff/pytest/pyjwt). The tool bodies defer their heavy imports, so importing
the module needs only mcp itself.
"""

import asyncio

import pytest

pytest.importorskip("mcp", reason="mcp SDK not installed")

from src import mcp_server  # noqa: E402


def _tools():
    return asyncio.run(mcp_server.mcp.list_tools())


def test_two_tools_registered():
    names = {t.name for t in _tools()}
    assert names == {"search_travel_corpus", "route_a_to_b"}


def test_search_schema_from_type_hints():
    tool = next(t for t in _tools() if t.name == "search_travel_corpus")
    props = tool.input_schema["properties"]
    assert "query" in props and "top_k" in props
    assert tool.description  # docstring became the description


def test_route_schema_from_type_hints():
    tool = next(t for t in _tools() if t.name == "route_a_to_b")
    props = tool.input_schema["properties"]
    assert {"from_place", "to_place", "mode"} <= set(props)


def test_route_rejects_unknown_place(monkeypatch):
    # The tool should turn a LookupError into a friendly error dict, not raise.
    def boom(*a, **k):
        raise LookupError("Atlantis")

    monkeypatch.setattr(mcp_server, "route", boom, raising=False)
    # route is imported lazily inside the function; patch at the source instead.
    import src.routing as routing
    monkeypatch.setattr(routing, "route", boom)
    out = mcp_server.route_a_to_b("Atlantis", "Berlin")
    assert "error" in out
