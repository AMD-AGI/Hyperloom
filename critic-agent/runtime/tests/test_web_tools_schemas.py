"""Tests for :func:`runtime.web_tools.tool_schemas.build_tool_schemas`.

Pure-function unit tests; covers gating, name + required fields, and
mutual-exclusion fields.
"""

from __future__ import annotations

from runtime.web_tools.config import WebToolsConfig
from runtime.web_tools.tool_schemas import build_tool_schemas


def _cfg(**overrides) -> WebToolsConfig:
    base = dict(
        critic_web_tools_enabled=True,
        search_provider="tavily",
        tavily_api_key="test-key",
        fetch_enabled=True,
    )
    base.update(overrides)
    return WebToolsConfig(**base)


def test_returns_empty_when_master_disabled():
    cfg = _cfg(critic_web_tools_enabled=False)
    assert build_tool_schemas(cfg) == []


def test_only_search_when_fetch_disabled():
    cfg = _cfg(fetch_enabled=False)
    schemas = build_tool_schemas(cfg)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["web_search"]


def test_only_fetch_when_no_search_provider():
    cfg = _cfg(search_provider="disabled", fetch_enabled=True)
    schemas = build_tool_schemas(cfg)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["web_fetch"]


def test_both_schemas_when_fully_enabled():
    schemas = build_tool_schemas(_cfg())
    names = [s["function"]["name"] for s in schemas]
    assert names == ["web_search", "web_fetch"]


def test_no_search_schema_without_api_key():
    cfg = _cfg(
        tavily_api_key="",
        serper_api_key="",
        brave_api_key="",
        fetch_enabled=True,
    )
    names = [s["function"]["name"] for s in build_tool_schemas(cfg)]
    assert names == ["web_fetch"]


def test_no_search_schema_for_unimplemented_brave_provider():
    cfg = _cfg(
        search_provider="brave",
        tavily_api_key="",
        serper_api_key="",
        brave_api_key="brave-key",
        fetch_enabled=False,
    )
    assert build_tool_schemas(cfg) == []


def test_search_schema_required_fields():
    schemas = build_tool_schemas(_cfg(fetch_enabled=False))
    search = schemas[0]
    assert search["type"] == "function"
    fn = search["function"]
    assert fn["name"] == "web_search"
    assert "description" in fn
    params = fn["parameters"]
    assert params["required"] == ["query"]
    props = params["properties"]
    assert {"query", "allowed_domains", "blocked_domains",
            "max_results", "site", "freshness"}.issubset(props)
    assert props["freshness"]["enum"] == ["day", "week", "month", "year", "any"]


def test_fetch_schema_required_fields():
    schemas = build_tool_schemas(_cfg(search_provider="disabled"))
    fetch = schemas[0]
    fn = fetch["function"]
    assert fn["name"] == "web_fetch"
    params = fn["parameters"]
    assert params["required"] == ["url"]
    props = params["properties"]
    assert {"url", "max_bytes", "raw"}.issubset(props)
    assert props["max_bytes"]["maximum"] == 10 * 1024 * 1024


def test_schemas_are_json_serializable():
    import json
    out = build_tool_schemas(_cfg())
    json.dumps(out)  # must not raise
