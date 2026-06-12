# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplemental coverage for gbrain_remote_client: MCP envelope parsing,
scan pagination, metric filters, label-match edges, and env construction."""
from __future__ import annotations

from typing import Any

import pytest

from inference_optimizer.recipe_kb import gbrain_remote_client as grc
from inference_optimizer.recipe_kb.gbrain_remote_client import (
    GbrainRemoteError,
    GbrainRemoteRecipeClient,
    _as_float,
    _best_config_from_attrs,
    _json_list,
    _labels_match,
    _passes_metric_filters,
    build_gbrain_remote_from_env,
)


class _RawResp:
    """urlopen context-manager stand-in returning a raw string body."""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw.encode()

    def __enter__(self) -> "_RawResp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _patch_raw(monkeypatch, raw: str) -> None:
    monkeypatch.setattr(
        grc.urllib.request, "urlopen", lambda req, timeout=None: _RawResp(raw),
    )


def _mcp() -> grc._GbrainMcp:
    return grc._GbrainMcp("http://gbrain.test", "tok", 2.0)


# -- _GbrainMcp.call envelope parsing -------------------------------------
def test_mcp_call_transport_error(monkeypatch) -> None:
    def _boom(req, timeout=None):
        raise grc.urllib.error.URLError("down")

    monkeypatch.setattr(grc.urllib.request, "urlopen", _boom)
    with pytest.raises(GbrainRemoteError, match="transport error"):
        _mcp().call("list_pages", {})


def test_mcp_call_bad_json_envelope(monkeypatch) -> None:
    _patch_raw(monkeypatch, "{not-json")
    with pytest.raises(GbrainRemoteError, match="bad envelope"):
        _mcp().call("get_page", {})


def test_mcp_call_event_stream_framing_non_dict(monkeypatch) -> None:
    # text/event-stream body whose first data line decodes to a list ->
    # unexpected envelope type guard.
    _patch_raw(monkeypatch, "data: [1, 2, 3]\n")
    with pytest.raises(GbrainRemoteError, match="unexpected envelope type"):
        _mcp().call("list_pages", {})


def test_mcp_call_event_stream_returns_parsed_content(monkeypatch) -> None:
    body = (
        'data: {"jsonrpc":"2.0","id":"1","result":'
        '{"content":[{"text":"{\\"ok\\": true}"}]}}\n'
    )
    _patch_raw(monkeypatch, body)
    assert _mcp().call("get_page", {}) == {"ok": True}


def test_mcp_call_content_text_non_json_returns_text(monkeypatch) -> None:
    _patch_raw(
        monkeypatch,
        '{"jsonrpc":"2.0","id":"1","result":{"content":[{"text":"plain"}]}}',
    )
    assert _mcp().call("get_page", {}) == "plain"


def test_mcp_call_no_content_returns_result(monkeypatch) -> None:
    _patch_raw(monkeypatch, '{"jsonrpc":"2.0","id":"1","result":{"x":1}}')
    assert _mcp().call("get_page", {}) == {"x": 1}


# -- pure helpers ----------------------------------------------------------
def test_as_float_edges() -> None:
    assert _as_float("3.5") == 3.5
    assert _as_float(None) == 0.0
    assert _as_float("nan-ish") == 0.0


def test_json_list_variants() -> None:
    assert _json_list([1, 2]) == [1, 2]  # already-decoded passthrough
    assert _json_list('[{"a":1}]') == [{"a": 1}]
    assert _json_list("not-json") == []  # malformed -> []
    assert _json_list('{"a":1}') == []  # decoded non-list -> []
    assert _json_list("") == []
    assert _json_list(None) == []


def test_best_config_from_attrs_empty_and_filled() -> None:
    assert _best_config_from_attrs({}) == {}
    out = _best_config_from_attrs(
        {"best_config_args": "--x 1", "best_config_envs": {"A": 2}}
    )
    assert out["extra_server_args"] == "--x 1"
    assert out["extra_envs"] == {"A": "2"}


def test_labels_match_empty_and_non_mapping() -> None:
    # empty match -> always True
    assert _labels_match({"labels": {"model": "m"}}, {}) is True
    # recipe with non-mapping labels still compares safely (no match key set)
    assert _labels_match({"labels": None}, {"model": "m"}) is True


def test_labels_match_mismatch_returns_false() -> None:
    recipe = {"labels": {"model": "qwen3-32b", "hardware": "mi300x"}}
    assert _labels_match(recipe, {"hardware": "mi355x"}) is False
    assert _labels_match(recipe, {"hardware": "mi300x"}) is True


def test_passes_metric_filters() -> None:
    recipe = {"metrics": {"throughput": 120.0}, "body": {"best_throughput": 120.0}}
    assert _passes_metric_filters(recipe, {"throughput": {"min": 100.0}}) is True
    assert _passes_metric_filters(recipe, {"throughput": {"min": 200.0}}) is False
    assert _passes_metric_filters(recipe, {"throughput": {"max": 50.0}}) is False
    # best_throughput alias resolves through body when metric absent
    recipe2 = {"metrics": {}, "body": {"best_throughput": 80.0}}
    assert _passes_metric_filters(recipe2, {"best_throughput": {"min": 50.0}}) is True
    # missing metric entirely -> filtered out
    assert _passes_metric_filters({"metrics": {}, "body": {}}, {"latency": {"min": 1}}) is False


# -- client read surface edges --------------------------------------------
class _FakeMcp:
    def __init__(self, pages: dict[str, dict[str, Any]], page_size: int = 100) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.page_size = page_size

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool == "list_pages":
            items = [
                {"slug": s, "type": "recipe", "updated_at": fm.get("updated_at", "")}
                for s, fm in self.pages.items()
            ]
            after = args.get("updated_after")
            if after is not None:
                items = [i for i in items if i["updated_at"] > after]
            return items[: args.get("limit", self.page_size)]
        if tool == "get_page":
            fm = self.pages.get(args.get("slug"))
            return {"frontmatter": fm} if fm is not None else {}
        return {}


def _recipe_page(model: str, hw: str, updated: str) -> dict[str, Any]:
    return {
        "attrs": {"model": model, "hardware": hw, "framework": "sglang"},
        "updated_at": updated,
    }


def _client(pages: dict[str, dict[str, Any]]) -> GbrainRemoteRecipeClient:
    c = GbrainRemoteRecipeClient(base_url="http://gbrain.test", token="tok", enabled=True)
    c._mcp = _FakeMcp(pages)  # type: ignore[assignment]
    return c


def test_close_releases_mcp() -> None:
    c = _client({"r1": _recipe_page("m", "mi300x", "t1")})
    c.close()
    assert c._mcp is None


def test_scan_cache_ttl_env(monkeypatch) -> None:
    c = _client({})
    monkeypatch.setenv("GBRAIN_RECIPE_SCAN_TTL_SEC", "12.5")
    assert c._scan_cache_ttl() == 12.5
    monkeypatch.setenv("GBRAIN_RECIPE_SCAN_TTL_SEC", "bogus")
    assert c._scan_cache_ttl() == grc._SCAN_CACHE_TTL_SEC


def test_get_recipe_validation() -> None:
    c = _client({"r1": _recipe_page("m", "mi300x", "t1")})
    with pytest.raises(ValueError):
        c.get_recipe(canonical_id="")
    # version other than 1 -> None for interface parity
    assert c.get_recipe(canonical_id="inference:m:mi300x:sglang:v:p", version=2) is None
    # malformed canonical id -> remote miss
    assert c.get_recipe(canonical_id="garbage-no-colons") is None


def test_search_updated_since_and_order(monkeypatch) -> None:
    c = _client({
        "r1": _recipe_page("Qwen3-32B", "mi300x", "2026-01-01T00:00:00Z"),
        "r2": _recipe_page("Qwen3-32B", "mi355x", "2026-03-01T00:00:00Z"),
    })
    # updated_since filters out the older row
    rows = c.search(label_match={"model": "Qwen3-32B"}, updated_since="2026-02-01T00:00:00Z")
    assert len(rows) == 1
    # ascending order reverses the default newest-first ordering
    rows_asc = c.search(
        label_match={"model": "Qwen3-32B"}, order_by=grc.C.ORDER_BY_UPDATED_AT_ASC,
    )
    assert [r["labels"]["hardware"] for r in rows_asc][0] == "mi300x"


def test_search_metric_filter() -> None:
    c = _client({"r1": _recipe_page("m", "mi300x", "t1")})
    # the page has no throughput attr -> filtered out by a min throughput bound
    assert c.search(metric_filters={"throughput": {"min": 1.0}}) == []


def test_list_recent_returns_rows() -> None:
    c = _client({
        "r1": _recipe_page("m1", "mi300x", "2026-01-01T00:00:00Z"),
        "r2": _recipe_page("m2", "mi355x", "2026-02-01T00:00:00Z"),
    })
    rows = c.list_recent(limit=10)
    assert len(rows) == 2
    # newest first
    assert rows[0]["labels"]["hardware"] == "mi355x"


def test_build_from_env_timeout(monkeypatch) -> None:
    monkeypatch.setenv("GBRAIN_BASE_URL", "http://gbrain.test")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    monkeypatch.setenv("GBRAIN_HTTP_TIMEOUT_SEC", "7.5")
    c = build_gbrain_remote_from_env()
    assert c is not None and c.timeout_sec == 7.5
    # invalid timeout falls back to the default budget
    monkeypatch.setenv("GBRAIN_HTTP_TIMEOUT_SEC", "bad")
    c2 = build_gbrain_remote_from_env()
    assert c2 is not None and c2.timeout_sec == grc.C.FOREGROUND_HTTP_TIMEOUT_SEC
