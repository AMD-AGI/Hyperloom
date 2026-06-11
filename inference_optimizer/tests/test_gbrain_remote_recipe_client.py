# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the gbrain read-side recipe-snapshot client (page->Recipe adaptation + read surface)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from inference_optimizer.recipe_kb import gbrain_remote_client as grc
from inference_optimizer.recipe_kb.gbrain_remote_client import (
    GbrainRemoteError,
    GbrainRemoteRecipeClient,
    _page_to_recipe,
    build_gbrain_remote_from_env,
)
from inference_optimizer.recipe_kb.remote_client import RemoteRecipeClientError


class _FakeMcp:
    """Stand-in for the gbrain MCP: serves canned list_pages / get_page."""

    def __init__(self, pages: dict[str, dict[str, Any]]) -> None:
        # pages: slug -> frontmatter dict
        self.pages = pages
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool == "list_pages":
            return [
                {"slug": s, "type": "recipe",
                 "updated_at": fm.get("updated_at", "")}
                for s, fm in self.pages.items()
            ]
        if tool == "get_page":
            fm = self.pages.get(args.get("slug"))
            return {"frontmatter": fm} if fm is not None else {}
        return {}


def _client(pages: dict[str, dict[str, Any]]) -> GbrainRemoteRecipeClient:
    c = GbrainRemoteRecipeClient(base_url="http://gbrain.test", token="tok", enabled=True)
    c._mcp = _FakeMcp(pages)  # type: ignore[assignment]
    return c


def _recipe_page(model: str, hw: str, framework: str = "sglang",
                 precision: str = "", args: str = "", gain: float = 0.0) -> dict[str, Any]:
    attrs: dict[str, Any] = {"model": model, "hardware": hw, "framework": framework}
    if precision:
        attrs["precision"] = precision
    if args:
        attrs["best_config_args"] = args
        attrs["best_config_envs"] = {"FOO": "1"}
    if gain:
        attrs["validated_gain_pct"] = gain
    return {
        "attrs": attrs,
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
        "updated_at": "2026-05-30T07:00:00Z",
    }


def test_page_to_recipe_maps_identity_and_config() -> None:
    fm = _recipe_page("Qwen/Qwen3-32B", "mi300x", "sglang", "fp8",
                      args="--cuda-graph-max-bs 256", gain=12.5)
    r = _page_to_recipe(fm)
    assert r is not None
    # Unified nested KB-interface envelope: identity under ``labels``,
    # champion under ``body.best_config``, throughput under ``metrics``.
    assert r["canonical_id"] == "inference:qwen3-32b:mi300x:sglang:unknown_version:fp8"
    assert r["labels"]["model"] == "qwen3-32b"
    assert r["labels"]["hardware"] == "mi300x"
    best_config = r["body"]["best_config"]
    assert best_config["extra_server_args"] == "--cuda-graph-max-bs 256"
    # Envs are nested under ``extra_envs`` (canonical warm-replay shape),
    # not flattened as sibling keys.
    assert best_config["extra_envs"] == {"FOO": "1"}
    assert "FOO" not in best_config
    assert r["metrics"]["validated_gain_pct"] == 12.5
    assert r["authority"] == "EXPERIENTIAL"


def test_page_to_recipe_requires_identity() -> None:
    assert _page_to_recipe({"attrs": {"model": "", "hardware": "mi300x"}}) is None
    assert _page_to_recipe({"attrs": {}}) is None


def test_get_recipe_roundtrip() -> None:
    c = _client({
        "cortex/recipe/qwen3-32b/mi300x": _recipe_page("Qwen3-32B", "mi300x", "sglang", "fp8"),
    })
    cid = "inference:qwen3-32b:mi300x:sglang:unknown_version:fp8"
    r = c.get_recipe(canonical_id=cid)
    assert r is not None and r["canonical_id"] == cid


def test_get_recipe_uses_direct_slug_fast_path() -> None:
    slug = "recipe-snapshot/inference/qwen3-32b/mi300x/sglang/unknown_version/fp8"
    c = _client({
        slug: _recipe_page("Qwen3-32B", "mi300x", "sglang", "fp8"),
        "recipe-snapshot/inference/other/mi300x/sglang/unknown_version/fp8": (
            _recipe_page("Other", "mi300x", "sglang", "fp8")
        ),
    })
    cid = "inference:qwen3-32b:mi300x:sglang:unknown_version:fp8"

    r = c.get_recipe(canonical_id=cid)

    assert r is not None and r["canonical_id"] == cid
    # Exact gbrain slugs should avoid the expensive broad list_pages scan.
    assert [tool for tool, _ in c._mcp.calls] == ["get_page"]  # type: ignore[union-attr]


def test_get_recipe_miss_on_unknown() -> None:
    c = _client({"cortex/recipe/a/b": _recipe_page("modelA", "mi300x")})
    assert c.get_recipe(canonical_id="inference:other:mi355x:vllm:v1:fp16") is None


def _hw(row: dict[str, Any]) -> str:
    return str((row.get("labels") or {}).get("hardware") or "")


def _fw(row: dict[str, Any]) -> str:
    return str((row.get("labels") or {}).get("framework") or "")


def _model(row: dict[str, Any]) -> str:
    return str((row.get("labels") or {}).get("model") or "")


def test_search_filters_by_label_match() -> None:
    c = _client({
        "r1": _recipe_page("Qwen3-32B", "mi300x", "sglang"),
        "r2": _recipe_page("Llama-3-70B", "mi300x", "vllm"),
        "r3": _recipe_page("Qwen3-32B", "mi355x", "sglang"),
    })
    # model-only filter → both Qwen rows. The gbrain adapter returns the
    # nested KB-interface shape, so identity lives under ``labels``.
    rows = c.search(label_match={"model": "Qwen3-32B"})
    assert {_hw(r) for r in rows} == {"mi300x", "mi355x"}
    # model + hardware → exactly one
    rows = c.search(label_match={"model": "Qwen3-32B", "hardware": "mi300x"})
    assert len(rows) == 1 and _fw(rows[0]) == "sglang"
    # framework filter
    rows = c.search(label_match={"framework": "vllm"})
    assert len(rows) == 1 and _model(rows[0]) == "llama-3-70b"


def test_search_reuses_scan_cache() -> None:
    c = _client({
        "r1": _recipe_page("Qwen3-32B", "mi300x", "sglang"),
        "r2": _recipe_page("Llama-3-70B", "mi300x", "vllm"),
    })

    assert len(c.search(label_match={"hardware": "mi300x"})) == 2
    first_call_count = len(c._mcp.calls)  # type: ignore[union-attr]
    assert any(tool == "list_pages" for tool, _ in c._mcp.calls)  # type: ignore[union-attr]

    assert len(c.search(label_match={"framework": "vllm"})) == 1
    # Second search should reuse the process-local scan cache; no extra MCP
    # calls are needed.
    assert len(c._mcp.calls) == first_call_count  # type: ignore[union-attr]


def test_disabled_client_returns_empty() -> None:
    c = GbrainRemoteRecipeClient(base_url="", token="", enabled=True)
    assert c.enabled is False
    assert c.get_recipe(canonical_id="inference:m:h:f:v:p") is None
    assert c.search(label_match={"model": "x"}) == []
    assert c.list_recent() == []
    assert c.health() is False


def test_get_history_and_attempts_are_empty() -> None:
    c = _client({"r1": _recipe_page("m", "mi300x")})
    assert c.get_history(canonical_id="inference:m:mi300x:sglang:v:p") == []
    assert c.list_attempts(canonical_id="inference:m:mi300x:sglang:v:p") == []
    assert c.list_session_attempts(session_id="s") == []
    assert c.session_summary(session_id="s") is None


def test_build_from_env(monkeypatch) -> None:
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)
    assert build_gbrain_remote_from_env() is None
    monkeypatch.setenv("GBRAIN_BASE_URL", "http://gbrain.test")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    c = build_gbrain_remote_from_env()
    assert c is not None and c.enabled is True


def test_client_returns_unified_nested_shape() -> None:
    # The unified KB interface: gbrain no longer advertises a flat-arbor
    # capability flag; every read returns the nested envelope the cortex
    # kb-service also emits, so the dispatcher runs ONE translation.
    c = GbrainRemoteRecipeClient(base_url="http://gbrain.test", token="tok", enabled=True)
    assert not hasattr(c, "returns_arbor_shape")
    fm = _recipe_page("Qwen3-32B", "mi300x", "sglang", "fp8",
                      args="--x 1")
    r = _page_to_recipe(fm)
    assert r is not None
    for key in ("labels", "body", "metrics", "findings", "failures", "gaps"):
        assert key in r, f"missing nested key {key!r}"


def test_gbrain_error_is_remote_recipe_client_error() -> None:
    # The dispatcher's ``except RemoteRecipeClientError`` fall-through only
    # catches gbrain failures if GbrainRemoteError subclasses it.
    assert issubclass(GbrainRemoteError, RemoteRecipeClientError)
    err = GbrainRemoteError("boom")
    assert isinstance(err, RemoteRecipeClientError)
    assert err.category == "transport"


class _Resp:
    """Minimal urlopen context-manager stand-in."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _patch_urlopen(monkeypatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(
        grc.urllib.request, "urlopen", lambda req, timeout=None: _Resp(payload),
    )


def test_mcp_call_raises_on_jsonrpc_error(monkeypatch) -> None:
    _patch_urlopen(monkeypatch, {
        "jsonrpc": "2.0", "id": "1",
        "error": {"code": -32000, "message": "boom"},
    })
    mcp = grc._GbrainMcp("http://gbrain.test", "tok", 2.0)
    with pytest.raises(GbrainRemoteError):
        mcp.call("put_page", {"slug": "s", "content": "c"})


def test_mcp_call_raises_on_tool_iserror(monkeypatch) -> None:
    _patch_urlopen(monkeypatch, {
        "jsonrpc": "2.0", "id": "1",
        "result": {"isError": True, "content": [{"text": "nope"}]},
    })
    mcp = grc._GbrainMcp("http://gbrain.test", "tok", 2.0)
    with pytest.raises(GbrainRemoteError):
        mcp.call("list_pages", {"type": "recipe"})


def test_health_false_on_rpc_error(monkeypatch) -> None:
    _patch_urlopen(monkeypatch, {
        "jsonrpc": "2.0", "id": "1",
        "error": {"code": -32000, "message": "boom"},
    })
    c = GbrainRemoteRecipeClient(base_url="http://gbrain.test", token="tok", enabled=True)
    assert c.health() is False
