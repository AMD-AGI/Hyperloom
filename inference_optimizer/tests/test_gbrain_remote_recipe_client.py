"""Unit tests for the gbrain read-side recipe-snapshot client.

Exercises the page->Recipe adaptation + the RemoteRecipeClient-compatible
read surface against a fake MCP (no network).
"""
from __future__ import annotations

from typing import Any

from inference_optimizer.recipe_kb.gbrain_remote_client import (
    GbrainRemoteRecipeClient,
    _page_to_recipe,
    build_gbrain_remote_from_env,
)


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
    assert r["canonical_id"] == "inference:qwen3-32b:mi300x:sglang:unknown_version:fp8"
    assert r["model"] == "Qwen/Qwen3-32B"
    assert r["best_config"]["extra_server_args"] == "--cuda-graph-max-bs 256"
    assert r["best_config"]["FOO"] == "1"
    assert r["validated_gain_pct"] == 12.5
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


def test_get_recipe_miss_on_unknown() -> None:
    c = _client({"cortex/recipe/a/b": _recipe_page("modelA", "mi300x")})
    assert c.get_recipe(canonical_id="inference:other:mi355x:vllm:v1:fp16") is None


def test_search_filters_by_label_match() -> None:
    c = _client({
        "r1": _recipe_page("Qwen3-32B", "mi300x", "sglang"),
        "r2": _recipe_page("Llama-3-70B", "mi300x", "vllm"),
        "r3": _recipe_page("Qwen3-32B", "mi355x", "sglang"),
    })
    # model-only filter → both Qwen rows
    rows = c.search(label_match={"model": "Qwen3-32B"})
    assert {r["hardware"] for r in rows} == {"mi300x", "mi355x"}
    # model + hardware → exactly one
    rows = c.search(label_match={"model": "Qwen3-32B", "hardware": "mi300x"})
    assert len(rows) == 1 and rows[0]["framework"] == "sglang"
    # framework filter
    rows = c.search(label_match={"framework": "vllm"})
    assert len(rows) == 1 and rows[0]["model"] == "Llama-3-70B"


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
