"""Unit tests for the local-recipe -> gbrain bulk ingest."""
from __future__ import annotations

from typing import Any

from inference_optimizer.recipe_kb.gbrain_ingest import (
    _scalar,
    ingest_local_to_gbrain,
    recipe_to_page,
)
from inference_optimizer.recipe_kb.gbrain_remote_client import _page_to_recipe


def _recipe(**over: Any) -> dict[str, Any]:
    base = {
        "canonical_id": "inference:qwen3-32b:mi300x:sglang:0_5_11:fp8",
        "model": "Qwen3-32B", "hardware": "mi300x", "framework": "sglang",
        "framework_version": "0_5_11", "precision": "fp8",
        "best_config": {"extra_sglang_args": "--cuda-graph-max-bs 256", "FOO": "1"},
        "best_throughput": 5800.5, "validated_gain_pct": 7.8,
        "authority": "EXPERIENTIAL", "confidence": 0.85,
    }
    base.update(over)
    return base


def test_scalar_quotes_risky_keeps_barewords() -> None:
    # digit-leading underscore token must be quoted (else YAML octal 329)
    assert _scalar("0_5_11") == '"0_5_11"'
    # tokens with ':' / spaces / yaml-keywords get quoted
    assert _scalar("kind:recipe") == '"kind:recipe"'
    assert _scalar("--x 1") == '"--x 1"'
    assert _scalar("no") == '"no"'
    # identifier-ish barewords stay unquoted
    assert _scalar("recipe") == "recipe"
    assert _scalar("mi300x") == "mi300x"
    assert _scalar("unknown_version") == "unknown_version"
    # real numbers emit bare
    assert _scalar(5800.5) == "5800.5"
    assert _scalar(True) == "true"
    assert _scalar(None) == "null"


def test_recipe_to_page_roundtrips_via_reader() -> None:
    slug, content = recipe_to_page(_recipe())
    assert slug == "recipe-snapshot/inference/qwen3-32b/mi300x/sglang/0_5_11/fp8"
    assert content.startswith("---\ntype: recipe\n")
    # the version token must be quoted so it survives YAML parse
    assert 'framework_version: "0_5_11"' in content
    # Simulate the gbrain get_page frontmatter the reader would see.
    fm = {
        "attrs": {
            "model": "Qwen3-32B", "hardware": "mi300x", "framework": "sglang",
            "framework_version": "0_5_11", "precision": "fp8",
            "best_config_args": "--cuda-graph-max-bs 256",
            "best_config_envs": {"FOO": "1"},
            "best_throughput": 5800.5, "validated_gain_pct": 7.8,
        },
        "authority": "EXPERIENTIAL", "confidence": 0.85,
    }
    r = _page_to_recipe(fm)
    assert r["canonical_id"] == "inference:qwen3-32b:mi300x:sglang:0_5_11:fp8"
    assert r["best_config"] == {"extra_sglang_args": "--cuda-graph-max-bs 256", "FOO": "1"}
    assert r["best_throughput"] == 5800.5


def test_recipe_to_page_skips_bare_anchor() -> None:
    # anchor with no best_config is not worth caching remotely
    assert recipe_to_page(_recipe(best_config={})) is None
    assert recipe_to_page(_recipe(best_config={}, canonical_id="")) is None


class _FakeMcp:
    def __init__(self) -> None:
        self.puts: list[str] = []

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "put_page":
            self.puts.append(args["slug"])
        return {}


def test_ingest_counts_and_gates() -> None:
    recipes = [
        _recipe(),  # has config -> ingest
        _recipe(canonical_id="inference:a:b:c:d:e", best_config={}),  # anchor -> skip
        _recipe(canonical_id="inference:m2:mi355x:vllm:v1:fp16",
                model="m2", hardware="mi355x", framework="vllm"),  # ingest
    ]
    mcp = _FakeMcp()
    stats = ingest_local_to_gbrain(recipes=recipes, mcp=mcp, dry_run=False)
    assert stats["total"] == 3
    assert stats["ingested"] == 2
    assert stats["skipped_no_config"] == 1
    assert stats["errors"] == 0
    assert len(mcp.puts) == 2


def test_ingest_dry_run_writes_nothing() -> None:
    mcp = _FakeMcp()
    stats = ingest_local_to_gbrain(recipes=[_recipe()], mcp=mcp, dry_run=True)
    assert stats["ingested"] == 1 and mcp.puts == []
