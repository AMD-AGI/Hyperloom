# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the local-recipe -> gbrain bulk ingest."""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_ingest import (
    _best_config_split,
    _recipe_slug_prefix,
    _scalar,
    recipe_to_page,
)
from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import _page_to_recipe


def _recipe(**over: Any) -> dict[str, Any]:
    base = {
        "canonical_id": "inference:qwen3-32b:mi300x:sglang:unknown_model_type:unknown_arch:0_5_11:fp8",
        "model": "Qwen3-32B",
        "hardware": "mi300x",
        "framework_name": "sglang",
        "framework_version": "0_5_11",
        "precision": "fp8",
        "best_config": {"extra_server_args": "--cuda-graph-max-bs 256", "FOO": "1"},
        "best_throughput": 5800.5,
        "validated_gain_pct": 7.8,
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
    }
    base.update(over)
    return base


def test_recipe_to_page_reads_legacy_framework_key() -> None:
    """The batch ingest reads raw on-disk recipe.json (no Recipe normalization),
    so rows persisted before the framework_name rename still carry the legacy
    ``framework`` key. The emitted page must surface it as ``framework_name``
    (attr + tag) instead of an empty framework dimension."""
    import yaml

    legacy = _recipe()
    legacy.pop("framework_name")
    legacy["framework"] = "sglang"
    slug, content = recipe_to_page(legacy)
    assert slug.split("/")[4] == "sglang"
    fm = yaml.safe_load(content.split("---", 2)[1])
    assert fm["attrs"]["framework_name"] == "sglang"
    assert "framework_name:sglang" in fm["tags"]
    recovered = _page_to_recipe(fm)
    assert recovered["labels"]["framework_name"] == "sglang"


def test_scalar_quotes_risky_keeps_barewords() -> None:
    # digit-leading underscore token must be quoted (else YAML octal 329)
    assert _scalar("0_5_11") == '"0_5_11"'
    assert _scalar("kind:recipe") == '"kind:recipe"'
    assert _scalar("--x 1") == '"--x 1"'
    assert _scalar("no") == '"no"'
    assert _scalar("recipe") == "recipe"
    assert _scalar("mi300x") == "mi300x"
    assert _scalar("unknown_version") == "unknown_version"
    assert _scalar(5800.5) == "5800.5"
    assert _scalar(True) == "true"
    assert _scalar(None) == "null"


def test_recipe_to_page_roundtrips_via_reader() -> None:
    slug, content = recipe_to_page(_recipe())
    assert slug == "hyperloom-recipe-kb/inference/qwen3-32b/mi300x/sglang/unknown_model_type/unknown_arch/0_5_11/fp8"
    # Check version token quoting survived
    assert content.startswith("---\ntype: recipe\n")
    # the version token must be quoted so it survives YAML parse
    assert 'framework_version: "0_5_11"' in content
    fm = {
        "attrs": {
            "model": "Qwen3-32B",
            "hardware": "mi300x",
            "framework_name": "sglang",
            "framework_version": "0_5_11",
            "precision": "fp8",
            "best_config_args": "--cuda-graph-max-bs 256",
            "best_config_envs": {"FOO": "1"},
            "best_throughput": 5800.5,
            "validated_gain_pct": 7.8,
        },
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
    }
    r = _page_to_recipe(fm)
    # Reader emits the unified nested KB-interface envelope: champion under
    # ``body.best_config``, throughput under ``metrics``/``body``.
    assert r["canonical_id"] == "inference:qwen3-32b:mi300x:sglang:unknown_model_type:unknown_arch:0_5_11:fp8"
    assert r["body"]["best_config"] == {
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"FOO": "1"},
    }
    assert r["body"]["best_throughput"] == 5800.5


def test_recipe_to_page_uses_configured_slug_prefix(monkeypatch) -> None:
    """The ingest write side must use the same prefix env as gbrain reads."""
    monkeypatch.setenv("GBRAIN_RECIPE_SLUG_PREFIX", "/hyperloom-session-kb/")

    slug, _content = recipe_to_page(_recipe())

    assert _recipe_slug_prefix() == "hyperloom-session-kb"
    assert slug == "hyperloom-session-kb/inference/qwen3-32b/mi300x/sglang/unknown_model_type/unknown_arch/0_5_11/fp8"


def test_best_config_split_unwraps_nested_extra_envs() -> None:
    # Nested extra_envs dict must be unwrapped, not str()'d into one bogus key.
    args, envs = _best_config_split(
        {
            "extra_server_args": "--cuda-graph-max-bs 256",
            "extra_envs": {"SGLANG_X": "1", "FOO": "bar"},
        }
    )
    assert args == "--cuda-graph-max-bs 256"
    assert envs == {"SGLANG_X": "1", "FOO": "bar"}
    assert "extra_envs" not in envs


def test_best_config_split_handles_flat_envs() -> None:
    args, envs = _best_config_split(
        {
            "extra_server_args": "--x 1",
            "FOO": "1",
            "BAR": "2",
        }
    )
    assert args == "--x 1"
    assert envs == {"FOO": "1", "BAR": "2"}


def test_best_config_split_skips_passthrough_metadata() -> None:
    # name/tput/accuracy are not envs and must not be serialized as such.
    args, envs = _best_config_split(
        {
            "extra_server_args": "--x 1",
            "name": "v1",
            "tput": 5400.0,
            "accuracy": 0.9,
            "extra_envs": {"E": "1"},
        }
    )
    assert args == "--x 1"
    assert envs == {"E": "1"}


def test_nested_extra_envs_roundtrips_via_reader() -> None:
    # End-to-end: nested-env recipe -> page -> read surfaces canonical extra_envs.
    import yaml

    rec = _recipe(
        best_config={
            "extra_server_args": "--cuda-graph-max-bs 256",
            "extra_envs": {"SGLANG_X": "1"},
        }
    )
    _slug, content = recipe_to_page(rec)
    fm = yaml.safe_load(content.split("---", 2)[1])
    r = _page_to_recipe(fm)
    assert r["body"]["best_config"] == {
        "extra_server_args": "--cuda-graph-max-bs 256",
        "extra_envs": {"SGLANG_X": "1"},
    }


def test_negative_knowledge_roundtrips() -> None:
    # Tier 0: pitfalls/lessons/what_failed must survive emit -> parse -> read.
    import yaml

    pitfalls = [{"flag": "--num-continuous-decode-steps 16", "reason": "noise -0.3%"}]
    lessons = [{"note": "fp8 baseline already near-optimal"}]
    what_failed = [{"variant": "ncds16", "gain_pct": -0.28}]
    what_worked = [{"flag": "--cuda-graph-max-bs 256", "gain_pct": 0.4}]
    _slug, content = recipe_to_page(
        _recipe(
            pitfalls=pitfalls,
            lessons=lessons,
            what_failed=what_failed,
            what_worked=what_worked,
        )
    )
    fm = yaml.safe_load(content.split("---", 2)[1])
    r = _page_to_recipe(fm)
    # pitfalls / lessons stay top-level in the nested envelope; the
    # success / failure lists map to findings / failures.
    assert r["pitfalls"] == pitfalls
    assert r["lessons"] == lessons
    assert r["failures"] == what_failed
    assert r["findings"] == what_worked


def test_negative_knowledge_absent_yields_empty() -> None:
    import yaml

    fm = yaml.safe_load(recipe_to_page(_recipe())[1].split("---", 2)[1])
    r = _page_to_recipe(fm)
    assert r["pitfalls"] == [] and r["lessons"] == []
    assert r["failures"] == [] and r["findings"] == []


def test_recipe_to_page_encodes_bare_anchor_by_default() -> None:
    assert recipe_to_page(_recipe(best_config={})) is not None
    assert recipe_to_page(_recipe(best_config={}, canonical_id="")) is None
