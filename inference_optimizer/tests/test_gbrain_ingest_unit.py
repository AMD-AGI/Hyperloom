# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``recipe_kb.gbrain_ingest``: YAML emitter edge shapes, the
seed-only shareable-signal gate, page mapping (stack fingerprint + negative
knowledge), bulk + single mirror flows, env-built MCP, the mirroring KB
wrapper, and the CLI ``main`` entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.recipe_kb import gbrain_ingest as gi


class _FakeMcp:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def call(self, method: str, params: dict) -> Any:
        self.calls.append((method, params))
        if self.fail:
            raise RuntimeError("transport down")
        return {"ok": True}


# -- _emit_yaml ------------------------------------------------------------
def test_emit_yaml_shapes() -> None:
    out = gi._emit_yaml({
        "empty_list": [],
        "empty_map": {},
        "nested": {"a": 1},
        "scalar_list": ["x", "y"],
        "flat": "v",
    })
    assert "empty_list: []" in out
    assert "empty_map: {}" in out
    assert "- x" in out and "- y" in out
    assert "flat: v" in out


# -- _has_shareable_signal -------------------------------------------------
def test_has_shareable_signal_session_throughput() -> None:
    assert gi._has_shareable_signal({"sessions": [{"throughput_after": 12.0}]}) is True


def test_has_shareable_signal_session_actions() -> None:
    assert gi._has_shareable_signal(
        {"sessions": [{"throughput_after": "bad", "actions_taken": ["x"]}]},
    ) is True


def test_has_shareable_signal_negative_knowledge_field() -> None:
    assert gi._has_shareable_signal({"what_worked": ["aiter"]}) is True


def test_has_shareable_signal_model_class() -> None:
    assert gi._has_shareable_signal({"architectures": ["DeepseekV3"]}) is True


def test_has_shareable_signal_none() -> None:
    assert gi._has_shareable_signal({"sessions": ["not-a-map", {}]}) is False


# -- recipe_to_page --------------------------------------------------------
def test_recipe_to_page_none_without_canonical() -> None:
    assert gi.recipe_to_page({"model": "m"}) is None


def test_recipe_to_page_strict_gate_skips_seed_only(monkeypatch) -> None:
    monkeypatch.setenv("RECIPE_KB_MIRROR_REQUIRE_SIGNAL", "1")
    page = gi.recipe_to_page({"canonical_id": "a:b:c:d:e"})
    assert page is None


def test_recipe_to_page_emits_fingerprint_and_negatives(monkeypatch) -> None:
    monkeypatch.delenv("RECIPE_KB_MIRROR_REQUIRE_SIGNAL", raising=False)
    slug, content = gi.recipe_to_page({
        "canonical_id": "sglang:qwen:mi300x:bf16:v1",
        "model": "qwen", "hardware": "mi300x", "framework": "sglang",
        "best_config": {"extra_server_args": "--tp 1", "extra_envs": {"A": "1"}},
        "best_throughput": 1000.0,
        "what_worked": ["aiter"],
        "pitfalls": [{"knob": "x"}],
        "stack_fingerprint": {"rocm": "6.2", "aiter": "abc"},
    })
    assert slug.startswith("recipe-snapshot/")
    assert "stack_fingerprint" in content
    assert "what_worked" in content
    assert "kind:recipe" in content


# -- ingest_local_to_gbrain ------------------------------------------------
def test_ingest_dry_run_and_skip_and_error() -> None:
    recipes = [
        {"canonical_id": "a:b:c:d:e", "model": "m"},   # mirrorable
        {"model": "no-canonical"},                       # skipped
    ]
    # dry-run: counts ingested without touching mcp
    stats = gi.ingest_local_to_gbrain(recipes=recipes, mcp=None, dry_run=True)
    assert stats["ingested"] == 1
    assert stats["skipped_unmirrorable"] == 1

    # write path with a failing mcp -> error counted, loop continues
    mcp = _FakeMcp(fail=True)
    stats2 = gi.ingest_local_to_gbrain(
        recipes=[{"canonical_id": "a:b:c:d:e"}], mcp=mcp, dry_run=False,
    )
    assert stats2["errors"] == 1
    assert mcp.calls  # put_page attempted


def test_ingest_write_success() -> None:
    mcp = _FakeMcp()
    stats = gi.ingest_local_to_gbrain(
        recipes=[{"canonical_id": "a:b:c:d:e"}], mcp=mcp, dry_run=False,
    )
    assert stats["ingested"] == 1
    assert mcp.calls[0][0] == "put_page"


# -- mirror_recipe ---------------------------------------------------------
def test_mirror_recipe_no_mcp() -> None:
    assert gi.mirror_recipe({"canonical_id": "a:b:c:d:e"}, None) is False


def test_mirror_recipe_page_none() -> None:
    assert gi.mirror_recipe({"model": "no-canon"}, _FakeMcp()) is False


def test_mirror_recipe_success_and_error() -> None:
    ok = _FakeMcp()
    assert gi.mirror_recipe({"canonical_id": "a:b:c:d:e"}, ok) is True
    bad = _FakeMcp(fail=True)
    assert gi.mirror_recipe({"canonical_id": "a:b:c:d:e"}, bad) is False


# -- build_mirror_mcp_from_env --------------------------------------------
def test_build_mirror_mcp_from_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)
    assert gi.build_mirror_mcp_from_env() is None


def test_build_mirror_mcp_from_env_present(monkeypatch) -> None:
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://gbrain.example")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    mcp = gi.build_mirror_mcp_from_env()
    assert mcp is not None


# -- GbrainMirroringRecipeKB ----------------------------------------------
def test_mirroring_kb_put_and_delegate() -> None:
    class _Inner:
        def __init__(self) -> None:
            self.put_calls: list[dict] = []

        def put_recipe(self, **kwargs: Any) -> str:
            self.put_calls.append(kwargs)
            return "wrote"

        def get_recipe(self, x: str) -> str:
            return f"got:{x}"

    inner = _Inner()
    mcp = _FakeMcp()
    kb = gi.GbrainMirroringRecipeKB(inner, mcp)
    assert kb.put_recipe(canonical_id="a:b:c:d:e") == "wrote"
    assert inner.put_calls  # local write happened first
    assert mcp.calls  # mirror happened
    # __getattr__ delegates unknown attributes
    assert kb.get_recipe("z") == "got:z"


def test_mirroring_kb_swallows_mirror_error(monkeypatch) -> None:
    class _Inner:
        def put_recipe(self, **kwargs: Any) -> str:
            return "wrote"

    def _raise(*_a, **_k):
        raise RuntimeError("mirror boom")

    monkeypatch.setattr(gi, "mirror_recipe", _raise)
    kb = gi.GbrainMirroringRecipeKB(_Inner(), _FakeMcp())
    assert kb.put_recipe(canonical_id="a:b:c:d:e") == "wrote"  # error swallowed


# -- main ------------------------------------------------------------------
def test_main_requires_local_kb_root(monkeypatch, capsys) -> None:
    monkeypatch.delenv("HYPERLOOM_LOCAL_KB_ROOT", raising=False)
    assert gi.main([]) == 2
    assert "requires --local-kb-root" in capsys.readouterr().out


def test_main_write_requires_creds(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "inference_optimizer.recipe_kb.local_store.LocalRecipeStore.list_recent",
        lambda self, limit: [],
    )
    rc = gi.main(["--local-kb-root", str(tmp_path), "--write",
                  "--gbrain-url", "", "--token", ""])
    assert rc == 2
    assert "GBRAIN_BASE_URL" in capsys.readouterr().out


def test_main_dry_run_success(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "inference_optimizer.recipe_kb.local_store.LocalRecipeStore.list_recent",
        lambda self, limit: [{"canonical_id": "a:b:c:d:e"}],
    )
    rc = gi.main(["--local-kb-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "ingested" in out
