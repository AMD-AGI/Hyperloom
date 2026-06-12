# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``cli_kb``: local KB root resolution, the RecipeKB dispatcher
across remote-mode branches (degraded / gbrain / cortex), the cortex T0
bootstrap (success + mid-flight failure), and the KnowledgePlane facade."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from inference_optimizer import cli_kb


def _args(**over):
    base = dict(
        local_kb_root=None, degraded_kb=False, cortex_kb_url=None,
        pr_monitor_enabled=True, pr_monitor_url=None, pr_monitor_mcp_url=None,
        pr_feed_window_days=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


# -- _resolve_local_kb_root ------------------------------------------------
def test_resolve_local_kb_root_explicit(tmp_path) -> None:
    out = cli_kb._resolve_local_kb_root(_args(local_kb_root=str(tmp_path / "kb")))
    assert out == tmp_path / "kb"


def test_resolve_local_kb_root_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "envkb"))
    out = cli_kb._resolve_local_kb_root(_args())
    assert out == tmp_path / "envkb"


def test_resolve_local_kb_root_default(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_LOCAL_KB_ROOT", raising=False)
    out = cli_kb._resolve_local_kb_root(_args())
    assert out.name == "kb"


# -- _build_recipe_kb_dispatcher -------------------------------------------
def test_dispatcher_degraded_kb(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    kb = cli_kb._build_recipe_kb_dispatcher(_args(degraded_kb=True))
    assert kb.remote is None


def test_dispatcher_local_only_no_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.delenv("RECIPE_KB_REMOTE", raising=False)
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert kb.remote is None


def test_dispatcher_cortex_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.delenv("RECIPE_KB_REMOTE", raising=False)
    kb = cli_kb._build_recipe_kb_dispatcher(_args(cortex_kb_url="http://cortex"))
    assert kb.remote is not None


def test_dispatcher_gbrain_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_REMOTE", "gbrain")
    monkeypatch.setenv("RECIPE_KB_MIRROR_MODE", "external")

    class _Remote:
        enabled = True

    from inference_optimizer.recipe_kb import gbrain_remote_client as grc
    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: _Remote())
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert isinstance(kb.remote, _Remote)


def test_dispatcher_gbrain_inline_mirror(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_REMOTE", "gbrain")
    monkeypatch.setenv("RECIPE_KB_MIRROR_MODE", "inline")

    class _Remote:
        enabled = True

    from inference_optimizer.recipe_kb import gbrain_remote_client as grc
    from inference_optimizer.recipe_kb import gbrain_ingest as gi
    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: _Remote())
    monkeypatch.setattr(gi, "build_mirror_mcp_from_env", lambda: object())
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    # inline mirror wraps the dispatcher
    assert isinstance(kb, gi.GbrainMirroringRecipeKB)


def test_dispatcher_gbrain_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_REMOTE", "gbrain")
    from inference_optimizer.recipe_kb import gbrain_remote_client as grc
    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: None)
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert kb.remote is None


def test_dispatcher_both_no_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_REMOTE", "both")
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    from inference_optimizer.recipe_kb import gbrain_remote_client as grc
    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: None)
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert kb.remote is None


# -- _bootstrap_cortex_kb --------------------------------------------------
def test_bootstrap_cortex_kb_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.delenv("RECIPE_KB_REMOTE", raising=False)
    calls = []
    monkeypatch.setattr(cli_kb, "run_t0_anchor", lambda *a, **k: calls.append(k))
    kb = cli_kb._bootstrap_cortex_kb(
        _args(), session_dir=tmp_path, manifest={"model_name": "m"}, resume=False,
    )
    assert kb is not None
    assert calls  # t0 anchor invoked


def test_bootstrap_cortex_kb_t0_failure_continues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.delenv("RECIPE_KB_REMOTE", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("t0 down")

    monkeypatch.setattr(cli_kb, "run_t0_anchor", _boom)
    args = _args()
    kb = cli_kb._bootstrap_cortex_kb(
        args, session_dir=tmp_path,
        manifest={"model_path": "/models/Qwen", "stack_fingerprint": {"rocm": "6.2"},
                  "image": "img@sha"},
        resume=False,
    )
    assert kb is not None
    assert args.kb_degraded_reason == "t0_runtime_fail"


# -- _bootstrap_knowledge_plane --------------------------------------------
def test_bootstrap_knowledge_plane_enabled(tmp_path) -> None:
    plane = cli_kb._bootstrap_knowledge_plane(
        _args(pr_monitor_enabled=True), session_dir=tmp_path,
    )
    assert plane is not None


def test_bootstrap_knowledge_plane_disabled(tmp_path) -> None:
    plane = cli_kb._bootstrap_knowledge_plane(
        _args(pr_monitor_enabled=False, pr_degraded_reason="explicit_flag"),
        session_dir=tmp_path,
    )
    assert plane is not None
