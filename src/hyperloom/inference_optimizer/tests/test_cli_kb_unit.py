# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``cli_kb``: local KB root resolution, the RecipeKB dispatcher
across remote-mode branches (degraded / gbrain / local-only), the T0
bootstrap (success + mid-flight failure), and the KnowledgePlane facade."""

from __future__ import annotations

import argparse


from hyperloom.inference_optimizer.cli import kb as cli_kb


def _args(**over):
    base = dict(
        local_kb_root=None,
        degraded_kb=False,
        cortex_kb_url=None,
        pr_monitor_enabled=True,
        pr_monitor_url=None,
        pr_monitor_mcp_url=None,
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


def test_dispatcher_local_only_no_gbrain(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: None)
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert kb.remote is None


def test_dispatcher_cortex_url_is_not_a_recipe_remote(tmp_path, monkeypatch) -> None:
    # CORTEX_KB_URL / --cortex-kb-url now only feed the critic agent; they must
    # NOT wire a recipe-KB remote. Without gbrain configured, the dispatcher
    # stays local-only even when a cortex URL is supplied.
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: None)
    kb = cli_kb._build_recipe_kb_dispatcher(_args(cortex_kb_url="http://cortex"))
    assert kb.remote is None


def test_dispatcher_gbrain_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_MIRROR_MODE", "external")

    class _Remote:
        enabled = True

    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: _Remote())
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert isinstance(kb.remote, _Remote)


def test_dispatcher_gbrain_inline_mirror(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_MIRROR_MODE", "inline")

    class _Remote:
        enabled = True

    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc
    from hyperloom.inference_optimizer.recipe_kb import gbrain_ingest as gi

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: _Remote())
    monkeypatch.setattr(gi, "build_mirror_mcp_from_env", lambda: object())
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    # inline mirror wraps the dispatcher
    assert isinstance(kb, gi.GbrainMirroringRecipeKB)


def test_dispatcher_gbrain_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: None)
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert kb.remote is None


# -- _attach_recipe_audit_hook ---------------------------------------------
def test_attach_recipe_audit_hook_appends_jsonl(tmp_path) -> None:
    import json

    from hyperloom.inference_optimizer.recipe_kb import LocalRecipeStore, RecipeKB
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    cli_kb._attach_recipe_audit_hook(kb, tmp_path)
    assert callable(kb.audit_hook)
    kb.audit_hook({"method": "get_recipe", "resolution": "local", "hit": False})
    path = recipe_snapshot_audit_jsonl(tmp_path)
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["method"] == "get_recipe"
    assert "ts" in row


def test_attach_recipe_audit_hook_unwraps_mirror(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("RECIPE_KB_MIRROR_MODE", "inline")

    class _Remote:
        enabled = True

    from hyperloom.inference_optimizer.recipe_kb import gbrain_ingest as gi
    from hyperloom.inference_optimizer.recipe_kb import gbrain_remote_client as grc

    monkeypatch.setattr(grc, "build_gbrain_remote_from_env", lambda: _Remote())
    monkeypatch.setattr(gi, "build_mirror_mcp_from_env", lambda: object())
    kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert isinstance(kb, gi.GbrainMirroringRecipeKB)
    cli_kb._attach_recipe_audit_hook(kb, tmp_path)
    # The hook lands on the inner RecipeKB whose reads emit the audit.
    assert callable(kb._inner.audit_hook)


def test_attach_recipe_audit_hook_noop_without_session_dir(tmp_path) -> None:
    from hyperloom.inference_optimizer.recipe_kb import LocalRecipeStore, RecipeKB

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    cli_kb._attach_recipe_audit_hook(kb, None)
    assert kb.audit_hook is None


# -- _bootstrap_cortex_kb --------------------------------------------------
def test_bootstrap_cortex_kb_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.delenv("RECIPE_KB_REMOTE", raising=False)
    calls = []
    monkeypatch.setattr(cli_kb, "run_t0_anchor", lambda *a, **k: calls.append(k))
    kb = cli_kb._bootstrap_cortex_kb(
        _args(),
        session_dir=tmp_path,
        manifest={"model_name": "m"},
        resume=False,
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
        args,
        session_dir=tmp_path,
        manifest={"model_path": "/models/Qwen", "stack_fingerprint": {"rocm": "6.2"}, "image": "img@sha"},
        resume=False,
    )
    assert kb is not None
    assert args.kb_degraded_reason == "t0_runtime_fail"


# -- _bootstrap_knowledge_plane --------------------------------------------
def test_bootstrap_knowledge_plane_enabled(tmp_path) -> None:
    plane = cli_kb._bootstrap_knowledge_plane(
        _args(pr_monitor_enabled=True),
        session_dir=tmp_path,
    )
    assert plane is not None


def test_bootstrap_knowledge_plane_disabled(tmp_path) -> None:
    plane = cli_kb._bootstrap_knowledge_plane(
        _args(pr_monitor_enabled=False, pr_degraded_reason="explicit_flag"),
        session_dir=tmp_path,
    )
    assert plane is not None


def test_bootstrap_knowledge_plane_with_kb_mcp(tmp_path, monkeypatch) -> None:
    """When a specialist KB MCP url resolves, the enabled branch (L302-303) runs."""
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_KB_MCP_URL", "http://kb.invalid/mcp")
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_KB_MCP_TOKEN", "tok")
    plane = cli_kb._bootstrap_knowledge_plane(_args(), session_dir=tmp_path)
    assert plane is not None


class _FailingParent:
    """A stand-in ``.parent`` whose mkdir always raises OSError."""

    def mkdir(self, *_a, **_k):
        raise OSError("no space")


class _MarkerPath:
    """Path-like whose parent.mkdir raises, exercising the OSError guard."""

    parent = _FailingParent()

    def write_text(self, *_a, **_k):  # pragma: no cover - never reached
        raise OSError("no space")


def test_bootstrap_knowledge_plane_marker_write_failure(tmp_path, monkeypatch) -> None:
    """An OSError writing the pr_monitor status marker is swallowed (L290-291)."""
    from hyperloom.inference_optimizer.session import session_paths as sp

    monkeypatch.setattr(sp, "pr_monitor_status_json", lambda _sd: _MarkerPath())
    plane = cli_kb._bootstrap_knowledge_plane(_args(pr_monitor_enabled=True), session_dir=tmp_path)
    assert plane is not None


# -- _attach_recipe_audit_hook extra branches ------------------------------
def test_attach_recipe_audit_hook_target_without_hook_attr(tmp_path) -> None:
    """A target lacking ``audit_hook`` is a no-op (L67-68)."""

    class _NoHook:
        pass

    obj = _NoHook()
    cli_kb._attach_recipe_audit_hook(obj, tmp_path)
    assert not hasattr(obj, "audit_hook")


def test_attach_recipe_audit_hook_write_error_is_swallowed(tmp_path, monkeypatch) -> None:
    """A write failure inside the hook is swallowed, not raised (L90-91)."""
    from hyperloom.inference_optimizer.recipe_kb import LocalRecipeStore, RecipeKB
    from hyperloom.inference_optimizer.session import session_paths as sp

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)

    class _BadPath:
        parent = None

        def mkdir(self, *_a, **_k):
            raise OSError("boom")

    bad = _BadPath()
    bad.parent = bad  # type: ignore[assignment]
    monkeypatch.setattr(sp, "recipe_snapshot_audit_jsonl", lambda _sd: bad)
    cli_kb._attach_recipe_audit_hook(kb, tmp_path)
    # Must not raise even though the underlying write path errors.
    kb.audit_hook({"method": "search"})


# -- _resolve_specialist_kb_mcp --------------------------------------------
def test_resolve_specialist_kb_mcp_override(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_KB_MCP_URL", "http://x/mcp")
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_KB_MCP_TOKEN", "secret")
    url, headers = cli_kb._resolve_specialist_kb_mcp(_args())
    assert url == "http://x/mcp"
    assert headers == {"Authorization": "Bearer secret"}


def test_resolve_specialist_kb_mcp_override_without_token(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_KB_MCP_URL", "http://x/mcp")
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_KB_MCP_TOKEN", raising=False)
    url, headers = cli_kb._resolve_specialist_kb_mcp(_args())
    assert url == "http://x/mcp"
    assert headers == {}


def test_resolve_specialist_kb_mcp_gbrain(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_KB_MCP_URL", raising=False)
    monkeypatch.setenv("GBRAIN_BASE_URL", "http://gbrain.invalid/")
    monkeypatch.setenv("GBRAIN_TOKEN", "gtok")
    url, headers = cli_kb._resolve_specialist_kb_mcp(_args())
    assert url == "http://gbrain.invalid/mcp"
    assert headers == {"Authorization": "Bearer gtok"}


def test_resolve_specialist_kb_mcp_nothing_configured(monkeypatch) -> None:
    for k in (
        "HYPERLOOM_SPECIALIST_KB_MCP_URL",
        "HYPERLOOM_SPECIALIST_KB_MCP_TOKEN",
        "GBRAIN_BASE_URL",
        "GBRAIN_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    url, headers = cli_kb._resolve_specialist_kb_mcp(_args())
    assert url == ""
    assert headers == {}
