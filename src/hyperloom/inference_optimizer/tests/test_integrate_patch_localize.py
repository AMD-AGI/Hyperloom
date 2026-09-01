# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the _stage_localize_source wiring in integrate_patch."""

from __future__ import annotations

import types

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.framework.stack_actions import EnablementStackAction


def _ctx(task_id: str = "t-1"):
    task = types.SimpleNamespace(task_id=task_id, params={})
    return types.SimpleNamespace(task=task, extra={})


@pytest.fixture()
def _executor(tmp_path):
    return ip.IntegratePatchExecutor(session_dir=tmp_path / "session")


def _pr_candidate(framework: str = "vllm") -> dict:
    return EnablementStackAction(
        kind="pr_backport",
        framework=framework,
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        repo_url="https://github.com/ROCm/vllm.git",
        pr_number=1234,
    ).to_state()


_PY_DIFF = (
    "diff --git a/vllm/model/deepseek_v4.py b/vllm/model/deepseek_v4.py\n"
    "--- a/vllm/model/deepseek_v4.py\n"
    "+++ b/vllm/model/deepseek_v4.py\n"
    "@@ -1 +1 @@\n-old\n+new\n"
)
_CUDA_DIFF = "diff --git a/csrc/attn.cu b/csrc/attn.cu\n--- a/csrc/attn.cu\n+++ b/csrc/attn.cu\n@@ -1 +1 @@\n-a\n+b\n"


# ---------------------------------------------------------------------------
# no-op / skip paths
# ---------------------------------------------------------------------------


async def test_no_candidate_is_noop(_executor):
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {}, "t-1")
    assert out is None
    assert ctx._ip_localization_patches == []


async def test_multi_node_skips(_executor, monkeypatch):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is None
    assert ctx._ip_localization_patches == []


# ---------------------------------------------------------------------------
# python-only -> patch written + staged
# ---------------------------------------------------------------------------


async def test_python_only_writes_patch(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _PY_DIFF)
    # Allow the touched path under a broad allowlist so the gate passes.
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: ("/",), raising=False)
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is None, out
    assert len(ctx._ip_localization_patches) == 1
    patch = ctx._ip_localization_patches[0]
    assert patch.exists()
    assert "deepseek_v4.py" in patch.read_text()
    assert ctx._ip_localization_touched == ["vllm/model/deepseek_v4.py"]


# ---------------------------------------------------------------------------
# compiled-closure deferral: reverted, no patch
# ---------------------------------------------------------------------------


async def test_compiled_closure_defers_rung5(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _CUDA_DIFF)
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_rung5_deferred"
    assert ctx._ip_localization_patches == []


async def test_fetch_failure_reverts(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: "")
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_fetch_failed"


# ---------------------------------------------------------------------------
# allowlist gate: path outside allowlist -> reverted (no global env mutation)
# ---------------------------------------------------------------------------


async def test_path_outside_allowlist_reverts(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    outside_diff = "diff --git a/etc/passwd b/etc/passwd\n--- a/etc/passwd\n+++ b/etc/passwd\n@@ -1 +1 @@\n-a\n+b\n"
    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: outside_diff)
    # A narrow allowlist that does NOT contain /etc, and no framework root.
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: ("/sgl-workspace/vllm/",), raising=False)
    monkeypatch.setattr(ip, "_resolve_framework_root", lambda *a, **k: None, raising=False)
    ctx = _ctx()
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is not None
    assert out["error_class"] == "localization_outside_allowlist"


async def test_attempt_root_added_to_allowlist_only(_executor, monkeypatch, tmp_path):
    """A path under the attempt root is allowed even when outside the global allowlist."""
    import hyperloom.agents.framework.sources.github as gh

    attempt_venv = tmp_path / "attempt" / "venv"
    attempt_venv.mkdir(parents=True, exist_ok=True)
    localized = "attempt/localized.py"  # relative to framework_root = attempt dir parent
    diff = f"diff --git a/{localized} b/{localized}\n--- a/{localized}\n+++ b/{localized}\n@@ -1 +1 @@\n-a\n+b\n"
    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: diff)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: ("/sgl-workspace/vllm/",), raising=False)
    # framework_root under tmp_path so the localized path resolves within it.
    monkeypatch.setattr(ip, "_resolve_framework_root", lambda *a, **k: tmp_path, raising=False)
    ctx = _ctx()
    ctx._ip_attempt_venv_root = str(attempt_venv)
    out = await _executor._stage_localize_source(ctx, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is None, out
    assert len(ctx._ip_localization_patches) == 1


def test_empty_allowlist_fail_closed():
    """No trusted write root: every non-empty path is out of bounds."""
    outside = ip._localization_paths_outside_allowlist(
        ["a/b.py", "/etc/passwd", "", "  "],
        None,
        [],
    )
    assert outside == ["a/b.py", "/etc/passwd"]


def test_empty_allowlist_still_uses_framework_root(tmp_path):
    """framework_root alone is a trusted root even when allow_roots is empty."""
    outside = ip._localization_paths_outside_allowlist(
        ["ok.py", "/etc/passwd"],
        tmp_path,
        [],
    )
    assert "ok.py" not in outside
    assert "/etc/passwd" in outside
