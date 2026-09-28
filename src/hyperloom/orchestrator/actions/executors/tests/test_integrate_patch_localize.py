# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the _stage_localize_source wiring in integrate_patch."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.enablement.runtime.stack_actions import EnablementStackAction


def _attempt(task_id: str = "t-1"):
    return ip.IntegrateAttempt(task_id=task_id)


@pytest.fixture(autouse=True)
def _stub_external_operations(monkeypatch):
    from hyperloom.agents.framework.sources import github
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    def forbidden(*_args, **_kwargs):
        pytest.fail("localization fetches must be stubbed by the test")

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    monkeypatch.setattr(github, "pr_patches", forbidden)
    monkeypatch.setattr(github, "fetch_raw_file", forbidden)


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
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, {}, "t-1")
    assert out is None
    assert attempt.localization_patches == []


async def test_multi_node_skips(_executor, monkeypatch):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is None
    assert attempt.localization_patches == []


# ---------------------------------------------------------------------------
# python-only -> patch written + staged
# ---------------------------------------------------------------------------


async def test_python_only_writes_patch(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _PY_DIFF)
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is None, out
    assert len(attempt.localization_patches) == 1
    patch = attempt.localization_patches[0]
    assert patch.exists()
    assert "deepseek_v4.py" in patch.read_text()
    assert attempt.localization_touched == ["vllm/model/deepseek_v4.py"]


# ---------------------------------------------------------------------------
# compiled-closure deferral: reverted, no patch
# ---------------------------------------------------------------------------


async def test_compiled_closure_defers_rung5(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _CUDA_DIFF)
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_rung5_deferred"
    assert attempt.localization_patches == []


async def test_fetch_failure_reverts(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: "")
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, {"localization_candidate": _pr_candidate()}, "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_fetch_failed"
