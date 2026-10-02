# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the _stage_localize_source wiring in integrate_patch."""

from __future__ import annotations

import types

import pytest

from hyperloom.common.failure_signature import classify_failure
from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

_ARCH_LOG = "ValueError: Model architectures ['DeepseekV4ForCausalLM'] are not supported for now."


def _params(**extra) -> dict:
    """Params for an enablement round dispatched on an architecture-miss verdict."""
    return {
        "enablement": True,
        "enablement_failure_signature": classify_failure(_ARCH_LOG).to_dict(),
        **extra,
    }


def _attempt(task_id: str = "t-1", *, candidate_refs: tuple[str, ...] = ("PR:1234",)):
    """An attempt whose shared state carries the round the executor derives from."""
    attempt = ip.IntegrateAttempt(task_id=task_id)
    attempt.shared_state = types.SimpleNamespace(
        framework="vllm",
        gpu_type="mi355x",
        enablement=EnablementRound(candidate_refs=list(candidate_refs)),
    )
    return attempt


@pytest.fixture(autouse=True)
def _stub_external_operations(monkeypatch):
    from hyperloom.agents.framework.sources import github

    def forbidden(*_args, **_kwargs):
        pytest.fail("localization fetches must be stubbed by the test")

    monkeypatch.setattr(github, "pr_patches", forbidden)
    monkeypatch.setattr(github, "fetch_raw_file", forbidden)


@pytest.fixture()
def _executor(tmp_path):
    return ip.IntegratePatchExecutor(session_dir=tmp_path / "session")


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


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="not_an_enablement_round"),
        pytest.param({"enablement": True}, id="no_dispatched_verdict"),
        pytest.param(_params(enablement_launch_only=True), id="launch_only_bench"),
    ],
)
async def test_rounds_that_acquire_nothing_are_a_noop(_executor, params):
    """Only an enablement round dispatched on a code-acquirable verdict localizes."""
    attempt = _attempt()
    assert await _executor._stage_localize_source(attempt, params, "t-1") is None
    assert attempt.localization_patches == []


async def test_without_a_bridging_candidate_nothing_is_fetched(_executor):
    """A round whose discovery found no upstream ref has nothing to backport."""
    attempt = _attempt(candidate_refs=())
    assert await _executor._stage_localize_source(attempt, _params(), "t-1") is None
    assert attempt.localization_patches == []


# ---------------------------------------------------------------------------
# python-only -> patch written + staged
# ---------------------------------------------------------------------------


async def test_python_only_writes_patch(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    fetched: list[tuple[str, int]] = []

    def _pr_patches(slug, num):
        fetched.append((slug, num))
        return _PY_DIFF

    monkeypatch.setattr(gh, "pr_patches", _pr_patches)
    attempt = _attempt(candidate_refs=("PR:1234",))
    out = await _executor._stage_localize_source(attempt, _params(), "t-1")
    assert out is None, out
    # The ref discovery recorded on the round is the PR that gets backported.
    assert fetched == [("ROCm/vllm", 1234)]
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
    out = await _executor._stage_localize_source(attempt, _params(), "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_rung5_deferred"
    assert attempt.localization_patches == []


async def test_fetch_failure_reverts(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: "")
    attempt = _attempt()
    out = await _executor._stage_localize_source(attempt, _params(), "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "localization_fetch_failed"
