# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the _stage_localize_source wiring in integrate_patch."""

from __future__ import annotations

import subprocess
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
        pytest.param(
            {
                "enablement": True,
                "enablement_failure_signature": classify_failure(
                    "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB"
                ).to_dict(),
            },
            id="resource_constraint",
        ),
        pytest.param(_params(enablement_launch_only=True), id="launch_only_bench"),
    ],
)
async def test_rounds_that_acquire_nothing_are_a_noop(_executor, params):
    """Only an enablement round dispatched on a code-acquirable verdict localizes."""
    attempt = _attempt()
    await _executor._stage_localize_source(attempt, params)
    assert attempt.localization_patches == []


async def test_without_a_bridging_candidate_nothing_is_fetched(_executor):
    """A round whose discovery found no upstream ref has nothing to backport."""
    attempt = _attempt(candidate_refs=())
    await _executor._stage_localize_source(attempt, _params())
    assert attempt.localization_patches == []


async def test_a_bridge_repo_candidate_is_not_localized(_executor):
    """A diff cut from aiter does not apply to the framework tree."""
    attempt = _attempt(candidate_refs=("https://github.com/ROCm/aiter/pull/99",))
    await _executor._stage_localize_source(attempt, _params())
    assert attempt.localization_patches == []


async def test_the_framework_ref_is_preferred_over_a_higher_ranked_bridge_ref(_executor, monkeypatch):
    """Discovery ranks both repos into one list; only the framework's own is applicable."""
    import hyperloom.agents.framework.sources.github as gh

    fetched: list[tuple[str, int]] = []
    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: (fetched.append((slug, num)), _PY_DIFF)[1])
    attempt = _attempt(
        candidate_refs=("https://github.com/ROCm/aiter/pull/99", "https://github.com/ROCm/vllm/pull/1234")
    )
    await _executor._stage_localize_source(attempt, _params())
    assert fetched == [("ROCm/vllm", 1234)]


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
    await _executor._stage_localize_source(attempt, _params())
    # The ref discovery recorded on the round is the PR that gets backported.
    assert fetched == [("ROCm/vllm", 1234)]
    assert len(attempt.localization_patches) == 1
    patch = attempt.localization_patches[0]
    assert patch.exists()
    assert "deepseek_v4.py" in patch.read_text()
    assert attempt.localization_touched == ["vllm/model/deepseek_v4.py"]


async def test_localizing_does_not_displace_the_runtime_action(_executor, monkeypatch):
    """The KEEP re-provisions from the runtime action; a backport is not something to provision."""
    import hyperloom.agents.framework.sources.github as gh
    from hyperloom.orchestrator.enablement.runtime.stack_actions import EnablementStackAction

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _PY_DIFF)
    runtime_action = EnablementStackAction(
        kind="runtime_candidate",
        framework="vllm",
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        acquisition_method="wheel",
        index_url="https://rocm.repo/whl",
    )
    attempt = _attempt()
    attempt.stack_action = runtime_action

    await _executor._stage_localize_source(attempt, _params())
    assert attempt.stack_action is runtime_action
    assert attempt.localization_action is not None
    assert attempt.localization_action.kind == "pr_backport"


# ---------------------------------------------------------------------------
# a change that cannot be localized is skipped, never reverted
# ---------------------------------------------------------------------------

_OUT_OF_PACKAGE_DIFF = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-a\n+b\n"


@pytest.mark.parametrize(
    "diff",
    [
        pytest.param(_CUDA_DIFF, id="compiled_closure"),
        pytest.param("", id="fetch_failed"),
    ],
)
async def test_an_unlocalizable_change_is_skipped(_executor, monkeypatch, diff):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: diff)
    attempt = _attempt()
    await _executor._stage_localize_source(attempt, _params())
    assert attempt.localization_patches == []
    assert attempt.localization_action is None


async def test_only_the_files_under_the_package_are_localized(_executor, monkeypatch):
    import hyperloom.agents.framework.sources.github as gh

    monkeypatch.setattr(gh, "pr_patches", lambda slug, num: _PY_DIFF + _OUT_OF_PACKAGE_DIFF)
    attempt = _attempt()
    await _executor._stage_localize_source(attempt, _params())
    assert attempt.localization_touched == ["vllm/model/deepseek_v4.py"]
    assert "tests/test_x.py" not in attempt.localization_patches[0].read_text()


# ---------------------------------------------------------------------------
# admission against the tree the round runs on
# ---------------------------------------------------------------------------


def _repo_with_model(tmp_path):
    root = tmp_path / "tree"
    (root / "vllm" / "model").mkdir(parents=True)
    (root / "vllm" / "model" / "deepseek_v4.py").write_text("old\n")
    for cmd in (
        ["init", "-q"],
        ["config", "user.name", "t"],
        ["config", "user.email", "t@t"],
        ["add", "-A"],
        ["commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(root), *cmd], check=True)
    return root


def _localized(attempt, tmp_path, diff):
    patch = tmp_path / "loc.patch"
    patch.write_text(diff)
    attempt.localization_patches = [patch]
    attempt.localization_touched = ["vllm/model/deepseek_v4.py"]
    attempt.localization_action = object()
    attempt.pending = {"patches": ["round.patch"]}
    return patch


def test_a_localization_that_applies_goes_ahead_of_the_rounds_patches(tmp_path):
    root = _repo_with_model(tmp_path)
    attempt = _attempt()
    loc = _localized(attempt, tmp_path, _PY_DIFF)
    own = tmp_path / "round.patch"

    out = ip.IntegratePatchExecutor._admit_localization(attempt, root, [own])

    assert out == [loc, own]
    assert attempt.pending["patches"] == [str(loc), str(own)]
    assert attempt.localization_action is not None


def test_a_localization_that_does_not_apply_is_dropped_and_the_rounds_patches_stand(tmp_path):
    root = _repo_with_model(tmp_path)
    (root / "vllm" / "model" / "deepseek_v4.py").write_text("drifted\n")
    subprocess.run(["git", "-C", str(root), "commit", "-qam", "drift"], check=True)
    attempt = _attempt()
    _localized(attempt, tmp_path, _PY_DIFF)
    own = tmp_path / "round.patch"

    out = ip.IntegratePatchExecutor._admit_localization(attempt, root, [own])

    assert out == [own]
    assert attempt.pending["patches"] == ["round.patch"]
    assert attempt.localization_action is None
    assert attempt.localization_touched == []
