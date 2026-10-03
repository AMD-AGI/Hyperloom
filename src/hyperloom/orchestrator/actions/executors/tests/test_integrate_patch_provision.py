# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the attempt-runtime provision-stage wiring in integrate_patch."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from hyperloom.common.failure_signature import classify_failure
from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _build_variant_yaml,
    apply_runtime_override,
)
from hyperloom.orchestrator.enablement.runtime.stack_actions import (
    EnablementStackAction,
    FrameworkRuntime,
    ProvisionResult,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.enablement.lane import EnablementLane

_ARCH_LOG = "ValueError: Model architectures ['DeepseekV4ForCausalLM'] are not supported for now."
_OOM_LOG = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB"


def _params(log: str = _ARCH_LOG, **extra) -> dict:
    """Params for an enablement round dispatched on the verdict ``log`` classifies to."""
    return {
        "enablement": True,
        "enablement_failure_signature": classify_failure(log).to_dict(),
        **extra,
    }


def _attempt(task_id: str = "t-1", *, kept: dict | None = None):
    """An attempt whose shared state carries the round the executor derives from."""
    attempt = ip.IntegrateAttempt(task_id=task_id)
    attempt.shared_state = types.SimpleNamespace(
        framework="vllm",
        gpu_type="mi355x",
        enablement=EnablementRound(kept_stack_action=dict(kept) if kept else {}),
    )
    return attempt


def _candidate(framework: str = "vllm") -> dict:
    return EnablementStackAction(
        kind="runtime_candidate",
        framework=framework,
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        acquisition_method="wheel",
        index_url="https://rocm.repo/whl",
        packages=("vllm",),
    ).to_state()


class _FakeAdapter:
    """Adapter double whose provision/probe outcomes are programmable."""

    def __init__(self, result: ProvisionResult, probe_ok: bool = True):
        self._result = result
        self._probe_ok = probe_ok
        self.provision_calls = 0

    def provision(self, action, attempt_dir):
        self.provision_calls += 1
        # Simulate an on-disk venv so GC has something to remove.
        (attempt_dir / "venv" / "bin").mkdir(parents=True, exist_ok=True)
        return self._result

    def probe(self, result, action):
        return self._probe_ok

    def build_stack_action(self, gap, *, gpu_type=""):
        return EnablementStackAction.from_state(_candidate())


def _ok_result(venv_root: str) -> ProvisionResult:
    return ProvisionResult(
        ok=True,
        runtime=FrameworkRuntime(
            bin_path=f"{venv_root}/bin",
            python_path=f"{venv_root}/bin/python",
            venv_root=venv_root,
        ),
        installed_versions={"vllm": "0.21.0"},
    )


@pytest.fixture()
def _executor(tmp_path):
    return ip.IntegratePatchExecutor(session_dir=tmp_path / "session")


@pytest.fixture(autouse=True)
def _neutralize_disk_preflight(monkeypatch):
    """Stop the real disk_preflight from leaking the runner's free-space into these tests."""
    import hyperloom.agents.framework.isolation as iso
    from hyperloom.orchestrator.enablement.runtime import adapters

    def forbidden(*_args, **_kwargs):
        pytest.fail("runtime acquisition must be stubbed by the test")

    monkeypatch.setattr(iso, "disk_preflight", lambda *_a, **_k: None)
    real_get_adapter = adapters.get_adapter

    def metadata_adapter(framework, **kwargs):
        kwargs["run"] = forbidden
        adapter = real_get_adapter(framework, **kwargs)
        monkeypatch.setattr(adapter, "provision", forbidden)
        monkeypatch.setattr(adapter, "probe", forbidden)
        return adapter

    monkeypatch.setattr(adapters, "get_adapter", metadata_adapter)


# provision stage: which rounds acquire a runtime at all


def _forbid_the_adapter(monkeypatch) -> None:
    def _forbidden(_fw):
        pytest.fail("adapter must not be consulted for a round that acquires nothing")

    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", _forbidden)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="not_an_enablement_round"),
        pytest.param(_params(enablement_launch_only=True), id="launch_only_bench"),
    ],
)
async def test_rounds_that_acquire_nothing_are_a_noop(_executor, monkeypatch, params):
    """Not even a kept runtime is re-provisioned outside an acquiring enablement round."""
    _forbid_the_adapter(monkeypatch)
    attempt = _attempt(kept=_candidate())
    assert await _executor._stage_provision_attempt_runtime(attempt, params, "t-1") is None
    assert attempt.provision_result is None


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"enablement": True}, id="no_dispatched_verdict"),
        pytest.param(_params(_OOM_LOG), id="resource_constraint"),
    ],
)
async def test_without_a_kept_runtime_only_a_code_gap_builds_a_candidate(_executor, monkeypatch, params):
    """A round that names no gap code closes has no runtime to acquire from scratch."""
    _forbid_the_adapter(monkeypatch)
    attempt = _attempt()
    assert await _executor._stage_provision_attempt_runtime(attempt, params, "t-1") is None
    assert attempt.provision_result is None


async def test_dispatched_verdict_drives_the_adapter_candidate(_executor, monkeypatch):
    """The verdict the round was dispatched on is the gap the adapter builds against."""
    seen: dict = {}

    class _Recorder(_FakeAdapter):
        def build_stack_action(self, gap, *, gpu_type=""):
            seen["gap"] = gap
            seen["gpu_type"] = gpu_type
            return EnablementStackAction.from_state(_candidate())

    adapter = _Recorder(_ok_result(str(_executor.session_dir / "v")))
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _attempt()
    assert await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1") is None
    assert seen["gap"].kind == "missing_model_arch"
    assert seen["gap"].requires_code_acquisition is True
    assert seen["gpu_type"] == "mi355x"
    assert attempt.stack_action is not None


@pytest.mark.parametrize(
    "log",
    [
        pytest.param(_ARCH_LOG, id="code_gap"),
        # The kept runtime is what got the boot this far; an OOM past it is
        # still to be fixed on top of it, not on the base install.
        pytest.param(_OOM_LOG, id="resource_constraint"),
    ],
)
async def test_kept_action_is_reprovisioned_instead_of_a_fresh_candidate(_executor, monkeypatch, log):
    """Serial stacking reuses the runtime the last KEEP promoted; no new candidate is built."""

    class _NoBuild(_FakeAdapter):
        def build_stack_action(self, gap, *, gpu_type=""):
            pytest.fail("a round with a kept stack action must not build a fresh candidate")

    adapter = _NoBuild(_ok_result(str(_executor.session_dir / "v")))
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    kept = _candidate()
    attempt = _attempt(kept=kept)
    assert await _executor._stage_provision_attempt_runtime(attempt, _params(log), "t-1") is None
    assert adapter.provision_calls == 1
    assert attempt.provision_result is not None
    assert attempt.stack_action is not None
    assert attempt.stack_action.capability == EnablementStackAction.from_state(kept).capability


# provision ok / fail


@pytest.mark.asyncio
async def test_provision_runs_off_the_event_loop_thread(_executor, monkeypatch):
    """Adapter provision (venv/pip, 1800s) must not occupy the event-loop thread."""
    import threading

    seen: dict[str, int] = {}
    loop_ident = threading.get_ident()
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    adapter = _FakeAdapter(_ok_result(venv))
    orig = adapter.provision

    def _spy_provision(action, attempt_dir):
        seen["ident"] = threading.get_ident()
        return orig(action, attempt_dir)

    adapter.provision = _spy_provision
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _attempt(kept=_candidate())
    out = await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1")
    assert out is None
    assert "ident" in seen
    assert seen["ident"] != loop_ident


async def test_provision_fail_returns_reverted_and_gcs(_executor, monkeypatch):
    adapter = _FakeAdapter(ProvisionResult(ok=False, error="pip failed"))
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _attempt(kept=_candidate())
    out = await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert out["error_class"] == "provision_failed"
    assert out["enablement"] is True
    # GC removed the attempt dir.
    attempt = _executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1"
    assert not attempt.exists()


async def test_probe_fail_returns_reverted(_executor, monkeypatch):
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    adapter = _FakeAdapter(_ok_result(venv), probe_ok=False)
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _attempt(kept=_candidate())
    out = await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1")
    assert out is not None
    assert out["status"] == "reverted"
    assert "probe" in out["error"]


async def test_disk_preflight_failure_returns_reverted(_executor, monkeypatch):
    import hyperloom.agents.framework.isolation as iso

    def _boom(*_a, **_k):
        raise iso.DiskPreflightError("no space")

    monkeypatch.setattr(iso, "disk_preflight", _boom)
    called = {"n": 0}
    monkeypatch.setattr(
        "hyperloom.orchestrator.enablement.runtime.adapters.get_adapter",
        lambda _fw: called.__setitem__("n", called["n"] + 1),
    )
    attempt = _attempt(kept=_candidate())
    out = await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1")
    assert out is not None
    assert out["error_class"] == "disk_preflight_failed"
    assert called["n"] == 0  # never reached the adapter


# apply stage: an acquired runtime is itself the round's change


def _apply_ready(executor, task_id: str = "t-1", *, kept: dict | None = None):
    """An attempt the apply stage can run against, carrying no deliverable."""
    attempt = _attempt(task_id, kept=kept)
    attempt.specialist_task_id = "t-spec-1"
    attempt.specialist_workspace = executor.session_dir / "ws"
    attempt.specialist_workspace.mkdir(parents=True, exist_ok=True)
    attempt.shared_state.save = lambda *_a, **_k: None
    return attempt


async def test_a_runtime_only_round_reaches_the_bench(_executor, monkeypatch):
    """Rung 3's whole deliverable is the runtime; benching it is how the round is judged."""
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    adapter = _FakeAdapter(_ok_result(venv))
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _apply_ready(_executor)
    assert await _executor._stage_provision_attempt_runtime(attempt, _params(), "t-1") is None
    assert attempt.attempt_venv_root == venv

    assert await _executor._stage_apply(attempt, _params(), {}) is None
    # The sentinel carries the runtime the launch has to boot into.
    assert attempt.pending["attempt_venv_root"] == venv


async def test_patches_apply_to_the_tree_the_runtime_imports(_executor, monkeypatch, tmp_path):
    """A patch to the shared framework tree is invisible to a server booting the attempt runtime."""
    runtime_tree = tmp_path / "attempt-src"
    runtime_tree.mkdir()
    shared_tree = tmp_path / "shared-framework"
    shared_tree.mkdir()
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    result = ProvisionResult(
        ok=True,
        runtime=FrameworkRuntime(
            bin_path=f"{venv}/bin",
            python_path=f"{venv}/bin/python",
            venv_root=venv,
            source_root=str(runtime_tree),
        ),
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: _FakeAdapter(result)
    )
    attempt = _apply_ready(_executor)
    params = _params(framework_source_root=str(shared_tree))
    assert await _executor._stage_provision_attempt_runtime(attempt, params, "t-1") is None

    assert await _executor._stage_apply(attempt, params, {}) is None
    assert attempt.pending["framework_source_root"] == str(runtime_tree)


async def test_a_round_that_acquired_nothing_is_still_no_patches(_executor):
    """No runtime and no deliverable leaves nothing for the bench to measure."""
    attempt = _apply_ready(_executor)
    out = await _executor._stage_apply(attempt, _params(), {})
    assert out is not None
    assert out["status"] == "no_patches"


async def test_the_kept_runtime_alone_is_still_no_patches(_executor, monkeypatch):
    """The last KEEP already graded the kept stack; booting it again with nothing added re-observes its wall."""
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    adapter = _FakeAdapter(_ok_result(venv))
    monkeypatch.setattr("hyperloom.orchestrator.enablement.runtime.adapters.get_adapter", lambda _fw: adapter)
    attempt = _apply_ready(_executor, kept=_candidate())
    assert await _executor._stage_provision_attempt_runtime(attempt, _params(_OOM_LOG), "t-1") is None
    assert attempt.attempt_venv_root == venv

    out = await _executor._stage_apply(attempt, _params(_OOM_LOG), {})
    assert out is not None
    assert out["status"] == "no_patches"


# decision gate: runtime lands in materialized YAML, not os.environ


def test_provisioned_runtime_lands_in_yaml_not_process_env(tmp_path, monkeypatch):
    import os

    venv = str(tmp_path / "attempt" / "venv")
    runtime = FrameworkRuntime(bin_path=f"{venv}/bin", python_path=f"{venv}/bin/python", venv_root=venv)

    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m", "envs": {}}}), encoding="utf-8")

    variant = GridVariant(name="v")
    variant.runtime_override = runtime.to_runtime_override()

    env_before = dict(os.environ)
    out_yaml = _build_variant_yaml(
        base_yaml_path=base, base_extra_args="", variant=variant, output_subdir=tmp_path / "out"
    )
    assert dict(os.environ) == env_before  # os.environ untouched

    materialized = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    envs = materialized["benchmark"]["envs"]
    # the server binary resolves the attempt runtime, proven via YAML.
    assert f"{venv}/bin" in envs["PATH"]
    assert envs["HYPERLOOM_FRAMEWORK_BIN"] == f"{venv}/bin"
    assert envs["HYPERLOOM_FRAMEWORK_VENV_ROOT"] == venv


def test_opt_venv_path_never_replaced(tmp_path):
    """The attempt bin is PREPENDED; the existing /opt/venv PATH survives."""
    venv = str(tmp_path / "attempt" / "venv")
    envs = {"PATH": "/opt/venv/bin:/usr/bin"}
    apply_runtime_override(envs, FrameworkRuntime(bin_path=f"{venv}/bin", venv_root=venv).to_runtime_override())
    parts = envs["PATH"].split(":")
    assert parts[0] == f"{venv}/bin"  # attempt bin first
    assert "/opt/venv/bin" in parts  # shared venv still present, not replaced


# rearm: KEEP'd stack action survives one rearm cycle


@pytest.mark.asyncio
async def test_kept_stack_action_survives_rearm(monkeypatch):
    from hyperloom.orchestrator.state.shared_state import SharedState

    # Simulate a coordinator with just enough surface for maybe_rearm_enablement.
    state = SharedState()
    coord = types.SimpleNamespace(shared_state=state, session_dir=Path("/tmp/does-not-matter"))
    coord.save = lambda *a, **k: None

    async def _no_round(*_a, **_k):
        """No round is open, so the rearm's settle is a no-op."""

    async def _no_stalls(*_a, **_k):
        """An empty ledger, which the rearm reads to stamp the round's row."""
        return 0

    coord.settle_enablement_round = _no_round
    coord.rounds = types.SimpleNamespace(consecutive_stalled=_no_stalls)

    action_state = _candidate()
    runtime_state = FrameworkRuntime(bin_path="/a/bin", venv_root="/a").to_state()
    res = {
        "enablement": True,
        "status": "kept",
        "enablement_kept_stack_action": action_state,
        "enablement_active_runtime": runtime_state,
    }

    monkeypatch.setattr(state, "save", lambda *a, **k: None, raising=False)
    await EnablementLane.maybe_rearm_enablement(coord, res)

    assert state.enablement.kept_stack_action == action_state
    assert state.enablement.active_runtime == runtime_state
    assert runtime_state in state.enablement.attempt_runtimes
