# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the attempt-runtime provision-stage wiring in integrate_patch.

Exercises _stage_provision_attempt_runtime and the YAML-layer runtime activation
in isolation, plus the KEEP/rearm stack-action survival and GC. All subprocess /
adapter calls are mocked so no ROCm / network / real venv is needed.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _build_variant_yaml,
    apply_runtime_override,
)
from hyperloom.orchestrator.framework.stack_actions import (
    EnablementStackAction,
    FrameworkRuntime,
    ProvisionResult,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def _ctx(task_id: str = "t-1"):
    task = types.SimpleNamespace(task_id=task_id, params={})
    return types.SimpleNamespace(task=task, extra={})


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
    """Stop the real disk_preflight from leaking the runner's free-space into
    these tests. The provision stage runs disk_preflight before consulting the
    adapter; on a space-constrained CI runner (< 20 GB free on /tmp) it would
    raise DiskPreflightError and short-circuit provision-logic tests that never
    intend to exercise it. Tests that DO exercise it re-patch disk_preflight
    themselves (that patch wins over this autouse no-op)."""
    import hyperloom.agents.framework.isolation as iso

    monkeypatch.setattr(iso, "disk_preflight", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# provision stage: no candidate / skip paths
# ---------------------------------------------------------------------------


async def test_no_candidate_is_noop(_executor):
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {}, "t-1")
    assert out is None
    assert ctx._ip_provision_result is None


async def test_multi_node_skips_provision(_executor, monkeypatch):
    monkeypatch.setattr(ip, "_read_done_payload", lambda *_a, **_k: {}, raising=False)
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    called = {"n": 0}

    def _get_adapter(_fw):
        called["n"] += 1
        return _FakeAdapter(_ok_result("/x"))

    monkeypatch.setattr("hyperloom.orchestrator.framework.adapters.get_adapter", _get_adapter)
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {"runtime_candidate": _candidate()}, "t-1")
    assert out is None
    assert called["n"] == 0  # adapter never consulted in multi-node


# ---------------------------------------------------------------------------
# provision ok / fail
# ---------------------------------------------------------------------------


async def test_provision_ok_sets_ctx(_executor, monkeypatch):
    venv = str(_executor.session_dir / "enablement" / "stacks" / "vllm" / "t-1" / "venv")
    adapter = _FakeAdapter(_ok_result(venv))
    monkeypatch.setattr("hyperloom.orchestrator.framework.adapters.get_adapter", lambda _fw: adapter)
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {"runtime_candidate": _candidate()}, "t-1")
    assert out is None
    assert ctx._ip_provision_result is not None
    assert ctx._ip_provision_result.ok is True
    assert ctx._ip_stack_action.attempt_venv_root == venv
    assert ctx._ip_attempt_venv_root == venv


async def test_provision_fail_returns_reverted_and_gcs(_executor, monkeypatch):
    adapter = _FakeAdapter(ProvisionResult(ok=False, error="pip failed"))
    monkeypatch.setattr("hyperloom.orchestrator.framework.adapters.get_adapter", lambda _fw: adapter)
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {"runtime_candidate": _candidate()}, "t-1")
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
    monkeypatch.setattr("hyperloom.orchestrator.framework.adapters.get_adapter", lambda _fw: adapter)
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {"runtime_candidate": _candidate()}, "t-1")
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
        "hyperloom.orchestrator.framework.adapters.get_adapter",
        lambda _fw: called.__setitem__("n", called["n"] + 1),
    )
    ctx = _ctx()
    out = await _executor._stage_provision_attempt_runtime(ctx, {"runtime_candidate": _candidate()}, "t-1")
    assert out is not None
    assert out["error_class"] == "disk_preflight_failed"
    assert called["n"] == 0  # never reached the adapter


# ---------------------------------------------------------------------------
# decision gate: runtime lands in materialized YAML, not os.environ
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# rearm: KEEP'd stack action survives one rearm cycle
# ---------------------------------------------------------------------------


def test_kept_stack_action_survives_rearm(monkeypatch):
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.enablement.params import _maybe_build_runtime_candidate  # noqa: F401

    # Simulate a coordinator with just enough surface for _maybe_rearm_enablement.
    state = SharedState()
    coord = types.SimpleNamespace(shared_state=state, session_dir=Path("/tmp/does-not-matter"))
    coord.save = lambda *a, **k: None

    action_state = _candidate()
    runtime_state = FrameworkRuntime(bin_path="/a/bin", venv_root="/a").to_state()
    res = {
        "enablement": True,
        "status": "kept",
        "enablement_kept_stack_action": action_state,
        "enablement_active_runtime": runtime_state,
    }

    # Bind the real method to our fake coordinator (SimpleNamespace has no save-to-disk).
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    monkeypatch.setattr(state, "save", lambda *a, **k: None, raising=False)
    Coordinator._maybe_rearm_enablement(coord, res)

    assert state.enablement.kept_stack_action == action_state
    assert state.enablement.active_runtime == runtime_state
    assert runtime_state in state.enablement.attempt_runtimes


def test_rearm_reactivation_threads_kept_action_into_next_params(monkeypatch):
    """A prior KEEP'd stack action is re-attached as runtime_candidate next round."""
    import hyperloom.agents.framework.sources as src
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    monkeypatch.setattr(src, "enumerate_candidates", lambda _req: [])

    kept = _candidate()
    state = types.SimpleNamespace(
        framework="vllm",
        model_name="deepseek-v4",
        gpu_type="mi355x",
        enablement=EnablementRound(
            kept_patches=[],
            setup_commands=[],
            kept_stack_action=kept,
            localization_manifest=[],
            last_build_failure={},
        ),
    )
    fake = types.SimpleNamespace(shared_state=state)
    fake._discover_enablement_candidate_refs = types.MethodType(Coordinator._discover_enablement_candidate_refs, fake)
    fake._read_enablement_source_context = lambda _sig: ""
    fake._derive_checkpoint_weight_facts = lambda _log: ""
    fake._framework_gpu_params = lambda: {}
    params = Coordinator._build_enablement_specialist_params(fake, "Model architecture 'Foo' is not supported")
    assert params is not None
    # The prior KEEP'd runtime is re-attached for the next round.
    assert params.get("runtime_candidate") == kept
