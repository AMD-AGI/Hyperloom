# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""S5 unit tests: targeted-build <-> enablement integration.

Tests escalation, outcome routing, failure_class injection into the mandate,
and the gpu_arch derivation helper.  No GPU, no network, no coordinator ticks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction, BuildResult, FrameworkRuntime
from hyperloom.orchestrator.loop.build_lifecycle import BuildLifecycleCollaborator
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework import _derive_gpu_arch
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry


# ---------------------------------------------------------------------------
# Helpers: minimal fake coordinator for framework-phase methods
# ---------------------------------------------------------------------------

class _FakeCoord:
    def __init__(self, session_dir, db, state):
        self.session_dir = session_dir
        self.tasks = TaskRegistry(db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(db))
        self.shared_state = state
        self._rearm_calls: list[dict] = []

    def _maybe_rearm_enablement(self, res):
        self._rearm_calls.append(dict(res) if isinstance(res, dict) else {})

    async def enqueue_targeted_build(self, action):
        # Delegate to real BuildLifecycleCollaborator
        return await self._bl.enqueue_targeted_build(action)

    def _setup_bl(self):
        self._bl = BuildLifecycleCollaborator(self)


@pytest.fixture
def coord(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    state = SharedState()
    fc = _FakeCoord(tmp_path, db, state)
    fc._setup_bl()
    yield fc
    db.close()


# ---------------------------------------------------------------------------
# _derive_gpu_arch
# ---------------------------------------------------------------------------

def test_derive_gpu_arch_mi355x():
    assert _derive_gpu_arch("mi355x") == "gfx950"

def test_derive_gpu_arch_mi300x():
    assert _derive_gpu_arch("mi300x") == "gfx942"

def test_derive_gpu_arch_unknown():
    assert _derive_gpu_arch("unknown_gpu") == ""

def test_derive_gpu_arch_empty():
    assert _derive_gpu_arch("") == ""

def test_derive_gpu_arch_case_insensitive():
    assert _derive_gpu_arch("MI355X") == "gfx950"


# ---------------------------------------------------------------------------
# _maybe_escalate_to_targeted_build
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalate_enqueues_for_compiled_gap(coord, monkeypatch):
    coord.shared_state.gpu_type = "mi355x"
    coord.shared_state.framework = "vllm"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    hip_kernel_log = "hipErrorNoBinaryForGpu: no kernel image is available"
    await Coordinator._maybe_escalate_to_targeted_build(coord, hip_kernel_log)

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    assert action.component == "aiter"
    assert action.gpu_arch == "gfx950"


@pytest.mark.asyncio
async def test_escalate_skipped_for_pure_python_gap(coord, monkeypatch):
    coord.shared_state.framework = "vllm"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    python_log = "Model architecture 'DeepseekV4ForCausalLM' is not supported"
    await Coordinator._maybe_escalate_to_targeted_build(coord, python_log)

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 0


@pytest.mark.asyncio
async def test_escalate_skipped_on_multi_node(coord, monkeypatch):
    coord.shared_state.framework = "vllm"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)

    log = "hipErrorNoBinaryForGpu"
    await Coordinator._maybe_escalate_to_targeted_build(coord, log)
    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0


@pytest.mark.asyncio
async def test_escalate_idempotent_same_gap(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi300x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    log = "hipErrorNoBinaryForGpu"
    await Coordinator._maybe_escalate_to_targeted_build(coord, log)
    await Coordinator._maybe_escalate_to_targeted_build(coord, log)

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1  # idempotent, not two rows


@pytest.mark.asyncio
async def test_escalate_disabled_by_env(coord, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD", "1")
    coord.shared_state.framework = "vllm"
    log = "hipErrorNoBinaryForGpu"
    await Coordinator._maybe_escalate_to_targeted_build(coord, log)
    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0


# ---------------------------------------------------------------------------
# _maybe_route_build_outcomes -> _maybe_rearm_enablement routing
# ---------------------------------------------------------------------------

async def _enqueue_and_transition(coord, action, state):
    task_id = await coord._bl.enqueue_targeted_build(action)
    await coord.tasks.transition(task_id, "running")
    await coord.tasks.transition(task_id, state)
    return task_id


@pytest.mark.asyncio
async def test_route_succeeded_row_calls_kept(coord, tmp_path):
    root = tmp_path / "attempt_s"
    root.mkdir(parents=True, exist_ok=True)

    # Write a result.json for the rich runtime
    rt = FrameworkRuntime(pythonpath_prefixes=(str(root),), runtime_env={"X": "1"})
    br = BuildResult(ok=True, attempt_root=str(root), runtime=rt)
    (root / "result.json").write_text(json.dumps(br.to_state()), encoding="utf-8")

    action = TargetedBuildAction(gap_id="g2", framework="vllm", component="aiter",
                                 capability="fp4_moe", ref="v1", attempt_root=str(root))
    task_id = await _enqueue_and_transition(coord, action, "succeeded")

    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "kept" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_route_failed_timeout_calls_advanced(coord):
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter",
                                 capability="fp4_moe", ref="v1")
    task_id = await _enqueue_and_transition(coord, action, "failed")
    # Simulate failure recorded by lifecycle
    coord.shared_state.enablement_last_build_failure = {
        "failure_class": "timeout",
        "failure_summary": "build exceeded budget",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "advanced" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_route_failed_compile_error_calls_reverted(coord):
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter",
                                 capability="fp4_moe", ref="v1")
    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement_last_build_failure = {
        "failure_class": "compile_error",
        "failure_summary": "hipcc failed",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "reverted" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_route_same_row_not_processed_twice(coord):
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter",
                                 capability="fp4_moe", ref="v1")
    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement_last_build_failure = {
        "failure_class": "compile_error", "failure_summary": "x"
    }
    await Coordinator._maybe_route_build_outcomes(coord)
    await Coordinator._maybe_route_build_outcomes(coord)

    assert len(coord._rearm_calls) == 1  # only once


# ---------------------------------------------------------------------------
# _build_enablement_specialist_params injects failure_class into notes/params
# ---------------------------------------------------------------------------

def _make_params_fake(**kw):
    import types
    state = types.SimpleNamespace(
        framework=kw.get("framework", "vllm"),
        model_name=kw.get("model_name", "deepseek-ai/DeepSeek-V4"),
        gpu_type=kw.get("gpu_type", "mi355x"),
        enablement_kept_patches=[],
        enablement_kept_stack_action={},
        enablement_setup_commands=[],
        enablement_localization_manifest=[],
        enablement_last_build_failure=kw.get("enablement_last_build_failure", {}),
    )
    fake = types.SimpleNamespace(shared_state=state, session_dir="/tmp")
    fake._build_enablement_specialist_params = types.MethodType(
        Coordinator._build_enablement_specialist_params, fake
    )
    fake._discover_enablement_candidate_refs = lambda req, plan: []
    fake._read_enablement_source_context = lambda _sig: ""
    fake._derive_checkpoint_weight_facts = lambda _log: ""
    fake._framework_gpu_params = lambda: {}
    return fake


def test_build_params_injects_last_build_failure_into_notes():
    fake = _make_params_fake(
        enablement_last_build_failure={
            "failure_class": "timeout",
            "failure_summary": "AITER ran out of time",
        }
    )
    log = "hipErrorNoBinaryForGpu: no kernel image"
    params = fake._build_enablement_specialist_params(log, attempt=1)
    assert params is not None
    notes = params.get("notes", "")
    assert "PREVIOUS TARGETED-BUILD" in notes
    assert "timeout" in notes
    assert params.get("enablement_last_build_failure", {}).get("failure_class") == "timeout"


def test_build_params_no_injection_when_no_build_failure():
    fake = _make_params_fake(enablement_last_build_failure={})
    log = "hipErrorNoBinaryForGpu: no kernel image"
    params = fake._build_enablement_specialist_params(log, attempt=0)
    assert params is not None
    notes = params.get("notes", "")
    assert "PREVIOUS TARGETED-BUILD" not in notes
    assert "enablement_last_build_failure" not in params


def test_build_params_failure_class_distinguishes_timeout_vs_defect():
    fake_timeout = _make_params_fake(
        enablement_last_build_failure={"failure_class": "timeout", "failure_summary": ""}
    )
    fake_defect = _make_params_fake(
        enablement_last_build_failure={"failure_class": "compile_error", "failure_summary": "bad code"}
    )
    log = "hipErrorNoBinaryForGpu"
    notes_timeout = fake_timeout._build_enablement_specialist_params(log)["notes"]
    notes_defect = fake_defect._build_enablement_specialist_params(log)["notes"]
    # Both should mention the build failure
    assert "timeout" in notes_timeout
    assert "compile_error" in notes_defect
    # The timeout note steers toward more time/smaller scope
    assert "budget" in notes_timeout.lower() or "time" in notes_timeout.lower()
