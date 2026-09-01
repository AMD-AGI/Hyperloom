# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Targeted-build <-> enablement integration in the framework phase.

Tests escalation, outcome routing, failure_class injection into the mandate,
and the gpu_arch derivation helper.  No GPU, no network, no coordinator ticks.
"""

from __future__ import annotations

import json
import types as _types

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction, BuildResult, FrameworkRuntime
from hyperloom.orchestrator.loop.build_lifecycle import BuildLifecycleCollaborator
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.enablement.build import (
    _derive_gpu_arch,
    _repo_matches_targeted_build_component,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


# ---------------------------------------------------------------------------
# Fixture: extend the shared build_coord with framework-phase routing methods
# ---------------------------------------------------------------------------


@pytest.fixture
def coord(build_coord):
    """``build_coord`` augmented with the routing-method surface the framework
    phase delegates to (launch-probe enqueue, rearm capture, build lifecycle).
    """
    build_coord._rearm_calls = []
    for name in (
        "_enqueue_build_launch_probe",
        "_route_succeeded_build",
        "_route_failed_build",
        "_build_routing_record",
        "_note_build_routed",
        "_build_probe_was_cancelled",
        "_open_row_past_spent_generations",
        "_time_budget_denial_for_action",
    ):
        setattr(build_coord, name, _types.MethodType(getattr(Coordinator, name), build_coord))
    # The real wall-clock gate, on the real catalogue: with no budget set it
    # admits everything, so a test that wants a denial sets one.
    build_coord.action_registry = ACTION_CATALOGUE

    def _maybe_rearm_enablement(res):
        build_coord._rearm_calls.append(dict(res) if isinstance(res, dict) else {})

    async def _enqueue_targeted_build(action):
        return await build_coord._bl.enqueue_targeted_build(action)

    build_coord._maybe_rearm_enablement = _maybe_rearm_enablement
    build_coord.enqueue_targeted_build = _enqueue_targeted_build
    build_coord._framework_gpu_params = lambda: {}
    build_coord._framework_authoring_lanes_ttl = lambda params, *, base_ttl_sec: (["research_lane"], base_ttl_sec)
    # The launch probe is an ``integrate_patch`` task, so it resolves its lanes
    # from that kind rather than from the specialist research lane.
    build_coord._registry_lanes_ttl = lambda kind: (
        ["server_lifecycle", "workspace_mutation", "benchmark_lane"],
        3600,
    )
    build_coord._coerce_needs_gpu = bool
    build_coord._bl = BuildLifecycleCollaborator(build_coord)
    return build_coord


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


def test_targeted_build_repo_match_ignores_origin():
    assert _repo_matches_targeted_build_component("https://example.test/forks/aiter.git", "aiter")
    assert _repo_matches_targeted_build_component("git@example.test:team/vllm.git", "vllm_source")


def test_targeted_build_repo_match_rejects_wrong_component():
    assert not _repo_matches_targeted_build_component(
        "https://example.test/forks/unrelated.git",
        "aiter",
    )


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
# _maybe_escalate_to_targeted_build: vLLM arch/weight deep-failure -> vllm_source
# (source patches keep hitting the arch wall)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arch_stall_escalates_to_vllm_source_after_attempts(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi355x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    log = "Model architecture 'DeepseekV4ForCausalLM' is not supported"
    # attempt 0: give the cheap source-patch path first crack -> no build yet.
    await Coordinator._maybe_escalate_to_targeted_build(coord, log, attempt=0)
    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0

    # attempt 1: source patches still hit the arch wall -> from-source vLLM build.
    await Coordinator._maybe_escalate_to_targeted_build(coord, log, attempt=1)
    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    assert action.component == "vllm_source"
    assert action.gpu_arch == "gfx950"


@pytest.mark.asyncio
async def test_arch_stall_not_escalated_on_non_vllm(coord, monkeypatch):
    coord.shared_state.framework = "sglang"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    log = "Model architecture 'FooForCausalLM' is not supported"
    # Even at a high attempt count, the from-source vLLM recipe is vLLM-only.
    await Coordinator._maybe_escalate_to_targeted_build(coord, log, attempt=5)
    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0


# ---------------------------------------------------------------------------
# _maybe_enqueue_specialist_requested_build
# (specialist asks for a compiled / from-source build in specialist_done)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_specialist_requested_build_enqueued(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi355x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    tid = "spec-abc123"
    wd = coord.session_dir / "runs" / "specialist" / tid
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "specialist_done.json").write_text(
        json.dumps(
            {
                "needs_targeted_build": {
                    "component": "aiter",
                    "capability": "deepseek_v4_nsa",
                    "repo_url": "https://github.com/ROCm/aiter",
                    "ref": "PR:1234",
                    "reason": "NSA index op missing on ROCm",
                }
            }
        )
    )
    coord.shared_state.enablement.last_specialist_task_id = tid

    await Coordinator._maybe_enqueue_specialist_requested_build(coord)

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    assert action.component == "aiter"
    assert action.capability == "deepseek_v4_nsa"
    assert action.ref == "PR:1234"
    assert action.gpu_arch == "gfx950"
    # Consume-once: the marker is cleared so the next tick does not re-enqueue.
    assert coord.shared_state.enablement.last_specialist_task_id == ""


@pytest.mark.asyncio
async def test_specialist_requested_build_enqueued_from_payload(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi355x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    payload = {
        "needs_targeted_build": {
            "component": "aiter",
            "capability": "deepseek_v4_decode",
            "repo_url": "https://github.com/ROCm/aiter",
            "ref": "v0.1.15.post2",
        }
    }
    await Coordinator._maybe_enqueue_specialist_requested_build(
        coord,
        task_id="spec-direct",
        payload=payload,
    )

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    assert TargetedBuildAction.from_state(queued[0].params).ref == "v0.1.15.post2"


@pytest.mark.asyncio
async def test_specialist_requested_build_rejects_repo_component_mismatch(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi355x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    tid = "spec-wrong-repo"
    wd = coord.session_dir / "runs" / "specialist" / tid
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "specialist_done.json").write_text(
        json.dumps(
            {
                "needs_targeted_build": {
                    "component": "aiter",
                    "repo_url": "https://example.test/forks/unrelated.git",
                }
            }
        )
    )
    coord.shared_state.enablement.last_specialist_task_id = tid

    await Coordinator._maybe_enqueue_specialist_requested_build(coord)

    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0


@pytest.mark.asyncio
async def test_specialist_requested_build_defaults_component_to_vllm_source(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi300x"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    tid = "spec-nocomp"
    wd = coord.session_dir / "runs" / "specialist" / tid
    wd.mkdir(parents=True, exist_ok=True)
    # A request with no (or invalid) component defaults to a from-source vLLM build.
    (wd / "specialist_done.json").write_text(
        json.dumps({"needs_targeted_build": {"capability": "deepseek_v4", "reason": "new arch"}})
    )
    coord.shared_state.enablement.last_specialist_task_id = tid

    await Coordinator._maybe_enqueue_specialist_requested_build(coord)

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    assert TargetedBuildAction.from_state(queued[0].params).component == "vllm_source"


@pytest.mark.asyncio
async def test_specialist_requested_build_noop_without_request(coord, monkeypatch):
    coord.shared_state.framework = "vllm"

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    tid = "spec-plain"
    wd = coord.session_dir / "runs" / "specialist" / tid
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "specialist_done.json").write_text(json.dumps({"empty": False, "patches_written": ["p.patch"]}))
    coord.shared_state.enablement.last_specialist_task_id = tid

    await Coordinator._maybe_enqueue_specialist_requested_build(coord)

    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0
    # Marker is still consumed (cleared) even when there is no request.
    assert coord.shared_state.enablement.last_specialist_task_id == ""


@pytest.mark.asyncio
async def test_specialist_requested_build_noop_when_no_task_id(coord, monkeypatch):
    coord.shared_state.framework = "vllm"
    coord.shared_state.enablement.last_specialist_task_id = ""
    await Coordinator._maybe_enqueue_specialist_requested_build(coord)
    assert len([t for t in await coord.tasks.queued() if t.kind == "targeted_build"]) == 0


# ---------------------------------------------------------------------------
# _maybe_route_build_outcomes -> _maybe_rearm_enablement routing
# ---------------------------------------------------------------------------


async def _enqueue_and_transition(coord, action, state):
    task_id = await coord._bl.enqueue_targeted_build(action)
    await coord.tasks.transition(task_id, "running")
    await coord.tasks.transition(task_id, state)
    return task_id


async def _verified_build(coord, root, *, gap_id, framework="vllm", ref="v1", runtime_env=None):
    """A succeeded ``targeted_build`` row whose result.json carries a usable runtime."""
    root.mkdir(parents=True, exist_ok=True)
    rt = FrameworkRuntime(pythonpath_prefixes=(str(root),), runtime_env=runtime_env or {})
    br = BuildResult(ok=True, attempt_root=str(root), runtime=rt)
    (root / "result.json").write_text(json.dumps(br.to_state()), encoding="utf-8")
    action = TargetedBuildAction(
        gap_id=gap_id,
        framework=framework,
        component="aiter",
        capability="fp4_moe",
        ref=ref,
        attempt_root=str(root),
    )
    return await _enqueue_and_transition(coord, action, "succeeded")


async def _queued_probes(coord):
    """The launch-probe rows currently waiting to be dispatched."""
    return [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]


def _spend_the_budget(coord, *, minutes=60):
    """Leave the session with no wall-clock budget for a probe to run in."""
    coord.shared_state.max_minutes = minutes
    coord.shared_state.elapsed_minutes = lambda **_kw: float(minutes)


def _restore_the_budget(coord):
    """Give the session a budget a probe fits in again, as a resume would."""
    coord.shared_state.max_minutes = 600
    coord.shared_state.elapsed_minutes = lambda **_kw: 0.0


@pytest.mark.asyncio
async def test_route_succeeded_row_enqueues_launch_probe(coord, tmp_path):
    """A succeeded build must enqueue an integrate_patch launch probe, not call rearm directly."""
    await _verified_build(coord, tmp_path / "attempt_s", gap_id="g2", runtime_env={"X": "1"})

    await Coordinator._maybe_route_build_outcomes(coord)

    # Must NOT directly rearm with "kept" — KEEP comes from the probe.
    assert not any(r.get("status") == "kept" for r in coord._rearm_calls)
    # Must have queued a launch-probe task.
    probes = await _queued_probes(coord)
    assert len(probes) == 1
    probe_params = probes[0].params
    assert probe_params.get("enablement_launch_only") is True
    assert probe_params.get("enablement") is True
    assert isinstance(probe_params.get("runtime_override"), dict)
    assert probe_params["runtime_override"]  # non-empty


@pytest.mark.asyncio
async def test_route_succeeded_row_probe_carries_config_path(coord, tmp_path):
    """Launch probe inherits baseline_config_path from shared state."""
    coord.shared_state.baseline_config_path = "/cfg/bench.yaml"
    await _verified_build(
        coord,
        tmp_path / "attempt_cfg",
        gap_id="g3",
        framework="sglang",
        ref="v2",
    )

    await Coordinator._maybe_route_build_outcomes(coord)

    probes = await _queued_probes(coord)
    assert len(probes) == 1
    assert probes[0].params.get("config_path") == "/cfg/bench.yaml"


@pytest.mark.asyncio
async def test_route_succeeded_missing_result_json_calls_reverted(coord, tmp_path):
    """A succeeded build whose result.json is absent routes reverted."""
    root = tmp_path / "attempt_missing"
    root.mkdir(parents=True, exist_ok=True)
    # No result.json written.

    action = TargetedBuildAction(
        gap_id="g4", framework="vllm", component="aiter", capability="fp4_moe", ref="v1", attempt_root=str(root)
    )
    await _enqueue_and_transition(coord, action, "succeeded")

    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "reverted" for r in coord._rearm_calls)
    # No probe should be queued.
    probes = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
    assert len(probes) == 0


@pytest.mark.asyncio
async def test_route_succeeded_empty_runtime_override_calls_reverted(coord, tmp_path):
    """A succeeded build with an empty runtime override routes reverted."""
    root = tmp_path / "attempt_empty"
    root.mkdir(parents=True, exist_ok=True)

    # FrameworkRuntime with all-default (empty) fields → to_runtime_override() == {}
    rt = FrameworkRuntime()
    br = BuildResult(ok=True, attempt_root=str(root), runtime=rt)
    (root / "result.json").write_text(json.dumps(br.to_state()), encoding="utf-8")

    action = TargetedBuildAction(
        gap_id="g5", framework="vllm", component="aiter", capability="fp4_moe", ref="v1", attempt_root=str(root)
    )
    await _enqueue_and_transition(coord, action, "succeeded")

    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "reverted" for r in coord._rearm_calls)
    probes = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
    assert len(probes) == 0


@pytest.mark.asyncio
async def test_route_succeeded_probe_idempotent(coord, tmp_path):
    """Calling _maybe_route_build_outcomes twice for the same row only enqueues one probe."""
    await _verified_build(coord, tmp_path / "attempt_idem", gap_id="g6")

    await Coordinator._maybe_route_build_outcomes(coord)
    await Coordinator._maybe_route_build_outcomes(coord)

    probes = await _queued_probes(coord)
    assert len(probes) == 1  # idempotent


@pytest.mark.asyncio
async def test_a_launch_probe_the_budget_cannot_fit_is_not_enqueued(coord, tmp_path):
    """A probe opened into a spent budget is cancelled at dispatch and lost.

    The probe is what declares KEEP for a build, and the queue scan drops a
    queued row the wall-clock budget can no longer fit. Opening one anyway spends
    the build's one routing pass on a row that will never run, so the build stays
    verified and unlaunched with nothing left to notice it.
    """
    build_tid = await _verified_build(coord, tmp_path / "attempt_broke", gap_id="g_budget")
    _spend_the_budget(coord)

    await Coordinator._maybe_route_build_outcomes(coord)

    assert await _queued_probes(coord) == []
    # Nothing was routed, so the build is still owed a probe.
    assert Coordinator._build_routing_record(coord, build_tid) is None
    assert coord._rearm_calls == []


@pytest.mark.asyncio
async def test_a_build_whose_probe_the_run_cancelled_is_still_unprobed(coord, tmp_path):
    """A cancelled probe is no evidence about the build, so the build gets another.

    The gate above narrows the window but cannot close it: a probe that fits when
    it is opened can still be dropped before it is dispatched, and a probe row
    cancelled that way owns this build's idempotency key for the rest of the
    session. Both halves have to hold -- the build is routed again, and the key it
    is routed on is a fresh generation rather than the cancelled row.
    """
    build_tid = await _verified_build(coord, tmp_path / "attempt_again", gap_id="g_again")
    await Coordinator._maybe_route_build_outcomes(coord)
    first = (await _queued_probes(coord))[0]
    await coord.tasks.transition(first.task_id, "cancelled", evidence={"reason": "time_budget"})
    _restore_the_budget(coord)

    await Coordinator._maybe_route_build_outcomes(coord)

    probes = await _queued_probes(coord)
    assert [p.task_id for p in probes] != [], "the build was left accounted for by a probe that never ran"
    assert first.task_id not in {p.task_id for p in probes}, "the window resolved to the cancelled row"
    record = Coordinator._build_routing_record(coord, build_tid) or {}
    assert record.get("probe_task_id") == probes[0].task_id
    assert int(record.get("probe_generation") or 0) == 1


@pytest.mark.asyncio
async def test_a_build_whose_probe_ran_and_failed_is_not_probed_again(coord, tmp_path):
    """A probe that ran said something about the build; only a cancel says nothing."""
    await _verified_build(coord, tmp_path / "attempt_failed", gap_id="g_failed")
    await Coordinator._maybe_route_build_outcomes(coord)
    probe = (await _queued_probes(coord))[0]
    await coord.tasks.transition(probe.task_id, "running")
    await coord.tasks.transition(probe.task_id, "failed")

    await Coordinator._maybe_route_build_outcomes(coord)

    assert await _queued_probes(coord) == []


@pytest.mark.asyncio
async def test_route_failed_timeout_calls_advanced(coord):
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", ref="v1")
    await _enqueue_and_transition(coord, action, "failed")
    # Simulate failure recorded by lifecycle
    coord.shared_state.enablement.last_build_failure = {
        "failure_class": "timeout",
        "failure_summary": "build exceeded budget",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "advanced" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_route_failed_compile_error_novel_calls_advanced(coord):
    """A compile_error not yet in the novelty ledger → advanced (novel attempt)."""
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", ref="v1")
    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement.last_build_failure = {
        "failure_class": "compile_error",
        "failure_summary": "hipcc failed",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "advanced" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_route_failed_compile_error_repeat_calls_reverted(coord, tmp_path):
    """A compile_error whose key is already in the novelty ledger → reverted."""
    from hyperloom.orchestrator.framework.build_actions import build_novelty_key

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", ref="v1")
    # Pre-seed the ledger with this exact novelty key.
    key = list(build_novelty_key(action))
    coord.shared_state.enablement.build_novelty = [key]

    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement.last_build_failure = {
        "failure_class": "compile_error",
        "failure_summary": "hipcc failed again",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    assert any(r.get("status") == "reverted" for r in coord._rearm_calls)


@pytest.mark.asyncio
async def test_novelty_ledger_is_appended_and_bounded(coord, tmp_path):
    """Ledger grows on each novel failure and is capped at 20 entries."""
    from hyperloom.orchestrator.framework.build_actions import build_novelty_key

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", ref="v1")
    # Seed ledger with 20 different entries so the cap truncates old ones.
    coord.shared_state.enablement.build_novelty = [["aiter", f"v{i}", "gfx950", []] for i in range(20)]
    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement.last_build_failure = {
        "failure_class": "compile_error",
        "failure_summary": "overflow",
    }
    await Coordinator._maybe_route_build_outcomes(coord)

    ledger = coord.shared_state.enablement.build_novelty
    assert len(ledger) == 20  # bounded
    assert list(build_novelty_key(action)) in ledger  # new entry present


@pytest.mark.asyncio
async def test_route_same_row_not_processed_twice(coord):
    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", ref="v1")
    await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement.last_build_failure = {"failure_class": "compile_error", "failure_summary": "x"}
    await Coordinator._maybe_route_build_outcomes(coord)
    await Coordinator._maybe_route_build_outcomes(coord)

    assert len(coord._rearm_calls) == 1  # only once


@pytest.mark.asyncio
async def test_route_failed_build_not_acked_when_rearm_raises(coord):
    """Failed builds stay unrouted when rearm fails so the next tick can retry."""
    action = TargetedBuildAction(
        gap_id="g",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        ref="v1",
    )
    task_id = await _enqueue_and_transition(coord, action, "failed")
    coord.shared_state.enablement.last_build_failure = {
        "failure_class": "compile_error",
        "failure_summary": "x",
    }
    attempts = {"n": 0}
    real_rearm = coord._maybe_rearm_enablement

    def _flaky_rearm(res):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("rearm failed")
        real_rearm(res)

    coord._maybe_rearm_enablement = _flaky_rearm

    await Coordinator._maybe_route_build_outcomes(coord)
    assert Coordinator._build_routing_record(coord, task_id) is None
    assert coord.shared_state.enablement.build_novelty == []

    await Coordinator._maybe_route_build_outcomes(coord)
    assert Coordinator._build_routing_record(coord, task_id) is not None
    assert attempts["n"] == 2
    assert len(coord._rearm_calls) == 1
    assert coord._rearm_calls[-1]["status"] == "advanced"


@pytest.mark.asyncio
async def test_route_oldest_unrouted_build_when_newer_already_routed(coord, tmp_path):
    """Older unseen builds must still route when a newer build is already accounted for."""
    older_tid = await _verified_build(coord, tmp_path / "attempt_older", gap_id="g_older", ref="v-older")
    newer_tid = await _verified_build(coord, tmp_path / "attempt_newer", gap_id="g_newer", ref="v-newer")
    Coordinator._note_build_routed(coord, newer_tid)

    await Coordinator._maybe_route_build_outcomes(coord)

    probes = await _queued_probes(coord)
    assert len(probes) == 1
    assert Coordinator._build_routing_record(coord, older_tid) is not None
    assert Coordinator._build_routing_record(coord, newer_tid) is not None


# ---------------------------------------------------------------------------
# _build_enablement_specialist_params injects failure_class into notes/params
# ---------------------------------------------------------------------------


def _make_params_fake(**kw):
    import types

    state = types.SimpleNamespace(
        framework=kw.get("framework", "vllm"),
        model_name=kw.get("model_name", "deepseek-ai/DeepSeek-V4"),
        gpu_type=kw.get("gpu_type", "mi355x"),
        enablement=EnablementRound(
            kept_patches=[],
            kept_stack_action={},
            setup_commands=[],
            localization_manifest=[],
            last_build_failure=kw.get("enablement_last_build_failure", {}),
        ),
    )
    fake = types.SimpleNamespace(shared_state=state, session_dir="/tmp")
    fake._build_enablement_specialist_params = types.MethodType(Coordinator._build_enablement_specialist_params, fake)
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
    fake_timeout = _make_params_fake(enablement_last_build_failure={"failure_class": "timeout", "failure_summary": ""})
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


# ---------------------------------------------------------------------------
# _maybe_escalate_to_targeted_build: discovery-driven ref selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_uses_discovery_ref_when_no_kept_ref(coord, monkeypatch):
    """When no kept/operator ref, top matching candidate from discovery is used."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.enablement.candidate_refs = [
        "https://github.com/ROCm/aiter/pull/77",
    ]

    await Coordinator._maybe_escalate_to_targeted_build(coord, "hipErrorNoBinaryForGpu")

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    assert action.ref == "PR:77"
    assert "aiter" in action.repo_url.lower()
    assert action.source_pr_url == "https://github.com/ROCm/aiter/pull/77"


@pytest.mark.asyncio
async def test_escalate_kept_ref_short_circuits_discovery(coord, monkeypatch):
    """When a kept stack action has a ref, discovery candidates are ignored."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.enablement.kept_stack_action = {
        "ref": "v0.99.0",
        "repo_url": "https://github.com/ROCm/aiter",
    }
    coord.shared_state.enablement.candidate_refs = [
        "https://github.com/ROCm/aiter/pull/77",
    ]

    await Coordinator._maybe_escalate_to_targeted_build(coord, "hipErrorNoBinaryForGpu")

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    assert action.ref == "v0.99.0"
    assert action.source_pr_url == ""


@pytest.mark.asyncio
async def test_escalate_skips_candidate_with_wrong_component_repo(coord, monkeypatch):
    """aiter component only accepts candidates whose repo contains 'aiter'."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    coord.shared_state.framework = "vllm"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.enablement.candidate_refs = [
        "https://github.com/sgl-project/sglang/pull/5",  # wrong repo for aiter
    ]

    await Coordinator._maybe_escalate_to_targeted_build(coord, "hipErrorNoBinaryForGpu")

    queued = [t for t in await coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(queued) == 1
    action = TargetedBuildAction.from_state(queued[0].params)
    # Falls back to empty ref (tag autoselect) since no matching candidate
    assert action.ref == ""
    assert action.source_pr_url == ""
