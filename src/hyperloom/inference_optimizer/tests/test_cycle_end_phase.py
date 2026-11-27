# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CYCLE_END phase tests: the stack work its entry settles, and the exit it takes once settled."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hyperloom.orchestrator.kernel import kernel_decisions as kd
from hyperloom.orchestrator.phases.machine import Transition

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.tests.mock_backend import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
from hyperloom.orchestrator.kernel.patch_lifecycle import lifecycle_complete
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.phases import phase_predicates
from hyperloom.orchestrator.phases import phase_transitions
from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import seed_kernel_keep


def _coordinator_with_a_pending_keep(tmp_path: Path, monkeypatch) -> tuple[Coordinator, list[str]]:
    """A coordinator whose queue holds the KEEP ``k004``, and the kernel ids its stubbed integrate handler is asked for."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.backends = {}
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
        macro_cycle=1,
    )
    seed_kernel_keep(c.shared_state, "k004", micro=4.21, source_file="/tmp/kernel.cu", artifact="/tmp/optimized.cu")
    calls: list[str] = []

    async def _fake_integrate_handler(payload, *, session_dir):
        calls.append(payload["kernel_id"])
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": payload["kernel_id"],
            "patch_path": "/tmp/optimized.cu",
            "target_file": "/tmp/kernel.cu",
            "base_tput": 100.0,
            "new_tput": 102.0,
            "gain_pct": 2.0,
            "workspace": str(tmp_path / "integrate-k004"),
        }

    async def _noop_roofline(*, reason: str):
        return None

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.integrate_handler",
        _fake_integrate_handler,
    )
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline
    return c, calls


@pytest.mark.asyncio
async def test_drain_pending_keep_integrates_records_result_once(
    tmp_path: Path,
    monkeypatch,
):
    """CYCLE_END entry drain must record integrate results so the same KEEP is not retried until cap."""
    c, calls = _coordinator_with_a_pending_keep(tmp_path, monkeypatch)

    await c.phase_kernel_stack.drain_pending_keep_integrates()

    assert calls == ["k004"]
    assert c.shared_state.kernel_integrate_attempts
    assert kd.pending_keep_kernel_ids(c.shared_state) == []
    assert c.shared_state.current_best["action"] == "integrate"
    assert c.shared_state.current_best["variant_name"] == "k004"


@pytest.mark.asyncio
async def test_cycle_end_entry_integrates_the_pending_keep_and_settles_the_cycle(tmp_path: Path, monkeypatch):
    """CYCLE_END entry lands the KEEP the KERNEL phase left queued, then records the cycle as settled."""
    c, calls = _coordinator_with_a_pending_keep(tmp_path, monkeypatch)

    await c.phase_cycle_end.on_enter_cycle_end(
        Transition(from_phase="KERNEL_AGENT", to_phase="CYCLE_END", reason="test", evidence={})
    )

    assert calls == ["k004"]
    assert c.shared_state.current_best["variant_name"] == "k004"
    assert kd.pending_keep_kernel_ids(c.shared_state) == []
    assert c.shared_state.cycle_end_settled_cycle == 1


def test_pending_keep_kernel_ids_prioritize_trace_impact_over_micro():
    """E2E integrate order should prefer trace impact over isolated micro speedup."""
    state = SharedState()
    state.last_trace_analyze = {
        "hot_kernels_top15": [
            {"kernel_id": "k001", "gpu_pct": 60.0},
            {"kernel_id": "k004", "gpu_pct": 10.0},
        ],
    }
    seed_kernel_keep(state, "k004", micro=4.21, source_file="/tmp/rmsnorm.cu", artifact="/tmp/k004_opt.cu")
    seed_kernel_keep(state, "k001", micro=1.51, source_file="/tmp/moe.cu", artifact="/tmp/k001_opt.cu")

    assert kd.pending_keep_kernel_ids(state) == ["k001", "k004"]


def test_pending_keep_kernel_ids_do_not_retry_needs_review():
    """A recorded NEEDS_REVIEW attempt should not auto-rerun the same patch."""
    state = SharedState()
    seed_kernel_keep(state, "k004", micro=4.21, source_file="/tmp/rmsnorm.cu", artifact="/tmp/k004_opt.cu")
    seed_kernel_keep(state, "k001", micro=1.51, source_file="/tmp/moe.cu", artifact="/tmp/k001_opt.cu")
    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "kernel_id": "k004",
            "patch_path": "/tmp/k004_opt.cu",
            "target_file": "/tmp/rmsnorm.cu",
            "new_tput": 100.8,
            "gain_pct": 0.8,
            "workspace": "/tmp/integrate-k004",
        }
    )

    assert kd.pending_keep_kernel_ids(state) == ["k001"]


@pytest.mark.asyncio
async def test_drain_uses_current_best_tput_not_baseline(
    tmp_path: Path,
    monkeypatch,
):
    """Drain should pass current_best.tput (not baseline) so multi-KEEP gain is incremental."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "integrate", "tput": 110.0, "kernel_id": "k_prev"},
    )
    c.shared_state.optimization_stack = [
        {"action": "integrate", "kernel_id": "k_prev", "tput": 110.0},
    ]
    seed_kernel_keep(c.shared_state, "k_new", micro=2.0, source_file="/tmp/new.cu", artifact="/tmp/new_opt.cu")
    captured_payloads = []

    async def _fake_integrate_handler(payload, *, session_dir):
        captured_payloads.append(payload)
        return {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": payload["kernel_id"],
            "patch_path": "/tmp/new_opt.cu",
            "target_file": "/tmp/new.cu",
            "base_tput": payload.get("base_tput", 0.0),
            "new_tput": 112.0,
            "gain_pct": (112.0 / payload.get("base_tput", 100.0) - 1) * 100,
            "workspace": str(tmp_path / "integrate-k_new"),
        }

    async def _noop_roofline(*, reason: str):
        return None

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.integrate_handler",
        _fake_integrate_handler,
    )
    c.phase_kernel.maybe_enqueue_watermark_roofline = _noop_roofline

    await c.phase_kernel_stack.drain_pending_keep_integrates()

    assert len(captured_payloads) == 1
    # use current_best.tput (110.0), not baseline (100.0)
    assert captured_payloads[0]["base_tput"] == 110.0


# End-to-end via the real phase machine
def _real_coordinator(tmp_path: Path) -> Coordinator:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=session_dir,
        backends={"orchestration": MockBackend(idle_plan), "critic": MockBackend(idle_plan)},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )


@pytest.mark.asyncio
async def test_a_transition_into_cycle_end_settles_the_cycle_and_dispatches_nothing(tmp_path: Path):
    coord = _real_coordinator(tmp_path)
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.macro_cycle = 2
    machine_state.record_phase_transition(
        coord.shared_state, to_phase="CYCLE_END", reason="kernel_no_more_leverage", evidence={}
    )

    await coord.phase_machine._on_phase_entered(from_phase="KERNEL_AGENT", to_phase="CYCLE_END")

    assert coord.shared_state.cycle_end_settled_cycle == 2
    assert SharedState.load_or_init(coord.session_dir).cycle_end_settled_cycle == 2
    assert await coord.tasks.queued() == []


# The exit CYCLE_END takes
_NOW = 1_800_000_000.0


def _cycle_end_state(*, settled_cycle: int, macro_cycle: int = 0, entered_min_ago: float = 1.0) -> SharedState:
    """A CYCLE_END entered ``entered_min_ago`` minutes ago, one hour into a ten-hour session with gain this cycle."""
    return SharedState(
        phase=machine_state.PHASE_CYCLE_END,
        phase_started_unix=_NOW - entered_min_ago * 60.0,
        start_ts=datetime.fromtimestamp(_NOW - 3600.0, tz=timezone.utc).isoformat(),
        max_minutes=600,
        phase_budget_pct=dict(machine_state.DEFAULT_PHASE_BUDGET_PCT),
        macro_cycle=macro_cycle,
        cumulative_gain_validated=5.0,
        gain_at_cycle_start=0.0,
        baseline_tput=100.0,
        cycle_end_settled_cycle=settled_cycle,
    )


def test_a_settled_cycle_end_loops_back_while_another_cycle_is_feasible():
    out = phase_transitions.compute_next_phase(_cycle_end_state(settled_cycle=0), now_unix=_NOW)

    assert out is not None
    target, reason, evidence = out
    assert (target, reason) == (machine_state.PHASE_FRAMEWORK_AGENT, "cycle_reloop")
    assert evidence["loopback"] is True
    assert evidence["predicate_inputs"]["cycle_end_result"]["settled"] is True
    assert phase_transitions.replay_next_phase(evidence["predicate_inputs"]) == out


def test_a_settled_cycle_end_with_no_further_cycle_closes_as_cycle_end_done():
    out = phase_transitions.compute_next_phase(
        _cycle_end_state(settled_cycle=0),
        flags=phase_predicates.PhaseFlags(optimize_enabled=False),
        now_unix=_NOW,
    )

    assert out is not None
    assert out[:2] == (machine_state.PHASE_CLOSE, "cycle_end_done")
    assert "terminal" not in out[2]


def test_a_settled_cycle_end_closes_on_the_rule_that_refuses_the_next_cycle():
    state = _cycle_end_state(settled_cycle=0)
    state.saturated_directions = {"kv_cache": {"saturated": True}}

    out = phase_transitions.compute_next_phase(state, now_unix=_NOW)

    assert out is not None
    assert out[:2] == (machine_state.PHASE_CLOSE, "global_converged")
    assert out[2]["reloop_blocked"] == "all_directions_saturated"


@pytest.mark.parametrize("settled_cycle", [-1, 0], ids=["never", "earlier_cycle"])
def test_an_unsettled_cycle_end_holds_until_its_entry_work_settles_the_cycle(settled_cycle):
    state = _cycle_end_state(settled_cycle=settled_cycle, macro_cycle=1, entered_min_ago=600.0)

    assert phase_transitions.compute_next_phase(state, now_unix=_NOW) is None


# ────────────────────────────────────────────────────────────────────────────── Patch lifecycle convergence — new
# tests (P1-19 fix verification) ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_integrate_handler_revert_partial_becomes_failed(
    tmp_path: Path,
    monkeypatch,
):
    """Non-KEEP + partial revert must set top-level status='failed'."""
    from hyperloom.orchestrator.kernel import _integrate_outcomes as integrate_outcomes
    import hyperloom.orchestrator.actions.executors.baseline as baseline_mod
    import hyperloom.orchestrator.actions.executors.benchmark_result as br

    monkeypatch.setattr(
        integrate_outcomes,
        "_maybe_revert_kernel_patch",
        lambda apply_result: {"status": "partial", "reason": "mn_revert_failed"},
    )
    monkeypatch.setattr(
        integrate_outcomes,
        "_maybe_apply_kernel_patch",
        lambda payload, *, session_dir, kernel_id=None: {
            "status": "ok",
            "kernel_id": str(kernel_id or ""),
            "manifest_path": "/tmp/fake.manifest",
        },
    )

    class _FakeBaseline:
        default_timeout_sec = baseline_mod.resolve_benchmark_timeouts()[1]

        def __init__(self, *, session_dir, shared_state=None):
            self.session_dir = session_dir
            self.shared_state = shared_state

        async def __call__(self, ctx):
            return {"output_throughput": 98.0}  # below base_tput -> REVERT

    monkeypatch.setattr(baseline_mod, "BaselineExecutor", _FakeBaseline)
    monkeypatch.setattr(br, "is_valid_measurement", lambda r: True)

    from hyperloom.orchestrator.kernel.request_handlers import integrate_handler

    result = await integrate_handler(
        {
            "task_id": "test-partial-revert",
            "kernel_id": "k-test",
            "patch_path": "/tmp/fake.patch",
            "target_file": "/tmp/fake.cu",
            "base_tput": 100.0,
            "config_path": str(tmp_path / "base.yaml"),
        },
        session_dir=tmp_path,
    )

    assert result["decision"] == "REVERT"
    assert result["status"] == "failed"
    assert result["patch_cleanup_action"] == "revert"
    assert result.get("error_class") == "patch_revert_incomplete"


def test_close_neither_rebenches_nor_profiles_a_tree_it_refused_to_trust():
    """CLOSE re-benches an unvalidated stack and runs a post-opt roofline on the live tree.

    Under this stop reason that tree is the one the halt declared untrustworthy.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
    from hyperloom.orchestrator.phases import close as close_phase

    assert PATCH_RECOVERY_INCOMPLETE_STOP_REASON in close_phase._NO_REVALIDATION_STOP_REASONS

    seen: list[str] = []

    async def _enqueue(**_kw):
        seen.append("enqueued")
        return

    phase = close_phase.ClosePhase.__new__(close_phase.ClosePhase)
    object.__setattr__(
        phase,
        "_coord",
        SimpleNamespace(
            shared_state=SimpleNamespace(
                closing_phase=False,
                stop_reason=PATCH_RECOVERY_INCOMPLETE_STOP_REASON,
                optimization_stack=[{"action": "integrate"}],
                phase_history=[],
                save=lambda _: None,
            ),
            session_dir=None,
            phase_prelude=SimpleNamespace(
                internal_analysis_kind=lambda: seen.append("analysis_kind") or "roofline",
                enqueue_internal_analysis_task=_enqueue,
            ),
        ),
    )

    asyncio.run(phase._maybe_run_close_post_opt_roofline())

    assert seen == []
    assert not any("roofline" in str(s) for s in seen)


def test_a_revert_that_already_completed_is_not_run_again(tmp_path: Path):
    """The first revert moves each backup back, so a second would fail on a clean tree.

    A caller that reverts an apply which already reverted itself hits exactly that.
    """
    from hyperloom.orchestrator.kernel.apply_kernel_patch import revert_kernel_patch

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "reverted",
                "reverted_at": "2026-01-01T00:00:00Z",
                "restored_paths": ["/framework/src.py"],
                # Present and "ok" is what drives the second restore attempt.
                "jit_build_backup": {"status": "ok", "backup_path": str(tmp_path / "gone")},
                "artifacts": [{"backup_path": str(tmp_path / "missing.bak"), "path": str(tmp_path / "missing.py")}],
            }
        ),
        encoding="utf-8",
    )

    result = revert_kernel_patch(manifest)

    assert result["status"] == "skipped"
    assert result["already_reverted"] is True
    assert result["restored_paths"] == ["/framework/src.py"]
    assert lifecycle_complete(result)
