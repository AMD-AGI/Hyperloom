# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lane-deferred integrate is retried by the coordinator and stops holding KERNEL after its bound.

The rehearsal replays the wait one session spent 3.8 hours in: the KERNEL pipeline holds the
benchmark lanes, a KEEP it produced waits on them, and the orchestrator re-sends ``integrate``
on every turn it gets. Time is virtual; one tick is 36 s, as in that session.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.kernel import _kernel_decisions as kd
from hyperloom.orchestrator.kernel import request_handlers
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.deferred_integrate import DeferredIntegrates
from hyperloom.orchestrator.phases.machine_state import kernel_work_pending
from hyperloom.orchestrator.rehearsal import VirtualClock, installed_clock
from hyperloom.orchestrator.roles import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState

KERNEL = "llm_input_residual_rmsnorm"
TICK_SEC = 36.0


def _integrate() -> Intent:
    return Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "kernel_agent", "kind": "integrate", "params": {"kernel_id": KERNEL}},
    )


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _queue_keep(state) -> None:
    patch = SimpleNamespace(
        kernel_name=KERNEL,
        patch_path="/ws/forge_experiments/best/iter_001/forge.patch",
        target_file="/repo/vllm/model_executor/models/qwen3_5.py",
        env_flag="",
        micro_speedup=13.09,
        snapshot_dir="",
        kernel_repo="/repo",
        base_commit="deadbeef",
    )
    assert kd.enqueue_nominated_patch(state, patch=patch, lane="fusion") is not None


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# --- the parking lot ---------------------------------------------------------
def test_a_parked_integrate_counts_deferrals_and_expires_on_either_bound():
    lot = DeferredIntegrates(enabled=True, max_attempts=3, max_minutes=60)
    payload = _integrate().payload
    entry = lot.park("orchestration", payload, ["benchmark_lane"], now=0.0)
    assert entry is not None and entry.kernel_id == KERNEL and entry.attempts == 1
    assert lot.park("orchestration", payload, ["benchmark_lane"], now=10.0) is entry
    assert entry.attempts == 2 and entry.first_deferred == 0.0
    assert lot.expiry_reason(entry, now=59 * 60) == ""
    assert lot.expiry_reason(entry, now=60 * 60) == "deferred_for_60_min"
    lot.park("orchestration", payload, ["benchmark_lane"], now=20.0)
    assert lot.expiry_reason(entry, now=30.0) == "deferred_3_times"
    lot.discard(KERNEL)
    assert len(lot) == 0


def test_parking_is_off_when_disabled(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_INTEGRATE_RETRY", "0")
    assert DeferredIntegrates().park("orchestration", _integrate().payload, ["benchmark_lane"], now=0.0) is None


# --- what an expired wait changes --------------------------------------------
def test_an_expired_wait_releases_kernel_but_keeps_the_keep_for_sweep():
    state = SharedState(session_id="s")
    _queue_keep(state)
    assert state.keep_pending_holds_kernel and kernel_work_pending(state)
    assert state.mark_integrate_wait_expired(KERNEL, reason="deferred_for_60_min", at="t")
    assert not state.keep_pending_holds_kernel
    assert state.has_keep_pending_integrate, "the SWEEP-entry drain must still see it"
    assert state.integrate_wait_expired_kernel_ids() == [KERNEL]
    summary = state.to_prompt_summary()
    assert "pending_keep_kernels=(none)" in summary
    assert "has_keep_pending_integrate=false" in summary
    assert f"integrate_wait_expired=['{KERNEL}']" in summary
    assert not state.mark_integrate_wait_expired(KERNEL, reason="again", at="t2")


def test_the_prompt_is_unchanged_while_nothing_expired():
    state = SharedState(session_id="s")
    _queue_keep(state)
    summary = state.to_prompt_summary()
    assert f"pending_keep_kernels=['{KERNEL}']" in summary
    assert "has_keep_pending_integrate=true" in summary
    assert "integrate_wait_expired" not in summary
    assert "pending_escalate_hint" not in summary


# --- the rehearsal -----------------------------------------------------------
def _resends_every_turn() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_integrate())


def _sends_once() -> ScriptedPlan:
    return ScriptedPlan(turns=[MockTurn(intents=[_integrate()])], default_intent=_heartbeat())


async def _kernel_wait(
    session_dir: Path, *, ticks: int, plan: ScriptedPlan, release_at: int | None = None
) -> SimpleNamespace:
    """Play ``ticks`` KERNEL ticks while the pipeline holds the lanes, optionally freeing them at ``release_at``."""
    orchestration = MockBackend(plan, name="o")
    critic = MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="c")
    clock = VirtualClock()
    with installed_clock(clock):
        c = Coordinator(session_dir, backends={"orchestration": orchestration, "critic": critic})

        async def _hold_phase() -> None:
            return None

        c._advance_phase_if_needed = _hold_phase
        try:
            state = c.shared_state
            state.baseline_tput = 800.0
            state.phase = "KERNEL_AGENT"
            _queue_keep(state)
            lanes = list(ACTION_CATALOGUE["integrate"].requires_lanes)
            pipeline = await c.locks.try_acquire_many(
                lanes, holder_id="kernel-pipeline", task_id="kernel-pipeline", action="kernel_agent", ttl_sec=86_400
            )
            assert pipeline is not None
            for i in range(ticks):
                if release_at is not None and i == release_at:
                    await c.locks.release(pipeline)
                await c.tick(1)
                clock.advance(TICK_SEC)
            requests = [m for m in await c.bus.tail(topic="request", n=-1) if m.payload.get("kind") == "integrate"]
            expired = [
                m
                for m in await c.bus.tail(topic="observation", n=-1)
                if m.payload.get("kind") == "integrate_wait_expired"
            ]
            return SimpleNamespace(
                llm_calls=len(orchestration.calls),
                integrate_requests=len(requests),
                expired_notices=len(expired),
                holds_kernel=state.keep_pending_holds_kernel,
                keep_pending=state.has_keep_pending_integrate,
            )
        finally:
            await c.stop()


async def test_before_the_change_every_tick_pays_an_llm_turn_to_resend_integrate(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "0")
    monkeypatch.setenv("HYPERLOOM_INTEGRATE_RETRY", "0")
    run = await _kernel_wait(session_dir, ticks=40, plan=_resends_every_turn())  # 24 virtual minutes
    assert run.llm_calls == 40
    assert run.integrate_requests == 40
    assert run.holds_kernel


async def test_after_the_change_the_wait_costs_a_few_turns_and_ends_at_its_bound(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "1")
    monkeypatch.setenv("HYPERLOOM_INTEGRATE_RETRY", "1")
    monkeypatch.setenv("HYPERLOOM_INTEGRATE_DEFER_MAX_MIN", "15")
    monkeypatch.delenv("HYPERLOOM_INTEGRATE_DEFER_MAX_ATTEMPTS", raising=False)
    run = await _kernel_wait(session_dir, ticks=40, plan=_resends_every_turn())
    # One turn to send it, one per 5-minute heartbeat, one to read the expiry notice.
    assert run.llm_calls <= 8, run
    assert run.integrate_requests == run.llm_calls, "the coordinator never re-sends into busy lanes"
    assert run.expired_notices == 1
    assert not run.holds_kernel
    assert run.keep_pending, "SWEEP entry still integrates it"


async def test_the_coordinator_dispatches_the_parked_integrate_once_the_lanes_free(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IDLE_TICK_GATE", "1")
    monkeypatch.setenv("HYPERLOOM_INTEGRATE_RETRY", "1")
    handled: list[dict] = []

    async def _integrate_handler(payload, **_kwargs):
        handled.append(dict(payload))
        return {"status": "skipped", "decision": "REVERT", "kernel_id": KERNEL, "error_class": "rehearsal"}

    monkeypatch.setitem(request_handlers.KERNEL_REQUEST_HANDLERS, "integrate", _integrate_handler)
    run = await _kernel_wait(session_dir, ticks=20, plan=_sends_once(), release_at=10)
    assert len(handled) == 1, "the lanes freed at tick 10 and the parked integrate ran without a re-send"
    assert run.expired_notices == 0
    assert run.llm_calls <= 4, run
