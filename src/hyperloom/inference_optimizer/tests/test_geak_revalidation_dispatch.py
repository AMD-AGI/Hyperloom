# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for GEAK same-harness revalidation dispatch (L1/L2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.phases import geak_rebench as gr
from hyperloom.orchestrator.phases import machine_state as ps


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    return Coordinator(sd, backends=backends)


def _geak_rebench_params(**extra: object) -> dict:
    return {
        "source": "resume_stack_revalidate",
        "geak_fallback": True,
        "reason": "geak_e2e_win",
        **extra,
    }


def _arm_kernel_to_sweep(st) -> None:
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.geak_pending = {}
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)


def test_geak_revalidate_idempotency_key_scopes_by_macro_cycle() -> None:
    assert gr.geak_revalidate_idempotency_key(0) == "geak-revalidate-c0"
    assert gr.geak_revalidate_idempotency_key(2) == "geak-revalidate-c2"
    assert gr.geak_revalidate_idempotency_key(2) != gr.geak_revalidate_idempotency_key(3)


@pytest.mark.asyncio
async def test_cancel_queued_not_allowed_spares_geak_rebench_into_sweep(coordinator) -> None:
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_SWEEP],
        reason="phase_transition:KERNEL_AGENT->SWEEP",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_SWEEP,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_queued_not_allowed_cancels_geak_rebench_into_close(coordinator) -> None:
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_CLOSE],
        reason="phase_transition:SWEEP->CLOSE",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_CLOSE,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_family_cancels_geak_rebench_explore(coordinator) -> None:
    """Explore-family prune must cancel GEAK rebench (no implicit spare)."""
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c1",
    )

    cancelled = await c.tasks.cancel_family(["explore"], reason="prune_branch")

    assert geak_task.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_geak_revalidate_idempotency_key_allows_retry_per_macro_cycle(
    coordinator,
) -> None:
    c = coordinator
    cycle0_key = gr.geak_revalidate_idempotency_key(0)
    cycle1_key = gr.geak_revalidate_idempotency_key(1)

    settled = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=cycle0_key,
        task_id="geak-revalidate-c0-task",
    )
    await c.tasks.transition(settled.task_id, "cancelled", evidence={"reason": "test"})

    fresh, was_existing = await c.tasks.create_or_return_existing(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=cycle1_key,
    )

    assert not was_existing
    assert fresh.task_id != settled.task_id
    assert fresh.state == "queued"


@pytest.mark.asyncio
async def test_enqueue_internal_stack_rebench_uses_macro_cycle_idempotency_key(
    coordinator,
) -> None:
    """Production enqueue path must stamp cycle-scoped GEAK rebench keys."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 8192", "env": ""},
        "bench_protocol": {
            "random_range_ratio": 1.0,
            "num_prompts": 192,
            "num_warmups": 8,
            "seed": 0,
        },
    }

    st.macro_cycle = 0
    first = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row0 = await c.tasks.get(str(first["task_id"]))
    assert row0.idempotency_key == "geak-revalidate-c0"
    assert row0.params["enable_stack_rebench"] is True
    assert row0.params["rebench_required"] is True
    assert row0.params["stack_rebench_repeats"] == 3
    assert row0.params["revalidation_protocol"]["prewarm_rounds"] == 2
    assert row0.params["revalidation_protocol"]["measured_repeats"] == 3
    assert row0.params["grid"][0]["extra_envs"]["RANDOM_RANGE_RATIO"] == "1.0"
    assert row0.params["grid"][0]["extra_envs"]["NUM_PROMPTS"] == "192"
    assert row0.params["grid"][0]["extra_envs"]["NUM_WARMUPS"] == "8"
    assert row0.params["grid"][0]["extra_envs"]["SEED"] == "0"
    # final_launch_script is optional: structured accepted_config is enough
    # to construct the orchestrator-harness replay.
    assert "final_launch_script" not in st.geak_result

    st.macro_cycle = 1
    second = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row1 = await c.tasks.get(str(second["task_id"]))
    assert row1.idempotency_key == "geak-revalidate-c1"
    assert row1.task_id != row0.task_id


@pytest.mark.asyncio
async def test_enqueue_geak_rebench_allows_container_only_config_path(
    coordinator,
    tmp_path,
) -> None:
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--max-num-batched-tokens 8192",
            "env": f"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE={tmp_path / 'missing.csv'}",
        },
    }

    out = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")

    assert out["task_state"] == "queued"
    task = await c.tasks.get(str(out["task_id"]))
    assert task.params["expected_config_file_digests"] == {}
    assert task.params["unverified_config_file_refs"] == [
        f"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE={tmp_path / 'missing.csv'}"
    ]


@pytest.mark.asyncio
async def test_rebench_can_be_rebuilt_after_cancel_within_same_cycle(coordinator) -> None:
    """A cancelled rebench must not block a fresh one in the same macro-cycle.

    ``create_or_return_existing`` hands back the cancelled row for a reused key,
    which KERNEL then reads as ``rebench_unavailable`` and the GEAK win stays
    audit-only for the rest of the cycle.
    """
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.macro_cycle = 0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 8192", "env": ""},
    }

    first = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    first_id = str(first["task_id"])
    assert first["task_state"] == "queued"

    # An explore-family prune settles the queued rebench mid-cycle.
    assert first_id in await c.tasks.cancel_family(["explore"], reason="prune_branch")

    second = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")

    assert second["task_id"] != first_id
    assert second["task_state"] == "queued"


@pytest.mark.asyncio
async def test_prune_settles_geak_pending_when_rebench_cancelled(coordinator) -> None:
    """Pruning the explore family must not leave the slot stuck awaiting."""
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}

    rebench = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="pruned-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": rebench.task_id}
    st.resume_pending_revalidation = True

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    await c._handle_prune_branch(
        "robustness",
        Intent(type=IntentType.PRUNE_BRANCH, payload={"family": "explore", "reason": "prune_branch"}),
    )

    assert (await c.tasks.get(rebench.task_id)).state == "cancelled"
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_settled_pending_rejects_late_rebench_result(coordinator) -> None:
    """A settled slot must not be revived by a late/orphan rebench completion."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
    }
    st.geak_pending = {"status": "rebench_cancelled", "revalidation_error": "close_sequence"}
    st.resume_pending_revalidation = False

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="late-rebench",
    )

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "abc"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["tput"] == 100.0
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert not any(e.get("action") == "geak_e2e" for e in st.optimization_stack)


@pytest.mark.asyncio
async def test_cleared_pending_without_resume_flag_rejects_result(coordinator) -> None:
    """An empty slot is only a resume signal when the resume flag is set."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
    }
    st.geak_pending = {}
    st.resume_pending_revalidation = False

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="post-clear-rebench",
    )

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "abc"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["tput"] == 100.0
    assert not any(e.get("action") == "geak_e2e" for e in st.optimization_stack)


@pytest.mark.asyncio
async def test_settle_preserves_candidate_audit_fields(coordinator) -> None:
    """Settling must not discard the self-reported numbers the report shows."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {"status": "ok"}
    c.phase_kernel._record_geak_candidate(
        {
            "status": "ok",
            "final_throughput_tok_s": 116.0,
            "throughput_speedup": 1.16,
            "accepted_config": {"flags": "--foo", "env": ""},
        }
    )
    st.geak_pending = {**st.geak_pending, "revalidation_task_id": "gone-task"}
    assert st.geak_pending["self_reported_gain_pct"] == pytest.approx(16.0)

    settled = await gr.settle_dangling_geak_pending(c.tasks, st, reason="close_sequence")

    assert settled is True
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.geak_pending["revalidation_error"] == "close_sequence"
    # The audit numbers survive so the report can name what was dropped.
    assert st.geak_pending["self_reported_gain_pct"] == pytest.approx(16.0)
    assert st.geak_pending["self_reported_tput"] == pytest.approx(116.0)
    # The id of a task that will never land must not outlive the slot.
    assert "revalidation_task_id" not in st.geak_pending


@pytest.mark.asyncio
async def test_wall_clock_closing_stops_rebench_and_settles(coordinator) -> None:
    """The wall-clock closing path never reaches ``_on_enter_close``.

    It cancels queued work but left the slot at ``awaiting_rebench`` and a
    running rebench alive, so the report claimed a rebench was still coming
    while the task had already been cancelled.
    """
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}

    queued = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="queued-at-timeout",
    )
    running = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0, 1),
        task_id="running-at-timeout",
    )
    await c.tasks.transition(running.task_id, "running")
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": running.task_id}
    st.resume_pending_revalidation = True

    await c._enter_closing_phase(grace_sec=30.0)

    assert (await c.tasks.get(queued.task_id)).state == "cancelled"
    assert (await c.tasks.get(running.task_id)).state == "cancelled"
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.resume_pending_revalidation is False


def _render_final(geak_pending: dict) -> tuple[list[str], list[str]]:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render

    section = render(
        {
            "final": {
                "throughput_tok_s_per_gpu": 140.0,
                "cumulative_gain_pct_validated": 0.0,
                "geak_pending": geak_pending,
            },
            "baseline": {"throughput_tok_s_per_gpu": 100.0},
        }
    )
    return list(section.key_facts), list(section.warnings)


def test_final_report_surfaces_cancelled_geak_revalidation() -> None:
    """A measured candidate dropped for a missed rebench must be visible."""
    facts, warnings = _render_final(
        {
            "status": "rebench_cancelled",
            "revalidation_error": "close_sequence",
            "self_reported_gain_pct": 12.5,
        }
    )

    blob = " ".join(facts + warnings).lower()
    assert "rebench" in blob
    assert any("close_sequence" in w or "could not" in w.lower() for w in warnings)
    # The dropped candidate must not be presented as awaiting anything.
    assert not any("awaiting" in f.lower() for f in facts)


def test_final_report_still_flags_awaiting_geak_revalidation() -> None:
    facts, warnings = _render_final(
        {"status": "awaiting_rebench", "self_reported_gain_pct": 12.5}
    )

    assert any("AWAITING" in f for f in facts)
    assert warnings


def test_legacy_placeholder_does_not_match_cycle_scoped_keys() -> None:
    """The legacy slot must not absorb a cycle-scoped rebench from another cycle."""
    from hyperloom.orchestrator.state.task_registry import Task

    def _task(key: str) -> Task:
        return Task(task_id="t1", kind="explore", state="queued", params={}, idempotency_key=key)

    assert gr.geak_rebench_tracks_pending_task(
        gr.LEGACY_GEAK_REVALIDATE_PLACEHOLDER,
        _task(gr.LEGACY_GEAK_REVALIDATE_PLACEHOLDER),
        macro_cycle=0,
    )
    assert not gr.geak_rebench_tracks_pending_task(
        gr.LEGACY_GEAK_REVALIDATE_PLACEHOLDER,
        _task("geak-revalidate-c3"),
        macro_cycle=0,
    )


@pytest.mark.asyncio
async def test_geak_rebench_survives_kernel_to_sweep_transition(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)

    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(st.macro_cycle),
        task_id="geak-rebench-survives-transition",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore-transition",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_SWEEP],
        reason="phase_transition:KERNEL_AGENT->SWEEP",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_SWEEP,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_kernel_budget_cap_transition_spares_just_enqueued_geak_rebench(
    coordinator,
    monkeypatch,
) -> None:
    """Issue #1239: the real budget-cap exit must not cancel the fresh 2b task."""
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)

    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(st.macro_cycle),
        task_id="geak-rebench-at-budget-cap",
    )
    st.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": geak_task.task_id,
    }
    st.resume_pending_revalidation = True

    # Reproduce the production branch named in #1239: KERNEL has wall-clock
    # remaining, but its absolute phase cap is exhausted immediately after the
    # GEAK handback enqueues the rebench.
    monkeypatch.setattr(ps, "phase_budget_remaining_seconds", lambda *_args, **_kwargs: 60.0)
    monkeypatch.setattr(ps, "phase_cap_exceeded", lambda *_args, **_kwargs: True)

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP
    persisted = await c.tasks.get(geak_task.task_id)
    assert persisted.state == "queued"
    assert not any(
        row.get("to") == "cancelled"
        and row.get("evidence", {}).get("reason") == "phase_transition:KERNEL_AGENT->SWEEP"
        for row in persisted.history
    )
    assert st.geak_pending["status"] == "awaiting_rebench"
    assert st.geak_pending["revalidation_task_id"] == geak_task.task_id


@pytest.mark.asyncio
async def test_duplicate_enqueue_skips_while_rebench_in_flight(coordinator, tmp_path) -> None:
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    inflight = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="cycle0-inflight",
    )

    coord = c
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run")),
    )
    try:
        await coord._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    assert len([t for t in await c.tasks.queued() if gr.is_geak_same_harness_rebench_task(t.kind, t.params)]) == 1
    assert st.geak_pending["revalidation_task_id"] == inflight.task_id
    created = await c.tasks.get(inflight.task_id)
    assert created.state == "queued"


@pytest.mark.asyncio
async def test_crash_recovery_tombstones_no_promote_result(coordinator, tmp_path) -> None:
    """AMD-AGI/Hyperloom#1240 (item 4): a result.json already adjudicated
    ``no_promote`` (measured, but did not beat current_best -- a legitimate
    rejection, not an inconclusive one) must not be re-recovered and
    re-enqueued on a later KERNEL entry, the same way ``no_material`` is
    already tombstoned. Without this gate, the stale idempotency key would
    resolve back to the already-``succeeded`` rebench row and the whitelist
    in ``_enqueue_geak_revalidation`` would overwrite the correct verdict
    with ``rebench_unavailable``.
    """
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    # The rebench for this exact candidate already ran earlier in this KERNEL
    # entry and was legitimately adjudicated no_promote.
    st.geak_result = {**result, "revalidation_status": "no_promote"}
    st.geak_pending = {}

    coord = c
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run for this test")),
    )
    try:
        await coord._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    # The tombstone must short-circuit BEFORE re-promoting / re-enqueuing: no
    # new same-harness rebench task must appear.
    assert not [t for t in await c.tasks.queued() if gr.is_geak_same_harness_rebench_task(t.kind, t.params)]


@pytest.mark.asyncio
async def test_geak_revalidation_collision_with_succeeded_task_reported_honestly(
    coordinator, tmp_path
) -> None:
    """AMD-AGI/Hyperloom#1240 (items 2/3): when the enqueue collides with a
    row that ``create_or_return_existing`` reports as pre-existing
    (``existing=True``) AND already ``succeeded``, the whitelist must not
    claim it was "settled before dispatch" -- that phrasing is backwards for
    a task that actually ran to completion, and is now reserved for the
    ``cancelled`` case only. When the collision cannot be reconciled against
    an already-recorded verdict, the error must name the task and say so
    honestly instead.
    """
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    st.geak_result = {}  # nothing recorded yet: a genuinely fresh win recovery
    st.geak_pending = {}

    coord = c
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None

    async def _fake_enqueue(*, reason: str) -> dict:
        # Simulates create_or_return_existing handing back a row that
        # already ran to completion under this idempotency key.
        return {
            "task_id": "stale-succeeded-task",
            "task_state": "succeeded",
            "existing": True,
            "mode": "geak_2b",
        }

    coord._enqueue_internal_stack_rebench = _fake_enqueue  # type: ignore[assignment]

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    assert st.geak_pending["status"] == "rebench_unavailable"
    error = str(st.geak_pending["revalidation_error"])
    assert "before dispatch" not in error
    assert "stale-succeeded-task" in error


@pytest.mark.asyncio
async def test_geak_rebench_failure_clears_pending_when_placeholder_tracked(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    placeholder = gr.geak_revalidate_idempotency_key(0)
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": placeholder}

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=placeholder,
        task_id="geak-rebench-fail",
    )
    fallback_calls: list[str] = []

    async def _fallback(*, reason: str) -> dict:
        fallback_calls.append(reason)
        return {"validated": False, "reason": "2a also failed"}

    c.writeback._validate_geak_via_geak_harness = _fallback  # type: ignore[method-assign]

    await c._handle_unpromotable_result(
        task,
        {"status": "failed", "error_class": "subprocess_nonzero", "error": "revalidation failed"},
    )

    assert not st.geak_pending
    assert fallback_calls == ["subprocess_nonzero"]
    assert st.geak_result["revalidation_status"] == "fallback_failed"


@pytest.mark.asyncio
async def test_resume_revalidation_with_empty_pending_still_promotes(coordinator) -> None:
    """Resume stack rebench with cleared geak_pending must still lift validated gain."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "geak_e2e", "tput": 116.0, "extra_server_args": "--foo"}
    st.optimization_stack = [{"action": "geak_e2e", "tput": 116.0}]
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.geak_pending = {}
    st.resume_pending_revalidation = True

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="resume-reverify",
    )

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 140.0,
            "best_variant": {"fingerprint": "abc"},
            "winners": [],
        },
        task=task,
    )

    assert st.cumulative_gain_validated == pytest.approx(40.0)
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_orphan_geak_rebench_success_does_not_promote(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.kernel_optimizer = "geak"
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
    }
    tracked = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="tracked-rebench",
    )
    orphan = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(1),
        task_id="orphan-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": tracked.task_id}

    await c._promote_to_shared_state(
        orphan.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "abc", "gain_pct": 50.0},
            "winners": [{"fingerprint": "abc", "gain_pct": 50.0}],
        },
        task=orphan,
    )

    assert st.current_best["tput"] == 100.0
    assert st.geak_pending["revalidation_task_id"] == tracked.task_id
    assert not any(e.get("action") == "geak_e2e" for e in st.optimization_stack)


@pytest.mark.asyncio
async def test_orphan_geak_rebench_inconclusive_does_not_run_2a(coordinator) -> None:
    """An untracked rebench must not trigger the GEAK-harness 2a fallback.

    A successful 2a writes the stack entry; a failed 2a clears the pending slot
    of the genuinely tracked rebench. Both bypass the orphan gate.
    """
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.kernel_optimizer = "geak"
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
    }
    tracked = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="tracked-rebench",
    )
    orphan = await c.tasks.create(
        kind="explore",
        # Fingerprint mismatch below makes the 2b decision inconclusive.
        params=_geak_rebench_params(expected_cfg_hash="expected-hash"),
        idempotency_key=gr.geak_revalidate_idempotency_key(1),
        task_id="orphan-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": tracked.task_id}

    fallback_calls: list[str] = []

    async def _record_fallback(*, reason: str) -> dict:
        fallback_calls.append(reason)
        return {"validated": False, "reason": "should not run"}

    c._validate_geak_via_geak_harness = _record_fallback  # type: ignore[assignment]

    await c._promote_to_shared_state(
        orphan.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "mismatched-hash"},
            "winners": [],
        },
        task=orphan,
    )

    assert fallback_calls == []
    assert st.geak_pending["revalidation_task_id"] == tracked.task_id
    assert st.geak_result.get("revalidation_status") != "fallback_failed"


@pytest.mark.asyncio
async def test_close_entry_settles_pending_after_phase_boundary_cancel(coordinator) -> None:
    """CLOSE must settle a dangling ``awaiting_rebench`` slot.

    The SWEEP->CLOSE transition already cancels the queued rebench, so the CLOSE
    sequencer finds nothing left to cancel and must still settle the slot.
    """
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}

    rebench = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="cancelled-by-phase-boundary",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": rebench.task_id}
    st.resume_pending_revalidation = True

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_CLOSE],
        reason="phase_transition:SWEEP->CLOSE",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_CLOSE,
            kind=kind,
            params=params,
        ),
    )
    assert rebench.task_id in cancelled

    settled = await gr.settle_dangling_geak_pending(c.tasks, st, reason="close_sequence")

    assert settled is True
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_advance_into_close_settles_pending_end_to_end(coordinator) -> None:
    """Real entry order: the transition cancels the rebench, CLOSE settles the slot."""
    c = coordinator
    st = c.shared_state
    st.phase = ps.PHASE_EXPLORE
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.set_stop_reason("target_reached")

    rebench = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="rebench-into-close",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": rebench.task_id}
    st.resume_pending_revalidation = True

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_CLOSE
    assert (await c.tasks.get(rebench.task_id)).state == "cancelled"
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_settle_waits_while_rebench_still_running(coordinator) -> None:
    """Settling is state-driven: a running rebench can still deliver, so wait."""
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}

    rebench = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="running-rebench",
    )
    await c.tasks.transition(rebench.task_id, "running")
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": rebench.task_id}

    settled = await gr.settle_dangling_geak_pending(c.tasks, st, reason="close_sequence")

    assert settled is False
    assert st.geak_pending["status"] == "awaiting_rebench"


@pytest.mark.asyncio
async def test_close_drain_cancels_running_rebench_and_settles(coordinator) -> None:
    """CLOSE writes reports only, so a running rebench is stopped, not awaited.

    Leaving it running would hold the GPU lane against the post-opt roofline and
    could still rewrite current_best after the report was generated.
    """
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}

    rebench = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="running-into-close",
    )
    await c.tasks.transition(rebench.task_id, "running")
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": rebench.task_id}
    st.resume_pending_revalidation = True

    await c._drain_geak_rebench_for_close()

    assert (await c.tasks.get(rebench.task_id)).state == "cancelled"
    assert st.geak_pending["status"] == "rebench_cancelled"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_prune_drain_leaves_running_rebench_alone(coordinator) -> None:
    """A backlog drain only clears queued work; running rebench keeps going."""
    c = coordinator
    running = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="running-during-prune",
    )
    await c.tasks.transition(running.task_id, "running")

    cancelled = await gr.cancel_geak_rebench_tasks(c.tasks, reason="prune_branch")

    assert cancelled == []
    assert (await c.tasks.get(running.task_id)).state == "running"
