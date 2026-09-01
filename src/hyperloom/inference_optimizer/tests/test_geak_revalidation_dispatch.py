# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for GEAK same-harness revalidation dispatch (L1/L2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant
from hyperloom.orchestrator.bus.message_bus import Message
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
async def test_geak_revalidate_idempotency_key_steps_past_succeeded_attempt(
    coordinator,
) -> None:
    c = coordinator
    settled = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="succeeded-cycle0-rebench",
    )
    await c.tasks.transition(settled.task_id, "running")
    await c.tasks.transition(settled.task_id, "succeeded")

    key = await gr.resolve_geak_revalidate_idempotency_key(c.tasks, 0)

    assert key == gr.geak_revalidate_idempotency_key(0, 1)


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
    }

    st.macro_cycle = 0
    first = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row0 = await c.tasks.get(str(first["task_id"]))
    assert row0.idempotency_key == "geak-revalidate-c0"
    # GEAK's revalidation is a plain explore over the stack: it names no
    # protocol of its own and takes the executor's grading as it stands.
    assert {
        "rebench_required",
        "revalidation_protocol",
        "expected_geak_ttft_ms",
        "expected_config_file_digests",
        "unverified_config_file_refs",
        "expected_current_best_cfg_hash",
        "expected_workload_signature",
    }.isdisjoint(row0.params)

    st.macro_cycle = 1
    second = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row1 = await c.tasks.get(str(second["task_id"]))
    assert row1.idempotency_key == "geak-revalidate-c1"
    assert row1.task_id != row0.task_id


@pytest.mark.asyncio
async def test_expected_cfg_hash_matches_the_variant_the_executor_builds(
    coordinator,
) -> None:
    """The pinned hash must describe the config the grid executor actually runs.

    ``accepted_config`` carries PATH so the benchmark resolves its own
    interpreter, but ``GridVariant`` drops shell/loader keys before it
    fingerprints. Hashing the unfiltered mapping made the 2b identity check miss
    on every GEAK win that shipped one, replaying a measured gain as
    inconclusive.
    """
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.macro_cycle = 0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--fp8-gemm-backend aiter",
            "env": "PATH=/opt/venv/bin:/usr/bin SGLANG_USE_AITER=1 TP=1",
        },
    }

    enqueued = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row = await c.tasks.get(str(enqueued["task_id"]))
    entry = row.params["grid"][0]
    ran = GridVariant(
        str(entry["name"]),
        str(entry["extra_args"]),
        dict(entry["extra_envs"]),
    )

    assert "PATH" not in ran.extra_envs
    assert ran.extra_envs == {"SGLANG_USE_AITER": "1", "TP": "1"}
    assert row.params["expected_cfg_hash"] == ran.fingerprint


def test_material_check_ignores_untrusted_env_names() -> None:
    """An untrusted key on one side only must not read as a config difference.

    ``accepted_config`` is a harness snapshot and carries PATH; ``current_best``
    holds the executor-filtered mapping. Comparing them raw made every echoed
    config look material, which is exactly the passthrough noise the gate exists
    to reject.
    """
    from hyperloom.orchestrator.loop.coordinator_helpers import _geak_result_has_material

    echoed = {
        "status": "ok",
        "accepted_config": {
            "flags": "--fp8-gemm-backend aiter",
            "env": "PATH=/opt/venv/bin SGLANG_USE_AITER=1",
        },
    }
    assert not _geak_result_has_material(
        echoed,
        prev_best_flags="--fp8-gemm-backend aiter",
        prev_best_envs={"SGLANG_USE_AITER": "1"},
    )
    # A real config delta still registers.
    assert _geak_result_has_material(
        echoed,
        prev_best_flags="--fp8-gemm-backend triton",
        prev_best_envs={"SGLANG_USE_AITER": "1"},
    )


@pytest.mark.asyncio
async def test_resume_stack_revalidate_promotes_material_geak_candidate(coordinator) -> None:
    """The legacy source label must not suppress a proven GEAK product."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "baseline",
        "tput": 110.0,
        "extra_server_args": "--incumbent",
        "extra_envs": {},
    }
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--candidate", "env": ""},
        "accepted_kernels": ["replacement_kernel"],
    }

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="material-geak-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 120.0,
            "best_variant": {"fingerprint": "candidate-hash"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["action"] == "geak_e2e"
    assert st.current_best["tput"] == pytest.approx(120.0)
    assert any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)
    assert st.geak_pending == {}


@pytest.mark.asyncio
async def test_resume_stack_revalidate_rejects_same_config_noise(coordinator) -> None:
    """A faster remeasure of unchanged config is not a GEAK optimization."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "explore",
        "tput": 110.0,
        "extra_server_args": "--same-config",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--same-config",
            "env": "SGLANG_USE_AITER=1",
        },
    }

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="same-config-geak-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 120.0,
            "best_variant": {"fingerprint": "same-config-hash"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["action"] == "explore"
    assert st.current_best["tput"] == pytest.approx(110.0)
    assert not any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)
    assert st.geak_result["revalidation_status"] == "no_material"
    assert st.geak_pending == {}


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


def _render_final(geak_pending: dict, *, geak: dict | None = None) -> tuple[list[str], list[str]]:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render

    section = render(
        {
            "final": {
                "throughput_tok_s_per_gpu": 140.0,
                "cumulative_gain_pct_validated": 0.0,
                "geak_pending": geak_pending,
            },
            "baseline": {"throughput_tok_s_per_gpu": 100.0},
            "geak": geak or {},
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


def test_final_report_surfaces_failed_geak_revalidation() -> None:
    facts, warnings = _render_final(
        {},
        geak={
            "revalidation_status": "failed",
            "revalidation_error": "subprocess_nonzero",
            "gain_pct": 3.2,
        },
    )

    blob = " ".join(facts + warnings).lower()
    assert "dropped" in blob
    assert "subprocess_nonzero" in blob


def test_final_report_still_flags_awaiting_geak_revalidation() -> None:
    facts, warnings = _render_final({"status": "awaiting_rebench", "self_reported_gain_pct": 12.5})

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
    """An adjudicated no_promote result must not be recovered and re-enqueued."""
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
    st.geak_result = {**result, "revalidation_status": "no_promote"}
    st.geak_pending = {}

    c.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run")),
    )
    try:
        await c._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    queued = await c.tasks.queued()
    assert not [task for task in queued if gr.is_geak_same_harness_rebench_task(task.kind, task.params)]


@pytest.mark.asyncio
async def test_geak_revalidation_collision_with_succeeded_task_reported_honestly(coordinator, tmp_path) -> None:
    """A succeeded idempotency collision must not be described as undispatched."""
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
    st.geak_result = {}
    st.geak_pending = {}
    c.phase_kernel._record_geak_kernel_journey = lambda _result: None

    async def _fake_enqueue(*, reason: str) -> dict:
        return {
            "task_id": "stale-succeeded-task",
            "task_state": "succeeded",
            "existing": True,
            "mode": "geak_2b",
        }

    c._enqueue_internal_stack_rebench = _fake_enqueue  # type: ignore[assignment]
    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert st.geak_pending["status"] == "rebench_unavailable"
    error = str(st.geak_pending["revalidation_error"])
    assert "before dispatch" not in error
    assert "stale-succeeded-task" in error


@pytest.mark.asyncio
async def test_geak_revalidation_cancelled_task_reports_cancelled_before_completion(coordinator, tmp_path) -> None:
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
    st.geak_result = {}
    st.geak_pending = {}
    c.phase_kernel._record_geak_kernel_journey = lambda _result: None

    async def _fake_enqueue(*, reason: str) -> dict:
        return {
            "task_id": "cancelled-rebench-task",
            "task_state": "cancelled",
            "existing": True,
            "mode": "geak_2b",
        }

    c._enqueue_internal_stack_rebench = _fake_enqueue  # type: ignore[assignment]
    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert st.geak_pending["status"] == "rebench_unavailable"
    error = str(st.geak_pending["revalidation_error"])
    assert "cancelled before completion" in error
    assert "succeeded" not in error


@pytest.mark.asyncio
async def test_geak_revalidation_collision_replays_persisted_succeeded_result(coordinator, tmp_path) -> None:
    """A succeeded collision replays its delegated result through normal adjudication."""
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "explore",
        "tput": 120.0,
        "extra_server_args": "--incumbent",
        "extra_envs": {},
    }
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--candidate", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    st.geak_result = {}
    st.geak_pending = {}
    c.phase_kernel._record_geak_kernel_journey = lambda _result: None

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(expected_cfg_hash="abc"),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="persisted-succeeded-task",
    )
    await c.tasks.transition(task.task_id, "running")
    await c.tasks.transition(task.task_id, "succeeded")
    await c.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": task.task_id,
                "kind": "explore",
                "state": "succeeded",
                "result": {
                    "output_throughput": 110.0,
                    "best_variant": {"fingerprint": "abc"},
                    "winners": [],
                },
                "error": None,
            },
        )
    )

    async def _fake_enqueue(*, reason: str) -> dict:
        return {
            "task_id": task.task_id,
            "task_state": "succeeded",
            "existing": True,
            "mode": "geak_2b",
        }

    c._enqueue_internal_stack_rebench = _fake_enqueue  # type: ignore[assignment]
    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert st.current_best["tput"] == pytest.approx(120.0)
    assert st.geak_result["revalidation_status"] == "no_promote"
    assert not st.geak_pending
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_geak_rebench_failure_releases_pending_and_preserves_result_diagnostic(coordinator) -> None:
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

    await c._handle_unpromotable_result(
        task,
        {"status": "failed", "error_class": "subprocess_nonzero", "error": "revalidation failed"},
    )

    assert not st.geak_pending
    assert st.geak_result["revalidation_status"] == "failed"
    assert st.geak_result["revalidation_error_class"] == "subprocess_nonzero"
    assert st.geak_result["revalidation_error"] == "revalidation failed"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_failed_geak_rebench_slot_rejects_late_success(coordinator) -> None:
    """A terminal result diagnostic rejects late success without occupying pending."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
        "revalidation_status": "failed",
        "revalidation_error": "subprocess_nonzero",
    }
    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="failed-then-late-rebench",
    )
    st.geak_pending = {}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "abc"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["tput"] == pytest.approx(100.0)
    assert not st.geak_pending
    assert st.geak_result["revalidation_status"] == "failed"
    assert not any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)


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
    st.phase = ps.PHASE_FRAMEWORK_AGENT
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
