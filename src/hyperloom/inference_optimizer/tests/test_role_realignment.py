# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Role realignment and phase-aware prompts tests."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.phases.machine_state import PHASE_NAMES
from hyperloom.orchestrator.loop.coordinator_helpers import _parse_iso_unix
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir, make_session_dir


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture
def registry() -> dict:
    return ACTION_CATALOGUE


# Static system prompts carry phase semantics
def test_orchestration_prompt_includes_phase_contract(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
    )
    assert "PHASE CONTRACT" in text
    # Driven off the phase table: a hand-written list keeps naming a phase the
    # build dropped, and passes on any other string that happens to contain it.
    for phase in PHASE_NAMES:
        assert phase in text, f"missing phase {phase} from orchestration prompt"
    assert "phase-allowed actions" in text.lower()
    assert "policy_denied" in text.lower()


def test_orchestration_prompt_defers_rescue_moves_to_reference(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )

    assert "kill_task" not in text
    assert "read_reference('specialist_rescue')" in text


def test_orchestration_prompt_no_kernel_marks_kernel_skipped(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=True),
        framework="sglang",
        max_minutes=120,
    )
    assert "(DISABLED: --no-kernel — phase skipped)" in text


def test_orchestration_prompt_no_framework_agent_marks_skipped_and_context(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=False,
        max_minutes=120,
    )
    assert "(DISABLED: --no-framework-agent — phase skipped)" in text
    assert "optimize_enabled : false" in text


def test_orchestration_prompt_all_enabled_session_context_true(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=True,
        max_minutes=120,
    )
    assert "optimize_enabled : true" in text
    assert "kernel_enabled   : true" in text
    assert "(DISABLED:" not in text


def test_orchestration_md_carries_phase_awareness():
    """The orchestration rules fragment names the phase chain it plans against."""
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

    body = (asset_system_prompts_dir() / "orchestration.md").read_text(encoding="utf-8")
    assert "Phase awareness" in body
    assert "PRELUDE" in body or "PHASE_PRELUDE" in body
    assert "FRAMEWORK_AGENT" in body
    assert "KERNEL_AGENT" in body


def test_critic_phase_orientation_is_delivered_not_inlined():
    """Critic phase awareness lives in the per-phase injector, not in critic.md.

    ``critic.md`` keeps the framing (how to treat a phase question) and points
    at the delivered fields; the per-phase contracts are injected one at a time
    so the Critic never reads five phases' rules to use one.
    """
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
    from hyperloom.orchestrator.phases import machine_state as _ps
    from hyperloom.orchestrator.roles.critic_agent import _PHASE_ORIENTATION

    body = (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")
    assert "Phase-specific rules" in body
    assert "judge_bundle.phase" in body
    assert "phase_orientation" in body
    # The five-phase bullet list must not come back as always-on text.
    assert "Per-phase orientation:" not in body

    assert set(_PHASE_ORIENTATION) == set(_ps.PHASE_NAMES)


# SharedState renderers
def test_shared_state_phase_status_summary_renders_compact_block():
    from datetime import datetime, timezone

    from hyperloom.orchestrator.phases import machine_state as _ps

    s = SharedState(max_minutes=60)
    # Pin start_ts to the same clock as now_unix so the (charge-back) budget math
    # is well-defined; session and phase both start at 1_000_000.
    s.start_ts = datetime.fromtimestamp(1_000_000.0, tz=timezone.utc).isoformat()
    phase = _ps.PHASE_FRAMEWORK_AGENT
    s.record_phase_transition(
        to_phase=phase,
        reason="prelude_done",
        evidence={"baseline_tput": 100},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1_000_000.0,
    )
    out = s.to_phase_status_summary(budget_pct={phase: 0.5}, now_unix=1_000_120.0)
    assert f"phase     : {phase}" in out
    assert "entered" in out
    assert "elapsed_sec=120" in out
    # Wiring: the rendered remaining must match the budget helper it delegates to.
    expected_rem = int(_ps.phase_budget_remaining_seconds(s, budget_pct={phase: 0.5}, now_unix=1_000_120.0))
    assert f"remaining_sec={expected_rem}" in out
    # The merged phase's allowlist carries both arms' levers.
    assert "explore" in out and "integrate_patch" in out and "specialist" in out


def test_shared_state_phase_status_summary_no_max_minutes_marks_unlimited():
    s = SharedState(max_minutes=0)
    s.record_phase_transition(
        to_phase="FRAMEWORK_AGENT",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    out = s.to_phase_status_summary(now_unix=10.0)
    assert "unlimited run" in out.lower()


def test_shared_state_phase_budget_telemetry_reports_per_phase_elapsed():
    s = SharedState(max_minutes=60)
    s.record_phase_transition(
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1_000_000.0,
    )
    s.record_phase_transition(
        to_phase="FRAMEWORK_AGENT",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:01:00+00:00",
        ts_unix=1_000_060.0,
    )
    out = s.to_phase_budget_telemetry(now_unix=1_000_300.0)
    # PRELUDE: 60s elapsed, cap 108s (3% of 3600s), used 56%.
    assert "PRELUDE: elapsed=60s" in out
    # FRAMEWORK_AGENT: 240s elapsed (300-60).
    assert "FRAMEWORK_AGENT: elapsed=240s" in out
    # Both lines present.
    assert out.count("elapsed=") == 2


def test_shared_state_warm_start_summary_empty_when_no_recipe():
    assert SharedState().to_warm_start_summary() == ""


def test_shared_state_warm_start_summary_renders_recipe_and_pitfalls():
    s = SharedState()
    s.warm_start_recipe = {
        "workload": "deepseek-r1",
        "hw": "mi300x",
        "raw": "recipe_id=42 stack=sglang/0.4.10\nbest_config={'foo':'bar'}\nwhat_worked=[A, B]",
    }
    s.warm_start_pitfalls = [
        {"raw": "OOM on fp8 expert_dtype — switch to fp4"},
        {"raw": "TP=8 + ISL>=8k causes nccl hang"},
    ]
    out = s.to_warm_start_summary()
    assert "workload=deepseek-r1" in out
    assert "hw=mi300x" in out
    assert "recipe_id=42" in out
    assert "pitfalls (2):" in out
    assert "OOM on fp8" in out


# Coordinator per-tick prompt assembly
def _silent_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


@pytest.fixture
def coordinator_with_mocks(session_dir):
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    return Coordinator(session_dir, backends=backends)


@pytest.mark.asyncio
async def test_compose_prompt_emits_phase_block_for_every_role(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        for role in ("orchestration", "critic", "robustness"):
            prompt = await c._compose_prompt(role)
            assert "=== Phase ===" in prompt, f"{role}: phase block missing"
            assert "phase     : PRELUDE" in prompt, f"{role}: phase value missing"
            assert "allowed" in prompt, f"{role}: allowed-actions line missing"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_renders_warm_start_when_set(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        c.shared_state.warm_start_recipe = {
            "workload": "qwen3-8b",
            "hw": "mi325x",
            "raw": "recipe_id=99 best_throughput=2100",
        }
        c.shared_state.save(session_dir)
        prompt = await c._compose_prompt("orchestration")
        assert "=== Warm start (Recipe KB T0) ===" in prompt
        assert "workload=qwen3-8b" in prompt
        assert "recipe_id=99" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_omits_warm_start_when_empty(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        prompt = await c._compose_prompt("orchestration")
        assert "=== Warm start" not in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_name", ["robustness", "orchestration"])
async def test_compose_prompt_omits_specialist_health_block(
    coordinator_with_mocks,
    agent_name,
):
    """The periodic specialist block is intentionally gone (see conversation.py)."""
    c = coordinator_with_mocks
    try:
        prompt = await c._compose_prompt(agent_name)
        assert "Specialist health" not in prompt
        assert "stale" not in prompt.lower()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_robustness_includes_budget_telemetry(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        # Force PRELUDE -> FRAMEWORK_AGENT so there is a segment to report.
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        prompt = await c._compose_prompt("robustness")
        assert "=== Phase budget telemetry ===" in prompt
        assert "PRELUDE: elapsed=" in prompt
        assert "FRAMEWORK_AGENT: elapsed=" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_reports_held_resources(coordinator_with_mocks):
    """Lease expiry, lanes and GPU ids reach the planner.

    These four fields are the whole point of the on-demand path: the prompt
    tells the planner to weigh "what is queued behind the lane or GPUs it
    holds" and to extend a lease that is near expiry, so each has to survive
    the join from ``leases`` / ``gpu_leases`` into the rendered line.
    """
    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-resources",
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        # Two lanes, deliberately out of sorted order and with different
        # expiries: the renderer must sort the lanes and report the SOONEST
        # expiry, because that is when reclaim starts.
        for lane, holder, expires in (
            ("research_lane", "h-late", "2099-12-31T23:59:59+00:00"),
            ("gpu_research_lane", "h-soon", "2099-01-01T00:00:00+00:00"),
        ):
            c.bus.db.raw.execute(
                "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
                "acquired_at, expires_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    lane,
                    holder,
                    task.task_id,
                    "specialist",
                    1,
                    "2026-05-19T18:00:00+00:00",
                    expires,
                    "2026-05-19T18:00:00+00:00",
                ),
            )
        for gpu_id in (3, 1):
            c.bus.db.raw.execute(
                "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, "
                "expires_at, heartbeat_at) VALUES (?,?,?,?,?,?)",
                (
                    gpu_id,
                    f"h-gpu{gpu_id}",
                    task.task_id,
                    "2026-05-19T18:00:00+00:00",
                    "2099-12-31T23:59:59+00:00",
                    "2026-05-19T18:00:00+00:00",
                ),
            )
        c.bus.db.raw.commit()

        out = c._context_running_tasks_reader()
        assert "lanes=['gpu_research_lane', 'research_lane']" in out
        assert "gpu_ids=[1, 3]" in out
        # Soonest expiry wins: reclaim starts at the FIRST lane to lapse, so
        # reporting the latest would overstate the remaining window by a year.
        reported = int(out.split("lease_expires_in_sec=")[1].split()[0])
        now = datetime.now(timezone.utc)
        soonest = int((datetime.fromisoformat("2099-01-01T00:00:00+00:00") - now).total_seconds())
        latest = int((datetime.fromisoformat("2099-12-31T23:59:59+00:00") - now).total_seconds())
        assert abs(reported - soonest) <= 5, f"expected soonest {soonest}, got {reported}"
        assert abs(reported - latest) > 1000
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_reports_heartbeat_age(
    coordinator_with_mocks,
    session_dir,
):
    """Heartbeat age is read from the same files the reap loop polls.

    ``process.log`` counts as proof of life alongside ``heartbeat.json`` — a
    specialist mid-benchmark can go minutes without restamping the heartbeat
    while its log grows, and treating that as silence would invite a spurious
    kill.
    """
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-heartbeat",
        )
        await c.tasks.transition(task.task_id, "running")
        # No workspace yet: the field is omitted rather than reported as zero.
        assert "heartbeat_age_sec=" not in c._context_running_tasks_reader()

        ws = runs_dir(c.session_dir, "specialist", task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "process.log").write_text("benchmarking\n", encoding="utf-8")
        out = c._context_running_tasks_reader()
        assert "heartbeat_age_sec=" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_skips_heartbeat_for_non_specialist(
    coordinator_with_mocks,
    session_dir,
):
    """The kind guard holds even when a same-named workspace exists.

    ``runs_dir`` is keyed on task_id, so a non-specialist whose id collides
    with a specialist workspace would otherwise inherit a heartbeat that
    describes someone else's process.
    """
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="explore",
            params={},
            idempotency_key="k-explore-hb",
        )
        await c.tasks.transition(task.task_id, "running")
        # Plant the liveness file the specialist path would have read.
        ws = runs_dir(c.session_dir, "specialist", task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "heartbeat.json").write_text("{}", encoding="utf-8")

        out = c._context_running_tasks_reader()
        assert task.task_id in out
        assert "kind='explore'" in out
        assert "heartbeat_age_sec=" not in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_survives_db_failure(coordinator_with_mocks):
    """A read failure degrades to a message, never an exception.

    This reader backs a context tool the planner calls on its own turn; an
    exception here would surface as an SDK stream failure and discard every
    intent already collected in that turn.
    """
    c = coordinator_with_mocks
    try:

        def _boom(*_a, **_k):
            raise RuntimeError("db gone")

        c.bus.db.fetchall_sync = _boom
        out = c._context_running_tasks_reader()
        assert "running tasks unavailable" in out
        assert "db gone" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_reports_in_flight_task(coordinator_with_mocks):
    """A running task is visible with its elapsed time and idempotency key."""
    c = coordinator_with_mocks
    try:
        assert "no tasks in flight" in c._context_running_tasks_reader()
        task = await c.tasks.create(
            kind="specialist",
            params={"domain": "serving_specialist", "gap_canonical_id": "gap.x"},
            idempotency_key="k-running-1",
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        out = c._context_running_tasks_reader()
        assert "=== Tasks in flight ===" in out
        assert task.task_id in out
        assert "kind='specialist'" in out
        assert "domain='serving_specialist'" in out
        assert "gap='gap.x'" in out
        assert "idempotency_key='k-running-1'" in out
        assert "lease_ttl_sec=1800" in out
        assert "running_sec=" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_grows_ttl_and_lane_rows(coordinator_with_mocks):
    """extend_lease pushes out both the task TTL and every lane row it holds."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-1",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        lease = await c.locks.acquire_many(
            ["research_lane"],
            holder_id=f"h-{task.task_id}",
            task_id=task.task_id,
            action="specialist",
            ttl_sec=1800,
        )
        assert lease is not None
        before = await c.tasks.get(task.task_id)

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600, "reason": "close to done"},
            ),
        )

        updated = await c.tasks.get(task.task_id)
        assert updated.lease_ttl_sec == 2400
        # updated_at marks when the task started running; extending must not move it.
        assert updated.updated_at == before.updated_at
        rows = await c.db.fetchall("SELECT lane, expires_at FROM leases WHERE task_id=?", (task.task_id,))
        assert [r["lane"] for r in rows] == ["research_lane"]
        # The lane must expire at the REMAINING budget (cumulative TTL minus the
        # elapsed run time), not at now + the full cumulative TTL.
        expires_in = _parse_iso_unix(str(rows[0]["expires_at"])) - time.time()
        assert expires_in <= 2400
        started = _parse_iso_unix(updated.updated_at)
        remaining_budget = 2400 - (time.time() - started)
        assert abs(expires_in - remaining_budget) < 5
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_does_not_regrant_elapsed_time(coordinator_with_mocks):
    """A task that already burned most of its TTL only gets the remainder back."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-elapsed",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        await c.locks.acquire_many(
            ["research_lane"],
            holder_id=f"h-{task.task_id}",
            task_id=task.task_id,
            action="specialist",
            ttl_sec=1800,
        )
        # Backdate the running mark so the task looks 1000s old.
        started_iso = datetime.fromtimestamp(time.time() - 1000, tz=timezone.utc).isoformat()
        await c.db.execute(
            "UPDATE tasks SET updated_at=? WHERE task_id=?",
            (started_iso, task.task_id),
        )

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        expires_in = _parse_iso_unix(str(rows[0]["expires_at"])) - time.time()
        # 1800 + 600 cumulative, 1000 already spent -> ~1400s left, not 2400.
        assert 1300 < expires_in < 1450
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_late_grant_keeps_new_increment_for_lanes_and_gpus(coordinator_with_mocks):
    """A late extension must not reduce newly granted resource time to one second."""
    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={"needs_gpu": True},
            idempotency_key="k-extend-late",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        await c.locks.acquire_many(
            ["research_lane"],
            holder_id=f"h-{task.task_id}",
            task_id=task.task_id,
            action="specialist",
            ttl_sec=1800,
        )
        await c.db.execute(
            "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, expires_at, heartbeat_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                0,
                f"h-gpu-{task.task_id}",
                task.task_id,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:30:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        started_iso = datetime.fromtimestamp(time.time() - 3000, tz=timezone.utc).isoformat()
        await c.db.execute("UPDATE tasks SET updated_at=? WHERE task_id=?", (started_iso, task.task_id))

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        lane_rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        gpu_rows = await c.db.fetchall("SELECT expires_at FROM gpu_leases WHERE task_id=?", (task.task_id,))
        for row in [*lane_rows, *gpu_rows]:
            expires_in = _parse_iso_unix(str(row["expires_at"])) - time.time()
            assert 550 < expires_in < 650
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_reports_degraded_when_gpu_refresh_fails(coordinator_with_mocks):
    """A swallowed GPU-refresh failure must not read as a clean extension."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={"needs_gpu": True},
            idempotency_key="k-extend-gpu-fail",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")

        async def _boom(*_a, **_kw):
            raise RuntimeError("gpu pool down")

        c.gpu_specialist_pool.extend = _boom  # type: ignore[method-assign]

        recorded: list[dict] = []
        original = c._record_observation

        async def _capture(agent, topic, payload):
            recorded.append(dict(payload))
            return await original(agent, topic, payload)

        c._record_observation = _capture  # type: ignore[method-assign]

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        kinds = [r.get("kind") for r in recorded]
        assert "extend_lease_degraded" in kinds
        assert "extend_lease" not in kinds
        degraded = next(r for r in recorded if r.get("kind") == "extend_lease_degraded")
        assert "gpu pool down" in degraded["gpu_refresh_error"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_grants_live_subprocess_extension(coordinator_with_mocks):
    """The handler must hand the grant to the reaper, not just move DB rows.

    The reap-loop side of this (that the deadline actually moves) is covered in
    test_specialist_subprocess.py; here we pin the wiring.
    """
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.specialists import subprocess_ as _sub

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-live",
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        _sub.clear_wall_budget_extension(task.task_id)

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )
        assert _sub.wall_budget_extension(task.task_id) == 600.0

        # Repeated extensions accumulate on the live deadline.
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 300},
            ),
        )
        assert _sub.wall_budget_extension(task.task_id) == 900.0
        _sub.clear_wall_budget_extension(task.task_id)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_survives_wall_budget_grant_failure(coordinator_with_mocks):
    """A wall-budget failure retains DB changes but reports a degraded extension."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.specialists import subprocess_ as _sub

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-grant-fail",
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")

        def _boom(*_a, **_kw):
            raise RuntimeError("registry unavailable")

        original = _sub.grant_wall_budget_extension
        _sub.grant_wall_budget_extension = _boom  # type: ignore[assignment]
        recorded: list[dict] = []
        original_record = c._record_observation

        async def _capture(agent, topic, payload):
            recorded.append(dict(payload))
            return await original_record(agent, topic, payload)

        c._record_observation = _capture  # type: ignore[method-assign]
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.EXTEND_LEASE,
                    payload={"task_id": task.task_id, "extra_sec": 600},
                ),
            )
        finally:
            _sub.grant_wall_budget_extension = original  # type: ignore[assignment]

        # The DB-side extension still stands.
        updated = await c.tasks.get(task.task_id)
        assert updated.lease_ttl_sec == 2400
        degraded = next(r for r in recorded if r.get("kind") == "extend_lease_degraded")
        assert "registry unavailable" in degraded["wall_budget_extension_error"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_survives_unreadable_running_age(coordinator_with_mocks):
    """A failed age lookup must degrade to the full TTL, not abort the extension."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-age-fail",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        await c.locks.acquire_many(
            ["research_lane"],
            holder_id=f"h-{task.task_id}",
            task_id=task.task_id,
            action="specialist",
            ttl_sec=1800,
        )

        # extend_lease() itself still works; only the follow-up age read fails.
        real_get = c.tasks.get
        calls = {"n": 0}

        async def _get(task_id):
            calls["n"] += 1
            if calls["n"] > 0 and task_id == task.task_id:
                raise RuntimeError("row vanished")
            return await real_get(task_id)

        c.tasks.get = _get  # type: ignore[method-assign]

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        c.tasks.get = real_get  # type: ignore[method-assign]
        # Lane still moved — falling back to the full TTL is the safe direction
        # (a lease that outlives the task beats one reaped mid-run).
        rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        assert _parse_iso_unix(str(rows[0]["expires_at"])) > time.time()
        updated = await c.tasks.get(task.task_id)
        assert updated.lease_ttl_sec == 2400
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_rejects_non_running_task(coordinator_with_mocks):
    """A queued task cannot be extended; the rejection is recorded, not raised."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={},
            idempotency_key="k-extend-2",
            lease_ttl_sec=1800,
        )
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )
        updated = await c.tasks.get(task.task_id)
        assert updated.lease_ttl_sec == 1800
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_send_message_to_specialist_writes_inbox(coordinator_with_mocks):
    """A message addressed to a running specialist lands in its workspace inbox."""
    import json as _json

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={
                    "to": "specialist:task-steer",
                    "topic": "observation",
                    "body_md": "the mandate changed; measure prefill instead",
                },
            ),
        )
        inbox = runs_dir(c.session_dir, "specialist", "task-steer") / "inbox.json"
        assert inbox.exists()
        entries = _json.loads(inbox.read_text(encoding="utf-8"))
        assert len(entries) == 1
        assert entries[0]["from"] == "orchestration"
        assert "prefill" in entries[0]["body"]["body_md"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_send_message_to_specialist_prefers_worktree_inbox(coordinator_with_mocks):
    """The production specialist has a worktree; the prompt advertises that path."""
    import json as _json

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        workspace = runs_dir(c.session_dir, "specialist", "task-wt")
        (workspace / "worktree").mkdir(parents=True, exist_ok=True)

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={
                    "to": "specialist:task-wt",
                    "topic": "observation",
                    "body_md": "steer toward decode",
                },
            ),
        )

        worktree_inbox = workspace / "worktree" / "inbox.json"
        assert worktree_inbox.exists()
        assert not (workspace / "inbox.json").exists()
        entries = _json.loads(worktree_inbox.read_text(encoding="utf-8"))
        assert entries[0]["body"]["body_md"] == "steer toward decode"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_also_pushes_gpu_rows(coordinator_with_mocks):
    """The GPU lease is extended with the lane rows so the TTL ordering holds."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task = await c.tasks.create(
            kind="specialist",
            params={"needs_gpu": True},
            idempotency_key="k-extend-gpu",
            requires_lanes=["research_lane"],
            lease_ttl_sec=1800,
        )
        await c.tasks.transition(task.task_id, "running")
        await c.db.execute(
            "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, expires_at, heartbeat_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                0,
                "h-gpu",
                task.task_id,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:30:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        rows = await c.db.fetchall("SELECT expires_at FROM gpu_leases WHERE task_id=?", (task.task_id,))
        assert rows and rows[0]["expires_at"] > "2026-01-01T00:30:00+00:00"
    finally:
        await c.stop()
