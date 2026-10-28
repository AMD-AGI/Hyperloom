# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Role realignment and phase-aware prompts tests."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.mcp_tool_names import qualified_context_tool_name
from hyperloom.orchestrator.phases.machine_state import PHASE_NAMES
from hyperloom.common.coerce import to_unix
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
    PromptAssets,
    SessionContext,
)
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir, make_session_dir
from hyperloom.orchestrator.state.task_registry import TaskSpec


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
        session=SessionContext(framework="sglang", objective_kind="gain_pct", objective_value=10.0, max_minutes=120),
    )
    assert "PHASE CONTRACT" in text
    # Driven off the phase table: a hand-written list keeps naming a phase the build dropped, and passes on any other
    # string that happens to contain it.
    for phase in PHASE_NAMES:
        assert phase in text, f"missing phase {phase} from orchestration prompt"
    assert "phase-allowed actions" in text.lower()
    assert "policy_denied" in text.lower()


def test_orchestration_prompt_defers_rescue_moves_to_reference(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(framework="sglang", objective_kind="gain_pct", objective_value=10.0, max_minutes=120),
        assets=PromptAssets(rules_fragment_path=asset_system_prompts_dir() / "orchestration.md"),
    )

    assert "kill_task" not in text
    read_reference = qualified_context_tool_name("read_reference")
    assert f"{read_reference}(name='specialist_rescue')" in text


def test_orchestration_prompt_no_kernel_marks_kernel_skipped(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=True),
        session=SessionContext(framework="sglang", max_minutes=120),
    )
    assert "(DISABLED: --no-kernel — phase skipped)" in text


def test_orchestration_prompt_no_framework_agent_marks_skipped_and_context(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(
            framework="sglang", kernel_enabled=True, framework_agent_phase_enabled=False, max_minutes=120
        ),
    )
    assert "(DISABLED: --no-framework-agent — phase skipped)" in text
    assert "optimize_enabled : false" in text


def test_orchestration_prompt_all_enabled_session_context_true(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(
            framework="sglang", kernel_enabled=True, framework_agent_phase_enabled=True, max_minutes=120
        ),
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
    """Critic phase awareness lives in the per-phase injector, not in critic.md."""
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
    from hyperloom.orchestrator.phases import machine_state as _ps
    from hyperloom.orchestrator.roles._review_constraints import _PHASE_ORIENTATION

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
    # Pin start_ts to the same clock as now_unix so the (charge-back) budget math is well-defined; session and phase
    # both start at 1_000_000.
    s.start_ts = datetime.fromtimestamp(1_000_000.0, tz=timezone.utc).isoformat()
    phase = _ps.PHASE_FRAMEWORK_AGENT
    _ps.record_phase_transition(
        s,
        to_phase=phase,
        reason="prelude_done",
        evidence={"baseline_tput": 100},
        at=_ps.PhaseClock(ts="2026-05-19T00:00:00+00:00", ts_unix=1_000_000.0),
    )
    s.phase_budget_pct = {phase: 0.5}
    out = _ps.phase_status_summary(s, now_unix=1_000_120.0)
    assert f"phase     : {phase}" in out
    assert "entered   : 1970-01-12T13:46:40Z" in out
    assert "elapsed_sec=120" in out
    # Wiring: the rendered remaining must match the budget helper it delegates to.
    expected_rem = int(_ps.phase_budget_remaining_seconds(s, now_unix=1_000_120.0))
    assert f"remaining_sec={expected_rem}" in out
    # The merged phase's allowlist carries both arms' levers.
    assert "explore" in out and "integrate_patch" in out and "specialist" in out


def test_shared_state_phase_status_summary_no_max_minutes_marks_unlimited():
    from hyperloom.orchestrator.phases import machine_state as _ps

    s = SharedState(max_minutes=0)
    _ps.record_phase_transition(
        s,
        to_phase="FRAMEWORK_AGENT",
        reason="prelude_done",
        evidence={},
        at=_ps.PhaseClock(ts="2026-05-19T00:00:00+00:00", ts_unix=1.0),
    )
    out = _ps.phase_status_summary(s, now_unix=10.0)
    assert "unlimited run" in out.lower()


def test_shared_state_warm_start_summary_empty_when_no_recipe():
    assert SharedState().to_warm_start_summary() == ""


def _t0_warm_started(tmp_path, **put_kwargs) -> SharedState:
    """Seed a recipe, run the real T0 anchor, return the state the renderer sees.

    Hand-built fixtures are how this block came to render fields no writer
    produces — a ``raw`` blob that has never existed on ``warm_start_recipe``.
    Driving ``run_t0_anchor`` means the context under assertion is the one a
    session actually builds, for whichever shape the KB and T0 agree on.
    """
    from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity
    from hyperloom.orchestrator.knowledge.recipe_kb import (
        RecipeUpsert,
        LocalRecipeStore,
        RecipeKB,
        recipe_canonical_id,
    )
    from hyperloom.orchestrator.knowledge.recipe_kb_t0 import T0AnchorRequest, run_t0_anchor

    state = SharedState()
    state.framework_name = "vllm"
    state.framework_version = "0.11.0"
    state.precision = "fp8"
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    if put_kwargs:
        kb.put_recipe(
            RecipeUpsert(
                canonical_id=recipe_canonical_id(
                    RecipeIdentity(
                        model="kimi-k3",
                        hardware="mi355x",
                        framework_name=state.framework_name,
                        framework_version=state.framework_version,
                        precision=state.precision,
                    )
                ),
                model="kimi-k3",
                hardware="mi355x",
                framework_name=state.framework_name,
                framework_version=state.framework_version,
                precision=state.precision,
                provenance={"source": "seed", "generator": "ut"},
                **put_kwargs,
            ),
        )
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    run_t0_anchor(
        kb,
        state,
        T0AnchorRequest(workload="kimi-k3", hw="mi355x", extra_attrs={"framework_name": state.framework_name}),
        session_dir=session_dir,
    )
    return state


def test_warm_start_summary_reports_the_recipe_a_hit_actually_matched(tmp_path):
    """On a KB hit the block must not claim there is nothing to go on.

    It read a ``raw`` text field that no writer produces, so a matched recipe
    rendered as "first session for this workload/hw" — worse than saying nothing.
    """
    state = _t0_warm_started(
        tmp_path,
        best_config={
            "extra_server_args": "--attention-backend=AITER",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        },
        best_throughput=875.0,
    )
    out = state.to_warm_start_summary()
    assert "first session" not in out
    assert "tier=exact" in out
    assert "--attention-backend=AITER" in out
    # Envs render as pairs, not a Python dict repr.
    assert "VLLM_ROCM_USE_AITER=1" in out
    assert "{'" not in out


def test_warm_start_summary_attributes_a_borrowed_config_to_its_donor(tmp_path):
    """A non-exact match must not read as this session's own measurement.

    ``tier``/``confidence`` sit in the context beside the numbers; rendering the
    numbers without them turns "wrongly says cold start" into "wrongly says this
    is your warm data", which is harder to catch.
    """
    state = _t0_warm_started(
        tmp_path,
        best_config={"extra_server_args": "--tp 8"},
        best_throughput=875.0,
    )
    out = state.to_warm_start_summary()
    assert "confidence=" in out
    assert "tier=" in out


def test_warm_start_summary_pitfall_count_matches_the_rows_it_prints(tmp_path):
    """A ``pitfalls (N):`` header with fewer rows beneath it is its own small lie."""
    state = _t0_warm_started(
        tmp_path,
        best_throughput=875.0,
        pitfalls=[
            {"description": "K=8 draft overprovision at conc=1", "severity": "regress"},
            {"description": "", "severity": "regress"},
        ],
    )
    out = state.to_warm_start_summary()
    assert "K=8 draft overprovision at conc=1" in out
    assert "pitfalls (1):" in out


def test_warm_start_summary_still_says_first_session_on_a_real_first_session(tmp_path):
    """The honest case for that message, and it is ``seed_only``, not ``miss``.

    T0 seeds a row for every session, so a genuine first session comes back as a
    seed-only match rather than a miss — the state a hand-built fixture would not
    have produced.
    """
    state = _t0_warm_started(tmp_path)
    assert state.warm_start_context.get("status") == "seed_only"
    assert "first session" in state.to_warm_start_summary()


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
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
    }
    return Coordinator(session_dir, backends=backends)


@pytest.mark.asyncio
async def test_compose_prompt_emits_phase_block_for_every_role(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        for role in ("orchestration", "critic"):
            prompt = (await c.conversation.compose_prompt(role)).text
            assert "=== Phase ===" in prompt, f"{role}: phase block missing"
            assert "phase     : PRELUDE" in prompt, f"{role}: phase value missing"
            assert "allowed" in prompt, f"{role}: allowed-actions line missing"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_keeps_per_tick_values_out_of_the_context(coordinator_with_mocks):
    c = coordinator_with_mocks
    try:
        before = await c.conversation.compose_prompt("orchestration")
        c.shared_state.tick = int(c.shared_state.tick or 0) + 7
        c.shared_state.agent_last_active["orchestration"] = 1.0
        c.shared_state.phase_started_unix = 1_767_225_600.0
        after = await c.conversation.compose_prompt("orchestration")

        assert after.context == before.context
        assert f"tick={c.shared_state.tick}" in after.turn
        for heading in ("=== Phase ===", "=== Mission progress ===", "=== Tick ===", "=== Inbox for orchestration"):
            assert heading in after.turn
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_keeps_the_session_ledger_out_of_the_context(coordinator_with_mocks):
    """The ledger moves whenever an action lands, so it must not sit in the cached prefix."""
    c = coordinator_with_mocks
    try:
        before = await c.conversation.compose_prompt("orchestration")
        c.shared_state.crash_count = int(c.shared_state.crash_count or 0) + 1
        after = await c.conversation.compose_prompt("orchestration")

        assert after.context == before.context
        assert after.turn != before.turn
        assert "=== Shared session state ===" not in after.context
        assert "target_gap_pct=" not in after.context
        assert "=== Resource pools ===" in after.context
        # Stable blocks first, then the per-tick ones, ledger just above the inbox tail.
        assert after.turn.index("=== Tick ===") < after.turn.index("=== Shared session state ===")
        assert after.turn.index("=== Shared session state ===") < after.turn.index("=== Inbox for orchestration")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_keeps_the_denial_and_proposal_blocks_out_of_the_context(coordinator_with_mocks):
    """A denial lands and a proposal is benched between two ticks; the cached prefix has to survive both."""
    from hyperloom.orchestrator.actions.executors._proposal_identity import controls_of, normalize_proposal
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    c = coordinator_with_mocks
    tried = {"name": "proposal-alpha", "extra_args": "--enable-fused-rmsnorm"}
    waiting = {"name": "proposal-beta", "extra_args": "--chunked-prefill-size 8192"}
    try:
        c.shared_state.phase = PHASE_FRAMEWORK_AGENT
        c.shared_state.macro_cycle = 0
        c.shared_state.specialist_rounds = [
            {"cycle": 0, "domain": "serving_specialist", "gap_canonical_id": "", "proposal_set": [tried, waiting]}
        ]
        first = await c.conversation.compose_prompt("orchestration")
        assert "=== Untested proposals (current cycle) ===" in first.turn
        assert tried["name"] in first.turn

        c.shared_state.record_policy_denial(
            action_name="explore",
            rule="phase_gate",
            hint="wait for the bench slot",
            intent_type="DISPATCH_TASK",
        )
        fields = normalize_proposal(tried)
        tested_row = {
            "extra_server_args": fields["extra_args"],
            "extra_envs": fields["extra_envs"],
            **controls_of(fields),
        }
        c.shared_state.explore_search = {"tested": {"explore-1": dict(tested_row, outcome="REVERT")}}
        c.shared_state.tick = int(c.shared_state.tick or 0) + 1
        second = await c.conversation.compose_prompt("orchestration")

        assert second.context == first.context
        assert "=== Recent policy denials" in second.turn
        assert "=== Recent policy denials" not in second.context
        assert "=== Untested proposals (current cycle) ===" not in second.context
        # The benched proposal left the queue; the prefix above it did not move.
        queue = second.turn.split("=== Untested proposals (current cycle) ===", 1)[1]
        assert tried["name"] not in queue
        assert waiting["name"] in queue
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [True, False])
async def test_reactor_turn_hands_a_caching_backend_the_context_as_its_system_prompt(coordinator_with_mocks, cached):
    c = coordinator_with_mocks
    backend = c.backends["orchestration"]
    backend.system_prompt_cached = cached
    try:
        composed = await c.conversation.compose_prompt("orchestration")
        await c.reactor_pass("orchestration", request="ASK")
        (call,) = backend.calls

        assert (composed.context in call["system_prompt"]) is cached
        # The ledger is a per-tick block, so it rides the prompt and never the cached half.
        assert "=== Shared session state ===" in call["prompt"]
        assert "=== Shared session state ===" not in call["system_prompt"]
        assert "=== Phase ===" in call["prompt"]
        assert call["prompt"].endswith("ASK")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_conversation_row_joins_the_context_to_the_backend_prompt(coordinator_with_mocks):
    import json

    from hyperloom.inference_optimizer.session.session_paths import conversations_path
    from hyperloom.orchestrator.roles.base import BackendTurnResult

    c = coordinator_with_mocks
    try:
        result = BackendTurnResult(intents=[], raw_text="", metadata={"prompt": "TURN", "response": "R"})
        c.conversation.record_reactor_conversation("orchestration", result, context="CONTEXT")
        (row,) = [json.loads(line) for line in conversations_path(c.session_dir).read_text().splitlines()]
        assert row["prompt"] == "CONTEXT\nTURN"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_renders_warm_start_when_set(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        c.shared_state.warm_start_context = {
            "status": "hit",
            "match": {"tier": "exact", "confidence": 1.0},
            "recommended_replay": {
                "best_throughput": 2100.0,
                "extra_server_args": "--tp 8",
                "config_tier": "self",
            },
        }
        c.shared_state.save(session_dir)
        prompt = (await c.conversation.compose_prompt("orchestration")).text
        assert "=== Warm start (Recipe KB T0) ===" in prompt
        assert "tier=exact" in prompt
        assert "best_throughput=2100" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_omits_warm_start_when_empty(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        prompt = (await c.conversation.compose_prompt("orchestration")).text
        assert "=== Warm start" not in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_name", ["orchestration"])
async def test_compose_prompt_omits_specialist_health_block(
    coordinator_with_mocks,
    agent_name,
):
    """The periodic specialist block is intentionally gone (see conversation.py)."""
    c = coordinator_with_mocks
    try:
        prompt = (await c.conversation.compose_prompt(agent_name)).text
        assert "Specialist health" not in prompt
        assert "stale" not in prompt.lower()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_reports_held_resources(coordinator_with_mocks):
    """Lease expiry, lanes and GPU ids reach the planner."""
    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-resources",
                lease_ttl_sec=1800,
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        # Two lanes, deliberately out of sorted order and with different expiries: the renderer must sort the lanes
        # and report the SOONEST expiry, because that is when reclaim starts.
        for lane, holder, expires in (
            ("research_lane", "h-late", "2099-12-31T23:59:59+00:00"),
            ("gpu_research_lane", "h-soon", "2099-01-01T00:00:00+00:00"),
        ):
            c.bus.db.raw.execute(
                "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
                "acquired_at, expires_at) VALUES (?,?,?,?,?,?,?)",
                (
                    lane,
                    holder,
                    task.task_id,
                    "specialist",
                    1,
                    "2026-05-19T18:00:00+00:00",
                    expires,
                ),
            )
        for gpu_id in (3, 1):
            c.bus.db.raw.execute(
                "INSERT INTO gpu_leases(gpu_id, holder_id, task_id) VALUES (?,?,?)",
                (gpu_id, f"h-gpu{gpu_id}", task.task_id),
            )
        c.bus.db.raw.commit()

        out = c.conversation._context_running_tasks_reader()
        assert "lanes=['gpu_research_lane', 'research_lane']" in out
        assert "gpu_ids=[1, 3]" in out
        # Soonest expiry wins: reclaim starts at the FIRST lane to lapse, so reporting the latest would overstate the
        # remaining window by a year.
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
    """Heartbeat age is read from the same files the reap loop polls."""
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-heartbeat",
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        # No workspace yet: the field is omitted rather than reported as zero.
        assert "heartbeat_age_sec=" not in c.conversation._context_running_tasks_reader()

        ws = runs_dir(c.session_dir, "specialist", task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "process.log").write_text("benchmarking\n", encoding="utf-8")
        out = c.conversation._context_running_tasks_reader()
        assert "heartbeat_age_sec=" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_counts_a_dsh_session_log_like_the_reaper(
    coordinator_with_mocks,
    session_dir,
):
    """dsh prints only its final message, so its session log is what shows it working."""
    import os
    import re
    import time

    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind="specialist", params={}, idempotency_key="k-dsh-heartbeat")
        )
        await c.tasks.transition(task.task_id, "running")
        ws = runs_dir(c.session_dir, "specialist", task.task_id)
        session_log = ws / ".dsh" / "sessions" / "ws" / "session-0" / "session.v3.jsonl"
        session_log.parent.mkdir(parents=True)
        stale = time.time() - 1200
        for name in ("heartbeat.json", "process.log"):
            (ws / name).write_text("{}\n", encoding="utf-8")
            os.utime(ws / name, (stale, stale))
        session_log.write_text('{"step": 1}\n', encoding="utf-8")

        age = re.search(r"heartbeat_age_sec=(\d+)", c.conversation._context_running_tasks_reader())

        assert age is not None
        assert int(age.group(1)) < 60
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_skips_heartbeat_for_non_specialist(
    coordinator_with_mocks,
    session_dir,
):
    """The kind guard holds even when a same-named workspace exists."""
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params={},
                idempotency_key="k-explore-hb",
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        # Plant the liveness file the specialist path would have read.
        ws = runs_dir(c.session_dir, "specialist", task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "heartbeat.json").write_text("{}", encoding="utf-8")

        out = c.conversation._context_running_tasks_reader()
        assert task.task_id in out
        assert "kind='explore'" in out
        assert "heartbeat_age_sec=" not in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_running_tasks_reader_reports_in_flight_task(coordinator_with_mocks):
    """A running task is visible with its elapsed time and idempotency key."""
    c = coordinator_with_mocks
    try:
        assert "no tasks in flight" in c.conversation._context_running_tasks_reader()
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={"domain": "serving_specialist", "gap_canonical_id": "gap.x"},
                idempotency_key="k-running-1",
                lease_ttl_sec=1800,
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        out = c.conversation._context_running_tasks_reader()
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
@pytest.mark.parametrize(
    ("phase", "kinds", "order_note"),
    [
        ("FRAMEWORK_AGENT", ("explore", "integrate_patch"), "integrate_patch dispatches before any queued explore"),
        ("FRAMEWORK_AGENT", ("explore",), None),
        ("SWEEP", ("explore", "integrate_patch"), None),
    ],
)
async def test_running_tasks_reader_states_the_backlog_dispatch_order(coordinator_with_mocks, phase, kinds, order_note):
    """A phase that dispatches one kind first says so, since a later-queued task of that kind still runs ahead."""
    c = coordinator_with_mocks
    try:
        c.shared_state.phase = phase
        for i, kind in enumerate(kinds):
            await c.tasks.create_or_return_existing(TaskSpec(kind=kind, params={}, idempotency_key=f"k-backlog-{i}"))
        out = c.conversation._context_running_tasks_reader()
        assert "queued backlog: " in out
        assert "dispatches before" in out if order_note else "dispatches before" not in out
        if order_note:
            assert order_note in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_grows_ttl_and_lane_rows(coordinator_with_mocks):
    """extend_lease pushes out both the task TTL and every lane row it holds."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-1",
                requires_lanes=["research_lane"],
                lease_ttl_sec=1800,
            ),
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

        await c.router.handle_intent(
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
        # The lane must expire at the REMAINING budget (cumulative TTL minus the elapsed run time), not at now + the
        # full cumulative TTL.
        expires_in = to_unix(str(rows[0]["expires_at"]), 0.0) - time.time()
        assert expires_in <= 2400
        started = to_unix(updated.updated_at, 0.0)
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
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-elapsed",
                requires_lanes=["research_lane"],
                lease_ttl_sec=1800,
            ),
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

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        expires_in = to_unix(str(rows[0]["expires_at"]), 0.0) - time.time()
        # 1800 + 600 cumulative, 1000 already spent -> ~1400s left, not 2400.
        assert 1300 < expires_in < 1450
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_late_grant_keeps_new_increment_for_lanes(coordinator_with_mocks):
    """A late extension must not reduce newly granted lane time to one second."""
    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-late",
                requires_lanes=["research_lane"],
                lease_ttl_sec=1800,
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        await c.locks.acquire_many(
            ["research_lane"],
            holder_id=f"h-{task.task_id}",
            task_id=task.task_id,
            action="specialist",
            ttl_sec=1800,
        )
        started_iso = datetime.fromtimestamp(time.time() - 3000, tz=timezone.utc).isoformat()
        await c.db.execute("UPDATE tasks SET updated_at=? WHERE task_id=?", (started_iso, task.task_id))

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        expires_in = to_unix(str(rows[0]["expires_at"]), 0.0) - time.time()
        assert 550 < expires_in < 650
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_grants_live_subprocess_extension(coordinator_with_mocks):
    """The handler must hand the grant to the reaper, not just move DB rows."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.specialists import wall_budget as _sub

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-live",
                lease_ttl_sec=1800,
            ),
        )
        await c.tasks.transition(task.task_id, "running")
        _sub.clear_wall_budget_extension(task.task_id)

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )
        assert _sub.wall_budget_extension(task.task_id) == 600.0

        # Repeated extensions accumulate on the live deadline.
        await c.router.handle_intent(
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
async def test_extend_lease_wall_budget_grant_failure_is_recorded_as_an_intent_exception(coordinator_with_mocks):
    """A wall-budget failure retains DB changes and reaches the intent boundary's exception record."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.loop import intent_router as _sub

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-grant-fail",
                lease_ttl_sec=1800,
            ),
        )
        await c.tasks.transition(task.task_id, "running")

        def _boom(*_a, **_kw):
            raise RuntimeError("registry unavailable")

        original = _sub.grant_wall_budget_extension
        _sub.grant_wall_budget_extension = _boom  # type: ignore[assignment]
        recorded: list[dict] = []
        original_record = c.bus.record_observation

        async def _capture(agent, topic, payload):
            recorded.append(dict(payload))
            return await original_record(agent, topic, payload)

        c.bus.record_observation = _capture  # type: ignore[method-assign]
        try:
            await c.router.handle_intent(
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
        failures = [r for r in recorded if r.get("kind") == "handle_intent_exception"]
        assert [(r["intent_type"], r["error"]) for r in failures] == [
            ("extend_lease", "RuntimeError: registry unavailable")
        ]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_enablement_rearm_failure_on_critic_reject_reaches_the_router(coordinator_with_mocks, monkeypatch):
    from hyperloom.orchestrator.loop.proposals import PendingProposal
    from hyperloom.orchestrator.loop.verdicts import CriticRuling

    c = coordinator_with_mocks
    try:

        async def _rearm_broke(_result):
            raise RuntimeError("enablement lane unavailable")

        monkeypatch.setattr(c.enablement_lane, "maybe_rearm_enablement", _rearm_broke)
        pending = PendingProposal(
            proposal_msg_id="m-reject",
            from_agent="orchestration",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload={"params": {"enablement": True}},
        )
        with pytest.raises(RuntimeError, match="enablement lane unavailable"):
            await c.router._apply_verdict_outcome(
                pending,
                ruling=CriticRuling(verdict="reject", reasoning="regresses accuracy"),
                pa_params={"enablement": True},
                sid_candidate="t-enable",
            )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_extend_lease_survives_unreadable_running_age(coordinator_with_mocks):
    """A failed age lookup must degrade to the full TTL, not abort the extension."""
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    c = coordinator_with_mocks
    try:
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-age-fail",
                requires_lanes=["research_lane"],
                lease_ttl_sec=1800,
            ),
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

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": task.task_id, "extra_sec": 600},
            ),
        )

        c.tasks.get = real_get  # type: ignore[method-assign]
        # Lane still moved — falling back to the full TTL is the safe direction (a lease that outlives the task beats
        # one reaped mid-run).
        rows = await c.db.fetchall("SELECT expires_at FROM leases WHERE task_id=?", (task.task_id,))
        assert to_unix(str(rows[0]["expires_at"]), 0.0) > time.time()
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
        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params={},
                idempotency_key="k-extend-2",
                lease_ttl_sec=1800,
            ),
        )
        await c.router.handle_intent(
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
async def test_prune_branch_on_a_catalogue_family_cancels_every_member_action(coordinator_with_mocks):
    """The ``family=`` tag the prompt renders reaches the queued rows of every action carrying it."""
    c = coordinator_with_mocks
    try:
        members = sorted(name for name, meta in ACTION_CATALOGUE.items() if meta.family == "prep")
        assert len(members) > 1
        queued = [
            (
                await c.tasks.create_or_return_existing(
                    TaskSpec(kind=kind, params={}, idempotency_key=f"k-prune-{kind}")
                )
            )[0]
            for kind in members
        ]
        bystander, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind="explore", params={}, idempotency_key="k-prune-bystander")
        )
        running, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind=members[0], params={}, idempotency_key="k-prune-running")
        )
        await c.tasks.transition(running.task_id, "running")

        await c.router.handle_intent(
            "orchestration",
            Intent(type=IntentType.PRUNE_BRANCH, payload={"family": "prep", "reason": "answered"}),
        )

        for task in queued:
            assert (await c.tasks.get(task.task_id)).state == "cancelled"
        assert (await c.tasks.get(bystander.task_id)).state == "queued"
        assert (await c.tasks.get(running.task_id)).state == "running"
        assert c.shared_state.is_pruned("prep")

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION, payload={"action_name": "target_analysis", "predicted_gain_pct": 5.0}
            ),
        )
        obs = await c.bus.tail(topic="observation")
        assert any(m.payload.get("kind") == "proposal_pruned_advisory" for m in obs)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_prune_branch_queued_scope_on_an_action_name_drains_only_that_kind(coordinator_with_mocks):
    """A single action name still works, and ``scope='queued'`` leaves the family usable."""
    c = coordinator_with_mocks
    try:
        target, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind="profile", params={}, idempotency_key="k-drain-profile")
        )
        sibling, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind="roofline", params={}, idempotency_key="k-drain-roofline")
        )

        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"family": "profile", "reason": "answered", "scope": "queued"},
            ),
        )

        assert (await c.tasks.get(target.task_id)).state == "cancelled"
        assert (await c.tasks.get(sibling.task_id)).state == "queued"
        assert not c.shared_state.is_pruned("profile")
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
        await c.router.handle_intent(
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

        await c.router.handle_intent(
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
async def test_send_message_to_a_list_writes_one_message_per_recipient(coordinator_with_mocks):
    """A ``to`` array is fanned out: one events row per recipient, and each specialist's copy in its inbox.

    Before the gate's check the raw list reached the events INSERT as ``to_agent`` and
    sqlite3.ProgrammingError escaped handle_intent, marking the running phase event failed.
    """
    import json as _json

    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    c = coordinator_with_mocks
    body = "redirecting both duplicates at once"
    try:
        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={
                    "to": ["specialist:dup-a", "specialist:dup-b", "critic", "specialist:dup-a"],
                    "topic": "observation",
                    "body_md": body,
                },
            ),
        )
        for name in ("specialist:dup-a", "specialist:dup-b", "critic"):
            mine = [
                m for m in c.bus.inbox_context_sync(name) if m.to_agent == name and m.payload.get("body_md") == body
            ]
            assert len(mine) == 1, name
        for task_id in ("dup-a", "dup-b"):
            inbox = runs_dir(c.session_dir, "specialist", task_id) / "inbox.json"
            entries = _json.loads(inbox.read_text(encoding="utf-8"))
            assert [e["body"]["body_md"] for e in entries] == [body]
        assert not any(m.payload.get("kind") == "policy_denied" for m in c.bus.inbox_context_sync("orchestration"))
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_send_message_to_a_list_with_an_empty_name_is_denied_not_crashed(coordinator_with_mocks):
    """A list holding an empty name is refused by the gate instead of faulting the phase event."""
    c = coordinator_with_mocks
    try:
        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={"to": ["critic", ""], "topic": "observation", "body_md": "one name is blank"},
            ),
        )
        denials = c.bus.inbox_context_sync("orchestration")
        kinds = [m.payload.get("kind") for m in denials if m.payload.get("kind") == "policy_denied"]
        assert kinds, "expected a policy_denied observation for the malformed recipient"
    finally:
        await c.stop()
