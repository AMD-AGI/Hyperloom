"""Tests for the M7 Conductor wiring — IMPL-CHECKLIST §7.5/7.10/7.11/7.16/7.18/7.19/7.23.

Covers:

    - Persona file is loaded into the prompt header
    - IronRules block is rendered in the prompt
    - ``_dispatcher_loop`` actually drains queued ``delegate`` tasks via
      ``SubAgentRunner`` (subset of §7.10)
    - 30-min checkpoint cadence wires through ``Checkpoint.create``
    - ``ephemeral_rca_via_critic`` extracts the rca_finding payload
    - ``_open_parliament`` returns approved / rejected based on votes
    - ``TokenBudgetMeter.record`` / ``should_throttle`` rules
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.backends.mock import ScriptStep
from inference_optimizer.orchestrator.conductor import (
    Conductor,
    StopReason,
    TokenBudgetMeter,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.storage.connection import SqliteConnection


PACKAGE_ACTIONS_DIR = asset_actions_dir()


# ---------------------------------------------------------------------------
# Persona index in compose_prompt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_includes_persona_file(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    personas = tmp_path / "personas"
    personas.mkdir(parents=True, exist_ok=True)
    (personas / "executor.md").write_text(
        "I prefer vllm on dense models.", encoding="utf-8"
    )

    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt("executor", msgs=[])
    assert "I prefer vllm on dense models" in prompt
    assert "Persona (running notes)" in prompt
    assert "Iron Rules" in prompt
    db.close()


@pytest.mark.asyncio
async def test_compose_prompt_iron_rules_block_present(tmp_path: Path):
    """The dynamic IronRules block is always rendered — its content shifts
    with mode, but the universal IR-4 / IR-5 lines are mandatory.

    (The executor's *static* role prompt also references IR-1..IR-7 as a
    recap, so we cannot assert on rule absence in the full prompt; we
    only check that the dynamic block prepares the universal rules.)
    """
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt("executor", msgs=[])
    assert "IR-4" in prompt
    assert "IR-5" in prompt
    # Plan A — render_for_prompt now tags each rule with its severity so
    # the LLM can prioritise. IR-4/IR-5 stay BLOCK so look for the
    # severity surface ("block" lower-case + "MUST" tone).
    assert "(block" in prompt  # accept "(block" or "(block — MUST)"
    assert "MUST" in prompt
    db.close()


@pytest.mark.asyncio
async def test_compose_prompt_sage_hint_block(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt(
        "executor",
        msgs=[],
        sage_hint="- prefer vllm for dense\n- watch out for kv-cache fp8",
    )
    assert "Sage hint" in prompt
    assert "kv-cache fp8" in prompt
    db.close()


# ---------------------------------------------------------------------------
# Inbox-body rendering — every published payload shape from the dispatcher
# (_handle_*) must produce a non-empty, informative body line. Regression
# guard for v0.7 bug: ``proposal`` / ``question`` / ``alert`` /
# ``reflection_tick`` etc. used to render as empty strings, which made
# Critic and Watchdog raise "I cannot evaluate" alerts.
# ---------------------------------------------------------------------------
class TestRenderMsgBody:
    def test_body_md_passthrough(self):
        out = Conductor._render_msg_body(
            "heartbeat", {"body_md": "Critic online, standing by."}
        )
        assert out == "Critic online, standing by."

    def test_body_md_long_text_not_truncated(self):
        long_text = "X" * 5000  # agent-authored content kept intact
        out = Conductor._render_msg_body("heartbeat", {"body_md": long_text})
        assert out == long_text

    def test_proposal_renders_action_and_gain(self):
        out = Conductor._render_msg_body(
            "proposal",
            {
                "task_id": "abc123def456",
                "action_name": "run_baseline",
                "predicted_gain_pct": 5,
                "params": {"backend": "sglang"},
            },
        )
        assert "action=run_baseline" in out
        assert "predicted_gain=5%" in out
        assert "backend" in out

    def test_proposal_delegate_kind_renders_distinctly(self):
        out = Conductor._render_msg_body(
            "proposal",
            {
                "kind": "delegate_queued",
                "task_id": "abc123def456789",
                "action_name": "kernel_opt",
                "params": {"file": "x.cu"},
            },
        )
        assert "delegate_queued" in out
        assert "action=kernel_opt" in out
        assert "task=abc123de" in out  # truncated to 8 chars

    def test_decision_renders_kind_changes_and_rationale(self):
        out = Conductor._render_msg_body(
            "decision",
            {
                "kind": "state_updated",
                "changes": {"current_action": "run_baseline"},
                "rationale": "establish baseline",
            },
        )
        assert "state_updated" in out
        assert "current_action" in out
        assert "rationale=establish baseline" in out

    def test_alert_renders_severity_summary_detail(self):
        out = Conductor._render_msg_body(
            "alert",
            {"severity": "high", "summary": "OOM on GPU 0", "detail": "VRAM=80GB"},
        )
        assert "[high]" in out
        assert "OOM on GPU 0" in out
        assert "VRAM=80GB" in out

    def test_alert_without_detail_renders_summary_only(self):
        out = Conductor._render_msg_body(
            "alert",
            {"severity": "low", "summary": "no actionable content", "detail": ""},
        )
        assert out.startswith("[low] no actionable content")
        assert "—" not in out  # no separator when detail is empty

    def test_question_renders_topic_and_question(self):
        out = Conductor._render_msg_body(
            "question",
            {"topic": "model_kb_recall", "question": "Best backend for Qwen3?"},
        )
        assert "[topic=model_kb_recall]" in out
        assert "Best backend for Qwen3?" in out

    def test_answer_renders_topic_and_answer(self):
        out = Conductor._render_msg_body(
            "answer",
            {"topic": "model_kb_recall", "answer": "no_prior_data"},
        )
        assert "[topic=model_kb_recall]" in out
        assert "no_prior_data" in out

    def test_kill_renders_task_and_reason(self):
        """v0.4 — replaces the deleted objection/vote topic renderers."""
        out = Conductor._render_msg_body(
            "kill",
            {
                "task_id": "deadbeef0123cafe",
                "reason": "stuck for 4x lease_ttl",
            },
        )
        assert "task=deadbeef" in out
        assert "reason=stuck for 4x lease_ttl" in out

    def test_reflection_tick_renders_elapsed_and_left(self):
        out = Conductor._render_msg_body(
            "reflection_tick",
            {"elapsed_minutes": 1.25, "time_left_minutes": 418.75},
        )
        assert "elapsed=1.25m" in out
        assert "time_left=418.75m" in out

    def test_observation_uses_kind_fallback(self):
        # _record_observation publishes {kind, agent, intent_type, rule, reason}
        out = Conductor._render_msg_body(
            "observation",
            {
                "kind": "policy_denied",
                "agent": "executor",
                "intent_type": "send_message",
                "rule": "topic",
                "reason": "topic='session_start' not in TOPIC_ALLOWLIST",
            },
        )
        assert "policy_denied" in out
        assert "agent=executor" in out
        assert "rule=topic" in out
        assert "TOPIC_ALLOWLIST" in out

    def test_event_persona_update_uses_kind_fallback(self):
        out = Conductor._render_msg_body(
            "event",
            {"kind": "persona_update", "agent": "executor", "chars": 240},
        )
        assert "persona_update" in out
        assert "agent=executor" in out
        assert "chars=240" in out

    def test_unknown_topic_with_kind_falls_back(self):
        out = Conductor._render_msg_body(
            "future_topic_xyz", {"kind": "newshape", "k1": "v1", "k2": 42}
        )
        assert "newshape" in out
        assert "k1=v1" in out
        assert "k2=42" in out

    def test_unknown_topic_without_kind_uses_kvs(self):
        out = Conductor._render_msg_body(
            "future_topic_xyz", {"a": 1, "b": "two"}
        )
        assert "a=1" in out
        assert "b=two" in out

    def test_synthesised_body_truncated_at_cap(self):
        long_params = {"x" * 50: "y" * 500}
        out = Conductor._render_msg_body(
            "proposal",
            {"action_name": "noop", "predicted_gain_pct": 0, "params": long_params},
        )
        assert len(out) <= Conductor._MSG_BODY_RENDER_CAP
        assert out.endswith("...")

    def test_non_dict_payload_coerced_to_string(self):
        assert Conductor._render_msg_body("heartbeat", None) == ""
        assert Conductor._render_msg_body("heartbeat", "raw text") == "raw text"

    def test_empty_body_md_falls_through_to_renderer(self):
        # body_md='' should NOT shadow the per-topic renderer (regression
        # for the case where send_message publishes ``body_md=""`` after
        # filtering).
        out = Conductor._render_msg_body(
            "alert", {"body_md": "", "severity": "info", "summary": "x"}
        )
        assert "[info]" in out
        assert "x" in out


@pytest.mark.asyncio
async def test_compose_prompt_renders_inbox_bodies_for_all_topics(tmp_path: Path):
    """End-to-end: a heterogeneous inbox of every published topic must
    show non-empty bodies in the rendered prompt."""
    from dataclasses import dataclass

    @dataclass
    class _Msg:
        seq: int
        from_agent: str
        topic: str
        payload: dict

    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()

    msgs: list[Any] = [
        _Msg(27, "executor", "question",
             {"topic": "model_kb_recall", "question": "Best backend?"}),
        _Msg(28, "executor", "proposal",
             {"task_id": "abc12345", "action_name": "run_baseline",
              "predicted_gain_pct": 0, "params": {}}),
        _Msg(29, "clock", "reflection_tick",
             {"elapsed_minutes": 1.25, "time_left_minutes": 418.75}),
        _Msg(30, "critic", "alert",
             {"severity": "low", "summary": "empty body", "detail": ""}),
        _Msg(31, "executor", "decision",
             {"kind": "state_updated", "changes": {"current_action": "run_baseline"},
              "rationale": "cold start"}),
        _Msg(32, "conductor", "observation",
             {"kind": "policy_denied", "agent": "executor",
              "intent_type": "send_message", "rule": "topic",
              "reason": "topic='X' not in TOPIC_ALLOWLIST"}),
    ]
    prompt = conductor._compose_prompt("critic", msgs=msgs)

    assert "seq=27 from=executor topic=question :: [topic=model_kb_recall] Best backend?" in prompt
    assert "seq=28 from=executor topic=proposal :: action=run_baseline predicted_gain=0%" in prompt
    assert "seq=29 from=clock topic=reflection_tick :: elapsed=1.25m time_left=418.75m" in prompt
    assert "seq=30 from=critic topic=alert :: [low] empty body" in prompt
    assert "seq=31 from=executor topic=decision :: state_updated" in prompt
    assert "seq=32 from=conductor topic=observation :: policy_denied" in prompt

    # Crucially: no line should end with the empty-body separator
    # ``"topic=X :: "`` followed by a newline (which is what triggered
    # the original Critic/Watchdog "empty body" alerts).
    for line in prompt.splitlines():
        line = line.rstrip()
        assert not line.endswith("::"), f"empty body line: {line!r}"
    db.close()


# ---------------------------------------------------------------------------
# Persona index refresh after update_persona intent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_update_persona_refreshes_in_memory_index(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    intent = Intent(
        type=IntentType.UPDATE_PERSONA,
        payload={"body_md": "Critical lesson: always check GPU memory"},
    )
    await conductor._handle_intent("executor", intent)
    assert "always check GPU memory" in conductor.ctx.persona_index["executor"]
    db.close()


# ---------------------------------------------------------------------------
# Dispatcher loop drains queued delegate tasks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_bench_enqueued_after_workspace_mutation(tmp_path: Path):
    """Phase C — ``_on_task_finished`` must auto-enqueue a bench_runner
    task whenever a workspace_mutation action succeeds and no bench_runner
    is already in-flight."""
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.05"},
        db=db,
        action_registry=registry,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # Pick any workspace_mutation action that exists.
    mutation_meta = next(
        a for a in registry.all()
        if "workspace_mutation" in (a.requires_lanes or ())
    )

    class _StubResult:
        status = "succeeded"
        metrics: dict = {}

    class _StubTask:
        task_id = "abcdef123456"
        params = {"action_name": mutation_meta.name}

    await conductor._on_task_finished(_StubTask(), _StubResult())
    queued = await conductor.ctx.tasks.list_by_state("queued")
    bench_tasks = [
        t for t in queued
        if (t.params or {}).get("action_name") == "bench_runner"
    ]
    assert bench_tasks, (
        f"expected an auto-enqueued bench_runner after {mutation_meta.name}, "
        f"got queued={[t.params for t in queued]}"
    )
    assert bench_tasks[0].params["auto_after"] == mutation_meta.name
    db.close()


@pytest.mark.asyncio
async def test_auto_bench_skipped_when_bench_already_running(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.05"},
        db=db,
        action_registry=registry,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # Pre-stage a bench_runner in queued state.
    await conductor.ctx.tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner", "params": {},
                "requested_by": "executor"},
        idempotency_key="manual-bench-1",
    )
    mutation_meta = next(
        a for a in registry.all()
        if "workspace_mutation" in (a.requires_lanes or ())
    )

    class _R:
        status = "succeeded"
        metrics: dict = {}

    class _T:
        task_id = "abcdef987654"
        params = {"action_name": mutation_meta.name}

    await conductor._on_task_finished(_T(), _R())
    bench_tasks = [
        t for t in await conductor.ctx.tasks.list_by_state("queued")
        if (t.params or {}).get("action_name") == "bench_runner"
    ]
    assert len(bench_tasks) == 1, (
        "auto-bench should skip when one is already in flight"
    )
    db.close()


@pytest.mark.asyncio
async def test_bench_done_event_falls_back_into_state_tput(tmp_path: Path):
    """Phase C — when a ``send_message{topic=event,kind=bench_done}``
    arrives the conductor mirrors the per-GPU throughput into
    SharedState even if no ``update_state`` reaches it (e.g. agent
    silently dropped that intent)."""
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    conductor.ctx.state.baseline_tput = 1000.0
    conductor.ctx.state.current_tput = 1000.0
    await conductor._handle_send_message(
        "executor",
        {
            "topic": "event",
            "body_md": "bench complete",
            "kind": "bench_done",
            "tput_per_gpu": 1500.0,
            "tput_total": 6000.0,
        },
    )
    assert conductor.ctx.state.current_tput == pytest.approx(1500.0)
    # Cumulative gain should also be recomputed: (1500/1000 - 1)*100 = 50%.
    assert conductor.ctx.state.cumulative_gain == pytest.approx(50.0)
    db.close()


@pytest.mark.asyncio
async def test_force_dispatch_promotes_queued_task_to_head(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.05"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # Two queued tasks; second one is the one we want force-dispatched.
    a = await conductor.ctx.tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner", "params": {},
                "requested_by": "executor"},
        idempotency_key="t1",
    )
    b = await conductor.ctx.tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner", "params": {"x": 1},
                "requested_by": "executor"},
        idempotency_key="t2",
    )
    await conductor._handle_force_dispatch(
        "triage", {"task_id": b.task_id, "reason": "test"},
    )
    rows = await db.fetchall(
        "SELECT task_id, created_at FROM tasks "
        "WHERE kind='delegate' AND state='queued' ORDER BY created_at"
    )
    assert rows[0]["task_id"] == b.task_id
    db.close()


@pytest.mark.asyncio
async def test_prune_branch_cancels_family_and_marks_state(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.05"},
        db=db,
        action_registry=registry,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # Pick a long-family action and queue one.
    long_action = next(a for a in registry.all() if a.family == "long")
    await conductor.ctx.tasks.create(
        kind="delegate",
        params={"action_name": long_action.name, "params": {},
                "requested_by": "executor"},
        idempotency_key="prune-test-1",
    )
    await conductor._handle_prune_branch(
        "triage", {"family": "long", "reason": "3 consecutive failures"},
    )
    assert "long" in conductor.ctx.state.pruned_families
    cancelled = await conductor.ctx.tasks.list_by_state("cancelled")
    assert any(
        (t.params or {}).get("action_name") == long_action.name for t in cancelled
    )
    db.close()


@pytest.mark.asyncio
async def test_escalate_strategy_change_emits_priority0_alert(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.05"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    await conductor._handle_escalate_strategy_change(
        "triage",
        {
            "reason": "GPU idle 12min",
            "next_action_hint": "switch to triton prefill",
            "severity": "high",
        },
    )
    bus = MessageBus(db)
    msgs = await bus.tail(n=20)
    strat = [
        m for m in msgs
        if m.topic == "alert"
        and (m.payload or {}).get("kind") == "strategy_change"
    ]
    assert strat, f"expected strategy_change alert, got {msgs}"
    assert strat[0].priority == 0
    assert strat[0].to_agent == "executor"
    db.close()


@pytest.mark.asyncio
async def test_dispatcher_loop_drains_queued_delegates(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()

    # Backend that emits one update_state intent — SubAgentRunner will
    # treat that as success metric extraction.
    backend = MockBackend(
        script=[
            ScriptStep(
                intents=[
                    Intent(
                        type=IntentType.UPDATE_STATE,
                        payload={"changes": {"current_tput": 5500.0}},
                    )
                ]
            )
        ]
    )
    conductor = Conductor(
        tmp_path,
        backend=backend,
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.005"},
        db=db,
        action_registry=registry,
        reactor_tick_s=0.1,
        clock_tick_s=0.1,
        enable_dispatcher=True,
    )

    async def kick_off():
        await conductor._bootstrap()
        # queue a delegate task directly
        ctx = conductor.ctx
        await ctx.tasks.create(
            kind="delegate",
            params={"action_name": "bench_runner",
                    "params": {}, "requested_by": "executor"},
            idempotency_key="dispatcher-test-1",
            requires_lanes=["benchmark_lane"],
            allowed_tools=["emit_intent", "Read", "Bash"],
            side_effects=["reads_server"],
            lease_ttl_sec=900,
        )

    await kick_off()
    # Now run the dispatcher one-shot.
    from inference_optimizer.orchestrator.sub_agent_runner import (
        dispatch_pending_delegates,
    )
    dispatched = await dispatch_pending_delegates(
        conductor.ctx.sub_agent_runner,
        db=db,
    )
    assert dispatched == 1
    finished = await conductor.ctx.tasks.list_by_state("succeeded")
    assert len(finished) == 1
    db.close()


# ---------------------------------------------------------------------------
# v0.4 — Self-review (parliament removed entirely; ephemeral RCA gone too)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_record_proposal_for_self_review_writes_event(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    await conductor._record_proposal_for_self_review(
        {"action_name": "bench_runner"}
    )
    bus = MessageBus(db)
    events = await bus.tail(n=10)
    proposals = [
        e for e in events
        if e.topic == "proposal"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "self_review"
    ]
    assert proposals
    db.close()


def test_v04_removed_parliament_and_ephemeral_rca():
    """v0.4 — confirm the deleted methods/symbols are actually gone."""
    assert not hasattr(Conductor, "_open_parliament"), (
        "v0.4 removed parliament — _open_parliament should not exist"
    )
    assert not hasattr(Conductor, "ephemeral_rca_via_critic"), (
        "v0.4 removed ephemeral RCA — method should not exist"
    )


# ---------------------------------------------------------------------------
# TokenBudgetMeter
# ---------------------------------------------------------------------------
def test_token_meter_record_increments():
    m = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    assert m.tokens_used == 0
    m.record(prompt_tokens=1000, completion_tokens=500)
    assert m.tokens_used == 1500
    assert m.remaining() == m.budget - 1500


def test_token_meter_throttle_at_80pct():
    m = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    m.record(prompt_tokens=int(m.warn_at * 0.5))
    assert m.should_throttle() is False
    m.record(prompt_tokens=int(m.warn_at * 0.6))
    assert m.should_throttle() is True


def test_token_meter_reset():
    m = TokenBudgetMeter(ExecutionMode.GUIDED_KERNEL_OPT)
    m.record(prompt_tokens=m.warn_at + 1)
    assert m.should_throttle() is True
    m.reset()
    assert m.tokens_used == 0
    assert m.should_throttle() is False


def test_token_meter_marathon_budget_higher():
    m_q = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    m_m = TokenBudgetMeter(ExecutionMode.MARATHON_MULTI_AGENT)
    assert m_m.budget > m_q.budget


# ---------------------------------------------------------------------------
# 30-min checkpoint cadence (only fires when enabled + interval passed)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_checkpoint_cadence_writes_on_first_call(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=True,
    )
    await conductor._bootstrap()
    handle = await conductor._maybe_checkpoint()
    assert handle is not None
    cps = list((tmp_path / "checkpoints").iterdir())
    assert cps
    db.close()


@pytest.mark.asyncio
async def test_checkpoint_cadence_throttles_within_30min(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=True,
    )
    await conductor._bootstrap()
    h1 = await conductor._maybe_checkpoint()
    h2 = await conductor._maybe_checkpoint()
    assert h1 is not None
    # second call should be throttled (within 30 min cadence)
    assert h2 is None
    db.close()


@pytest.mark.asyncio
async def test_checkpoint_cadence_disabled_returns_none(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=False,
    )
    await conductor._bootstrap()
    assert await conductor._maybe_checkpoint() is None
    db.close()


# ---------------------------------------------------------------------------
# resume_from_session_dir wiring
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_init_session_resume_loads_existing_state(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # seed a cursor row so resume has something to find
    from inference_optimizer.orchestrator.cursor_store import CursorStore
    cs = CursorStore(db)
    await cs.advance("executor", seq=99, msg_id="m99")
    state = await conductor._init_session_resume()
    assert state is not None
    assert state.cursors.get("executor") == 99
    db.close()


# ---------------------------------------------------------------------------
# Action catalogue + first-action hint injected into prompt (L3)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_injects_action_catalogue(
    tmp_path: Path,
):
    """The prompt must list the available actions so the LLM can name
    them in `propose_action` / `delegate` intents."""
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "1"},
        db=db,
        action_registry=registry,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt("executor", [])
    # Locate the catalogue section so we can assert on it without
    # fighting the role's static system prompt (executor.md mentions
    # kernel_opt in its mode table for context, which is fine). The
    # catalogue runs from its H2 header to the next H2 header.
    header = "## Available actions for this mode"
    assert header in prompt
    cat_start = prompt.index(header)
    next_h2 = prompt.find("\n## ", cat_start + len(header))
    cat_end = next_h2 if next_h2 != -1 else len(prompt)
    catalogue = prompt[cat_start:cat_end]
    # Quick mode (MAX_HOURS=1 → quick_param_sweep) → baseline, sweep,
    # params, classify, target_analysis, report should be present; the
    # deep_kernel family (kernel_opt, deep_kernel_analysis, ...) must
    # be excluded from the catalogue.
    assert "`baseline`" in catalogue
    assert "`sweep`" in catalogue
    assert "`kernel_opt`" not in catalogue
    assert "`deep_kernel_analysis`" not in catalogue
    db.close()


@pytest.mark.asyncio
async def test_compose_prompt_first_action_hint_only_for_executor_at_t0(
    tmp_path: Path,
):
    """First-action hint should fire for the executor when baseline_tput=0,
    and disappear once baseline lands."""
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "1"},
        db=db,
        action_registry=registry,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # The dynamic hint marker — distinct from the static mention in
    # executor.md which says "First action hint" (without "(live)")
    # to keep these two coupled but unambiguously separable.
    HINT_MARKER = "## First action hint (live)"

    # Cold-start: hint visible, baseline_tput=0
    prompt = conductor._compose_prompt("executor", [])
    assert HINT_MARKER in prompt

    # Critic / sage / watchdog should never see the hint even at t=0
    for other_role in ("critic", "watchdog", "sage"):
        prompt_other = conductor._compose_prompt(other_role, [])
        assert HINT_MARKER not in prompt_other

    # After baseline has been measured, hint disappears.
    conductor.ctx.state.baseline_tput = 5000.0
    prompt = conductor._compose_prompt("executor", [])
    assert HINT_MARKER not in prompt
    db.close()
