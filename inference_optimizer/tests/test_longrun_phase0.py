# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase 0 (long-run infrastructure) acceptance tests.

Covers the "live long safely" reworks that must hold before the cyclic phase
machine (Phase 1) is introduced:

* R4 — bounded SharedState ledgers + events/tasks DB retention (resume-safe).
* R5 — active GPU-lease reaper + coordinator maintenance tick.
* R6a — bounded transient-failure retry/backoff for LLM backend calls.

All deterministic + offline (no GPU, no network, fake clock).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from inference_optimizer.orchestrator import db_maintenance as dbm
from inference_optimizer.orchestrator import shared_state as ss_mod
from inference_optimizer.orchestrator.shared_state import SharedState, _cap_tested_ledger
from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.orchestrator.gpu_pool import SpecialistGpuPool
from inference_optimizer.orchestrator.message_bus import Message, MessageBus
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage import SqliteConnection
from inference_optimizer.storage.schema import ensure_schema


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield db
    db.close()


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


# ==========================================================================
# R4 — SharedState ledger caps
# ==========================================================================
def test_intervention_mix_capped():
    st = SharedState()
    cap = ss_mod._INTERVENTION_MIX_CAP
    for i in range(cap + 250):
        st.record_intervention(change_type="config", action="explore", task_id=str(i))
    assert len(st.intervention_mix) == cap
    # Most-recent retained (tail-trim), oldest evicted.
    assert st.intervention_mix[-1]["task_id"] == str(cap + 250 - 1)
    # Authoritative control counter is NOT trimmed.
    assert st.consecutive_config_only_rounds == cap + 250


def test_specialist_rounds_capped():
    st = SharedState()
    cap = ss_mod._SPECIALIST_ROUNDS_CAP
    for i in range(cap + 80):
        st.record_specialist_round({"round_id": f"r{i}", "n": i})
    assert len(st.specialist_rounds) == cap
    assert st.specialist_rounds[-1]["round_id"] == f"r{cap + 80 - 1}"


def test_seen_pr_ids_capped():
    st = SharedState()
    cap = ss_mod._SEEN_PR_IDS_CAP
    st.register_seen_pr_ids([f"pr{i}" for i in range(cap + 500)])
    assert len(st.research_scout_seen_pr_ids) == cap
    # Newest kept (FIFO eviction of oldest).
    assert st.has_seen_pr_id(f"pr{cap + 500 - 1}")


def test_winners_history_capped_via_explore_update():
    st = SharedState()
    cap = ss_mod._WINNERS_HISTORY_CAP
    big = [{"variant_name": f"v{i}", "gain_pct": i} for i in range(cap + 120)]
    st.apply_explore_search_update({"winners_history": big})
    assert len(st.explore_search["winners_history"]) == cap


def test_tested_ledger_cap_helper():
    cap = ss_mod._EXPLORE_TESTED_CAP
    tested = {f"fp{i}": {"i": i} for i in range(cap + 300)}
    out = _cap_tested_ledger(tested)
    assert len(out) == cap
    # Oldest insertion-order keys evicted; newest retained.
    assert f"fp{cap + 300 - 1}" in out
    assert "fp0" not in out


def test_tested_ledger_cap_applied_on_merge():
    st = SharedState()
    cap = ss_mod._EXPLORE_TESTED_CAP
    tested = {f"fp{i}": {"i": i} for i in range(cap + 10)}
    st.apply_explore_search_update({"tested": tested})
    assert len(st.explore_search["tested"]) == cap


# ==========================================================================
# R4 — DB retention (events + tasks), resume-safe
# ==========================================================================
@pytest.mark.asyncio
async def test_prune_events_respects_min_cursor_and_recent_window(conn):
    bus = MessageBus(conn)
    cursors = CursorStore(conn)
    for i in range(100):
        await bus.append_and_seq(
            Message.new("orchestration", "*", "heartbeat", {"i": i})
        )
    # No cursor yet => nothing is safe to prune.
    assert await dbm.prune_events(conn, cursors, keep_recent=0) == 0

    # Advance the (single) min cursor to seq=60; keep_recent=10 protects the tail.
    await cursors.advance("orchestration", seq=60, msg_id="x")
    deleted = await dbm.prune_events(conn, cursors, keep_recent=10)
    # delete_below = min(60, 100-10=90) = 60 => seq 1..60 removed.
    assert deleted == 60
    remaining = await conn.fetchall("SELECT seq FROM events ORDER BY seq")
    assert [r["seq"] for r in remaining] == list(range(61, 101))


@pytest.mark.asyncio
async def test_prune_events_never_crosses_resume_anchor(conn):
    """An event above any agent cursor must survive (still needs replay)."""
    bus = MessageBus(conn)
    cursors = CursorStore(conn)
    for i in range(50):
        await bus.append_and_seq(Message.new("orchestration", "kernel", "request", {"i": i}))
    # Two agents; the laggard cursor (kernel=20) is the safe watermark.
    await cursors.advance("orchestration", seq=50, msg_id="a")
    await cursors.advance("kernel", seq=20, msg_id="b")
    await dbm.prune_events(conn, cursors, keep_recent=0)
    rows = await conn.fetchall("SELECT MIN(seq) AS m FROM events")
    # Nothing at/below the laggard's unprocessed frontier (seq>20) is removed.
    assert int(rows[0]["m"]) == 21
    # Resume replay for the laggard still sees all its unprocessed events.
    replay = await bus.replay_for("kernel", after_seq=20)
    assert len(replay) == 30


# --------------------------------------------------------------------------
# Issue 3 — pruning must not delete a still-pending proposal
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prune_events_protects_pending_proposal(conn):
    """A processed-but-undecided proposal survives pruning; once a verdict
    targets it, it becomes prunable."""
    bus = MessageBus(conn)
    cursors = CursorStore(conn)
    # A proposal with no verdict yet (semantically pending).
    prop = Message.new("orchestration", "*", "proposal", {"action_name": "x"})
    await bus.append_and_seq(prop)
    # Filler so the proposal is well below the recent window.
    for i in range(50):
        await bus.append_and_seq(Message.new("orchestration", "*", "heartbeat", {"i": i}))
    # Advance all cursors past everything; small keep_recent.
    await cursors.advance("orchestration", seq=100, msg_id="z")

    # Pending proposal must survive even though it's processed + old.
    await dbm.prune_events(conn, cursors, keep_recent=0)
    rows = await conn.fetchall(
        "SELECT msg_id FROM events WHERE topic='proposal'"
    )
    assert [r["msg_id"] for r in rows] == [prop.msg_id]

    # Now a verdict decides it → it becomes prunable.
    await bus.append_and_seq(Message.new(
        "critic", "*", "review_verdict",
        {"target_proposal_msg_id": prop.msg_id, "verdict": "approve"},
    ))
    await cursors.advance("orchestration", seq=200, msg_id="z2")
    await dbm.prune_events(conn, cursors, keep_recent=0)
    rows = await conn.fetchall(
        "SELECT msg_id FROM events WHERE topic='proposal'"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_pending_proposal_seqs_matches_reconstruct_logic(conn):
    """The pruning guard's pending set must agree with the resume reconstruct
    logic: a proposal is decided iff a verdict has a NON-EMPTY target equal to
    its msg_id (empty/missing targets do not decide anything)."""
    bus = MessageBus(conn)
    p_pending = Message.new("orchestration", "*", "proposal", {"action_name": "a"})
    p_decided = Message.new("orchestration", "*", "proposal", {"action_name": "b"})
    await bus.append_and_seq(p_pending)
    await bus.append_and_seq(p_decided)
    # Decided one gets a real verdict.
    await bus.append_and_seq(Message.new(
        "critic", "*", "review_verdict",
        {"target_proposal_msg_id": p_decided.msg_id, "verdict": "reject"},
    ))
    # An empty-target verdict must NOT decide any proposal (NULL-trap guard).
    await bus.append_and_seq(Message.new(
        "critic", "*", "review_verdict",
        {"target_proposal_msg_id": "", "verdict": "approve"},
    ))

    pending = await dbm.pending_proposal_seqs(conn)

    # Cross-check against the same join replay_for_resume uses.
    proposals = await bus.tail(topic="proposal", n=1000)
    verdicts = await bus.tail(topic="review_verdict", n=1000)
    decided = {
        v.payload.get("target_proposal_msg_id")
        for v in verdicts
        if v.payload.get("target_proposal_msg_id")
    }
    expected_pending_seqs = {
        p.seq for p in proposals if p.msg_id not in decided
    }
    assert pending == expected_pending_seqs
    # Concretely: only the undecided proposal is protected.
    assert pending == {p_pending.seq}


@pytest.mark.asyncio
async def test_prune_tasks_keeps_recent_done_and_spares_inflight(conn):
    reg = TaskRegistry(conn)
    # In-flight tasks that must never be pruned.
    q = await reg.create(kind="explore", params={}, idempotency_key="q1")
    r = await reg.create(kind="explore", params={}, idempotency_key="r1")
    await reg.transition(r.task_id, "running")
    # Many done (succeeded) tasks.
    done_ids = []
    for i in range(30):
        t = await reg.create(kind="explore", params={}, idempotency_key=f"d{i}")
        await reg.transition(t.task_id, "running")
        await reg.transition(t.task_id, "succeeded")
        done_ids.append(t.task_id)
    deleted = await dbm.prune_tasks(conn, keep_done=10)
    assert deleted == 20
    # Queued + running survive.
    assert (await reg.get(q.task_id)).state == "queued"
    assert (await reg.get(r.task_id)).state == "running"
    # Exactly keep_done most-recent succeeded remain.
    rows = await conn.fetchall("SELECT COUNT(*) AS c FROM tasks WHERE state='succeeded'")
    assert int(rows[0]["c"]) == 10


@pytest.mark.asyncio
async def test_run_db_retention_aggregates(conn):
    bus = MessageBus(conn)
    cursors = CursorStore(conn)
    for i in range(40):
        await bus.append_and_seq(Message.new("orchestration", "*", "heartbeat", {"i": i}))
    await cursors.advance("orchestration", seq=40, msg_id="z")
    res = await dbm.run_db_retention(conn, cursors, events_keep_recent=5)
    assert res.events_deleted > 0
    assert res.total == res.events_deleted + res.tasks_deleted


# ==========================================================================
# R5 — GPU lease reaper
# ==========================================================================
@pytest.mark.asyncio
async def test_gpu_pool_reap_expired(conn):
    pool = SpecialistGpuPool(conn, gpu_ids=[0, 1, 2, 3])
    now = datetime.now(timezone.utc)
    # Insert one live + two expired leases directly.
    rows = [
        (0, "h0", "t0", _iso(now), _iso(now + timedelta(hours=1)), _iso(now)),
        (1, "h1", "t1", _iso(now - timedelta(hours=2)), _iso(now - timedelta(hours=1)), _iso(now)),
        (2, "h2", "t2", _iso(now - timedelta(hours=2)), _iso(now - timedelta(minutes=1)), _iso(now)),
    ]
    for row in rows:
        await conn.execute(
            "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, "
            "expires_at, heartbeat_at) VALUES (?,?,?,?,?,?)",
            row,
        )
    reaped = await pool.reap_expired()
    assert reaped == 2
    remaining = await conn.fetchall("SELECT gpu_id FROM gpu_leases")
    assert [r["gpu_id"] for r in remaining] == [0]


# ==========================================================================
# R6a — retry/backoff
# ==========================================================================
@pytest.mark.asyncio
async def test_retry_with_backoff_recovers_then_succeeds():
    from inference_optimizer.orchestrator.backends.base import (
        RetryPolicy,
        retry_with_backoff,
    )

    calls = {"n": 0}
    slept: list[float] = []

    async def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise asyncio.TimeoutError("transient")
        return "ok"

    async def _fake_sleep(d):
        slept.append(d)

    out = await retry_with_backoff(
        _flaky,
        policy=RetryPolicy(max_attempts=5, base_delay_s=1.0, jitter_s=0.0),
        retry_on=(asyncio.TimeoutError,),
        sleep=_fake_sleep,
    )
    assert out == "ok"
    assert calls["n"] == 3
    assert slept == [1.0, 2.0]  # exponential, no jitter


@pytest.mark.asyncio
async def test_retry_with_backoff_exhausts_and_raises():
    from inference_optimizer.orchestrator.backends.base import (
        RetryPolicy,
        retry_with_backoff,
    )

    async def _always_fail():
        raise ConnectionError("down")

    async def _fake_sleep(d):
        return None

    with pytest.raises(ConnectionError):
        await retry_with_backoff(
            _always_fail,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_s=0.0),
            retry_on=(ConnectionError,),
            sleep=_fake_sleep,
        )


def test_retry_policy_from_env(monkeypatch):
    from inference_optimizer.orchestrator.backends.base import RetryPolicy

    monkeypatch.setenv("INFERENCE_OPTIMIZER_LLM_RETRY_ATTEMPTS", "1")
    pol = RetryPolicy.from_env()
    assert pol.max_attempts == 1  # disables retry


# ==========================================================================
# R5 — coordinator maintenance tick (cadence + wiring)
# ==========================================================================
@pytest.mark.asyncio
async def test_coordinator_maintenance_tick_cadence_and_reaps(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MAINTENANCE_EVERY_TICKS", "10")
    from inference_optimizer.paths import make_session_dir
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.backends import (
        MockBackend, MockCriticBackend, MockKernelBackend,
        MockRobustnessBackend, ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = make_session_dir()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "kernel": MockKernelBackend(),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(sd, backends=backends)
    try:
        # Seed an expired GPU lease + some events with an advanced cursor.
        now = datetime.now(timezone.utc)
        await c.db.execute(
            "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, "
            "expires_at, heartbeat_at) VALUES (?,?,?,?,?,?)",
            (0, "h", "t", _iso(now - timedelta(hours=2)),
             _iso(now - timedelta(hours=1)), _iso(now)),
        )
        c.gpu_specialist_pool = SpecialistGpuPool(c.db, gpu_ids=[0, 1])
        for i in range(30):
            await c.bus.append_and_seq(
                Message.new("orchestration", "*", "heartbeat", {"i": i})
            )
        await c.cursors.advance("orchestration", seq=30, msg_id="m")

        # Off-cadence ticks are a no-op.
        assert await c._maybe_run_maintenance_tick(tick=7) is None
        # On-cadence tick reaps leases + runs DB retention (events under the
        # 5000-event keep-recent default are intentionally retained; the prune
        # threshold itself is unit-tested in test_prune_events_* above).
        summary = await c._maybe_run_maintenance_tick(tick=10)
        assert summary is not None
        assert summary["gpu_leases_reaped"] == 1
        assert "events_pruned" in summary and "tasks_pruned" in summary
        rows = await c.db.fetchall("SELECT COUNT(*) AS c FROM gpu_leases")
        assert int(rows[0]["c"]) == 0
    finally:
        await c.stop()


# --------------------------------------------------------------------------
# R6a — ClaudeBackend retries a transient SDK failure end-to-end
# --------------------------------------------------------------------------
class ToolUseBlock:  # class name must be exactly "ToolUseBlock" (backend checks type name)
    def __init__(self, name: str, inp: dict[str, Any]):
        self.name = name
        self.input = inp


@dataclass
class _Msg:
    content: list[Any] = field(default_factory=list)
    session_id: str | None = "s1"
    result: str = ""


@dataclass
class _Options:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_claude_backend_retries_transient_then_succeeds():
    from inference_optimizer.orchestrator.backends.claude import ClaudeBackend
    from inference_optimizer.orchestrator.backends.base import RetryPolicy
    from inference_optimizer.orchestrator.backends.mcp_emit_intent import (
        EMIT_INTENT_TOOL_QUALIFIED,
    )

    state = {"n": 0}

    def _factory(*, prompt, options):
        state["n"] += 1
        attempt = state["n"]

        async def _gen():
            if attempt == 1:
                raise asyncio.TimeoutError("proxy stall")
            block = ToolUseBlock(
                EMIT_INTENT_TOOL_QUALIFIED,
                {"intent_type": "send_message",
                 "payload": {"topic": "heartbeat", "body_md": "ok"}},
            )
            yield _Msg(content=[block])
        return _gen()

    backend = ClaudeBackend(
        sdk_query_factory=_factory,
        sdk_options_cls=_Options,
        enable_mcp_emit_intent=False,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_s=0.0),
    )
    # Force a fast timeout so the first attempt's wait_for path is exercised.
    backend.call_timeout_s = 5.0
    backend.mcp_tool_name = EMIT_INTENT_TOOL_QUALIFIED

    result = await backend.run("hi", allow_no_intent=True)
    assert state["n"] == 2  # one transient failure, one success
    assert len(result.intents) == 1
