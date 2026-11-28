# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-run infrastructure acceptance tests."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from hyperloom.orchestrator.state._shared_state import phase_state as ss_mod
from hyperloom.orchestrator.state._shared_state.phase_state import _cap_tested_ledger
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
from hyperloom.orchestrator.bus.message_bus import Message, MessageBus
from hyperloom.orchestrator.state.task_registry import TaskSpec
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom_common.timeutil import now_iso


@pytest.fixture
def conn(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield db
    db.close()


# SharedState ledger caps
def test_intervention_mix_capped():
    st = SharedState()
    cap = ss_mod._INTERVENTION_MIX_CAP
    for _ in range(cap + 250):
        st.record_intervention(change_type="config")
    st.record_intervention(change_type="code_patch")
    assert len(st.intervention_mix) == cap
    assert st.intervention_mix[-1]["change_type"] == "code_patch"


def test_specialist_rounds_capped():
    st = SharedState()
    cap = ss_mod._SPECIALIST_ROUNDS_CAP
    for i in range(cap + 80):
        st.record_specialist_round({"round_id": f"r{i}", "n": i})
    assert len(st.specialist_rounds) == cap
    assert st.specialist_rounds[-1]["round_id"] == f"r{cap + 80 - 1}"


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
    # Oldest keys evicted; newest retained.
    assert f"fp{cap + 300 - 1}" in out
    assert "fp0" not in out


def test_tested_ledger_cap_applied_on_merge():
    st = SharedState()
    cap = ss_mod._EXPLORE_TESTED_CAP
    tested = {f"fp{i}": {"i": i} for i in range(cap + 10)}
    st.apply_explore_search_update({"tested": tested})
    assert len(st.explore_search["tested"]) == cap


@pytest.mark.asyncio
async def test_append_rejects_a_non_string_recipient(conn):
    """``to_agent`` becomes a column: a list must be refused, not bound.

    Binding an unsupported type raises sqlite3.ProgrammingError from inside
    the INSERT, which faults the caller's open phase event.
    """
    bus = MessageBus(conn)
    with pytest.raises(ValueError, match="to_agent"):
        await bus.append_and_seq(Message.new("orchestration", ["specialist", "kernel_agent"], "observation", {"x": 1}))
    rows = await conn.fetchall("SELECT COUNT(*) AS n FROM events")
    assert rows[0]["n"] == 0


# Retained GPU ownership
@pytest.mark.asyncio
async def test_gpu_pool_retains_old_leases(conn):
    pool = SpecialistGpuPool(conn, gpu_ids=[0, 1, 2, 3])
    for gpu_id in (0, 1, 2):
        async with conn.transaction() as cur:
            cur.execute(
                "INSERT INTO gpu_leases(gpu_id, holder_id, task_id) VALUES (?,?,?)",
                (gpu_id, f"h{gpu_id}", f"t{gpu_id}"),
            )
    lease = await pool.try_acquire(count=1, holder_id="new", task_id="new")
    assert lease is not None and lease.gpu_ids == (3,)
    assert await pool.try_acquire(count=1, holder_id="full", task_id="full") is None
    remaining = await conn.fetchall("SELECT gpu_id FROM gpu_leases")
    assert sorted(r["gpu_id"] for r in remaining) == [0, 1, 2, 3]


# retry/backoff
@pytest.mark.asyncio
async def test_retry_with_backoff_recovers_then_succeeds():
    from hyperloom.orchestrator.roles.base import (
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
    from hyperloom.orchestrator.roles.base import (
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
    from hyperloom.orchestrator.roles.base import RetryPolicy

    monkeypatch.setenv("INFERENCE_OPTIMIZER_LLM_RETRY_ATTEMPTS", "1")
    pol = RetryPolicy.from_env()
    assert pol.max_attempts == 1  # disables retry


# coordinator maintenance tick (cadence + wiring)
@pytest.mark.asyncio
async def test_coordinator_maintenance_reports_leases_and_keeps_every_record(tmp_path, monkeypatch):
    """Crash recovery and the final report read a session's first result, and its first finished task keeps its key."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    from hyperloom.orchestrator.loop.coordinator import MAINTENANCE_INTERVAL_SEC, Coordinator
    from hyperloom.orchestrator.tests.mock_backend import MockBackend, ScriptedPlan
    from .conftest import seed_target_analysis_marker

    sd = make_session_dir()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
    }
    c = Coordinator(sd, backends=backends)
    try:
        # Seed a GPU lease, then a long session's worth of finished tasks and events behind the first of each.
        async with c.db.transaction() as cur:
            cur.execute("INSERT INTO gpu_leases(gpu_id, holder_id, task_id) VALUES (?,?,?)", (0, "h", "t"))
        c.gpu_specialist_pool = SpecialistGpuPool(c.db, gpu_ids=[0, 1])
        task, _ = await c.tasks.create_or_return_existing(TaskSpec(kind="explore", params={}, idempotency_key="k1"))
        await c.tasks.transition(task.task_id, "running")
        await c.tasks.transition(task.task_id, "succeeded")
        first = await c.bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", {"task_id": "t"}))
        later = now_iso()
        async with c.db.transaction() as cur:
            cur.executemany(
                "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                ((f"later-{i}", "explore", "succeeded", "{}", f"later-{i}", later, later) for i in range(2000)),
            )
            cur.executemany(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, payload, ts) VALUES (?,?,?,?,?,?)",
                (Message.new("orchestration", "*", "observation", {"i": i}).to_db_row() for i in range(6000)),
            )

        summary = await c.maintenance.run(tick=10)
        assert summary is not None
        assert "gpu_leases_reaped" not in summary
        assert "leases_reaped" in summary and "running_tasks_reclaimed" in summary
        assert [m.seq for m in await c.bus.tail(topic="delegated_result", n=-1)] == [first]
        assert (await c.tasks.get(task.task_id)).state == "succeeded"
        assert len(await c.tasks.by_state("succeeded")) == 2001
        rows = await c.db.fetchall("SELECT COUNT(*) AS c FROM gpu_leases")
        assert int(rows[0]["c"]) == 1
        # The wall-clock gate is seeded at construction, so tick 1 of a fresh
        # session must not already be past the interval.
        assert time.monotonic() - c._last_maintenance_ts < MAINTENANCE_INTERVAL_SEC
    finally:
        await c.stop()


# ClaudeBackend retries a transient SDK failure end-to-end
class ToolUseBlock:  # the backend dispatches on the type name, so do not rename
    def __init__(self, name: str, inp: dict[str, Any]):
        self.name = name
        self.input = inp


@dataclass
class _Msg:
    content: list[Any] = field(default_factory=list)
    session_id: str | None = "s1"
    result: str = ""


@pytest.mark.asyncio
async def test_claude_backend_retries_transient_then_succeeds(claude_sdk):
    from hyperloom.orchestrator.roles.claude import ClaudeBackend
    from hyperloom.orchestrator.roles.base import RetryPolicy
    from hyperloom.orchestrator.roles.mcp_emit_intent import (
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
                {"intent_type": "send_message", "payload": {"topic": "observation", "body_md": "ok"}},
            )
            yield _Msg(content=[block])

        return _gen()

    claude_sdk.query = _factory
    backend = ClaudeBackend(
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_s=0.0),
    )
    # Fast timeout so the first attempt's wait_for path is exercised.
    backend.call_timeout_s = 5.0

    result = await backend.run("hi")
    assert state["n"] == 2  # one transient failure, one success
    assert len(result.intents) == 1
