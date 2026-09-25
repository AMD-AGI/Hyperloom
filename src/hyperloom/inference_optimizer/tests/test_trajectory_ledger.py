# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``reports/trace/trajectory/`` event ledger and its Langfuse projection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.session_paths import trajectory_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.trace import langfuse_emitter as lfe
from hyperloom.orchestrator.trace import trajectory_projection as trajmap
from hyperloom.orchestrator.trace import trajectory_trace as tt

from .test_langfuse_emitter import _enable_env, _FakeClient, _install_fake_sdk, _write_manifest


@pytest.fixture(autouse=True)
def _clear_registry():
    lfe._REGISTRY.clear()
    yield
    lfe._REGISTRY.clear()


def _rows(session_dir: Path) -> list[dict]:
    return tt.load_events(session_dir)


def test_record_event_is_a_noop_without_a_session_in_scope(tmp_path):
    assert tt.record_event(tt.EVENT_SESSION) is None
    assert not trajectory_dir(tmp_path).exists()


def test_rows_carry_the_closed_schema_and_ambient_keys(tmp_path):
    sd = tmp_path / "SID"
    with tt.trajectory_scope(session_dir=sd, component="coordinator", agent="orchestration", tick=4, phase="EXPLORE"):
        tt.record_event(tt.EVENT_SESSION, attributes={"name": "x", "path": Path("/p")})
        tt.record_event(tt.EVENT_SESSION, task_id="t-1")

    shards = tt.trajectory_shards(sd)
    assert len(shards) == 1
    first, second = _rows(sd)
    assert set(first) == tt._ROW_FIELDS
    assert first["schema_version"] == tt.SCHEMA_VERSION
    assert first["session_id"] == "SID"
    assert (first["component"], first["agent"], first["tick"], first["phase"]) == (
        "coordinator",
        "orchestration",
        4,
        "EXPLORE",
    )
    assert first["attributes"] == {"name": "x", "path": "/p"}
    assert first["status"] == tt.STATUS_POINT
    assert second["task_id"] == "t-1"
    assert first["writer"] == second["writer"] == shards[0].stem
    assert second["seq"] == first["seq"] + 1


def test_unknown_vocabulary_is_rejected(tmp_path):
    with tt.trajectory_scope(session_dir=tmp_path):
        with pytest.raises(tt.TrajectoryRowError):
            tt.record_event("not-an-event")
        with pytest.raises(tt.TrajectoryRowError):
            tt.record_event(tt.EVENT_SESSION, status="done")
        with pytest.raises(tt.TrajectoryRowError):
            tt.record_event(tt.EVENT_SESSION, component="nobody")


def test_phase_and_tick_follow_the_live_source_when_unset(tmp_path):
    live = {"phase": "BASELINE", "tick": 1}
    with tt.trajectory_scope(session_dir=tmp_path, phase_tick_source=lambda: (live["phase"], live["tick"])):
        tt.record_event(tt.EVENT_SESSION)
        live.update(phase="EXPLORE", tick=2)
        tt.record_event(tt.EVENT_SESSION)
        tt.record_event(tt.EVENT_SESSION, tick=9)
    assert [(r["phase"], r["tick"]) for r in _rows(tmp_path)] == [("BASELINE", 1), ("EXPLORE", 2), ("EXPLORE", 9)]


def test_span_pairs_rows_and_parents_its_children(tmp_path):
    with tt.trajectory_scope(session_dir=tmp_path):
        with tt.trajectory_span(tt.EVENT_SESSION, attributes={"name": "outer"}) as span:
            tt.record_event(tt.EVENT_SESSION)
            span.finish(stop_reason="max_ticks")
    started, child, closed = _rows(tmp_path)
    assert (started["status"], closed["status"]) == (tt.STATUS_STARTED, tt.STATUS_COMPLETED)
    assert started["span_id"] == closed["span_id"] == span.span_id
    assert closed["start_ts"] <= started["ts"] <= closed["ts"]
    assert closed["attributes"] == {"stop_reason": "max_ticks"}
    assert child["parent_span_id"] == span.span_id
    assert started["parent_span_id"] is None


def test_span_records_failure_and_cancellation(tmp_path):
    with tt.trajectory_scope(session_dir=tmp_path):
        with pytest.raises(RuntimeError):
            with tt.trajectory_span(tt.EVENT_SESSION):
                raise RuntimeError("boom")
        with pytest.raises(asyncio.CancelledError):
            with tt.trajectory_span(tt.EVENT_SESSION):
                raise asyncio.CancelledError()
    closed = [r for r in _rows(tmp_path) if r["status"] in tt.TERMINAL_STATUSES]
    assert [r["status"] for r in closed] == [tt.STATUS_FAILED, tt.STATUS_CANCELLED]
    assert closed[0]["attributes"] == {"error_type": "RuntimeError", "error_message": "boom"}


@pytest.mark.asyncio
async def test_asyncio_tasks_and_threads_inherit_the_scope(tmp_path):
    def _in_thread() -> None:
        tt.record_event(tt.EVENT_SESSION, attributes={"where": "thread"})

    async def _in_task() -> None:
        tt.record_event(tt.EVENT_SESSION, attributes={"where": "task"})

    with tt.trajectory_scope(session_dir=tmp_path, task_id="t-9"):
        await asyncio.create_task(_in_task())
        await asyncio.to_thread(_in_thread)
    rows = _rows(tmp_path)
    assert sorted(r["attributes"]["where"] for r in rows) == ["task", "thread"]
    assert {r["task_id"] for r in rows} == {"t-9"}


def _row(**over) -> dict:
    base = {
        "event_type": "session",
        "status": tt.STATUS_POINT,
        "ts": "2026-06-09T15:00:10+00:00",
        "span_id": "s1",
        "parent_span_id": None,
        "component": "coordinator",
        "agent": None,
        "phase": "EXPLORE",
        "tick": 2,
        "task_id": None,
        "call_id": None,
        "attributes": {},
    }
    base.update(over)
    return base


def test_projection_starts_a_span_at_its_started_row():
    queued = _row(status=tt.STATUS_QUEUED, ts="2026-06-09T15:00:00+00:00", parent_span_id="p0")
    started = _row(status=tt.STATUS_STARTED, ts="2026-06-09T15:00:05+00:00", attributes={"name": "go"})
    failed = _row(status=tt.STATUS_FAILED, ts="2026-06-09T15:00:09+00:00", attributes={"error_message": "bad"})
    openings = trajmap.span_openings([queued, started, failed])

    assert trajmap.project_row(started, openings) is None
    spec = trajmap.project_row(failed, openings)
    assert spec is not None
    assert spec.name == "session:go"
    assert spec.start.isoformat() == "2026-06-09T15:00:05+00:00"
    assert spec.end.isoformat() == "2026-06-09T15:00:09+00:00"
    assert spec.level == "ERROR" and spec.status_message == "bad"
    assert spec.metadata["parent_span_id"] == "p0"
    assert spec.metadata["queued_ts"] == "2026-06-09T15:00:00+00:00"
    assert spec.agent == "coordinator"


def _write_shard(sd: Path, name: str, rows: list[dict]) -> Path:
    shard = trajectory_dir(sd) / name
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return shard


def test_flush_projects_spans_only_and_resumes_the_shard_cursor(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    _write_shard(sd, "1-a.jsonl", [_row(status=tt.STATUS_STARTED), _row(status=tt.STATUS_COMPLETED)])

    first = _FakeClient()
    _install_fake_sdk(monkeypatch, first)
    lfe.LangfuseEmitter(sd).flush_session()
    assert first.generations == []
    assert [s.kwargs["name"] for s in first.spans if s.kwargs.get("metadata", {}).get("kind") == "trajectory"] == [
        "session"
    ]
    assert lfe.read_receipt(sd)["trajectory_rows_sent"] == {"1-a.jsonl": 2}

    _write_shard(sd, "1-a.jsonl", [_row(span_id="s2", attributes={"name": "later"})])
    second = _FakeClient()
    _install_fake_sdk(monkeypatch, second)
    lfe._REGISTRY.clear()
    lfe.LangfuseEmitter(sd).flush_session()
    projected = [s.kwargs["name"] for s in second.spans if s.kwargs.get("metadata", {}).get("kind") == "trajectory"]
    assert projected == ["session:later"]
    assert lfe.read_receipt(sd)["trajectory_rows_sent"] == {"1-a.jsonl": 3}


def test_a_repeat_flush_ships_the_trajectory_tail_the_close_flush_preceded(tmp_path, monkeypatch):
    _enable_env(monkeypatch)
    sd = tmp_path / "SID"
    _write_manifest(sd)
    _write_shard(sd, "1-a.jsonl", [_row(status=tt.STATUS_STARTED)])
    client = _FakeClient()
    _install_fake_sdk(monkeypatch, client)
    emitter = lfe.LangfuseEmitter(sd)
    emitter.flush_session()
    assert emitter._flushed is True
    flushed = client.flushed

    _write_shard(
        sd,
        "1-a.jsonl",
        [
            _row(span_id="close", phase="CLOSE", attributes={"name": "tail"}),
            _row(status=tt.STATUS_COMPLETED, attributes={"stop_reason": "time_exhausted"}),
        ],
    )
    emitter.flush_session()

    projected = [s for s in client.spans if s.kwargs.get("metadata", {}).get("kind") == "trajectory"]
    assert [s.kwargs["name"] for s in projected] == ["session:tail", "session"]
    assert client.flushed == flushed + 1
    assert all(s.ended for s in client.spans)
    assert lfe.read_receipt(sd)["trajectory_rows_sent"] == {"1-a.jsonl": 3}

    emitter.flush_session()
    assert len([s for s in client.spans if s.kwargs.get("metadata", {}).get("kind") == "trajectory"]) == 2
    assert client.flushed == flushed + 1


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


@pytest.mark.asyncio
async def test_coordinator_run_is_one_session_span(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    c = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    try:
        assert await c.run(max_ticks=2) == "max_ticks"
    finally:
        await c.stop()
    sessions = [r for r in _rows(session_dir) if r["event_type"] == tt.EVENT_SESSION]
    assert [r["status"] for r in sessions] == [tt.STATUS_STARTED, tt.STATUS_COMPLETED]
    assert sessions[1]["attributes"]["stop_reason"] == "max_ticks"
    assert sessions[1]["tick"] == 2
    assert {r["component"] for r in sessions} == {"coordinator"}
