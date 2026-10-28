# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The host tools against a session directory built by the real writers, and over HTTP with real tokens."""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from hyperloom.hosted.events import HostedProgressEvents
from hyperloom.hosted.mcp_host_tools import register_host_tools
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.inference_optimizer.session.optimization_journal import Journal, JournalEntry
from hyperloom.inference_optimizer.session.paths import db_path_for
from hyperloom.inference_optimizer.trace.llm_trace import LLMCallRecord, LLMCallSite, append_llm_call
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.loop.signals import SignalDrain
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.plugs import progress

HOST_TOKEN = "host-token-0123456789abcdef"
RUN_ID = "run-1"
CANDIDATES = [("task-a", "REVERT", -1.5, "FRAMEWORK_AGENT"), ("task-b", "KEEP", 10.0, "FRAMEWORK_AGENT")]


def _tasks(session: Path) -> None:
    async def build() -> None:
        path = db_path_for(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = SqliteConnection(path)
        ensure_schema(db.raw)
        registry = TaskRegistry(db)
        for task_id, *_ in CANDIDATES:
            task, _ = await registry.create_or_return_existing(
                TaskSpec(kind="explore", params={}, idempotency_key=task_id, task_id=task_id)
            )
            await registry.transition(task.task_id, "running")
            await registry.transition(task.task_id, "succeeded")
        live, _ = await registry.create_or_return_existing(
            TaskSpec(kind="specialist", params={}, idempotency_key="live", task_id="task-live")
        )
        await registry.transition(live.task_id, "running")
        db.close()

    asyncio.run(build())


@pytest.fixture
def session(tmp_path, monkeypatch) -> Path:
    """A session directory written by SharedState.save, Journal, TaskRegistry, append_llm_call and the events writer."""
    root = tmp_path / "session"
    root.mkdir()
    state = SharedState(session_id="s-1", phase="FRAMEWORK_AGENT", macro_cycle=2, max_minutes=120, baseline_tput=1000.0)
    state.tick = 7
    state.elapsed_charged_sec = 600.0
    state.current_best = {"action": "explore", "tput": 1100.0, "variant_name": "v-b", "extra_server_args": "--x 1"}
    state.current_best["extra_envs"] = {"A": "1"}
    state.current_best_measurement = {"tput": 1100.0, "declared_launch_identity": "id-1", "benchmark_workspace": "/w/b"}
    state.optimization_stack = [{"stack_entry_id": "e1", "action": "explore", "variant_name": "v-b", "tput": 1100.0}]
    state.gain_per_stack_entry = [10.0]
    state.save(root)

    journal = Journal.load_or_create(root, Journal(session_id="s-1", model="m", hardware="gpu"))
    for task_id, outcome, gain, phase in CANDIDATES:
        journal.append_entry(
            JournalEntry(
                phase=phase,
                iter=1,
                kind="param",
                change=f"change {task_id}",
                outcome=outcome,
                gain_pct=gain,
                task_id=task_id,
            )
        )
    journal.append_entry(
        JournalEntry(phase="KERNEL_AGENT", iter=2, kind="kernel_file", change="k", outcome="KEEP", task_id="task-k")
    )

    _tasks(root)

    site = LLMCallSite(session_id="s-1", component="orchestration")
    append_llm_call(
        session_dir=root,
        record=LLMCallRecord.from_metadata(
            site, {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 5}
        ),
    )
    append_llm_call(
        session_dir=root, record=LLMCallRecord.from_metadata(site, {"input_tokens": 50, "output_tokens": 20})
    )
    append_llm_call(session_dir=root, record=LLMCallRecord.for_failure(site, RuntimeError("boom")))

    events = HostedProgressEvents(tmp_path / "events", RUN_ID)
    monkeypatch.setattr(progress, "emit", events.emit)
    progress.best_updated(1050.0, state, "req-old")
    progress.best_updated(1100.0, state, "req-new")
    events.close()
    return root


@pytest.fixture
def events_dir(session) -> Path:
    return session.parent / "events"


class Rig:
    def __init__(self, session: Path, events_dir: Path, stops: list[str]) -> None:
        self.session = session
        self.server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign=RUN_ID)
        self.tools = register_host_tools(self.server, session, events_dir, stop=lambda: stops.append("stop"))
        self.server.start()
        self.initialized = False

    def post(self, token: str | None, method: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Hyperloom-Token"] = token
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
        connection = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=10)
        connection.request("POST", "/mcp", body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, json.loads(raw)

    def call(self, tool: str, arguments: dict[str, Any] | None = None, token: str = HOST_TOKEN) -> Any:
        status, reply = self.post(token, "tools/call", {"name": tool, "arguments": arguments or {}})
        assert status == 200
        if "error" in reply:
            return reply
        return (
            json.loads(reply["result"]["content"][0]["text"])
            if not reply["result"]["isError"]
            else reply["result"]["content"][0]["text"]
        )


@pytest.fixture
def stops() -> list[str]:
    return []


@pytest.fixture
def rig(session, events_dir, stops):
    running = Rig(session, events_dir, stops)
    yield running
    running.server.stop()


def test_get_status_reads_state_and_tasks(rig):
    status = rig.call("get_status")
    assert status["phase"] == "FRAMEWORK_AGENT" and status["macro_round"] == 2 and status["tick"] == 7
    assert status["total_sec"] == 7200.0 and status["elapsed_sec"] >= 600.0
    assert status["stop_reason"] is None and status["stopped"] is False
    assert status["best"]["value"] == 1100.0 and status["best"]["lift_pct"] == 10.0
    assert [t["task_id"] for t in status["in_flight"]] == ["task-live"]
    assert status["in_flight"][0]["state"] == "running" and status["warnings"] == []


def test_get_status_reports_a_stop_reason(rig):
    state = json.loads((rig.session / "state.json").read_text())
    state.update(stop_reason="target_reached", stop_ts="2026-10-26T00:00:00Z")
    (rig.session / "state.json").write_text(json.dumps(state))
    status = rig.call("get_status")
    assert status["stop_reason"] == "target_reached" and status["stopped"] is True


def test_get_status_without_state_is_an_error_not_an_empty_answer(rig):
    (rig.session / "state.json").unlink()
    assert "has not written its state" in rig.call("get_status")


def test_get_status_flags_an_unreadable_task_database(rig):
    db_path_for(rig.session).write_bytes(b"not a database" * 100)
    status = rig.call("get_status")
    assert status["in_flight"] is None and "in-flight tasks cannot be read" in status["warnings"][0]
    assert status["phase"] == "FRAMEWORK_AGENT"


def test_get_best(rig):
    best = rig.call("get_best")
    assert best["metric"] == "output_throughput" and best["value"] == 1100.0 and best["lift_pct"] == 10.0
    assert best["variant"] == {"extra_server_args": "--x 1", "extra_envs": {"A": "1"}}
    assert best["patches"] == [
        {"stack_entry_id": "e1", "action": "explore", "variant_name": "v-b", "tput": 1100.0, "gain_pct": 10.0}
    ]
    assert best["measurement"] == {
        "request_id": "req-new",
        "sandbox": None,
        "launch_identity": "id-1",
        "workspace": "/w/b",
    }


def test_get_best_before_any_best(rig):
    state = json.loads((rig.session / "state.json").read_text())
    state["current_best"] = {}
    (rig.session / "state.json").write_text(json.dumps(state))
    assert rig.call("get_best")["best"] is None


def test_list_candidates_newest_first_with_task_kind_and_duration(rig):
    rows = rig.call("list_candidates")["candidates"]
    assert [r["task_id"] for r in rows] == ["task-k", "task-b", "task-a"]
    b = rows[1]
    assert b["action"] == "explore" and b["decision"] == "KEEP" and b["lift_pct"] == 10.0 and b["duration_sec"] >= 0
    assert rows[0]["action"] == "kernel_file" and rows[0]["duration_sec"] is None  # no task row: not invented


def test_list_candidates_limit_and_phase(rig):
    assert [r["task_id"] for r in rig.call("list_candidates", {"limit": 1})["candidates"]] == ["task-k"]
    by_phase = rig.call("list_candidates", {"phase": "framework_agent"})
    assert [r["task_id"] for r in by_phase["candidates"]] == ["task-b", "task-a"] and by_phase["total"] == 2


def test_list_candidates_rejects_a_bad_limit(rig):
    assert rig.call("list_candidates", {"limit": 0})["error"]["code"] == -32602


def test_get_measurement_from_the_events_and_current_best(rig):
    new = rig.call("get_measurement", {"request_id": "req-new"})
    assert new["best_update"]["value"] == 1100.0 and new["best_update"]["patch_count"] == 1
    assert new["current_best_measurement"]["launch_identity"] == "id-1"
    old = rig.call("get_measurement", {"request_id": "req-old"})
    assert old["best_update"]["value"] == 1050.0 and "current_best_measurement" not in old


def test_get_measurement_unknown_request_is_an_error(rig):
    assert "no measurement is recorded" in rig.call("get_measurement", {"request_id": "nope"})


def test_get_measurement_includes_a_recorded_sandbox(rig):
    path = rig.session / "hosted" / "measurement_sandboxes.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"request_id": "req-new", "sandbox_id": "sb-1", "gpu_model": "MI300X"}) + "\n")
    assert rig.call("get_measurement", {"request_id": "req-new"})["sandbox"]["sandbox_id"] == "sb-1"
    assert rig.call("get_best")["measurement"]["sandbox"]["sandbox_id"] == "sb-1"


def test_get_costs_sums_ok_rows_and_counts_failures(rig):
    costs = rig.call("get_costs")
    assert costs["tokens"]["input_tokens"] == 150 and costs["tokens"]["output_tokens"] == 30
    assert costs["tokens"]["cache_read_input_tokens"] == 5
    assert costs["llm_calls"] == 2 and costs["llm_calls_failed"] == 1 and costs["skipped_lines"] == 0
    assert costs["gpu_seconds"] is None and costs["subrun_count"] is None


def test_get_costs_counts_a_corrupt_line_instead_of_hiding_it(rig):
    ledger = next(rig.session.rglob("llm_calls.jsonl"))
    with ledger.open("a") as handle:
        handle.write("{not json\n")
    assert rig.call("get_costs")["skipped_lines"] == 1


def test_request_stop_writes_stop_json_then_signals(rig, stops):
    order = []
    rig.tools._stop = lambda: order.append((rig.session / "hosted" / "control" / "stop.json").exists())
    assert rig.call("request_stop", {"reason": "operator asked"}) == {"accepted": True}
    assert order == [True]  # the file is on disk before the signal goes out
    written = json.loads((rig.session / "hosted" / "control" / "stop.json").read_text())
    assert written["reason"] == "operator asked" and written["by"] == "host"


def test_request_stop_with_an_empty_reason_writes_nothing_and_signals_nothing(rig, stops):
    assert rig.call("request_stop", {"reason": "  "})
    assert not (rig.session / "hosted" / "control").exists() and stops == []


def test_request_stop_does_not_signal_when_the_file_cannot_be_written(rig, stops):
    (rig.session / "hosted").write_text("a file where the control directory must go")
    assert "could not be written" in rig.call("request_stop", {"reason": "x"})
    assert stops == []


def test_request_stop_takes_the_real_signal_path(session, events_dir):
    """No stub: the default stopper SIGTERMs this process and SignalDrain turns it into the operator stop."""
    loop = asyncio.new_event_loop()
    drain = SignalDrain(loop=loop, stop_event=asyncio.Event())
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign=RUN_ID)
    register_host_tools(server, session, events_dir)
    server.start()
    try:
        assert drain.armed
        outcome: list[Any] = []

        def ask() -> None:
            outcome.append(
                Rig.post(
                    type("R", (), {"server": server})(),
                    HOST_TOKEN,
                    "tools/call",
                    {"name": "request_stop", "arguments": {"reason": "stop now"}},
                )
            )

        thread = threading.Thread(target=ask)
        thread.start()
        deadline = time.monotonic() + 10
        while not drain.requested.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        thread.join(timeout=10)
        assert drain.requested.is_set()
        assert (session / "hosted" / "control" / "stop.json").is_file()
        assert outcome[0][0] == 200
    finally:
        received = drain.close()
        server.stop()
        loop.close()
    assert received


def test_initialize_then_every_tool_over_http_and_auth_boundaries(rig, stops):
    status, reply = rig.post(
        HOST_TOKEN,
        "initialize",
        {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    )
    assert status == 200 and reply["result"]["serverInfo"]["name"] == "hyperloom"
    _, listed = rig.post(HOST_TOKEN, "tools/list")
    names = sorted(t["name"] for t in listed["result"]["tools"])
    assert names == ["get_best", "get_costs", "get_measurement", "get_status", "list_candidates", "request_stop"]
    for tool, arguments in [
        ("get_status", {}),
        ("get_best", {}),
        ("list_candidates", {"limit": 5}),
        ("get_measurement", {"request_id": "req-new"}),
        ("get_costs", {}),
        ("request_stop", {"reason": "done"}),
    ]:
        status, reply = rig.post(HOST_TOKEN, "tools/call", {"name": tool, "arguments": arguments})
        assert status == 200 and reply["result"]["isError"] is False, tool
    assert stops == ["stop"]

    for bad in (None, "", "wrong-token-0123456789"):
        status, reply = rig.post(bad, "tools/call", {"name": "get_status", "arguments": {}})
        assert status == 401 and reply == {"error": "unauthorized"}

    specialist = rig.server.issue_token("spec-1", "branch-1")
    _, listed = rig.post(specialist, "tools/list")
    assert listed["result"]["tools"] == []
    for tool in names:
        _, reply = rig.post(specialist, "tools/call", {"name": tool, "arguments": {"reason": "x", "request_id": "r"}})
        assert reply["error"]["code"] == -32001 and "result" not in reply, tool
    assert stops == ["stop"]  # the specialist's request_stop never ran


def test_elapsed_time_is_the_budget_rule_of_the_state_itself(session):
    from datetime import datetime, timezone

    recorded = json.loads((session / "state.json").read_text())
    for fields in ({}, {"leg_anchor_unix": 5000.0}, {"elapsed_charged_sec": 0.0, "start_ts": "1970-01-01T00:00:00Z"}):
        recorded.update(fields)
        live = SharedState.from_dict(recorded)
        live.leg_anchor_unix = recorded.get("leg_anchor_unix", 0.0)  # a loaded state starts no leg of its own
        now = datetime.fromtimestamp(9000.0, timezone.utc)
        assert SharedState.elapsed_sec_of(recorded, now_unix=9000.0) == pytest.approx(
            live.elapsed_minutes(now=now) * 60.0
        )
