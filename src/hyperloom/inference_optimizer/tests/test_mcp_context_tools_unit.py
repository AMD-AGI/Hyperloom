# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the in-process context-pull MCP tool surface."""

from __future__ import annotations

import asyncio
import contextvars
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import mcp_context_tools as mct


def _shared_state():
    """Build a fake SharedState exposing the to_*_summary projections."""
    return SimpleNamespace(
        to_mission_summary=lambda: "mission",
        to_prompt_summary=lambda: "prompt",
        to_gaps_summary=lambda max_attempts=0: "gaps",
        to_warm_start_summary=lambda: "warm",
        to_proposal_scores_summary=lambda: "scores",
        to_intervention_mix_summary=lambda: "mix",
        to_policy_denial_summary=lambda top_k=6: f"denials({top_k})",
        failures=[],
        find_failure=lambda fid: None,
        failures_for_task=lambda tid: [],
    )


def test_qualified():
    assert mct._qualified("foo") == "mcp__inference_optimizer_context__foo"


def test_provider_projections():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.mission_status() == "mission"
    assert p.shared_state_summary() == "prompt"
    assert p.gaps() == "gaps"
    assert p.warm_start() == "warm"
    assert p.proposal_scores() == "scores"
    assert p.intervention_mix() == "mix"


def test_safe_handles_exception():
    def boom():
        raise RuntimeError("x")

    p = mct.ContextProvider(shared_state=SimpleNamespace())
    out = p._safe(boom, "lbl")
    assert "unavailable" in out


def test_safe_empty_marker():
    p = mct.ContextProvider(shared_state=SimpleNamespace())
    assert p._safe(lambda: "", "lbl") == "(lbl: empty)"
    assert p._safe(lambda: None, "lbl") == "(lbl: empty)"


def test_why_denied_via_shared_state():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.why_denied(top_k=3) == "denials(3)"


def test_why_denied_via_reader():
    p = mct.ContextProvider(
        shared_state=_shared_state(),
        denial_reader=lambda k: f"reader({k})",
    )
    assert p.why_denied(top_k=2) == "reader(2)"


def test_analysis_md_not_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.analysis_md()


def test_analysis_md_wired():
    p = mct.ContextProvider(shared_state=_shared_state(), analysis_reader=lambda: "md")
    assert p.analysis_md() == "md"


def test_inbox_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.inbox()
    p2 = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    assert p2.inbox(5) == "inbox(5)"


def test_recent_outcomes_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.recent_outcomes()
    p2 = mct.ContextProvider(
        shared_state=_shared_state(),
        recent_outcomes_reader=lambda k: f"out({k})",
    )
    assert p2.recent_outcomes(4) == "out(4)"


async def test_run_action_now_not_wired_and_wired():
    async def run(name, params):
        return f"ran {name} {params}"

    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in await p.run_action_now("a")
    p2 = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    assert await p2.run_action_now("act", {"k": 1}) == "ran act {'k': 1}"


def test_tool_name_tuples():
    assert "get_mission_status" in mct.CONTEXT_TOOL_NAMES
    assert mct.CONTEXT_TOOL_QUALIFIED_NAMES[0].startswith("mcp__")
    assert len(mct.CONTEXT_TOOL_NAMES) == len(mct.CONTEXT_TOOL_SPECS)


# ---- synchronous conversation context readers ----


@pytest.fixture
def context_reader(tmp_path, monkeypatch):
    from hyperloom.orchestrator.bus.message_bus import MessageBus
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.loop import conversation
    from hyperloom.orchestrator.state.task_registry import TaskRegistry

    db = SqliteConnection(tmp_path / "context.db", journal_mode="DELETE")
    coordinator = SimpleNamespace(bus=MessageBus(db), tasks=TaskRegistry(db), session_dir=tmp_path)
    reader = conversation.ConversationCollaborator(coordinator)
    monkeypatch.setattr(conversation, "_format_inbox_event", lambda msg, **_kwargs: msg.msg_id)
    monkeypatch.setattr(conversation.time, "time", lambda: 1000.0)
    try:
        yield reader, db, conversation
    finally:
        db.close()


def _context_event(db, name, *, topic="observation", sender="worker", recipient="orchestration", payload=None):
    from hyperloom.orchestrator.bus.message_bus import Message

    msg = Message(name, sender, recipient, topic, payload or {})
    cursor = db.raw.execute(
        "INSERT INTO events (msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts) VALUES (?,?,?,?,?,?,?)",
        msg.to_db_row(),
    )
    return cursor.lastrowid


def _context_task(db, name, *, state="running", updated_at="1970-01-01T00:15:00+00:00"):
    db.raw.execute(
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, lease_ttl_sec, created_at, updated_at) "
        "VALUES (?, 'explore', ?, ?, ?, 60, ?, ?)",
        (name, state, json.dumps({"domain": "kernel", "gap_canonical_id": "gap-1"}), name, updated_at, updated_at),
    )


def test_context_reader_inbox_keeps_exact_recipient_cursor_and_unfiltered_topics(context_reader):
    reader, db, _ = context_reader
    since = _context_event(db, "at-cursor")
    _context_event(db, "other-recipient", recipient="critic")
    _context_event(db, "self-request", topic="request", sender="orchestration")
    _context_event(db, "broadcast", topic="response", recipient="*")
    for index in range(205):
        _context_event(db, f"event-{index}")

    assert reader._context_inbox_reader(str(since)).splitlines() == [
        "self-request",
        "broadcast",
        *(f"event-{index}" for index in range(205)),
    ]
    assert reader._context_inbox_reader(9999) == "(no inbox events)"
    assert reader._context_inbox_reader(None).splitlines()[0] == "at-cursor"
    assert reader._context_inbox_reader("invalid").startswith("(inbox unavailable: ValueError(")


@pytest.mark.parametrize("top_k, count", [(0, 8), (-5, 1), (999, 50), (3, 3), (None, 8), ("bad", 8)])
def test_context_reader_recent_outcomes_filters_only_topics_and_keeps_latest_chronological(
    context_reader, top_k, count
):
    reader, db, _ = context_reader
    for index in range(55):
        _context_event(
            db,
            f"outcome-{index}",
            topic=("delegated_result", "review_verdict")[index % 2],
            sender="orchestration",
            recipient="critic",
        )
        _context_event(db, f"ignored-{index}", topic="observation")

    assert reader._context_recent_outcomes_reader(top_k).splitlines() == [
        "=== Recent action outcomes (newest last) ===",
        *(f"outcome-{index}" for index in range(55 - count, 55)),
    ]


def test_context_reader_recent_outcomes_keeps_variant_and_total_tail_caps(context_reader, monkeypatch):
    reader, db, conversation = context_reader
    calls = []

    def format_event(msg, *, max_variant_rows):
        calls.append((msg.msg_id, max_variant_rows))
        return "\n".join(f"{msg.msg_id}:{index}" for index in range(max_variant_rows + 1))

    monkeypatch.setattr(conversation, "_format_inbox_event", format_event)
    for index in range(10):
        _context_event(db, f"outcome-{index}", topic="delegated_result")
    expected = [f"outcome-{event}:{row}" for event in range(10) for row in range(13)]

    assert reader._context_recent_outcomes_reader(10).splitlines() == [
        "=== Recent action outcomes (newest last) ===",
        *expected[-120:],
        "(truncated at 120 lines; re-query with a smaller top_k)",
    ]
    assert calls == [(f"outcome-{index}", 12) for index in range(10)]


@pytest.mark.parametrize(
    "method, empty, unavailable",
    [
        ("_context_inbox_reader", "(no inbox events)", "inbox"),
        ("_context_recent_outcomes_reader", "(no recent outcomes)", "recent outcomes"),
        ("_context_running_tasks_reader", "(no tasks in flight)", "running tasks"),
    ],
)
def test_context_reader_empty_and_first_query_failure(context_reader, monkeypatch, method, empty, unavailable):
    reader, db, _ = context_reader
    assert getattr(reader, method)() == empty

    def fail(*_args):
        raise RuntimeError("read failed")

    monkeypatch.setattr(db, "fetchall_sync", fail)
    assert getattr(reader, method)() == f"({unavailable} unavailable: RuntimeError('read failed'))"


@pytest.mark.parametrize("method", ["_context_inbox_reader", "_context_recent_outcomes_reader"])
def test_context_reader_message_decode_error_is_not_a_query_failure(context_reader, method):
    reader, db, _ = context_reader
    _context_event(db, "malformed", topic="delegated_result")
    db.raw.execute("UPDATE events SET payload='not-json'")
    with pytest.raises(json.JSONDecodeError):
        getattr(reader, method)()


def test_context_reader_running_empty_short_circuits_resource_reads(context_reader, monkeypatch):
    reader, db, _ = context_reader
    _context_task(db, "queued", state="queued")
    reads = []
    fetch = db.fetchall_sync

    def read(sql, params=()):
        reads.append(sql)
        return fetch(sql, params)

    monkeypatch.setattr(db, "fetchall_sync", read)
    assert reader._context_running_tasks_reader() == "(no tasks in flight)"
    assert len(reads) == 1
    assert "FROM tasks" in reads[0]


def test_context_reader_running_order_resources_and_separate_reads(context_reader, monkeypatch):
    reader, db, _ = context_reader
    _context_task(db, "newer", updated_at="1970-01-01T00:16:00+00:00")
    _context_task(db, "older")
    _context_task(db, "finished", state="succeeded")
    reads = []
    fetch = db.fetchall_sync

    def read(sql, params=()):
        rows = fetch(sql, params)
        reads.append(sql)
        if "FROM tasks" in sql:
            for lane, expiry in [
                ("profile_lane", "1970-01-01T00:17:20+00:00"),
                ("benchmark_lane", "1970-01-01T00:17:00+00:00"),
            ]:
                db.raw.execute(
                    "INSERT INTO leases (lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at) "
                    "VALUES (?, 'holder', 'older', 'explore', 1, '', ?, '')",
                    (lane, expiry),
                )
        elif "FROM leases" in sql:
            for gpu in (3, 1):
                db.raw.execute(
                    "INSERT INTO gpu_leases VALUES (?, 'holder', 'older', '', '', '')",
                    (gpu,),
                )
        return rows

    monkeypatch.setattr(db, "fetchall_sync", read)
    assert reader._context_running_tasks_reader().splitlines() == [
        "=== Tasks in flight ===",
        "  - task_id=older kind='explore' running_sec=100 domain='kernel' gap='gap-1' "
        "idempotency_key='older' lease_ttl_sec=60 lease_expires_in_sec=20 "
        "lanes=['benchmark_lane', 'profile_lane'] gpu_ids=[1, 3]",
        "  - task_id=newer kind='explore' running_sec=40 domain='kernel' gap='gap-1' "
        "idempotency_key='newer' lease_ttl_sec=60",
    ]
    assert len(reads) == 3
    assert all(f"FROM {table}" in sql for table, sql in zip(("tasks", "leases", "gpu_leases"), reads))
    assert db.raw.in_transaction is False


@pytest.mark.parametrize("failing_table, expected_reads", [("leases", 2), ("gpu_leases", 3)])
def test_context_reader_running_resource_errors_propagate(context_reader, monkeypatch, failing_table, expected_reads):
    reader, db, _ = context_reader
    _context_task(db, "running")
    fetch = db.fetchall_sync
    reads = []

    def read(sql, params=()):
        reads.append(sql)
        if f"FROM {failing_table}" in sql:
            raise RuntimeError(f"{failing_table} failed")
        return fetch(sql, params)

    monkeypatch.setattr(db, "fetchall_sync", read)
    with pytest.raises(RuntimeError, match=f"{failing_table} failed"):
        reader._context_running_tasks_reader()
    assert len(reads) == expected_reads


def test_context_reader_running_task_decode_follows_resource_reads(context_reader, monkeypatch):
    reader, db, _ = context_reader
    _context_task(db, "malformed")
    db.raw.execute("UPDATE tasks SET params='not-json'")
    fetch = db.fetchall_sync
    reads = []

    def read(sql, params=()):
        reads.append(sql)
        return fetch(sql, params)

    monkeypatch.setattr(db, "fetchall_sync", read)
    with pytest.raises(json.JSONDecodeError):
        reader._context_running_tasks_reader()
    assert len(reads) == 3


@pytest.mark.parametrize("format_fails", [False, True])
def test_context_reader_running_formats_each_task_before_decoding_the_next(context_reader, monkeypatch, format_fails):
    reader, db, _ = context_reader
    _context_task(db, "first")
    _context_task(db, "malformed", updated_at="1970-01-01T00:16:00+00:00")
    db.raw.execute("UPDATE tasks SET params='not-json' WHERE task_id='malformed'")
    formatted = []

    def heartbeat(task, *, now_unix):
        formatted.append(task.task_id)
        if format_fails:
            raise RuntimeError("first format failed")
        return None

    monkeypatch.setattr(reader, "_task_heartbeat_age_sec", heartbeat)
    error = RuntimeError if format_fails else json.JSONDecodeError
    with pytest.raises(error):
        reader._context_running_tasks_reader()
    assert formatted == ["first"]


# ---- _resolve_sdk ----


def test_resolve_sdk_explicit():
    sentinel = object()
    assert mct._resolve_sdk(sentinel) is sentinel


def test_resolve_sdk_import_error(monkeypatch):
    def boom(_name):
        raise ImportError("no sdk")

    monkeypatch.setattr(mct.importlib, "import_module", boom)
    assert mct._resolve_sdk(None) is None


# ---- _make_handler ----


async def test_make_handler_success():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "mission_status")
    out = await handler({})
    assert out["content"][0]["text"] == "mission"


async def test_make_handler_forwards_kwargs():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "why_denied")
    out = await handler({"top_k": 2})
    assert out["content"][0]["text"] == "denials(2)"


async def test_make_handler_forwards_since_seq():
    p = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    handler = mct._make_handler(p, "inbox")
    out = await handler({"since_seq": 9})
    assert out["content"][0]["text"] == "inbox(9)"


async def test_make_handler_forwards_action_args():
    async def run(name, params):
        return f"{name}:{params}"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    handler = mct._make_handler(p, "run_action_now")
    out = await handler({"action_name": "act", "params": {"k": 1}})
    assert out["content"][0]["text"] == "act:{'k': 1}"


async def test_make_handler_non_str_result():
    provider = SimpleNamespace(foo=lambda **k: {"a": 1})
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert json.loads(out["content"][0]["text"]) == {"a": 1}


async def test_make_handler_exception():
    def boom(**k):
        raise RuntimeError("nope")

    provider = SimpleNamespace(foo=boom)
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert out["is_error"] is True


async def test_action_handler_awaits_on_loop_and_preserves_contextvars():
    trace = contextvars.ContextVar("mcp_test_trace", default="missing")
    owner_thread = threading.get_ident()
    seen = []

    async def execute(name, params):
        seen.append((threading.get_ident(), trace.get(), name, params))
        trace.set("action-only")
        return "done"

    async def run(name, params):
        return await asyncio.create_task(execute(name, params))

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    token = trace.set("caller-trace")
    try:
        out = await mct._make_handler(p, "run_action_now")({"action_name": "probe", "params": {"k": 1}})
        assert seen == [(owner_thread, "caller-trace", "probe", {"k": 1})]
        assert trace.get() == "caller-trace"
        assert out == {"content": [{"type": "text", "text": "done"}]}
    finally:
        trace.reset(token)


async def test_read_projection_stays_on_the_calling_thread():
    owner_thread = threading.get_ident()
    seen = []

    def read():
        seen.append(threading.get_ident())
        return "mission"

    p = mct.ContextProvider(shared_state=SimpleNamespace(to_mission_summary=read))
    out = await mct._make_handler(p, "mission_status")({})
    assert seen == [owner_thread]
    assert out == {"content": [{"type": "text", "text": "mission"}]}


@pytest.mark.parametrize("params", [{"k": 1}, None, "not an object"])
async def test_action_handler_keeps_argument_normalization(params):
    seen = []

    async def run(name, arguments):
        seen.append((name, arguments))
        return "done"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    await mct._make_handler(p, "run_action_now")({"action_name": 123, "params": params, "ignored": True})
    assert seen == [("123", params if isinstance(params, dict) else {})]
    if isinstance(params, dict):
        assert seen[0][1] is not params


async def test_action_handler_keeps_provider_error_envelope():
    async def run(_name, _params):
        raise ValueError("action failed")

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {
        "content": [{"type": "text", "text": "(context tool run_action_now unavailable: ValueError('action failed'))"}]
    }


@pytest.mark.parametrize("value", ["", None, {"not": "text"}])
async def test_action_handler_keeps_empty_marker(value):
    async def run(_name, _params):
        return value

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {"content": [{"type": "text", "text": "(run_action_now: empty)"}]}


async def test_action_handler_keeps_handler_error_envelope(monkeypatch):
    async def run(**_kwargs):
        raise RuntimeError("bridge failed")

    p = mct.ContextProvider(shared_state=_shared_state())
    monkeypatch.setattr(p, "run_action_now", run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {"content": [{"type": "text", "text": "error: RuntimeError('bridge failed')"}], "is_error": True}


async def test_action_handler_propagates_provider_cancellation():
    async def run(_name, _params):
        raise asyncio.CancelledError("action cancelled")

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    with pytest.raises(asyncio.CancelledError, match="action cancelled"):
        await mct._make_handler(p, "run_action_now")({"action_name": "probe"})


@pytest.fixture
async def loop_action():
    """Await a real scheduled Future without lending it the caller's cancellation."""
    loop = asyncio.get_running_loop()
    probe = SimpleNamespace(
        started=asyncio.Event(),
        release=asyncio.Event(),
        finished=asyncio.Event(),
        submitted=[],
        calls=[],
        owner_thread=threading.get_ident(),
        bridge_threads=[],
    )

    async def execute(name, params):
        probe.calls.append((name, params, threading.get_ident()))
        probe.started.set()
        await probe.release.wait()
        probe.finished.set()
        return "action complete"

    async def run(name, params):
        probe.bridge_threads.append(threading.get_ident())
        future = asyncio.run_coroutine_threadsafe(execute(name, params), loop)
        probe.submitted.append(future)
        return await asyncio.shield(asyncio.wrap_future(future))

    probe.provider = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    try:
        yield probe
    finally:
        probe.release.set()
        await asyncio.wait_for(
            asyncio.gather(*(asyncio.wrap_future(future) for future in probe.submitted), return_exceptions=True),
            2.0,
        )


async def _sdk_request(server, request):
    """Exercise the request handler registered by the SDK on MCP 1.x or 2.x."""
    if hasattr(server, "get_request_handler"):
        entry = server.get_request_handler(request.method)
        assert entry is not None
        params = entry.params_type.model_validate(request.params.model_dump() if request.params else {})
        return await entry.handler(None, params)
    result = await server.request_handlers[type(request)](request)
    return result.root


@pytest.fixture(params=["direct", "sdk"])
async def action_handler(request, loop_action):
    if request.param == "direct":
        handler = mct._make_handler(loop_action.provider, "run_action_now")

        async def call(arguments):
            result = await handler(arguments)
            assert not result.get("is_error")
            return result["content"][0]["text"]

        return call

    sdk = pytest.importorskip("claude_agent_sdk")
    from mcp import types

    config = mct.build_context_tools_server(loop_action.provider, sdk_module=sdk)
    assert config["type"] == "sdk"
    assert config["name"] == mct.MCP_SERVER_NAME
    server = config["instance"]
    listed = await _sdk_request(server, types.ListToolsRequest())
    assert {tool.name for tool in listed.tools} == set(mct.CONTEXT_TOOL_NAMES)
    action = next(tool for tool in listed.tools if tool.name == "run_action_now")
    assert action.model_dump(by_alias=True)["inputSchema"] == mct._RUN_ACTION_SCHEMA

    async def call(arguments):
        result = await _sdk_request(
            server,
            types.CallToolRequest(
                method="tools/call", params=types.CallToolRequestParams(name="run_action_now", arguments=arguments)
            ),
        )
        assert not result.model_dump(by_alias=True)["isError"]
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        return result.content[0].text

    return call


async def test_action_handler_keeps_loop_timers_moving_before_action_completes(loop_action, action_handler):
    probe = loop_action
    call = asyncio.create_task(action_handler({"action_name": "probe", "params": {"k": 1}}))
    timer = asyncio.Event()
    handle = None
    try:
        await asyncio.wait_for(probe.started.wait(), 2.0)
        handle = asyncio.get_running_loop().call_later(0.01, timer.set)
        await asyncio.wait_for(timer.wait(), 0.5)
        assert not call.done(), "the handler must still be waiting while the loop timer advances"
        assert not probe.finished.is_set()
        assert probe.calls == [("probe", {"k": 1}, probe.owner_thread)]
        assert probe.bridge_threads == [probe.owner_thread]
        probe.release.set()
        out = await asyncio.wait_for(call, 1.0)
        assert out == "action complete"
    finally:
        if handle is not None:
            handle.cancel()
        probe.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_action_handler_cancellation_does_not_cancel_submitted_action(loop_action, action_handler):
    probe = loop_action
    call = asyncio.create_task(action_handler({"action_name": "probe"}))
    try:
        await asyncio.wait_for(probe.started.wait(), 2.0)
        assert not call.done(), "the handler must yield before the bridge wait expires"
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 0.5)
        assert not probe.finished.is_set()
        assert not probe.submitted[0].cancelled()
        probe.release.set()
        result = await asyncio.wait_for(asyncio.wrap_future(probe.submitted[0]), 1.0)
        assert result == "action complete"
        assert probe.calls == [("probe", {}, probe.owner_thread)]
    finally:
        probe.release.set()
        await asyncio.gather(call, return_exceptions=True)


# ---- build_context_tools_server ----


def test_build_server_unavailable_without_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert mct.build_context_tools_server(p, sdk_module=object()) is None


def test_build_server_with_fake_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    created = {}

    def tool_factory(name, desc, schema):
        def decorator(handler):
            return (name, handler)

        return decorator

    def server_factory(name, version, tools):
        created["name"] = name
        created["tools"] = tools
        return "SERVER"

    out = mct.build_context_tools_server(
        p,
        tool_factory=tool_factory,
        server_factory=server_factory,
    )
    assert out == "SERVER"
    assert created["name"] == mct.MCP_SERVER_NAME
    assert len(created["tools"]) == len(mct.CONTEXT_TOOL_SPECS)


async def test_action_provider_awaits_async_callback():
    async def run(name, params):
        await asyncio.sleep(0)
        return f"{name}:{params}"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    assert await p.run_action_now("probe", {"k": 1}) == "probe:{'k': 1}"


async def _run_saturation_probe(db_path, use_sdk):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    db = SqliteConnection(db_path)
    executed, submitted = [], []
    all_started, worker_entered = asyncio.Event(), asyncio.Event()
    release_worker = threading.Event()
    started = 0

    def hold_worker():
        loop.call_soon_threadsafe(worker_entered.set)
        release_worker.wait(5)

    async def execute(name, params):
        nonlocal started
        started += 1
        if started == 20:
            all_started.set()
        row = await db.fetchone("SELECT ? AS value", (params["index"],))
        executed.append(row["value"])
        return f"{name}:{row['value']}"

    async def run(name, params):
        future = asyncio.run_coroutine_threadsafe(execute(name, params), loop)
        submitted.append(future)
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), 3.0)

    provider = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    handler = mct._make_handler(provider, "run_action_now")
    if use_sdk:
        import claude_agent_sdk
        from mcp import types

        server = mct.build_context_tools_server(provider, sdk_module=claude_agent_sdk)["instance"]

        async def call(index):
            result = await _sdk_request(
                server,
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(
                        name="run_action_now", arguments={"action_name": "probe", "params": {"index": index}}
                    ),
                ),
            )
            assert not result.model_dump(by_alias=True)["isError"]
            return result.content[0].text

    else:

        async def call(index):
            result = await handler({"action_name": "probe", "params": {"index": index}})
            assert not result.get("is_error")
            return result["content"][0]["text"]

    blocker = asyncio.create_task(asyncio.to_thread(hold_worker))
    calls = []
    try:
        await asyncio.wait_for(worker_entered.wait(), 1.0)
        calls = [asyncio.create_task(call(index)) for index in range(20)]
        await asyncio.wait_for(all_started.wait(), 1.0)
        assert not blocker.done()
        assert all(not task.done() for task in calls)
        release_worker.set()
        results = await asyncio.wait_for(asyncio.gather(*calls), 2.0)
        assert results == [f"probe:{index}" for index in range(20)]
        assert sorted(executed) == list(range(20))
    finally:
        release_worker.set()
        await asyncio.gather(blocker, *calls, return_exceptions=True)
        await asyncio.gather(*(asyncio.wrap_future(future) for future in submitted), return_exceptions=True)
        db.close()


@pytest.mark.parametrize("use_sdk", [False, True], ids=["direct", "sdk"])
def test_concurrent_actions_leave_default_executor_available(tmp_path, use_sdk):
    if use_sdk:
        pytest.importorskip("claude_agent_sdk")
    source_root = Path(mct.__file__).resolve().parents[3]
    command = (
        "import asyncio,runpy,sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        f"module=runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"asyncio.run(module['_run_saturation_probe']({str(tmp_path / 'probe.db')!r}, {use_sdk!r}))"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


# ---- real dispatcher lifecycle (POSIX coordinator dependencies) ----


@pytest.fixture
async def inline_coordinator(session_dir, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("Coordinator imports POSIX-only resource/fcntl modules")
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "1")
    coord = Coordinator(
        session_dir,
        backends={name: MockBackend(ScriptedPlan(turns=[]), name=name) for name in ("orchestration", "critic")},
    )
    coord._coordinator_loop = asyncio.get_running_loop()
    try:
        yield coord
    finally:
        await coord.stop()


async def test_action_handler_rejects_baseline_with_default_whitelist(inline_coordinator):
    coord = inline_coordinator
    assert "baseline" not in coord._inline_action_whitelist()
    p = mct.ContextProvider(shared_state=coord.shared_state, action_runner=coord.dispatcher._run_action_now_wait)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "baseline"})
    assert "not inline-eligible" in out["content"][0]["text"]
    assert await coord.tasks.queued() == []
    assert await coord.tasks.running() == []
    assert not coord.dispatcher._executions
    assert not coord.dispatcher._inflight_actions


@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_action_handler_keeps_real_dispatcher_execution_exactly_once(inline_coordinator, monkeypatch, abandon):
    from hyperloom.orchestrator.actions.cancel_channel import stop_was_asked_for

    coord = inline_coordinator
    started, release = asyncio.Event(), asyncio.Event()
    trace = contextvars.ContextVar("mcp_dispatch_trace", default="missing")
    calls, bridge_threads = [], []
    owner_thread = threading.get_ident()
    cancelled = []

    async def execute(ctx):
        calls.append((ctx.task.task_id, threading.get_ident(), trace.get()))
        trace.set("action-only")
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        assert not stop_was_asked_for()
        return {"status": "ok", "gain_pct": 1.5}

    async def run(name, params):
        bridge_threads.append(threading.get_ident())
        return await coord.dispatcher._run_action_now_wait(name, params)

    coord.sub.register_executor("target_analysis", execute)
    assert "target_analysis" in coord._inline_action_whitelist()
    monkeypatch.setattr(coord.policy, "validate_intent", lambda *_args: None)
    monkeypatch.setattr(coord.dispatcher, "_admission_denial_for_action", lambda _action: None)
    if abandon == "timeout":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "0.05")
    p = mct.ContextProvider(shared_state=coord.shared_state, action_runner=run)
    handler = mct._make_handler(p, "run_action_now")
    arguments = {"action_name": "target_analysis", "params": {"probe": "same action"}}
    token = trace.set("request-trace")
    call = asyncio.create_task(handler(arguments))
    execution_handle = None
    try:
        await asyncio.wait_for(started.wait(), 2.0)
        task_id = calls[0][0]
        execution_handle = coord.dispatcher._inflight_actions[task_id].atask
        if abandon == "timeout":
            out = await asyncio.wait_for(call, 1.0)
            assert "still running after" in out["content"][0]["text"]
        else:
            assert not call.done()
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call, 0.5)
        assert not cancelled
        assert (await coord.tasks.get(task_id)).state == "running"
        assert not execution_handle.done()
        assert coord.dispatcher._executions
        repeated = await handler(arguments)
        assert "already 'running'" in repeated["content"][0]["text"]
        release.set()
        await asyncio.wait_for(asyncio.shield(execution_handle), 2.0)
        completed = await handler(arguments)
        assert "inline run complete" in completed["content"][0]["text"]
        assert "gain=1.5" in completed["content"][0]["text"]
        assert calls == [(task_id, owner_thread, "request-trace")]
        assert all(tid == owner_thread for tid in bridge_threads)
        assert trace.get() == "request-trace"
        assert (await coord.tasks.get(task_id)).state == "succeeded"
        outcomes = await coord.bus.tail(topic="delegated_result")
        assert len(outcomes) == 1
        assert outcomes[0].payload["task_id"] == task_id
        assert outcomes[0].payload["inline"] is True
        assert not coord.dispatcher._executions
        assert not coord.dispatcher._inflight_actions
        assert not await coord.locks.lane_holders()
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        if execution_handle is not None:
            await asyncio.wait_for(asyncio.shield(execution_handle), 2.0)
        trace.reset(token)


# ---- new tools: get_failure / get_variant_failures ----


def test_spec_methods_all_exist_on_provider():
    """Every CONTEXT_TOOL_SPECS entry must name a real ContextProvider method."""
    for _, _, _, method_name in mct.CONTEXT_TOOL_SPECS:
        assert callable(getattr(mct.ContextProvider, method_name, None)), (
            f"ContextProvider missing method {method_name!r}"
        )


def test_get_failure_not_found_points_at_the_disk_mirror():
    """An evicted packet is still reachable, so the miss must say where."""
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_failure("fail.t1.abc")
    assert "reports/failures/fail.t1.abc.json" in out


def test_get_failure_returns_json():
    import json

    fe = {"failure_id": "fail.t1.abc", "task_id": "t1", "error_class": "x"}
    ss = _shared_state()
    ss.find_failure = lambda fid: fe if fid == "fail.t1.abc" else None
    p = mct.ContextProvider(shared_state=ss)
    out = p.get_failure("fail.t1.abc")
    data = json.loads(out)
    assert data["failure_id"] == "fail.t1.abc"


def test_get_failure_requires_failure_id():
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_failure("")
    assert "required" in out


def test_get_variant_failures_empty():
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_variant_failures()
    assert "no failure" in out


def test_get_variant_failures_returns_entries():
    import json

    fe1 = {"failure_id": "fail.t1.a", "task_id": "t1"}
    fe2 = {"failure_id": "fail.t1.b", "task_id": "t1"}
    ss = _shared_state()
    ss.failures = [fe1, fe2]
    ss.failures_for_task = lambda tid: [fe for fe in [fe1, fe2] if fe["task_id"] == tid]
    p = mct.ContextProvider(shared_state=ss)
    out = p.get_variant_failures(task_id="t1")
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2
    ids = {json.loads(l)["failure_id"] for l in lines}
    assert ids == {"fail.t1.a", "fail.t1.b"}


async def test_make_handler_forwards_failure_id():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "get_failure")
    out = await handler({"failure_id": "fail.t1.abc"})
    assert "fail.t1.abc" in out["content"][0]["text"]
