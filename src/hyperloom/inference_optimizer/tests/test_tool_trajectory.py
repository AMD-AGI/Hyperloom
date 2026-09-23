# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tool call / result rows on the trajectory ledger, from the reactor SDK stream and from specialist logs."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.roles.claude_requests import ClaudeRequestTracker
from hyperloom.orchestrator.trace import tool_events as te
from hyperloom.orchestrator.trace import trajectory_trace as tt

sdk_types = pytest.importorskip("claude_agent_sdk.types")

_SECRET = "sk-ant-api03-" + "A" * 40


def _clock(*ticks: float):
    it = iter(ticks)
    return lambda: next(it)


def _tools(session_dir) -> list[dict]:
    return [r for r in tt.load_events(session_dir) if r["event_type"] == tt.EVENT_TOOL]


def test_reactor_tools_are_timed_from_use_to_result_under_their_request(tmp_path):
    # t=0 open; assistant m1 at 1 (request) and 1 (tool uses); results at 4; attempt end at 9.
    tracker = ClaudeRequestTracker(clock=_clock(0.0, 1.0, 1.0, 4.0, 9.0))
    tracker.observe(
        sdk_types.AssistantMessage(
            content=[
                sdk_types.ToolUseBlock(id="tu-1", name="Bash", input={"command": f"curl -H 'x-api-key: {_SECRET}'"}),
                sdk_types.ToolUseBlock(id="tu-2", name="Read", input={"path": "/tmp/a"}),
                sdk_types.ToolUseBlock(id="tu-3", name="Grep", input={"pattern": "x"}),
            ],
            model="claude-x",
            message_id="m1",
        )
    )
    tracker.observe(
        sdk_types.UserMessage(
            content=[
                sdk_types.ToolResultBlock(tool_use_id="tu-1", content="done", is_error=False),
                sdk_types.ToolResultBlock(
                    tool_use_id="tu-2", content=[{"type": "text", "text": f"denied {_SECRET}"}], is_error=True
                ),
            ]
        )
    )
    with tt.trajectory_scope(session_dir=tmp_path, component="orchestration", call_id="c-1"):
        tracker.record_trajectory(attempt=1, fallback_model=None)

    (request,) = [r for r in tt.load_events(tmp_path) if r["event_type"] == tt.EVENT_LLM_REQUEST]
    bash, read, grep = _tools(tmp_path)
    assert [r["status"] for r in (bash, read, grep)] == [tt.STATUS_COMPLETED, tt.STATUS_FAILED, tt.STATUS_CANCELLED]
    assert {r["parent_span_id"] for r in (bash, read, grep)} == {request["span_id"]}
    assert {r["call_id"] for r in (bash, read, grep)} == {"c-1"}
    assert bash["attributes"]["name"] == "Bash"
    assert bash["attributes"]["latency_ms"] == 3000
    assert (bash["attributes"]["result_chars"], bash["attributes"]["is_error"]) == (4, False)
    assert _SECRET not in json.dumps(bash)
    assert read["attributes"]["is_error"] is True
    assert "denied" in read["attributes"]["error_preview"]
    assert _SECRET not in read["attributes"]["error_preview"]
    assert grep["attributes"]["latency_ms"] is None
    assert "result_chars" not in grep["attributes"]
    assert grep["start_ts"] < grep["ts"]


def test_input_summary_is_clipped_and_sized():
    attributes = te.tool_attributes(name="Write", tool_use_id="t", tool_input={"content": "x" * 5000})
    assert attributes["input_chars"] > 5000
    assert len(attributes["input_summary"]) <= 241


def _write_log(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{torn", encoding="utf-8")


def test_specialist_stream_json_tools_pair_use_and_result(tmp_path):
    log = tmp_path / "process.log"
    _write_log(
        log,
        [
            {"type": "system", "subtype": "init", "model": "claude-x"},
            {
                "type": "assistant",
                "message": {
                    "id": "m1",
                    "content": [
                        {"type": "text", "text": "looking"},
                        {"type": "tool_use", "id": "a", "name": "WebSearch", "input": {"query": "rocm gemm"}},
                        {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "false"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "results"},
                        {"type": "tool_result", "tool_use_id": "b", "content": "exit 1", "is_error": True},
                        {"type": "tool_result", "tool_use_id": "unknown", "content": "?"},
                    ]
                },
            },
            {
                "type": "assistant",
                "parent_tool_use_id": "a",
                "message": {"id": "m2", "content": [{"type": "tool_use", "id": "c", "name": "Read", "input": {}}]},
            },
        ],
    )
    with tt.trajectory_scope(session_dir=tmp_path, component="specialist", task_id="task-1", parent_span_id="task-1"):
        assert te.record_stream_json_tools(log) == 3

    search, bash, read = _tools(tmp_path)
    assert [r["attributes"]["name"] for r in (search, bash, read)] == ["WebSearch", "Bash", "Read"]
    assert [r["status"] for r in (search, bash, read)] == [tt.STATUS_COMPLETED, tt.STATUS_FAILED, tt.STATUS_CANCELLED]
    assert {r["task_id"] for r in (search, bash, read)} == {"task-1"}
    assert {r["parent_span_id"] for r in (search, bash, read)} == {"task-1"}
    assert search["attributes"]["input_summary"] == "rocm gemm"
    assert search["attributes"]["message_id"] == "m1"
    assert search["attributes"]["timing_source"] == te.TIMING_NONE
    assert read["attributes"]["parent_tool_use_id"] == "a"


def test_specialist_tools_are_a_noop_outside_a_session(tmp_path):
    assert te.record_stream_json_tools(tmp_path / "missing.log") == 0
    log = tmp_path / "process.log"
    _write_log(log, [{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "a", "name": "X"}]}}])
    assert te.record_stream_json_tools(log) == 0
