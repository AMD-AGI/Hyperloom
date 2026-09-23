# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prompt manifests, prefix-break detection, and context compaction on the trajectory ledger."""

from __future__ import annotations

import json

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.roles.claude_requests import ClaudeRequestTracker
from hyperloom.orchestrator.trace import context_events as ce
from hyperloom.orchestrator.trace import trajectory_trace as tt

_PROMPT = "\n".join(
    [
        "SESSION_DIR=/s",
        "=== Phase ===",
        "EXPLORE tick 3",
        "=== Recent policy denials (newest last, total=2) ===",
        "- a",
        "=== Inbox for orchestration (newest last) ===",
        "  m1",
    ]
)


def _rows(session_dir, event_type: str) -> list[dict]:
    return [r for r in tt.load_events(session_dir) if r["event_type"] == event_type]


def test_sections_split_at_headings_and_drop_per_render_detail():
    sections = ce.prompt_sections(_PROMPT)
    assert [s.key for s in sections] == [ce.PREAMBLE, "Phase", "Recent policy denials", "Inbox for orchestration"]
    assert sum(s.chars for s in sections) == len(_PROMPT)
    assert sections[2].title == "Recent policy denials (newest last, total=2)"


def test_a_tail_change_keeps_the_prefix_and_names_the_section_that_broke_it():
    tracker = ce.PromptSnapshotTracker()
    first = tracker.observe("orchestration", prompt=_PROMPT, system_prompt="sys", tools=["b", "a"])
    assert first["first_prompt"] is True
    assert first["prompt_tokens_est"] == -(-len(_PROMPT) // ce.CHARS_PER_TOKEN)
    assert [s["key"] for s in first["sections"]] == [s.key for s in ce.prompt_sections(_PROMPT)]

    grown = _PROMPT.replace("total=2) ===\n- a", "total=3) ===\n- a\n- b") + "\n  m2"
    second = tracker.observe("orchestration", prompt=grown, system_prompt="sys", tools=["a", "b"])
    assert second["first_prompt"] is False
    assert second["first_changed_section"] == "Recent policy denials"
    assert second["changed_sections"] == ["Recent policy denials", "Inbox for orchestration"]
    assert (second["added_sections"], second["removed_sections"]) == ([], [])
    assert second["lcp_chars"] == _PROMPT.index("total=2") + len("total=")
    assert 0 < second["lcp_ratio"] < 1
    assert (second["system_prompt_changed"], second["tools_changed"], second["prefix_break"]) == (False, False, False)

    third = tracker.observe("orchestration", prompt=grown, system_prompt="sys v2", tools=["a", "b"])
    assert third["first_changed_section"] is None
    assert third["lcp_ratio"] == 1.0
    assert third["prefix_break"] is True

    other = tracker.observe("critic", prompt=grown, system_prompt="sys v2", tools=None)
    assert other["first_prompt"] is True
    assert other["tools_count"] is None


def test_added_and_removed_sections_are_reported():
    tracker = ce.PromptSnapshotTracker()
    tracker.observe("o", prompt="=== A ===\nx\n=== B ===\ny", system_prompt=None, tools=None)
    diff = tracker.observe("o", prompt="=== A ===\nx\n=== C ===\nz", system_prompt=None, tools=None)
    assert (diff["added_sections"], diff["removed_sections"]) == (["C"], ["B"])
    assert diff["first_changed_section"] == "C"


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


@pytest.mark.asyncio
async def test_each_reactor_call_records_its_prompt_snapshot_under_the_call(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    c = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    try:
        assert await c.run(max_ticks=2) == "max_ticks"
    finally:
        await c.stop()

    calls = {r["span_id"]: r for r in _rows(session_dir, tt.EVENT_LLM_CALL) if r["status"] == tt.STATUS_STARTED}
    snapshots = _rows(session_dir, tt.EVENT_PROMPT_SNAPSHOT)
    assert snapshots
    for row in snapshots:
        call = calls[row["parent_span_id"]]
        assert (row["call_id"], row["agent"]) == (call["call_id"], call["agent"])
        assert row["attributes"]["name"] == row["agent"]
        assert "prompt" not in row["attributes"]
    orchestration = [r for r in snapshots if r["agent"] == "orchestration"]
    assert [r["attributes"]["first_prompt"] for r in orchestration][:2] == [True, False]


def test_an_sdk_compact_boundary_is_a_compaction_event(tmp_path):
    sdk_types = pytest.importorskip("claude_agent_sdk.types")
    tracker = ClaudeRequestTracker(clock=iter([0.0, 5.0, 6.0]).__next__)
    tracker.observe(
        sdk_types.SystemMessage(
            subtype="compact_boundary",
            data={"type": "system", "compact_metadata": {"trigger": "auto", "pre_tokens": 180000}},
        )
    )
    tracker.observe(sdk_types.SystemMessage(subtype="init", data={}))
    with tt.trajectory_scope(session_dir=tmp_path, component="orchestration"):
        tracker.record_trajectory(attempt=1, fallback_model=None)
    (row,) = _rows(tmp_path, tt.EVENT_CONTEXT_COMPACTION)
    assert row["status"] == tt.STATUS_POINT
    assert row["attributes"] == {"name": "auto", "trigger": "auto", "pre_tokens": 180000, "attempt": 1}
    assert row["ts"].startswith("1970-01-01T00:00:05")


def test_specialist_stream_json_compactions_are_recorded(tmp_path):
    log = tmp_path / "process.log"
    rows = [
        {"type": "system", "subtype": "init"},
        {"type": "system", "subtype": "compact_boundary", "compact_metadata": {"trigger": "manual", "pre_tokens": 9}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "compact_boundary"}]}},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert ce.record_stream_json_compactions(log) == 0
    with tt.trajectory_scope(session_dir=tmp_path, task_id="t-1"):
        assert ce.record_stream_json_compactions(log) == 1
    (row,) = _rows(tmp_path, tt.EVENT_CONTEXT_COMPACTION)
    assert row["task_id"] == "t-1"
    assert row["attributes"] == {"name": "manual", "trigger": "manual", "pre_tokens": 9, "timing_source": "none"}
