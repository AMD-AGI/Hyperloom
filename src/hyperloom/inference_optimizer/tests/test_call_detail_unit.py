# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace import call_detail


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_row_derives_isl_osl_and_tool_count():
    row = call_detail.CallDetailRecord(
        session_id="s1",
        component="orchestration",
        call_id="c1",
        api_call_index=0,
        model="unknown-to-the-card",
        input_tokens=10,
        output_tokens=4,
        reasoning_output_tokens=6,
        cache_read_input_tokens=100,
        cache_creation_input_tokens=20,
        tool_calls=[{"tool": "Bash"}, {"tool": "Read"}],
    ).to_row()
    assert row["isl"] == 130
    assert row["osl"] == 10
    assert row["tool_call_count"] == 2
    # An unpriced model must not be reported as a free call.
    assert row["cost_source"] == "unavailable"
    assert row["cost_usd"] is None


def test_row_field_set_is_exactly_the_closed_schema():
    row = call_detail.CallDetailRecord(session_id="s1", component="orchestration").to_row()
    assert set(row) == set(call_detail._ROW_FIELDS)


def test_task_path_depth_tracks_the_separator():
    row = call_detail.CallDetailRecord(
        session_id="s1", component="geak", task_path="geak/HeadKernel/lane-a"
    ).to_row()
    assert row["task_path"] == "geak/HeadKernel/lane-a"
    assert row["task_depth"] == 3


def test_append_writes_one_line_per_call_to_the_requested_destination(tmp_path: Path):
    dest = tmp_path / "shard.detail.jsonl"
    for i in range(3):
        call_detail.append_call_detail(
            session_dir=tmp_path,
            record=call_detail.CallDetailRecord(
                session_id="s1", component="geak", call_id="c1", api_call_index=i
            ),
            dest=dest,
        )
    rows = _rows(dest)
    assert [r["api_call_index"] for r in rows] == [0, 1, 2]
    assert {r["call_id"] for r in rows} == {"c1"}
    # The session ledger is untouched: the shard is the whole point.
    assert not (tmp_path / "reports").exists()


def test_bad_component_is_rejected_by_the_closed_schema_guard(tmp_path: Path):
    with pytest.raises(call_detail.CallDetailRowError):
        call_detail.append_call_detail(
            session_dir=tmp_path,
            record=call_detail.CallDetailRecord(session_id="s1", component="not-a-component"),
        )
