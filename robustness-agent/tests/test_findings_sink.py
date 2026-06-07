# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the findings JSONL sink."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness_agent.decision.action_ladder import Finding
from robustness_agent.findings import FindingSink, FindingSinkConfig


def _finding(**overrides) -> Finding:
    base = dict(
        tick_index=1,
        timestamp_unix=1.0,
        symptom_name="x",
        severity="medium",
        summary="s",
        intents=[{"intent_type": "alert", "payload": {"severity": "medium", "summary": "s"}}],
        evidence={"k": 1},
        rca_text="",
    )
    base.update(overrides)
    return Finding(**base)


@pytest.mark.asyncio
async def test_sink_appends_jsonl_rows(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    written = await sink.append_many([_finding(tick_index=1), _finding(tick_index=2)])
    assert written == 2
    path = sink.file_path
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert [r["tick_index"] for r in rows] == [1, 2]
    assert all(r["intents"][0]["intent_type"] == "alert" for r in rows)


@pytest.mark.asyncio
async def test_sink_appends_across_calls(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-2"))
    await sink.append_many([_finding(tick_index=1)])
    await sink.append_many([_finding(tick_index=2)])
    rows = sink.file_path.read_text().splitlines()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_sink_creates_subdirectories_when_missing(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="abc"))
    assert not sink.file_path.exists()
    await sink.append_many([_finding()])
    assert sink.file_path.exists()
    assert sink.file_path.parent.name == "findings"


@pytest.mark.asyncio
async def test_sink_is_resilient_to_io_errors(tmp_path: Path, caplog):
    cfg = FindingSinkConfig(session_dir=tmp_path / "non-existent-mount")
    sink = FindingSink(cfg)
    # Block creation by pre-creating a file where the parent dir would go
    blocker = tmp_path / "blocked"
    blocker.write_text("dummy")
    cfg2 = FindingSinkConfig(session_dir=blocker / "x", session_id="sess")
    sink2 = FindingSink(cfg2)
    with caplog.at_level("WARNING"):
        written = await sink2.append_many([_finding()])
    assert written == 1
    assert any("findings sink" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_sink_no_op_on_empty_iterable(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess"))
    assert await sink.append_many([]) == 0
    assert not sink.file_path.exists()
