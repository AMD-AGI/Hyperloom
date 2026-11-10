# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The status file a tracelens_analysis run keeps current, and what a terminal state adds to it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis._run_status import RunStatus


def _status(tmp_path: Path, started_at: str) -> RunStatus:
    log = tmp_path / "run.log"
    log.write_text("first line\nsecond line\n", encoding="utf-8")
    return RunStatus(
        status_path=tmp_path / "status.json",
        log_path=log,
        artifact_paths={"trace_report": str(tmp_path / "report.json")},
        run_id="run-1",
        started_at=started_at,
    )


def _written(status: RunStatus) -> dict:
    return json.loads(status.status_path.read_text(encoding="utf-8"))


def test_a_running_state_carries_no_end(tmp_path: Path):
    """A collector builds the timeline from ``ended_at``, so a live run must not look finished."""
    status = _status(tmp_path, "2026-01-01T00:00:00+00:00")

    status.write(state="running", current_step="analysing")

    payload = _written(status)
    assert payload["state"] == "running"
    assert payload["run_id"] == "run-1"
    assert payload["artifact_paths"] == {"trace_report": str(tmp_path / "report.json")}
    assert "ended_at" not in payload
    assert "duration_seconds" not in payload


@pytest.mark.parametrize("state", ["succeeded", "failed", "aborted", "cancelled"])
def test_every_terminal_state_records_an_end_and_a_duration(tmp_path: Path, state: str):
    """The timeline event is built from these two fields; a terminal state that omits them has no duration at all."""
    status = _status(tmp_path, "2026-01-01T00:00:00+00:00")

    status.write(state=state, current_step="done", error="boom" if state == "failed" else None)

    payload = _written(status)
    assert payload["ended_at"] == payload["updated_at"]
    assert payload["duration_seconds"] >= 0.0
    if state == "failed":
        assert payload["error"] == "boom"


def test_a_naive_start_time_is_read_as_utc(tmp_path: Path):
    """A start time written without an offset still has to subtract from an aware one."""
    status = _status(tmp_path, "2026-01-01T00:00:00")

    status.write(state="succeeded", current_step="done")

    assert _written(status)["duration_seconds"] >= 0.0


def test_a_start_time_that_cannot_be_parsed_leaves_the_duration_unknown(tmp_path: Path):
    """An unparsable stamp is reported as no duration, not as a crash while writing the status file."""
    status = _status(tmp_path, "not-a-timestamp")

    status.write(state="succeeded", current_step="done")

    payload = _written(status)
    assert payload["ended_at"] == payload["updated_at"]
    assert payload["duration_seconds"] is None
