# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the AgentX workload timeline built from aiperf's per-request export.

The record shape below is aiperf's own documented one (``docs/tutorials/working-with-profile-exports.md`` at the pinned
commit), not one invented here: getting the field names wrong is the whole risk in reading somebody else's artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.actions.executors._agentx_timeline import (
    TIMELINE_ARTIFACT_NAME,
    build_events,
    correlate_rows,
    find_profile_export,
    parse_profile_export,
    write_timeline,
)

BASE_NS = 1_759_813_207_000_000_000
BASE = BASE_NS / 1e9


def _record(
    *,
    request_id: str,
    correlation: str,
    turn: int,
    start_offset_ms: float,
    duration_ms: float,
    ttft_ms: float | None = 250.0,
    conversation: str = "conv-1",
    error: dict | None = None,
) -> dict:
    """One line of aiperf's ``profile_export.jsonl``, in its documented shape."""
    start_ns = BASE_NS + int(start_offset_ms * 1e6)
    metrics: dict = {"input_sequence_length": {"value": 550, "unit": "tokens"}}
    if ttft_ms is not None:
        metrics["time_to_first_token"] = {"value": ttft_ms, "unit": "ms"}
    return {
        "metadata": {
            "session_num": 45,
            "x_request_id": request_id,
            "x_correlation_id": correlation,
            "conversation_id": conversation,
            "turn_index": turn,
            "request_start_ns": start_ns,
            "request_ack_ns": start_ns + 100_000_000,
            "request_end_ns": start_ns + int(duration_ms * 1e6),
            "worker_id": "worker_359d423a",
            "record_processor_id": "record_processor_1fa47cd7",
            "benchmark_phase": "profiling",
            "was_cancelled": False,
            "cancellation_time_ns": None,
        },
        "metrics": metrics,
        "error": error,
    }


def _export(tmp_path: Path, records: list[dict], *, nested: bool = False) -> Path:
    """Write an export where aiperf would put it."""
    art = tmp_path / ("benchmark_agentx_1" if nested else ".") / "aiperf_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    path = art / "profile_export.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_export_is_found_beside_the_round(tmp_path):
    """The same resolution the progress address and the aiperf log use."""
    _export(tmp_path, [_record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=500)])
    assert find_profile_export(tmp_path) is not None


def test_export_is_found_in_a_nested_benchmark_directory(tmp_path):
    """An AgentX round nests its benchmark one level below the server log."""
    _export(
        tmp_path,
        [_record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=500)],
        nested=True,
    )
    assert find_profile_export(tmp_path) is not None


def test_absent_export_is_not_an_error(tmp_path):
    """Every synthetic (non-AgentX) round is this case."""
    assert find_profile_export(tmp_path) is None


def test_records_parse_with_the_identifiers_the_hierarchy_needs(tmp_path):
    """Trajectory, turn and request identity all come off one record."""
    path = _export(
        tmp_path,
        [_record(request_id="req-a", correlation="traj-1", turn=2, start_offset_ms=1000, duration_ms=400)],
    )
    (record,) = parse_profile_export(path)

    assert record.request_id == "req-a"
    assert record.trajectory_id == "traj-1"
    assert record.conversation_id == "conv-1"
    assert record.turn_index == 2
    assert record.start == BASE + 1.0
    assert record.end == BASE + 1.4
    # Derived from TTFT, not from the ack: the ack is acceptance, not the first token.
    assert record.first_token == BASE + 1.25


def test_a_truncated_last_line_does_not_cost_the_rest(tmp_path):
    """The export is written as the round ends; a partial final line is normal."""
    path = _export(tmp_path, [_record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=100)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"metadata": {"x_request_id": "half')

    assert len(parse_profile_export(path)) == 1


def test_records_without_usable_times_are_dropped(tmp_path):
    """A record that cannot be placed on the timeline is worse than absent."""
    bad = _record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=100)
    bad["metadata"]["request_start_ns"] = None
    good = _record(request_id="r2", correlation="t1", turn=0, start_offset_ms=10, duration_ms=100)

    assert [r.request_id for r in parse_profile_export(_export(tmp_path, [bad, good]))] == ["r2"]


def test_event_stream_covers_every_layer(tmp_path):
    """phase / trajectory / turn / request, which is the timeline that was asked for."""
    records = parse_profile_export(
        _export(
            tmp_path,
            [
                _record(request_id="r1", correlation="traj-1", turn=0, start_offset_ms=0, duration_ms=500),
                _record(request_id="r2", correlation="traj-1", turn=1, start_offset_ms=600, duration_ms=500),
                _record(request_id="r3", correlation="traj-2", turn=0, start_offset_ms=200, duration_ms=900),
            ],
        )
    )
    events, summary = build_events(records, {"measured": (BASE, BASE + 2.0)})

    kinds = {e["event"] for e in events}
    assert kinds == {
        "phase_start",
        "phase_end",
        "trajectory_start",
        "trajectory_end",
        "turn_start",
        "turn_end",
        "request_start",
        "first_token",
        "request_end",
    }
    assert summary["requests"] == 3
    assert summary["trajectories"] == 2
    assert summary["turns"] == 3
    assert summary["request_events_complete"] is True
    # Time-ordered, so a consumer can merge it against the KV rows in one pass.
    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)


def test_trajectory_span_runs_from_its_first_request_to_its_last(tmp_path):
    """A trajectory outlives any single turn, which is the point of the layer."""
    records = parse_profile_export(
        _export(
            tmp_path,
            [
                _record(request_id="r1", correlation="traj-1", turn=0, start_offset_ms=0, duration_ms=500),
                _record(request_id="r2", correlation="traj-1", turn=1, start_offset_ms=2000, duration_ms=500),
            ],
        )
    )
    events, _ = build_events(records)
    spans = {e["event"]: e["ts"] for e in events if e["event"].startswith("trajectory")}

    assert spans["trajectory_start"] == BASE
    assert spans["trajectory_end"] == BASE + 2.5


def test_a_request_without_ttft_emits_no_first_token_event(tmp_path):
    """A failed request has no first token, and inventing one would misdate a decode rise."""
    records = parse_profile_export(
        _export(
            tmp_path,
            [
                _record(
                    request_id="r1",
                    correlation="t1",
                    turn=0,
                    start_offset_ms=0,
                    duration_ms=10,
                    ttft_ms=None,
                    error={"code": 499, "type": "RequestCancellationError", "message": "cancelled"},
                )
            ],
        )
    )
    events, _ = build_events(records)

    assert not [e for e in events if e["event"] == "first_token"]
    assert [e for e in events if e["event"] == "request_end"][0]["ok"] is False


def test_request_events_are_sampled_but_the_stream_says_so(tmp_path, monkeypatch):
    """A long high-concurrency round would otherwise outweigh the whole bundle."""
    import hyperloom.orchestrator.actions.executors._agentx_timeline as tl

    monkeypatch.setattr(tl, "_MAX_REQUEST_EVENTS", 2)
    records = parse_profile_export(
        _export(
            tmp_path,
            [
                _record(request_id=f"r{i}", correlation="t1", turn=i, start_offset_ms=i * 10, duration_ms=5)
                for i in range(10)
            ],
        )
    )
    events, summary = tl.build_events(records)

    assert summary["requests"] == 10  # the count stays exact
    assert summary["request_events_complete"] is False
    assert summary["request_event_stride"] == 5
    assert len([e for e in events if e["event"] == "request_start"]) == 2
    # The coarse layers stay complete: they are bounded by trajectory count, not by request count.
    assert len([e for e in events if e["event"] == "turn_start"]) == 10


def test_rows_learn_which_trajectories_were_in_flight(tmp_path):
    """The question this exists for: which trajectory was live when the pool spiked."""
    records = parse_profile_export(
        _export(
            tmp_path,
            [
                _record(request_id="r1", correlation="traj-1", turn=0, start_offset_ms=0, duration_ms=1000),
                _record(request_id="r2", correlation="traj-2", turn=0, start_offset_ms=500, duration_ms=1000),
                _record(request_id="r3", correlation="traj-3", turn=0, start_offset_ms=5000, duration_ms=100),
            ],
        )
    )
    rows = [
        {"scrape_start_unix": BASE + 0.6, "scrape_end_unix": BASE + 0.7},
        {"scrape_start_unix": BASE + 5.02, "scrape_end_unix": BASE + 5.05},
        {"scrape_start_unix": BASE + 100.0, "scrape_end_unix": BASE + 100.1},
    ]
    annotated = correlate_rows(rows, records)

    assert annotated == 2
    assert rows[0]["workload"]["in_flight"]["requests"] == 2
    assert rows[0]["workload"]["in_flight"]["trajectory_ids"] == ["traj-1", "traj-2"]
    assert rows[1]["workload"]["in_flight"]["trajectory_ids"] == ["traj-3"]
    # A window with nothing running is left alone rather than annotated with zeros.
    assert "workload" not in rows[2]


def test_correlation_counts_a_request_that_spans_the_whole_window(tmp_path):
    """A long request is in flight during a scrape it neither starts nor ends in."""
    records = parse_profile_export(
        _export(
            tmp_path,
            [_record(request_id="r1", correlation="traj-1", turn=0, start_offset_ms=0, duration_ms=60_000)],
        )
    )
    rows = [{"scrape_start_unix": BASE + 30.0, "scrape_end_unix": BASE + 30.1}]

    assert correlate_rows(rows, records) == 1
    assert rows[0]["workload"]["in_flight"]["requests"] == 1


def test_rows_without_a_scrape_window_are_skipped(tmp_path):
    """Older rows carried only a duration; they cannot be placed on the timeline."""
    records = parse_profile_export(
        _export(tmp_path, [_record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=500)])
    )
    rows = [{"scrape_sec": 0.01}]

    assert correlate_rows(rows, records) == 0
    assert "workload" not in rows[0]


def test_timeline_is_written_as_one_json_object_per_line(tmp_path):
    """JSONL so a consumer can stream it rather than hold a long round in memory."""
    records = parse_profile_export(
        _export(tmp_path, [_record(request_id="r1", correlation="t1", turn=0, start_offset_ms=0, duration_ms=500)])
    )
    events, _ = build_events(records)
    out = tmp_path / TIMELINE_ARTIFACT_NAME

    assert write_timeline(out, events) is True
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(events)
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_artifact_is_in_the_package_globs():
    """It lives in the round workspace, which the bundle does not otherwise reach."""
    from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_GLOBS

    assert "runs/**/agentx_timeline.jsonl" in PACKAGE_GLOBS
