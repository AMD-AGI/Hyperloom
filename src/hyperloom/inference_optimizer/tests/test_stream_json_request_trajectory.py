# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-request ``llm.request`` rows recovered from a specialist's stream-json log."""

from __future__ import annotations

import json

from hyperloom.inference_optimizer.trace import request_events as re_
from hyperloom.inference_optimizer.trace import trajectory_trace as tt
from hyperloom.inference_optimizer.trace.tool_events import TIMING_NONE


def _write_log(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{torn", encoding="utf-8")


def _assistant(message_id: str, *, output: int, model: str = "claude-x", **extra) -> dict:
    usage = {
        "input_tokens": 3,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 40,
        "output_tokens": output,
    }
    return {
        "type": "assistant",
        "message": {"id": message_id, "model": model, "usage": usage, **extra},
    }


def _result(output: int) -> dict:
    return {"type": "result", "usage": {"output_tokens": output}}


def test_one_row_per_message_id_with_the_last_usage_snapshot(tmp_path):
    log = tmp_path / "process.log"
    _write_log(
        log,
        [
            {"type": "system", "subtype": "init", "model": "claude-init"},
            _assistant("m1", output=1),
            _assistant("m1", output=12, stop_reason="tool_use"),
            {**_assistant("m2", output=5), "parent_tool_use_id": "tu-1"},
            _assistant("synthetic", output=0, model="<synthetic>"),
            {"type": "assistant", "message": {"model": "claude-x"}},
            {"type": "assistant", "message": "not a dict"},
            _result(17),
        ],
    )
    first, second = re_.stream_json_requests(log)
    assert (first["message_id"], second["message_id"]) == ("m1", "m2")
    assert (first["output_tokens"], first["stop_reason"]) == (12, "tool_use")
    assert (first["cache_read_input_tokens"], first["cache_creation_input_tokens"]) == (900, 40)
    assert (first["request_index"], second["request_index"]) == (0, 1)
    assert second["parent_tool_use_id"] == "tu-1"
    assert first["model"] == "claude-x"
    assert first["output_tokens_exact"] is second["output_tokens_exact"] is True


def test_outputs_that_miss_the_result_total_are_flagged_inexact(tmp_path):
    log = tmp_path / "process.log"
    _write_log(log, [_assistant("m1", output=1), _assistant("m2", output=1), _result(40)])
    assert {r["output_tokens_exact"] for r in re_.stream_json_requests(log)} == {False}

    no_result = tmp_path / "no-result.log"
    _write_log(no_result, [_assistant("m1", output=1)])
    assert re_.stream_json_requests(no_result)[0]["output_tokens_exact"] is False


def test_model_falls_back_to_the_init_row(tmp_path):
    log = tmp_path / "process.log"
    _write_log(
        log,
        [
            {"type": "system", "subtype": "init", "model": "claude-init"},
            {"type": "assistant", "message": {"id": "m1", "usage": {"output_tokens": 2}}},
            _result(2),
        ],
    )
    (row,) = re_.stream_json_requests(log)
    assert row["model"] == "claude-init"


def test_requests_land_on_the_trajectory_untimed_under_the_task(tmp_path):
    log = tmp_path / "process.log"
    _write_log(log, [_assistant("m1", output=4), _assistant("m2", output=6), _result(10)])
    with tt.trajectory_scope(session_dir=tmp_path, component="specialist", task_id="task-1", parent_span_id="task-1"):
        assert re_.record_stream_json_requests(log) == 2

    rows = [r for r in tt.load_events(tmp_path) if r["event_type"] == tt.EVENT_LLM_REQUEST]
    assert [r["attributes"]["message_id"] for r in rows] == ["m1", "m2"]
    assert {r["status"] for r in rows} == {tt.STATUS_COMPLETED}
    assert {r["task_id"] for r in rows} == {"task-1"}
    assert {r["component"] for r in rows} == {"specialist"}
    assert {r["attributes"]["timing_source"] for r in rows} == {TIMING_NONE}
    assert rows[0]["attributes"]["name"] == "claude-x"
    assert all(r["attributes"]["output_tokens_exact"] for r in rows)


def test_recording_is_a_noop_outside_a_session_or_without_a_log(tmp_path):
    assert re_.record_stream_json_requests(tmp_path / "missing.log") == 0
    log = tmp_path / "process.log"
    _write_log(log, [_assistant("m1", output=1)])
    assert re_.record_stream_json_requests(log) == 0
