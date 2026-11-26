# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector import derive
from hyperloom.inspector.collectors import parse_ts


@pytest.mark.parametrize(
    ("result", "health"),
    [
        (None, "ok"),
        ({"status": "succeeded"}, "ok"),
        ({"runner_status": "stale"}, "crashed"),
        ({"runner_status": "Crashed", "reason": "boom"}, "crashed"),
        ({"exit_code": 1}, "crashed"),
        ({"subprocess_exit_code": 0}, "ok"),
        ({"reason": "subprocess_exit_code:1"}, "crashed"),
        ({"reason": "subprocess_exit_code:0"}, "ok"),
        ({"runner_status": "partial"}, "crashed"),
        ({"error": "subprocess_timeout: killed after 3600s"}, "crashed"),
        ({"error": "subprocess_stale_heartbeat"}, "crashed"),
        ({"error": "subprocess_error:claude: rate limited"}, "crashed"),
        ({"status": "error", "error": "x"}, "failed"),
    ],
)
def test_result_health(result, health):
    assert derive.result_health(result)[0] == health


def test_task_health_catches_crashed_but_succeeded_from_the_specialist_round():
    task = {"state": "succeeded"}
    out = derive.task_health(task, {"status": "ok"}, "subprocess_exit_code:1", None, 600)
    assert out == {"health": "crashed", "reason": "subprocess_exit_code:1"}


@pytest.mark.parametrize(
    "reason",
    ["subprocess_timeout: killed after 3600s", "subprocess_stale_heartbeat", "subprocess_error:claude: rate limited"],
)
def test_task_health_catches_every_subprocess_failure_from_the_specialist_round(reason):
    out = derive.task_health({"state": "succeeded"}, {"status": "ok"}, reason, None, 600)
    assert out == {"health": "crashed", "reason": reason}
    assert derive.task_health({"state": "succeeded"}, None, "subprocess_exit_code:0", None, 600)["health"] == "ok"


def test_task_health_running_vs_stalled():
    running = {"state": "running"}
    assert derive.task_health(running, None, None, 30, 600)["health"] == "running"
    assert derive.task_health(running, None, None, 1200, 600) == {"health": "stalled", "reason": "no output for 20 min"}
    assert derive.task_health({"state": "cancelled"}, None, None, None, 600)["health"] == "cancelled"


def test_task_state_at_walks_history():
    task = {
        "created_at": "1970-01-01T00:00:10Z",
        "history": [
            {"to": "running", "ts": "1970-01-01T00:00:20Z"},
            {"to": "succeeded", "ts": "1970-01-01T00:00:30Z"},
        ],
    }
    assert derive.task_state_at(task, 5, parse_ts) is None
    assert derive.task_state_at(task, 15, parse_ts)["state"] == "queued"
    assert derive.task_state_at(task, 25, parse_ts)["state"] == "running"
    at = derive.task_state_at(task, 30, parse_ts)
    assert at["state"] == "succeeded" and at["updated_at"] == "1970-01-01T00:00:30Z"


def test_llm_summary_aggregates_per_component():
    calls = [
        {"component": "critic", "input_tokens": 10, "output_tokens": 2, "latency_ms": 100, "model": "m"},
        {"component": "critic", "input_tokens": 5, "status": "error", "latency_ms": 300, "model": "m"},
        {"role": "orchestration", "cache_read_input_tokens": 7},
    ]
    out = derive.llm_summary(calls)
    assert out["critic"]["calls"] == 2 and out["critic"]["errors"] == 1
    assert out["critic"]["input_tokens"] == 15 and out["critic"]["latency_ms_avg"] == 200
    assert out["critic"]["models"] == {"m": 2}
    assert out["orchestration"]["cache_read_tokens"] == 7


def test_budget():
    assert derive.budget({"elapsed_charged_sec": 1800, "max_minutes": 120})["pct"] == 25.0
    assert derive.budget({})["pct"] is None


def test_state_summary_trims_and_keeps_best():
    state = {"phase": "SYSTEM", "secret_field": 1, "current_best": {"action": "a", "hot_tput": 5, "x": 1}}
    out = derive.state_summary(state)
    assert out["phase"] == "SYSTEM" and "secret_field" not in out
    assert out["current_best"] == {"action": "a", "hot_tput": 5, "cold_tput": None, "e2el_mean_ms": None}
    assert derive.state_summary(None) == {}


def test_throughput_series_reads_throughput_after():
    journal = {
        "entries": [{"kind": "baseline", "throughput_after": 100}, {"throughput_after": 120, "variant_name": "v"}]
    }
    journal["entries"].append({"kind": "param", "outcome": "FAILED"})
    series = derive.throughput_series(journal)
    assert [e["throughput"] for e in series] == [100, 120, None]
    assert [e["variant_name"] for e in series] == [None, "v", None]
    assert "throughput_before" not in series[0]
    assert derive.throughput_series(None) == []


def test_best_throughput_falls_back_when_state_has_none():
    explore = {"status": "succeeded", "output_throughput": 1706.8, "best_variant": {"name": "mnbt_16k"}}
    results = [("baseline", {"status": "succeeded", "output_throughput": 1685.2}), ("explore", explore)]
    results.append(("explore", {"status": "failed", "output_throughput": 9999}))
    series = [{"throughput": 1700, "outcome": "KEEP", "change": "x"}, {"throughput": 1800, "outcome": "REVERT"}]
    state = {"action": "explore", "hot_tput": None}
    assert derive.best_throughput(state, results, series) == {
        "action": "explore: mnbt_16k",
        "hot_tput": 1706.8,
        "source": "result",
    }
    assert derive.best_throughput(state, [], series)["source"] == "journal"
    assert derive.best_throughput({"hot_tput": 5}, results, series) == {"hot_tput": 5, "source": "state"}
    assert derive.best_throughput({}, [], [])["source"] is None


def test_task_progress_units_current_step_and_failure_reason():
    def note(ts, **kw):
        return {"progress": kw, "ts": ts}

    history = [
        {"from": "queued", "to": "running", "ts": 1},
        note(2, unit="variant_step", label="k7:variant", status="started"),
        note(3, unit="variant", label="k7", status="failed", output_throughput=None),
        note(4, unit="variant", label="q4", status="succeeded", output_throughput=1052.4),
        note(4.5, unit="variant", label="q4", status="succeeded", output_throughput=1689.2),
        note(5, unit="variant_step", label="mtp:benchmark", status="started", index=1, total=1),
    ]
    outcomes = [{"variant_name": "k7", "outcome": "FAILED", "reason": "warmup_failed", "stage": "warmup"}]
    out = derive.task_progress(history, None, lambda t: t, outcomes)
    assert out["current"] == {
        "unit": "variant_step",
        "label": "mtp:benchmark",
        "status": "started",
        "index": 1,
        "total": 1,
        "ts": 5,
    }
    assert [(u["label"], u["status"]) for u in out["units"]] == [("k7", "failed"), ("q4", "succeeded")]
    assert out["units"][0]["reason"] == "warmup_failed" and out["failed"] == 1
    assert out["units"][1]["output_throughput"] == 1689.2 and out["units"][1]["earlier"] == [1052.4]
    assert derive.task_progress(history, 3.5, lambda t: t)["current"]["label"] == "k7"
    assert derive.task_progress([{"to": "running", "ts": 1}], None, lambda t: t) == {}
