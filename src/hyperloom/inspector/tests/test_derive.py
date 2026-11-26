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


def test_budget_and_staleness():
    assert derive.budget({"elapsed_charged_sec": 1800, "max_minutes": 120})["pct"] == 25.0
    assert derive.budget({})["pct"] is None
    agents = derive.agent_staleness({"agent_last_active": {"a": 100, "b": 990, "c": "x"}}, 1000, 300)
    assert {a["agent"]: a["stale"] for a in agents} == {"a": True, "b": False}
    # Measured at a run's end, an agent active just after its last recorded activity was not quiet at all.
    assert derive.agent_staleness({"agent_last_active": {"a": 1010}}, 1000, 300)[0]["age_sec"] == 0


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


def _raw(**over):
    raw = {"output_throughput": 1800.5, "max_concurrency": 64, "benchmark_end_time_unix": 1000.0}
    for kind, (mean, median, p99) in {
        "ttft": (76.2, 64.4, 217.4),
        "tpot": (3.3, 3.2, 3.4),
        "e2el": (3500.0, 3498.0, 3614.0),
    }.items():
        raw |= {f"mean_{kind}_ms": mean, f"median_{kind}_ms": median, f"p99_{kind}_ms": p99}
    return raw | over


def test_benchmark_point_reads_the_path_the_throughput_and_the_latency_stats():
    measured = ("explore", "t1", "v02_fp8-kv", "variant_00_fp8-kv", "benchmark_vllm_1", "inferencex_result.json")
    point = derive.benchmark_point(measured, _raw(), mtime=5.0)
    assert point == {
        "ts": 1000.0,
        "kind": "explore",
        "task_id": "t1",
        "variant": "fp8-kv",
        "output_throughput": 1800.5,
        "latency": {
            "ttft": {"mean": 76.2, "median": 64.4, "p99": 217.4},
            "tpot": {"mean": 3.3, "median": 3.2, "p99": 3.4},
            "e2el": {"mean": 3500.0, "median": 3498.0, "p99": 3614.0},
        },
        "concurrency": 64,
    }
    warmup = ("explore", "t1", "v02_fp8-kv", "warmup_round", "variant_00_fp8-kv", "b", "inferencex_result.json")
    assert derive.benchmark_point(warmup, _raw(), 5.0) is None
    patch = ("integrate_patch", "t0", "measure_round", "benchmark_vllm_1", "inferencex_result.json")
    assert derive.benchmark_point(patch, _raw(benchmark_end_time_unix=None), mtime=5.0)["ts"] == 5.0
    assert derive.benchmark_point(patch, _raw(), 5.0)["variant"] is None


def test_benchmark_point_keeps_a_missing_stat_as_none_and_needs_throughput_and_some_latency():
    parts = ("explore", "t1", "b", "inferencex_result.json")
    only_ttft_mean = {"output_throughput": 10, "mean_ttft_ms": 5.0}
    latency = derive.benchmark_point(parts, only_ttft_mean, 1.0)["latency"]
    assert latency["ttft"] == {"mean": 5.0, "median": None, "p99": None}
    assert latency["e2el"] == {"mean": None, "median": None, "p99": None}
    assert derive.benchmark_point(parts, {"output_throughput": 10}, 1.0) is None
    assert derive.benchmark_point(parts, {"mean_ttft_ms": 5.0}, 1.0) is None
    assert derive.benchmark_point(parts, {"output_throughput": 10, "mean_ttft_ms": None}, 1.0) is None
    assert derive.benchmark_point(parts, {"output_throughput": True, "mean_ttft_ms": 5}, 1.0) is None
    assert derive.benchmark_point(parts, ["not", "a", "dict"], 1.0) is None
    assert derive.benchmark_point(("explore", "x.json"), _raw(), 1.0) is None


def test_baseline_point_comes_from_the_timeline_record_and_borrows_percentiles_from_the_raw_file():
    timeline = {
        "kind": "baseline",
        "ext": {
            "actions": [
                {"task_id": "t0", "end_time": "5", "measurement": {"ttft_mean_ms": 60.2}},
                {
                    "task_id": "t0",
                    "end_time": "7",
                    "measurement": {
                        "output_throughput": 13736.3,
                        "ttft_mean_ms": 60.2,
                        "tpot_mean_ms": 4.6,
                        "e2el_mean_ms": 4759.6,
                    },
                },
            ]
        },
    }
    point = derive.baseline_point(timeline, float, lambda task_id: None)
    assert point == {
        "ts": 7.0,
        "kind": "baseline",
        "task_id": "t0",
        "variant": None,
        "output_throughput": 13736.3,
        "latency": {
            "ttft": {"mean": 60.2, "median": None, "p99": None},
            "tpot": {"mean": 4.6, "median": None, "p99": None},
            "e2el": {"mean": 4759.6, "median": None, "p99": None},
        },
        "concurrency": None,
    }
    seen = []
    full = derive.baseline_point(timeline, float, lambda task_id: seen.append(task_id) or _raw())
    assert seen == ["t0"]
    assert full["output_throughput"] == 13736.3
    assert full["latency"]["ttft"] == {"mean": 76.2, "median": 64.4, "p99": 217.4}
    assert derive.baseline_point({**timeline, "kind": "stack"}, float, lambda t: None) is None
    assert (
        derive.baseline_point({"kind": "baseline", "ext": {"actions": [{"task_id": "t0"}]}}, float, lambda t: None)
        is None
    )
    assert derive.baseline_point(None, float, lambda t: None) is None
    measure = ("baseline", "t0", "measure_round", "b", "inferencex_result.json")
    assert derive.benchmark_point(measure, _raw(), 1.0) is None


def test_baseline_point_reads_a_session_recorded_before_the_measurement_key_was_renamed():
    # Hyperloom before 1.2.0 wrote the same whole-server rate as ``throughput_tok_s_per_gpu``.
    measurement = {"throughput_tok_s_per_gpu": 1685.2, "ttft_mean_ms": 60.2}
    timeline = {
        "kind": "baseline",
        "ext": {"actions": [{"task_id": "t0", "end_time": "5", "measurement": measurement}]},
    }
    point = derive.baseline_point(timeline, float, lambda task_id: None)
    assert point["output_throughput"] == 1685.2
    renamed = {**measurement, "output_throughput": 1700.0}
    timeline["ext"]["actions"][0]["measurement"] = renamed
    assert derive.baseline_point(timeline, float, lambda task_id: None)["output_throughput"] == 1700.0


def test_health_badges_order_and_thresholds():
    tasks = [{"health": "crashed", "label": "scout"}, {"health": "stalled"}]
    memory = {"used_bytes": 96, "limit_bytes": 100}
    agents = [{"agent": "critic", "stale": True}]
    badges = derive.health_badges({"crash_count": 1}, tasks, memory, agents)
    assert badges[0] == {"level": "error", "text": "1 task(s) crashed but recorded as succeeded: scout"}
    assert {"level": "error", "text": "container memory at 96% of limit (OOM risk)"} in badges
    assert any(b["text"] == "agents idle: critic" for b in badges)
    assert derive.health_badges({"stop_reason": "done"}, [], {}, agents) == []


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
