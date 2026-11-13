# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the AgentX load: every replay knob explicit, one seed, counts that agree with the aggregate."""

from __future__ import annotations

import json
import stat
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import pytest

from inference_suite.contracts import FailedStep, RunLayout, SuiteConfig, WorkloadKind
from inference_suite.load import agentx
from inference_suite.load.agentx import (
    AgentxRound,
    AgentxTool,
    ProgressFollower,
    client_argv,
    export_counts,
    progress_api_port,
    refusal,
    run_agentx,
)
from inference_suite.staging import AGENTX_PYTHON
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import RoundGate, ServerWatch
from inference_suite.tests.test_config_contract import _AGENTX
from inference_suite.tests.test_load import _DEADLINES, _flag, _round_config

_WORKLOAD = replace(_AGENTX, warmup_seed=_AGENTX.seed)
_FAKE_ENV = ("FAKE_CALLS", "FAKE_EXIT", "FAKE_EXPORT", "FAKE_AGGREGATE", "FAKE_LOG", "FAKE_PACE")
_FAKE_HARNESS = textwrap.dedent(
    """
    import json, os, shutil, sys, time
    argv = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a") as calls:
        calls.write(json.dumps(argv) + "\\n")
    if "agentx_cli" in argv:
        artifacts = argv[argv.index("--output-artifact-dir") + 1]
        os.makedirs(artifacts, exist_ok=True)
        shutil.copy(os.environ["FAKE_EXPORT"], os.path.join(artifacts, "profile_export.jsonl"))
        for pause in os.environ.get("FAKE_PACE", "").split():
            time.sleep(float(pause))
            print("progress", flush=True)
        print(os.environ.get("FAKE_LOG", "replayed"))
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    with open(os.environ["FAKE_CALLS"] + ".aggregator_tp", "w") as seen:
        seen.write(os.environ.get("TP", ""))
    target = os.path.join(os.environ["RESULT_DIR"], os.environ["RESULT_FILENAME"] + ".json")
    shutil.copy(os.environ["FAKE_AGGREGATE"], target)
    """
)


def _record(phase: str | None, error: bool = False, error_osl: bool = False) -> dict[str, object]:
    lengths = {"input_sequence_length": {"value": 1000, "unit": "tokens"}, "output_sequence_length": {"value": 200}}
    record: dict[str, object] = {"metadata": {} if phase is None else {"benchmark_phase": phase}, "metrics": lengths}
    if error:
        record["error"] = {"code": 500}
    if error_osl:
        record["metrics"] = {"error_osl": {"value": 12}}
    return record


_EXPORT = (
    _record("warmup"),
    _record("warmup", error=True),
    _record("profiling"),
    _record("profiling"),
    _record(None),
    _record("profiling", error=True, error_osl=True),
)
_COUNTS = {
    "records_total": 6,
    "successful": 3,
    "error_dropped": 2,
    "warmup_dropped": 2,
    "error_osl_exported": 1,
    "dropped_total": 3,
    "input_tokens": 3000,
    "output_tokens": 600,
}


def _aggregate(counts: dict[str, int] = _COUNTS) -> dict[str, object]:
    return {
        "num_requests_successful": counts["successful"],
        "request_accounting": {
            "records_total": counts["records_total"],
            "records_warmup_dropped": counts["warmup_dropped"],
            "records_error_dropped": counts["error_dropped"],
            "records_dropped_total": counts["dropped_total"],
        },
        "request_metrics": {
            "qps": {"mean": 1.5},
            "throughput": {"output": {"tokens_per_second": 900.0}, "duration_seconds": 60.0},
            "latency": {"ttft": {"p99": 2.0}, "e2e_norm_intvty": {"p50": 0.1}},
        },
    }


def _config(tmp_path: Path) -> SuiteConfig:
    config = _round_config(tmp_path, WorkloadKind.AGENTX, agentx=_WORKLOAD)
    return replace(config, server=replace(config.server, env_inherit=_FAKE_ENV))


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentxTool:
    root = tmp_path / "ix"
    python = root / AGENTX_PYTHON
    python.parent.mkdir(parents=True)
    script = tmp_path / "fake_harness.py"
    script.write_text(_FAKE_HARNESS)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    export = tmp_path / "export.jsonl"
    export.write_text("".join(json.dumps(record) + "\n" for record in _EXPORT))
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps(_aggregate()))
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_EXPORT", str(export))
    monkeypatch.setenv("FAKE_AGGREGATE", str(aggregate))
    return AgentxTool.staged(root)


def _round(tmp_path: Path, tool: AgentxTool, config: SuiteConfig | None = None) -> AgentxRound:
    server = Server("http://127.0.0.1:30000", True, None, lambda: True)
    gate = RoundGate(_DEADLINES, lambda: "", time.monotonic())
    return AgentxRound(config or _config(tmp_path), tool, server, gate, PhaseClock())


def test_client_argv_passes_every_replay_knob_explicitly(tmp_path: Path):
    config = _config(tmp_path)
    argv = client_argv(AgentxTool.staged(Path("/ix")), config)
    assert argv[:5] == (f"/ix/{AGENTX_PYTHON}", "-I", "-m", "agentx_cli", "profile")
    expected = {
        "--scenario": "agentx",
        "--url": "http://127.0.0.1:30000",
        "--model": "glm",
        "--tokenizer": "/models/GLM-5.3",
        "--public-dataset": "agentx-v1",
        "--concurrency": "8",
        "--benchmark-duration": "600",
        "--random-seed": "42",
        "--warmup-requests-per-lane": "2",
        "--failed-request-threshold": "0.1",
        "--server-metrics": "http://127.0.0.1:30000/metrics",
        "--output-artifact-dir": str(RunLayout(tmp_path / "run").aiperf_artifacts),
        "--api-host": "127.0.0.1",
        "--api-port": str(progress_api_port(config.server)),
    }
    assert {flag: _flag(argv, flag) for flag in expected} == expected
    assert "--tokenizer-trust-remote-code" in argv
    assert "--request-count" not in argv


@pytest.mark.parametrize(("duration", "override"), [(600, True), (899, True), (900, False), (3600, False)])
def test_the_replay_runs_for_its_duration_and_overrides_the_scenario_floor_below_it(tmp_path, duration, override):
    config = _config(tmp_path)
    config = replace(config, agentx=replace(config.agentx, duration_sec=duration))
    argv = client_argv(AgentxTool.staged(Path("/ix")), config)
    assert _flag(argv, "--benchmark-duration") == str(duration)
    assert ("--unsafe-override" in argv) is override
    assert "--request-count" not in argv


def test_the_progress_api_port_stays_in_range():
    server = _config(Path("/r")).server
    assert progress_api_port(server) == server.port + 1000
    assert progress_api_port(replace(server, port=65000)) == 64000


@pytest.mark.parametrize(
    ("changes", "match"),
    [({"warmup_seed": _AGENTX.seed + 1}, "warmup_seed"), ({"warmup_requests_per_lane": 0}, "warmup request")],
)
def test_a_workload_the_harness_cannot_honour_is_refused(changes, match):
    stop = refusal(replace(_WORKLOAD, **changes))
    assert stop.step is FailedStep.CONFIG_INVALID and match in stop.detail
    assert refusal(_WORKLOAD) is None


def test_export_counts_count_a_failed_warmup_as_both_and_drop_their_union(tmp_path: Path):
    export = tmp_path / "export.jsonl"
    export.write_text("".join(json.dumps(record) + "\n" for record in _EXPORT) + "\n")
    counts = export_counts(export)
    assert counts == _COUNTS
    assert counts["successful"] + counts["dropped_total"] == counts["records_total"]
    assert counts["warmup_dropped"] <= counts["dropped_total"]


def test_a_successful_request_without_a_length_leaves_its_token_count_unread(tmp_path: Path):
    export = tmp_path / "export.jsonl"
    unmeasured = {**_record("profiling"), "metrics": {"input_sequence_length": {"value": 7}}}
    export.write_text("".join(json.dumps(record) + "\n" for record in (*_EXPORT, unmeasured)))
    counts = export_counts(export)
    assert counts["input_tokens"] == 3007 and counts["output_tokens"] is None


def test_export_counts_refuse_a_line_that_is_not_a_record(tmp_path: Path):
    export = tmp_path / "export.jsonl"
    export.write_text("[1]\n")
    with pytest.raises(ValueError, match="line 1"):
        export_counts(export)


def test_run_replays_then_aggregates_and_keeps_counts_that_agree(tmp_path, harness):
    outcome = run_agentx(_round(tmp_path, harness))
    assert outcome.stop is None and outcome.returncode == 0
    replay, aggregator = (json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines())
    assert tuple(replay) == outcome.invocation.argv[1:]
    assert aggregator == ["-s", "-P", "-m", "infx.results.agentic.process_agentic_result"]
    assert outcome.invocation.seed == outcome.invocation.warmup_seed == 42
    assert outcome.invocation.env == {"AIPERF_EXPERIMENTAL_FAST": "0"}
    result = outcome.result
    assert dict(result.counts) == _COUNTS
    assert result.throughput["request_throughput"] == 1.5 and result.throughput["input_tokens_per_second"] is None
    assert result.latency_seconds["ttft_p99"] == 2.0 and result.e2e_norm_intvty["p50"] == 0.1
    assert not result.stopped_by_threshold and not result.fast_mode
    assert result.request_export == "client/aiperf_artifacts/profile_export.jsonl"


def test_the_aggregator_divides_per_gpu_throughput_by_the_cards_the_server_holds(tmp_path, harness):
    config = _config(tmp_path)
    config = replace(config, server=replace(config.server, gpus=(4, 5)))
    run_agentx(_round(tmp_path, harness, config))
    assert (tmp_path / "calls.jsonl.aggregator_tp").read_text() == "2"


def test_counts_that_disagree_with_the_aggregate_fail_the_output(tmp_path, harness):
    (tmp_path / "aggregate.json").write_text(json.dumps(_aggregate({**_COUNTS, "warmup_dropped": 1})))
    outcome = run_agentx(_round(tmp_path, harness))
    assert outcome.result is None
    assert outcome.stop.step is FailedStep.CLIENT_OUTPUT_MISSING
    assert "warmup_dropped 2 (aggregate 1)" in outcome.stop.detail


def test_a_threshold_stop_keeps_the_aggregated_result_and_fails_the_round(tmp_path, harness, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    monkeypatch.setenv("FAKE_LOG", f"aborting: {agentx.THRESHOLD_STOP_MARKER}")
    outcome = run_agentx(_round(tmp_path, harness))
    assert outcome.returncode == 1 and outcome.stop.step is FailedStep.CLIENT_FAILED
    assert outcome.result is not None and outcome.result.stopped_by_threshold


def test_a_missing_aggregate_fails_the_output(tmp_path, harness, monkeypatch):
    monkeypatch.setenv("FAKE_AGGREGATE", str(tmp_path / "absent.json"))
    outcome = run_agentx(_round(tmp_path, harness))
    assert outcome.result is None and outcome.stop.step is FailedStep.CLIENT_OUTPUT_MISSING


def test_the_follower_marks_phases_from_progress(monkeypatch):
    now_ns = time.time() * 1e9
    phases = {
        "warmup": {"start_ns": now_ns - 20e9},
        "profiling": {"start_ns": now_ns - 10e9, "requests_end_ns": 0},
    }
    readings = iter([None, phases])
    monkeypatch.setattr(agentx, "read_aiperf_phases", lambda url: next(readings))
    clock = PhaseClock()
    follower = ProgressFollower("http://127.0.0.1:31000", clock)
    follower.check()
    assert clock.marks == ()
    follower.check()
    warmup, measured = clock.first(Phase.WARMUP), clock.first(Phase.MEASURED)
    assert measured.mono - warmup.mono == pytest.approx(10.0, abs=0.5)
    assert measured.ts == pytest.approx((now_ns - 10e9) / 1e9, abs=0.5)
    assert clock.first(Phase.FINISHED) is None


def test_an_unreachable_progress_api_marks_nothing(monkeypatch):
    def unreachable(url):
        raise OSError("refused")

    monkeypatch.setattr(agentx, "read_aiperf_phases", unreachable)
    clock = PhaseClock()
    ProgressFollower("http://127.0.0.1:31000", clock).check()
    assert clock.marks == ()


def test_listeners_hear_the_finished_mark_before_the_export_is_aggregated(tmp_path, harness):
    calls = tmp_path / "calls.jsonl"
    agentx_round = _round(tmp_path, harness)
    clock, heard = agentx_round.clock, []

    def listener(now: float) -> None:
        replayed = calls.read_text().splitlines() if calls.exists() else []
        heard.append((clock.first(Phase.FINISHED) is not None, len(replayed)))

    run_agentx(replace(agentx_round, listeners=(listener,)))
    assert heard[-1] == (True, 1), "the last poll sees the finished mark while only the replay has run"
    assert len(calls.read_text().splitlines()) == 2


def test_client_output_while_a_listener_blocks_past_the_silence_bound_is_not_a_stall(tmp_path, harness, monkeypatch):
    monkeypatch.setenv("FAKE_PACE", "0.6 5.1")
    deadlines = replace(_DEADLINES, silence_sec=3.0)
    log = tmp_path / "server.log"
    log.write_text("")
    now = time.monotonic()
    server = Server("http://127.0.0.1:30000", True, ServerWatch(log, 0, deadlines, now), lambda: True)
    agentx_round = AgentxRound(_config(tmp_path), harness, server, RoundGate(deadlines, lambda: "", now), PhaseClock())
    polls: list[float] = []

    def stop_profile(now: float) -> None:
        # A window closes past the measured mark, never on the first poll, whose output check counts as activity anyway.
        polls.append(now)
        if len(polls) == 2:
            time.sleep(3.5)

    outcome = run_agentx(replace(agentx_round, listeners=(stop_profile,)))
    assert outcome.stop is None and outcome.returncode == 0, outcome.stop
