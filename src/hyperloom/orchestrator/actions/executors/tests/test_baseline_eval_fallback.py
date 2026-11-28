# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline accuracy-eval handling: eval-rooted failure classification, enablement routing and the accuracy stop."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
)


def _make_ctx(params: dict, *, enablement_mode: str = "off") -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-eval-1", params=params)
    return SimpleNamespace(task=task, extra={"shared_state": SharedState(enablement_mode=enablement_mode)})


def _run(coro):
    return asyncio.run(coro)


# --- _is_eval_rooted_failure ----------------------------------------------
def test_eval_rooted_failure_from_error_tail():
    result = {"status": "failed", "error": "...\nERROR: run_eval failed with exit code 1\n"}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_from_warning():
    result = {
        "status": "failed",
        "error": "boom",
        "nonfatal_warnings": ["Unknown parameter: --concurrent-requests"],
    }
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_negative():
    result = {"status": "failed", "error": "CUDA out of memory"}
    assert BaselineExecutor._is_eval_rooted_failure(result) is False


def test_eval_rooted_failure_scans_logs(tmp_path: Path):
    out = tmp_path / "task"
    ws = out / "benchmark_sglang_x"
    ws.mkdir(parents=True)
    (ws / "benchmark_stderr.log").write_text("+ run_eval ...\nrun_eval failed with exit code 1\n", encoding="utf-8")
    result = {"status": "failed", "error": "generic", "output_dir": str(out)}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_climbs_from_round_subdir(tmp_path: Path):
    # The result points at measure_round but the eval marker lives in the sibling warmup_round; the scan must climb to
    # the task root.
    task = tmp_path / "task"
    warm_ws = task / "warmup_round" / "benchmark_sglang_x"
    warm_ws.mkdir(parents=True)
    (warm_ws / "server.log").write_text("Unknown parameter: --concurrent-requests\n", encoding="utf-8")
    measure = task / "measure_round"
    measure.mkdir(parents=True)
    result = {"status": "failed", "error": "100% request failures", "output_dir": str(measure)}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


# --- eval-failure fallback end-to-end --------------------------------------


# --- baseline accuracy missing -> stop the whole run -----------------------


def _stop_ctx(framework: str, recorder, params: dict | None = None) -> SimpleNamespace:
    task = SimpleNamespace(
        task_id="t-bl",
        kind="baseline",
        params={"framework": framework, **(params or {})},
    )
    return SimpleNamespace(task=task, extra={"shared_state": recorder})


def _stopped(
    framework: str,
    result: dict,
    *,
    params: dict | None = None,
) -> str:
    """Run ``_maybe_stop_on_missing_baseline_accuracy`` and return the reason."""
    executor = BaselineExecutor()
    rec = SharedState(enablement_mode="off")
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx(framework, rec, params), result)
    return rec.stop_reason


# --- accuracy-stop decision matrix -----------------------------------------
@pytest.mark.parametrize(
    ("framework", "result"),
    [
        ("xdit", {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": True}),
        ("sglang", {"status": "succeeded", "run_eval_disabled": False}),
        ("sglang", {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": False}),
        ("sglang", {"status": "succeeded", "run_eval_disabled": True}),
        ("sglang", {"status": "succeeded", "run_eval_disabled": True, "accuracy_source": "eval_unavailable"}),
    ],
    ids=[
        "scriptable_fail_closed_zero",
        "serving_no_accuracy_eval_on",
        "serving_zero_accuracy",
        "serving_eval_disabled_via_config",
        "serving_eval_failure_fallback",
    ],
)
def test_stop_when_a_genuine_baseline_has_no_usable_accuracy(framework, result):
    """A genuine baseline has no opt-out: however the eval was turned off, a missing reference still stops the run."""
    assert _stopped(framework, result) == "baseline_accuracy_failed"


@pytest.mark.parametrize(
    ("result", "params"),
    [
        ({"status": "succeeded", "accuracy": 0.85, "run_eval_disabled": False}, None),
        # Synthetic kernel-lane re-baselines (kind="baseline" + quality_ref_exempt) are throughput-only A/B probes.
        ({"status": "succeeded", "run_eval_disabled": True}, {"quality_ref_exempt": True}),
    ],
    ids=["valid_accuracy", "quality_ref_exempt"],
)
def test_no_stop_when_accuracy_is_usable_or_exempt(result, params):
    assert _stopped("sglang", result, params=params) == ""


def test_no_stop_when_not_genuine_baseline():
    executor = BaselineExecutor()
    rec = SharedState(enablement_mode="off")
    task = SimpleNamespace(task_id="t", kind="replay_warm_recipe", params={"framework": "sglang"})
    ctx = SimpleNamespace(task=task, extra={"shared_state": rec})
    executor._maybe_stop_on_missing_baseline_accuracy(ctx, {"status": "succeeded", "run_eval_disabled": False})
    assert rec.stop_reason == ""


def _dead_server_log(tmp_path, name="server.log"):
    """A round whose server booted, served the benchmark, then lost its engine."""
    p = tmp_path / name
    p.write_text(
        "INFO:     Application startup complete\n"
        "INFO 01:40:00 Avg generation throughput: 188.0 tokens/s\n"
        "ERROR 01:57:28 [serving.py:448] "
        "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.\n"
        "INFO:     Shutting down\n",
        encoding="utf-8",
    )
    return p


def test_accuracy_stop_names_the_server_death(tmp_path, caplog):
    # The round's throughput succeeded and only then did the engine die, so the eval hit a closed port and wrote no
    # results*.json. The stop still fires -- there is no usable accuracy -- but it must not send the operator after a
    # "broken baseline setup" while the engine's traceback sits in that round's own server.log.
    log_path = _dead_server_log(tmp_path)
    with caplog.at_level(logging.WARNING):
        reason = _stopped(
            "vllm",
            {
                "status": "succeeded",
                "run_eval_disabled": False,
                "server_log_path": str(log_path),
            },
        )
    assert reason == "baseline_accuracy_failed"
    assert "EngineDeadError" in caplog.text
    # The context the stop is recorded under carries the distinction too, not just the prose above it.
    assert "baseline:vllm:server_died" in caplog.text
    # And the stop's own line must not assert the opposite of what the line above it just established.
    assert "broken baseline setup" not in caplog.text
    assert "a fatal engine death on record for this round" in caplog.text
    # The report must stay evidence, not a verdict: the marker carries no ordering against the eval, so nothing here
    # may claim the death is why the reference is missing.
    assert "because the server died" not in caplog.text


def test_accuracy_stop_keeps_a_measured_zero_out_of_the_death_story(tmp_path, caplog):
    # A zero is a measurement: the eval ran and wrote its results. Saying "no eval could measure this and no
    # results*.json was ever written" would contradict the very artifacts the zero came from -- however dead the
    # server became afterwards.
    log_path = _dead_server_log(tmp_path)
    with caplog.at_level(logging.WARNING):
        reason = _stopped(
            "vllm",
            {
                "status": "succeeded",
                "run_eval_disabled": False,
                "accuracy": 0.0,
                "server_log_path": str(log_path),
            },
        )
    assert reason == "baseline_accuracy_failed"
    assert "produced no result" in caplog.text
    assert "server_died" not in caplog.text
    assert "fatal engine death" not in caplog.text


def test_accuracy_stop_stays_plain_when_the_server_lived(tmp_path, caplog):
    # A server that never died must not be accused of it: the missing accuracy is then a genuine setup problem.
    p = tmp_path / "server.log"
    p.write_text(
        "INFO:     Application startup complete\nINFO 01:40:00 Avg generation throughput: 188.0 tokens/s\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        reason = _stopped(
            "vllm",
            {"status": "succeeded", "run_eval_disabled": False, "server_log_path": str(p)},
        )
    assert reason == "baseline_accuracy_failed"
    # Positive control: an absence assertion over caplog is satisfied by an empty caplog, so pin the line that must
    # be there before trusting the one that must not.
    assert "produced no result" in caplog.text
    assert "broken baseline setup" in caplog.text
    assert "server_died" not in caplog.text


def test_accuracy_stop_survives_an_unreadable_server_log(caplog):
    # The probe is diagnostics: a path that is missing, or a directory, must not turn a clean stop into a crash.
    with caplog.at_level(logging.WARNING):
        reason = _stopped(
            "vllm",
            {"status": "succeeded", "run_eval_disabled": False, "server_log_path": "/nonexistent/server.log"},
        )
    assert reason == "baseline_accuracy_failed"
    assert "produced no result" in caplog.text
    assert "broken baseline setup" in caplog.text
    assert "server_died" not in caplog.text


def test_no_stop_when_baseline_failed():
    reason = _stopped("sglang", {"status": "failed", "error": "boom"})
    assert reason == ""


# --- the eval reader ---------------------------------------------------------
def _write_gsm8k_results(round_dir: Path, score: float) -> None:
    """Write a minimal lm-eval ``results*.json`` under a round dir."""
    d = round_dir / "benchmark_vllm_x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_x.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}),
        encoding="utf-8",
    )


def test_the_eval_reader_prefers_a_measured_rounds_file_over_a_warmups(tmp_path):
    attempt = tmp_path / "runs" / "baseline" / "786a793e"
    _write_gsm8k_results(attempt / "warmup_round", 0.10)
    _write_gsm8k_results(attempt / "measure_round", 0.90)

    from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results

    parsed = parse_eval_results(attempt, framework="vllm")
    assert parsed["accuracy"] == pytest.approx(0.90)


# --- eval-origin enablement routing (flag on) ------------------------------
from hyperloom.common.gain_math import DEFAULT_ENABLEMENT_ACCURACY_FLOOR
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    EVAL_KIND_ACCURACY_BELOW_FLOOR,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
    EVAL_KIND_GENERATION_PATHOLOGY,
)


def _write_minimal_route_yaml(tmp_path: Path, framework: str = "sglang") -> Path:
    """Write a minimal materialized YAML for _route tests."""
    p = tmp_path / "route_config.yaml"
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": "/path/models/test",
            "benchmark_script": f"{framework}_mi300x.sh",
            "precision": "bf16",
            "envs": {"CONC": 64, "ISL": 1024, "OSL": 1024, "TP": 8, "RUN_EVAL": "true"},
        }
    }
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _route(monkeypatch, framework, result, *, nodes=None, tmp_path=None):
    if nodes is not None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", str(nodes))
    if tmp_path is not None and "materialized_config" not in result:
        result["materialized_config"] = str(_write_minimal_route_yaml(tmp_path, framework))
    executor = BaselineExecutor()
    rec = SharedState(enablement_mode="eval")
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx(framework, rec), result)
    return rec.stop_reason


def test_eval_enablement_missing_accuracy_routes_not_stop(monkeypatch, tmp_path):
    cfg_path = str(_write_minimal_route_yaml(tmp_path))
    result = {"status": "succeeded", "run_eval_disabled": False, "materialized_config": cfg_path}
    reason = _route(monkeypatch, "sglang", result, tmp_path=tmp_path)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_UNAVAILABLE
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] is None
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == DEFAULT_ENABLEMENT_ACCURACY_FLOOR
    assert result[BASELINE_EVAL_EVIDENCE_KEY]
    assert result["eval_origin"] == "eval"


@pytest.mark.parametrize("observed", [0.0, DEFAULT_ENABLEMENT_ACCURACY_FLOOR / 2], ids=["zero", "positive"])
def test_eval_enablement_accuracy_below_floor(monkeypatch, observed):
    result = {"status": "succeeded", "accuracy": observed, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_BELOW_FLOOR
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == observed
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == DEFAULT_ENABLEMENT_ACCURACY_FLOOR


def test_eval_enablement_probe_reports_generation_pathology(monkeypatch):
    """A tripped probe changes what a ~0 score means: the eval was cut short because the model never stopped generating, not because it answered and got them wrong."""
    result = {
        "status": "succeeded",
        "accuracy": 0.0,
        "run_eval_disabled": False,
        "eval_probe": {
            "kind": EVAL_KIND_GENERATION_PATHOLOGY,
            "tripped": True,
            "samples_seen": 128,
            "length_ratio_observed": 1.0,
        },
    }
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_GENERATION_PATHOLOGY
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == 0.0
    evidence = result[BASELINE_EVAL_EVIDENCE_KEY]
    assert "100% of 128 sampled responses" in evidence


def test_eval_enablement_accuracy_at_floor_passes(monkeypatch):
    result = {
        "status": "succeeded",
        "accuracy": DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
        "run_eval_disabled": False,
    }
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_eval_enablement_multi_node_falls_back_to_stop(monkeypatch):
    result = {"status": "succeeded", "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result, nodes=2)
    assert reason == "baseline_accuracy_failed"
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_eval_enablement_operator_optout_is_routed(monkeypatch):
    """A disabled eval is not an opt-out: with enablement on it routes there."""
    result = {"status": "succeeded", "run_eval_disabled": True}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True


def test_eval_enablement_quality_ref_exempt_not_routed(monkeypatch):
    """Synthetic kernel-lane re-baselines are neither routed nor stopped."""
    result = {"status": "succeeded", "run_eval_disabled": True}
    executor = BaselineExecutor()
    rec = SharedState(enablement_mode="eval")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx("sglang", rec, {"quality_ref_exempt": True}), result)
    assert rec.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_scriptable_baseline_without_a_quality_gate_stops_instead_of_enablement(tmp_path, caplog):
    """An operator script that writes no ``quality_gate`` is no capability gap a specialist can fix: stop at once."""
    from hyperloom.measure.tests.run_dirs import write_scriptable_run_dir
    from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results
    from inference_suite.contracts import RunLayout
    from hyperloom.orchestrator.actions.executors.benchmark_run.records import record_eval_score

    attempt = tmp_path / "runs" / "baseline" / "attempt"
    report = RunLayout(write_scriptable_run_dir(attempt / "suite_run_0", None)).report
    result = {"status": "succeeded", "output_dir": str(attempt)}
    record_eval_score(result, parse_eval_results(attempt, framework="custom"))

    rec = SharedState(enablement_mode="all")
    with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.actions.executors._accuracy_gate"):
        BaselineExecutor()._maybe_stop_on_missing_baseline_accuracy(_stop_ctx("custom", rec), result)

    assert rec.stop_reason == "baseline_accuracy_failed"
    assert BASELINE_EVAL_FAILED_KEY not in result
    stop_line = next(r.getMessage() for r in caplog.records if "stopping run" in r.getMessage())
    assert "quality_gate" in stop_line
    assert str(report) in stop_line


# --- the ruling reads the attempt's own result, never a results file outside it ---
def _attempt_beside_a_scored_one(tmp_path: Path) -> Path:
    """A single-round attempt's directory, next to an earlier attempt whose measured round scored 0.93."""
    runs_baseline = tmp_path / "runs" / "baseline"
    _write_gsm8k_results(runs_baseline / "786a793e" / "measure_round", 0.93)
    deciding = runs_baseline / "retry2_bootsafe"
    deciding.mkdir(parents=True)
    return deciding


def _measured_round_beside_a_scored_warmup(tmp_path: Path) -> Path:
    """A pair's measured round directory, next to a warm-up round directory that holds a 0.93 results file."""
    attempt = tmp_path / "runs" / "baseline" / "786a793e"
    _write_gsm8k_results(attempt / "warmup_round", 0.93)
    deciding = attempt / "measure_round"
    deciding.mkdir(parents=True)
    return deciding


_SCORED_OUTSIDE_THE_RESULT = pytest.mark.parametrize(
    "output_dir",
    [_attempt_beside_a_scored_one, _measured_round_beside_a_scored_warmup],
    ids=["by_another_attempt", "in_the_pairs_warmup_directory"],
)


def _ruled(
    output_dir: Path, *, enablement_mode: str, accuracy: float | None = None, reference: float = 0.0
) -> tuple[dict, SharedState]:
    """Rule a succeeded genuine baseline whose result carries ``accuracy``, in a session that holds ``reference``."""
    state = SharedState(enablement_mode=enablement_mode, baseline_accuracy=reference)
    result = {"status": "succeeded", "run_eval_disabled": False, "output_dir": str(output_dir)}
    if accuracy is not None:
        result["accuracy"] = accuracy
    BaselineExecutor()._maybe_stop_on_missing_baseline_accuracy(_stop_ctx("vllm", state), result)
    return result, state


@_SCORED_OUTSIDE_THE_RESULT
@pytest.mark.parametrize("reference", [0.0, 0.81], ids=["first_baseline", "session_holds_a_reference"])
def test_a_baseline_without_a_score_stops_whatever_was_scored_outside_its_result(tmp_path, output_dir, reference):
    result, state = _ruled(output_dir(tmp_path), enablement_mode="off", reference=reference)

    assert state.stop_reason == "baseline_accuracy_failed"
    assert "accuracy" not in result
    assert "nonfatal_warnings" not in result
    assert state.baseline_accuracy == reference


@_SCORED_OUTSIDE_THE_RESULT
def test_a_baseline_without_a_score_is_an_eval_failure_whatever_was_scored_outside_its_result(tmp_path, output_dir):
    result, state = _ruled(output_dir(tmp_path), enablement_mode="eval", reference=0.81)

    assert state.stop_reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_UNAVAILABLE
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] is None
    assert "accuracy" not in result
    assert state.baseline_accuracy == 0.81


@pytest.mark.parametrize(
    ("enablement_mode", "measured", "stop_reason"),
    [
        ("off", 0.0, "baseline_accuracy_failed"),
        ("eval", 0.0, ""),
        ("eval", DEFAULT_ENABLEMENT_ACCURACY_FLOOR / 2, ""),
    ],
    ids=["zero_stops", "zero_is_an_eval_failure", "under_the_floor_is_an_eval_failure"],
)
def test_a_baseline_is_ruled_on_the_unusable_score_it_measured(tmp_path, enablement_mode, measured, stop_reason):
    result, state = _ruled(_attempt_beside_a_scored_one(tmp_path), enablement_mode=enablement_mode, accuracy=measured)

    assert state.stop_reason == stop_reason
    assert result["accuracy"] == measured
    assert result.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY) == (measured if enablement_mode == "eval" else None)
    assert state.baseline_accuracy == 0.0


@pytest.mark.parametrize("enablement_mode", ["off", "eval"])
def test_an_eval_that_never_reached_the_server_is_neither_stopped_nor_an_eval_failure(
    tmp_path, monkeypatch, enablement_mode
):
    """A refused connection is a broken measurement: it carries no verdict, and no other attempt's score fills in."""
    state = SharedState(enablement_mode=enablement_mode, baseline_accuracy=0.81)
    played = {
        "status": "failed",
        "run_eval_disabled": False,
        "output_dir": str(_attempt_beside_a_scored_one(tmp_path)),
        # The client's request is refused first, and only then does run_eval report a non-zero exit.
        "error": (
            "aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host 0.0.0.0:41099\n"
            "ERROR: run_eval failed with exit code 1\n"
        ),
    }
    executor = BaselineExecutor()

    async def run_pass(ctx, *, recorder=None):
        return played

    monkeypatch.setattr(executor, "_run_pass", run_pass)
    result = _run(executor._run_with_accuracy(_stop_ctx("vllm", state)))

    assert state.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result
    assert "accuracy" not in result
    assert result["nonfatal_warnings"] == ["eval_failed_server_unreachable"]
    assert state.baseline_accuracy == 0.81


# --- a bench that measured, then an eval that left no score ----------------
def _bench_then_eval(error_class: str, error: str = "the eval exited with code 1") -> dict:
    """The result of a pair whose measured bench stands and whose eval, run after it, stopped at ``error_class``."""
    return {
        "status": "succeeded",
        "output_throughput": 4701.6,
        "run_eval_disabled": False,
        "nonfatal_warnings": ["post_measure_accuracy_failed"],
        "accuracy_stage": {"status": "failed", "error_class": error_class, "error": error, "workspace": "/w/run"},
    }


def _ruled_after_its_pass(played: dict, *, enablement_mode: str, params: dict | None = None):
    """Rule on ``played`` as ``_run_with_accuracy`` does once the attempt's rounds returned it."""
    state = SharedState(enablement_mode=enablement_mode, baseline_accuracy=0.81)
    executor = BaselineExecutor()

    async def run_pass(ctx, *, recorder=None):
        return played

    executor._run_pass = run_pass
    result = _run(executor._run_with_accuracy(_stop_ctx("vllm", state, params)))
    assert result is played
    assert (result["status"], result["output_throughput"]) == ("succeeded", 4701.6)
    return result, state


def test_a_failed_eval_after_the_bench_stops_a_genuine_baseline_and_keeps_its_bench():
    result, state = _ruled_after_its_pass(_bench_then_eval("eval_failed"), enablement_mode="off")

    assert state.stop_reason == "baseline_accuracy_failed"
    assert result["accuracy_source"] == "eval_unavailable"
    assert "eval_failed_no_fallback_baseline_requires_accuracy" in result["nonfatal_warnings"]
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_a_failed_eval_after_the_bench_goes_to_enablement_as_a_runtime_failure_with_the_evals_own_error():
    result, state = _ruled_after_its_pass(_bench_then_eval("eval_failed"), enablement_mode="eval")

    assert state.stop_reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == "eval_runtime_failure"
    assert result[BASELINE_EVAL_EVIDENCE_KEY] == "the eval exited with code 1"
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] is None


def test_a_failed_eval_after_the_bench_of_a_candidate_neither_stops_the_run_nor_opens_enablement():
    """A kernel re-baseline, a stack validation: the caller's accuracy gate rules on the missing score."""
    played = _bench_then_eval("eval_failed")
    result, state = _ruled_after_its_pass(played, enablement_mode="eval", params={"quality_ref_exempt": True})

    assert state.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result
    assert result["accuracy_source"] == "eval_unavailable"
    assert "accuracy" not in result


_STOPPED_EVAL_CLASSES = pytest.mark.parametrize(
    "error_class", ["session_time_exhausted", "round_timeout", "server_died", "server_stalled"]
)


@_STOPPED_EVAL_CLASSES
def test_an_eval_stopped_after_the_bench_stops_a_genuine_baseline_as_a_missing_score(error_class):
    result, state = _ruled_after_its_pass(_bench_then_eval(error_class, "stopped"), enablement_mode="off")

    assert state.stop_reason == "baseline_accuracy_failed"
    assert result["nonfatal_warnings"] == ["post_measure_accuracy_failed"]
    assert "accuracy_source" not in result


@_STOPPED_EVAL_CLASSES
def test_an_eval_stopped_after_the_bench_goes_to_enablement_as_an_unavailable_score(error_class):
    result, state = _ruled_after_its_pass(_bench_then_eval(error_class, "stopped"), enablement_mode="eval")

    assert state.stop_reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_UNAVAILABLE
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] is None


@pytest.mark.parametrize(("enablement_mode", "stop_reason"), [("off", "baseline_accuracy_failed"), ("eval", "")])
def test_an_eval_that_never_reached_the_server_after_the_bench_is_a_missing_score(enablement_mode, stop_reason):
    """A refused connection is no verdict of the eval: the bench is ruled like any result without a score."""
    refused = "aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host 0.0.0.0:41099"
    played = _bench_then_eval("eval_failed")
    played["nonfatal_warnings"].append(refused)
    result, state = _ruled_after_its_pass(played, enablement_mode=enablement_mode)

    assert state.stop_reason == stop_reason
    assert result["nonfatal_warnings"][-1] == "eval_failed_server_unreachable"
    assert "eval_failed_no_fallback_baseline_requires_accuracy" not in result["nonfatal_warnings"]
    assert result.get(BASELINE_EVAL_FAILURE_KIND_KEY) == (EVAL_KIND_ACCURACY_UNAVAILABLE if not stop_reason else None)


@pytest.mark.parametrize("enablement_mode", ["off", "eval"])
def test_a_baseline_standing_on_its_cold_warmup_is_not_ruled_on_accuracy(enablement_mode):
    """The budget refused the round that evaluates, so the anchor has no score: that is no accuracy failure."""
    cold = {
        "status": "succeeded",
        "output_throughput": 4701.6,
        "run_eval_disabled": True,
        "nonfatal_warnings": ["baseline_measure_round_dropped_low_budget"],
        "measure_round_dropped": {"priced_by": "session_hot_pass", "expected_cost_sec": 400.0},
    }
    result, state = _ruled_after_its_pass(cold, enablement_mode=enablement_mode)

    assert state.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result
    assert result["nonfatal_warnings"] == ["baseline_measure_round_dropped_low_budget"]
    assert state.baseline_accuracy == 0.81
