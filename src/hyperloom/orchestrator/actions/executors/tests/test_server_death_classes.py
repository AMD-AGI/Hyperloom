# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A server that served and then died is named a runtime death, never an init one; vLLM's execute-model RPC timeout
is a class of its own whose failure says which timeout ran out, and a roofline that hit it is not re-run on its recipe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.actions.executors._grid_base import failed_step_class
from hyperloom.orchestrator.actions.executors.benchmark_run.launch import _suite_round_failure
from hyperloom.orchestrator.bringup import observe_bringup
from hyperloom.orchestrator.state.analysis_failures import (
    RECIPE_PARAM,
    remember_roofline_failure,
    repeated_roofline_failure,
)
from hyperloom.orchestrator.state.failure_evidence import (
    ENGINE_EXECUTE_TIMEOUT,
    SERVER_DIED,
    SERVER_INIT_DEAD,
    classify_root_cause_type,
)
from inference_suite.contracts import Deadlines
from inference_suite.runtime.watch import ServerWatch, Stop

_BOOT = """\
(APIServer pid=10) INFO 11-11 23:10:00 [api_server.py:1] vLLM API server version 0.32.1
(EngineCore pid=20) INFO 11-11 23:10:05 [core.py:1] Initializing a V1 LLM engine with config: model='/m'
(Worker pid=30) INFO 11-11 23:11:00 [default_loader.py:1] Loading weights took 40.00 seconds
(EngineCore pid=20) INFO 11-11 23:12:00 [kv_cache_utils.py:1] GPU KV cache size: 1,000,000 tokens
"""
_SERVED = """\
INFO:     Application startup complete.
INFO:     127.0.0.1:5000 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=10) INFO 11-11 23:20:00 [loggers.py:1] Avg generation throughput: 812.4 tokens/s
"""
_RPC_TIMEOUT = """\
(EngineCore pid=20) INFO 11-11 23:31:21 [shm_broadcast.py:878] No available shared memory broadcast block found in 60 seconds.
(EngineCore pid=20) INFO 11-11 23:32:21 [shm_broadcast.py:878] No available shared memory broadcast block found in 60 seconds.
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] EngineCore encountered a fatal error.
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] Traceback (most recent call last):
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586]   File "/site/vllm/v1/executor/multiproc_executor.py", line 509, in get_response
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586]     raise TimeoutError(f"RPC call to {method} timed out.") from e
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] TimeoutError: RPC call to sample_tokens timed out.
(APIServer pid=10) ERROR 11-11 23:51:21 [async_llm.py:943] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
INFO:     Shutting down
"""
_ENGINE_DEATH = """\
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] EngineCore encountered a fatal error.
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] Traceback (most recent call last):
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586]   File "/site/vllm/v1/engine/core.py", line 1683, in _process_engine_step
(EngineCore pid=20) ERROR 11-11 23:51:21 [core.py:1586] RuntimeError: Worker failed with error 'shape mismatch in sampler'
(APIServer pid=10) ERROR 11-11 23:51:21 [async_llm.py:943] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
"""
_INIT_DEATH = """\
(EngineCore pid=20) ERROR 11-11 23:12:30 [core.py:1] EngineCore failed to start.
(EngineCore pid=20) ERROR 11-11 23:12:30 [core.py:1] Traceback (most recent call last):
(EngineCore pid=20) ERROR 11-11 23:12:30 [core.py:1]   File "/site/vllm/v1/engine/core.py", line 90, in __init__
(EngineCore pid=20) ERROR 11-11 23:12:30 [core.py:1] RuntimeError: Engine core initialization failed. See root cause above.
"""

_READY_THEN_RPC_TIMEOUT = _BOOT + _SERVED + _RPC_TIMEOUT
_READY_THEN_ENGINE_DEATH = _BOOT + _SERVED + _ENGINE_DEATH
_NEVER_READY = _BOOT + _INIT_DEATH

_DEADLINES = Deadlines(boot_sec=3600, round_sec=36000, eval_sec=60, silence_sec=3600, pre_ready_silence_sec=3600)


def _server_stop(tmp_path: Path, log_text: str) -> Stop:
    """The stop inference_suite records for a server whose log reads ``log_text`` and which then exited."""
    log = tmp_path / "server.log"
    log.write_text(log_text, encoding="utf-8")
    watch = ServerWatch(log, 0, _DEADLINES, now=0.0)
    watch.advance(1.0)
    if not watch.ready_logged:
        stop = watch.before_ready(1.0, 1)
        assert stop is not None
        return stop
    return watch.died()


@pytest.mark.parametrize(
    ("log_text", "error_class"),
    [
        (_READY_THEN_RPC_TIMEOUT, ENGINE_EXECUTE_TIMEOUT),
        (_READY_THEN_ENGINE_DEATH, SERVER_DIED),
        (_NEVER_READY, SERVER_INIT_DEAD),
    ],
    ids=["ready-then-rpc-timeout", "ready-then-engine-death", "never-ready"],
)
def test_a_server_death_is_classed_by_whether_it_served_and_how_it_died(tmp_path, log_text, error_class):
    stop = _server_stop(tmp_path, log_text)

    assert failed_step_error_class(stop.step, stop.detail) == error_class
    measurement = {"failed_step": stop.step.value, "failure_detail": stop.detail}
    assert failed_step_class(measurement) == error_class


@pytest.mark.parametrize(
    ("log_text", "served"),
    [(_READY_THEN_RPC_TIMEOUT, True), (_READY_THEN_ENGINE_DEATH, True), (_NEVER_READY, False)],
    ids=["ready-then-rpc-timeout", "ready-then-engine-death", "never-ready"],
)
def test_the_boot_observation_places_a_death_after_ready_above_it(log_text, served):
    observation = observe_bringup(server_log=log_text).observation

    assert observation.stage_failed is not None
    assert (observation.stage_failed > LadderStage.HTTP_READY) == served


def _round_failure(tmp_path: Path, stop: Stop) -> dict:
    round_ = SimpleNamespace(output_dir=tmp_path / "round", capture_meta={})
    finished = SimpleNamespace(returncode=1, workspace=tmp_path / "run", runtime_sec=1.0, boot_observation_ref="")
    measurement = {"failed_step": stop.step.value, "failure_detail": stop.detail, "run_status": "failed"}
    return _suite_round_failure(round_, finished, measurement, [])


def test_a_rounds_execute_timeout_names_the_timeout_that_ran_out_beside_the_death(tmp_path):
    result = _round_failure(tmp_path, _server_stop(tmp_path, _READY_THEN_RPC_TIMEOUT))

    assert result["error_class"] == ENGINE_EXECUTE_TIMEOUT
    assert "RPC call to sample_tokens timed out" in result["error"]
    assert "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" in result["error"]


@pytest.mark.parametrize("log_text", [_READY_THEN_ENGINE_DEATH, _NEVER_READY], ids=["ready-then-death", "never-ready"])
def test_any_other_server_death_keeps_its_own_excerpt_without_the_timeout_hint(tmp_path, log_text):
    stop = _server_stop(tmp_path, log_text)
    result = _round_failure(tmp_path, stop)

    assert result["error"] == stop.detail
    assert "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" not in result["error"]


@pytest.mark.parametrize(
    ("error_class", "log_text", "root_cause"),
    [
        (ENGINE_EXECUTE_TIMEOUT, _RPC_TIMEOUT, ENGINE_EXECUTE_TIMEOUT),
        (SERVER_DIED, _ENGINE_DEATH, "engine_runtime_death"),
        (SERVER_INIT_DEAD, _INIT_DEATH, "engine_core_init"),
    ],
)
def test_only_a_server_that_never_served_has_an_engine_init_root_cause(error_class, log_text, root_cause):
    assert classify_root_cause_type(error_class, log_text) == root_cause


def _failed_roofline(profile_error_class: str) -> dict:
    return {
        "status": "failed",
        "error_class": "profile_failed",
        "phase": "profile",
        "error": "the profile's server died",
        "sub_result": {"status": "failed", "error_class": profile_error_class, "error": "the profile's server died"},
    }


def test_a_roofline_whose_profile_hit_the_execute_timeout_is_not_re_run_on_its_recipe():
    state = SimpleNamespace(analysis_failures={})

    failure = remember_roofline_failure(state, {RECIPE_PARAM: "recipe-3"}, _failed_roofline(ENGINE_EXECUTE_TIMEOUT))

    assert failure == ENGINE_EXECUTE_TIMEOUT
    assert repeated_roofline_failure(state, "recipe-3")["failure"] == ENGINE_EXECUTE_TIMEOUT
    assert repeated_roofline_failure(state, "recipe-4") is None


def test_a_roofline_whose_profile_server_died_otherwise_may_run_again():
    state = SimpleNamespace(analysis_failures={})

    assert remember_roofline_failure(state, {RECIPE_PARAM: "recipe-3"}, _failed_roofline(SERVER_DIED)) == ""
    assert repeated_roofline_failure(state, "recipe-3") is None
