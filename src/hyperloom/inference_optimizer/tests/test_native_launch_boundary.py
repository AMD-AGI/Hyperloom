# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate the final launch inputs after watchdog preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.agentx.identity import native_workload_fingerprint, verified_workload_fingerprint
from hyperloom.orchestrator.actions.executors import baseline
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("run_eval_disabled", [False, True])
async def test_baseline_checks_final_watchdog_and_process_inputs_before_spawn(
    tmp_path, monkeypatch, native, run_eval_disabled
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    benchmark = {"framework": "sglang", "timeout_seconds": 7200, "envs": {"CONC": 8}}
    if native:
        benchmark["agentx"] = {"enabled": True, "launch_overrides": {"version": 1}}
        benchmark["workload_spec"] = {
            "execution": {
                "static_execution_fingerprint": "a" * 64,
                "workload_fingerprint": native_workload_fingerprint(benchmark, "a" * 64),
            }
        }
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"benchmark": benchmark}))
    original = config.read_bytes()
    output = tmp_path / "output"
    checkout = str(tmp_path / "InferenceX")
    checked = []

    def prepare(**kwargs):
        env = kwargs["env"]
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["MAGPIE_INFERENCEX_PATH"] == checkout
        assert env["RESULT_DIR"] == str(output)
        assert env["SERVER_LOG"] == str(output / "server.log")
        assert ("EVAL_RESULT_DIR" in env) is not native
        materialized = yaml.safe_load(kwargs["config_path"].read_text())["benchmark"]
        if native:
            assert config.read_bytes() == original
            assert verified_workload_fingerprint(materialized) is not None
        else:
            assert materialized["timeout_seconds"] == 7800.0
            assert materialized["envs"]["PYTHONUNBUFFERED"] == "1"
        checked.append(dict(env))

    def spawn(cmd, **kwargs):
        assert checked == [kwargs["env"]]
        assert kwargs["timeout"] == 7800.0
        assert Path(cmd[cmd.index("--benchmark-config") + 1]) == config
        raise RuntimeError("verified process boundary reached")

    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.runtime.maybe_prepare_agentx", prepare)
    monkeypatch.setattr(baseline, "run_with_session_kill", spawn)
    executor = baseline.BaselineExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    monkeypatch.setattr(executor, "_preflight_server_argv", lambda **kwargs: None)
    task = Task(task_id="boundary", kind="baseline", state="running", idempotency_key="boundary", params={})
    context = RunnerContext(task=task, lease=None, extra={"mn_round_restarted": True})

    with pytest.raises(RuntimeError, match="verified process boundary reached"):
        await executor._run_single_benchmark(
            config_path=config,
            output_dir=output,
            timeout_sec=7200,
            override_result_dir=None,
            resolved_model="example/model",
            materialized_config_path=config,
            inferencex_path=checkout,
            effective_extra_server_args="",
            params={},
            ctx=context,
            run_eval_disabled=run_eval_disabled,
            server_already_ready=True,
        )
