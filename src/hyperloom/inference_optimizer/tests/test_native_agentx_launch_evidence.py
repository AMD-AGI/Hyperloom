# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native baseline and grid evidence retain the engine's resolved configuration."""

import asyncio
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.orchestrator.actions.executors import _grid_runner, _ray_serving, baseline
from hyperloom.orchestrator.actions.executors._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS
from hyperloom.orchestrator.state.shared_state import SharedState


def _report(framework, model, flag):
    module = "sglang.launch_server" if framework == "sglang" else "vllm.entrypoints.openai.api_server"
    model_flag = "--model-path" if framework == "sglang" else "--model"
    argv = ["python3", "-m", module, model_flag, model, *([flag] if flag else [])]
    return {
        "scenario": "agentx",
        "framework": framework,
        "model": model,
        **dict.fromkeys(("success", "benchmark_valid", "publishable"), True),
        "agentx_metrics": {
            "mode": "canonical",
            "execution_owner": "magpie",
            "server_launch": {"effective_argv": argv, "effective_env": {}},
            "recipe": {"tp": 1, "ep": 1, "conc": 8},
            "requests": {"successful": 100, "total": 100},
            "throughput": {"output_tokens_per_second": 50.0},
        },
    }


def _server_record(framework, cache_off):
    if framework == "sglang":
        return f"server_args=ServerArgs(model_path='m', disable_radix_cache={cache_off}, tp_size=1)\n"
    return f"INFO api_server.py:1 non-default args: {{'model': 'm', 'enable_prefix_caching': {not cache_off}}}\n"


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("cache_off,exempt", [(True, False), (False, False), (True, True)])
def test_native_baseline_enforces_resolved_cache_gate(
    native_benchmark, tmp_path, monkeypatch, framework, cache_off, exempt
):
    config = yaml.safe_load(native_benchmark.read_text())
    config["benchmark"]["framework"] = framework
    native_benchmark.write_text(yaml.safe_dump(config))
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, str(native_benchmark))
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **kwargs: None)
    cache_flag = "--disable-radix-cache" if framework == "sglang" else "--no-enable-prefix-caching"
    flag = cache_flag if cache_off else ""
    model = config["benchmark"]["model"]
    report = _report(framework, model, flag)

    def run(cmd, **kwargs):
        rendered = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())["benchmark"]
        watchdog = ["--watchdog-timeout", "1800"] if framework == "sglang" else []
        assert rendered["agentx"]["launch_overrides"]["append_args"] == ([flag] if flag else []) + watchdog
        output = Path(cmd[cmd.index("--output-dir") + 1])
        workspace = output / "benchmark_native_cache"
        workspace.mkdir(parents=True)
        (workspace / "benchmark_report.json").write_text(json.dumps(report))
        (workspace / "server.log").write_text(_server_record(framework, cache_off))
        return subprocess.CompletedProcess(cmd, 0, "complete", "")

    monkeypatch.setattr(baseline, "run_with_session_kill", run)
    state = SharedState(benchmark_mode="agentx", agentx_backend="magpie", enablement_mode="off")
    task = SimpleNamespace(
        task_id="native-cache-baseline",
        kind="baseline",
        params={
            "framework": framework,
            "model_path": model,
            "output_dir": str(tmp_path / "round"),
            "extra_server_args": flag,
            "quality_ref_exempt": exempt,
            "timeout_sec": 10,
        },
    )
    executor = baseline.BaselineExecutor(
        magpie_python="python3", default_config_path=native_benchmark, session_dir=tmp_path
    )
    result = asyncio.run(executor(SimpleNamespace(task=task, extra={"shared_state": state})))
    refused = cache_off and not exempt
    assert result["status"] == ("failed" if refused else "succeeded"), result
    assert result.get("error_class") == (PREFIX_CACHE_OFF_ERROR_CLASS if refused else None)
    observed = result["launch_evidence"]["observed_server_config"]
    assert observed["disable_radix_cache" if framework == "sglang" else "enable_prefix_caching"] is (
        cache_off if framework == "sglang" else not cache_off
    )
    assert Path(result["server_log_path"]).name == "server.log"
    persisted = json.loads(Path(result["launch_evidence_path"]).read_text())
    assert persisted["actual_server_log_path"] == result["server_log_path"]
    assert persisted["magpie_server_launch"] == report["agentx_metrics"]["server_launch"]


def test_native_grid_persists_engine_observations(native_benchmark, tmp_path):
    slot = tmp_path / "variant_00_candidate"
    workspace = slot / "benchmark_sglang_candidate"
    workspace.mkdir(parents=True)
    (slot / "config.yaml").write_text(native_benchmark.read_text())
    report = _report("sglang", "m", "--disable-radix-cache")
    (workspace / "benchmark_report.json").write_text(json.dumps(report))
    log = workspace / "server.log"
    log.write_text(_server_record("sglang", True) + "chunked prefill size is adjusted from 65536 to 8192\n")
    result = _grid_runner.VariantResult(
        name="candidate",
        extra_server_args="--disable-radix-cache",
        extra_envs={},
        status="succeeded",
        benchmark_backend="magpie_agentx",
        workspace=str(workspace),
    )
    _grid_runner._attach_grid_launch_evidence(
        [result], grid=[_grid_runner.GridVariant("candidate")], output_root=tmp_path, caller_reused_ready_server=False
    )
    evidence = json.loads(Path(result.launch_evidence_path).read_text())
    assert evidence["actual_server_log_path"] == str(log)
    assert evidence["observed_server_config"]["disable_radix_cache"] is True
    assert evidence["observed_server_identity"]
    assert evidence["engine_adjusted_settings"]
    assert evidence["magpie_server_launch"] == report["agentx_metrics"]["server_launch"]
