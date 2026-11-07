# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MLPerf AgentX client's server is an inference_suite boot round: its config, and the client script's use of it."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.common.agentx_workload import (
    MLPERF_SERVER_CONFIG_ENV,
    MLPERF_SERVER_PID_DIR_ENV,
    MLPERF_SUITE_PYTHON_ENV,
)
from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.inference_optimizer.agentx import deploy, preflight
from hyperloom.orchestrator.actions.executors import benchmark_config as ms
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.tests._session_base import freeze_session_base
from inference_suite.contracts import GoldenUse, LifecycleMode, SuiteConfig, WorkloadKind

_CLIENT_ENVS = {
    "HYPERLOOM_AGENTIC_BACKEND": "mlperf",
    "MLPERF_AGENTIC_MODEL": "kimi-k3",
    "MLPERF_AGENTIC_FLOW": "smoke_test",
    "AGENTIC_NUM_TRAJECTORIES": "150",
    "AGENTIC_CONCURRENCY": "16",
    "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
    "MODEL": "/m",
    "FRAMEWORK": "sglang",
}


def _point(**envs: str) -> dict:
    return {
        "benchmark": {
            "framework": "sglang",
            "model": "/m",
            "benchmark_script": "mlperf_agentic_client.sh",
            "timeout_seconds": 3600,
            "envs": {
                **_CLIENT_ENVS,
                "CONC": "16",
                "PORT": "30000",
                "RUN_EVAL": "false",
                "EXTRA_SGLANG_ARGS": "--mem-fraction-static 0.8 --served-model-name kimi-k3",
                **envs,
            },
        }
    }


@pytest.fixture
def session(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(root))
    freeze_session_base(root, args=("--kv-cache-dtype", "fp8_e4m3"))
    return root


_MANIFEST_SHA = "b" * 64


@pytest.fixture
def env(tmp_path, monkeypatch) -> dict[str, str]:
    """A benchmark env whose tool cache inference_suite's staging check accepts."""
    cache = tmp_path / "tools"
    cache.mkdir()
    monkeypatch.setattr(ms, "_manifest_sha256", lambda tool_cache: _MANIFEST_SHA)
    return {"INFERENCE_SUITE_TOOL_CACHE": str(cache), "PATH": os.environ["PATH"], "HIP_VISIBLE_DEVICES": "0"}


def test_the_server_config_boots_the_base_with_the_harness_model_name_and_no_load(session, env, tmp_path):
    config = ms.mlperf_server_config(_point(), tmp_path / "slot" / ms.MLPERF_SERVER_DIR, env)

    assert config.lifecycle.mode is LifecycleMode.BOOT
    assert config.lifecycle.pid_dir == str(tmp_path / "slot" / ms.MLPERF_SERVER_DIR / ms.SERVER_PID_DIR)
    assert (config.workload, config.fixed, config.eval.enabled) == (WorkloadKind.NONE, None, False)
    server = config.server
    assert (server.served_model_name, server.port, server.golden) == ("kimi-k3", 30000, GoldenUse.APPLY)
    assert server.base.path == str(session / "base_launch.json")
    assert "--served-model-name" not in server.user.args_add
    assert ("--mem-fraction-static", "0.8") == server.user.args_add[-2:]
    assert server.user.env_set["SGLANG_TIMEOUT_KEEP_ALIVE"] == "900"
    assert not set(_CLIENT_ENVS) & set(server.user.env_set)
    assert set(server.env_inherit) == {"INFERENCE_SUITE_TOOL_CACHE", "PATH"}
    assert (config.paths.tool_cache, config.paths.tool_manifest_sha256) == (
        env["INFERENCE_SUITE_TOOL_CACHE"],
        _MANIFEST_SHA,
    )


def test_an_operator_keep_alive_is_kept(session, env, tmp_path):
    config = ms.mlperf_server_config(_point(SGLANG_TIMEOUT_KEEP_ALIVE="60"), tmp_path, env)

    assert config.server.user.env_set["SGLANG_TIMEOUT_KEEP_ALIVE"] == "60"


def test_a_session_that_froze_no_base_is_refused(tmp_path, env, monkeypatch):
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(tmp_path))
    SharedState(session_id="old").save(tmp_path)

    with pytest.raises(ms.MlperfServerError, match="no base launch"):
        ms.mlperf_server_config(_point(), tmp_path, env)


def test_an_unset_tool_cache_is_refused(session, env, tmp_path):
    del env["INFERENCE_SUITE_TOOL_CACHE"]

    with pytest.raises(ms.MlperfServerError, match="install.sh"):
        ms.mlperf_server_config(_point(), tmp_path, env)


def test_a_tool_cache_the_staging_check_refuses_is_refused(session, env, tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))

    with pytest.raises(ms.MlperfServerError, match="staging_missing"):
        ms.mlperf_server_config(_point(), tmp_path, env)


@pytest.fixture
def prepared(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(deploy, "deploy_agentx_assets", lambda benchmarks: calls.append(f"deploy {benchmarks}"))
    monkeypatch.setattr(preflight, "check_mlperf_harness", lambda env: calls.append("preflight"))
    return calls


def _write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_preparing_a_round_writes_the_boot_config_and_names_it_in_the_point(session, env, tmp_path, prepared):
    point = _write(tmp_path / "slot" / "config.yaml", _point())

    ms.prepare_mlperf_round(point, env, str(tmp_path / "ix"))

    envs = yaml.safe_load(point.read_text())["benchmark"]["envs"]
    written = Path(envs[MLPERF_SERVER_CONFIG_ENV])
    assert written == tmp_path / "slot" / ms.MLPERF_SERVER_DIR / ms.SERVER_CONFIG_FILE
    config = SuiteConfig.from_document(yaml.safe_load(written.read_text()))
    assert envs[MLPERF_SERVER_PID_DIR_ENV] == config.lifecycle.pid_dir
    assert Path(envs[MLPERF_SUITE_PYTHON_ENV]).name.startswith("python")
    assert prepared == [f"deploy {tmp_path / 'ix' / 'benchmarks'}", "preflight"]


def test_a_second_round_rewrites_the_config_without_leaking_its_names_into_the_server(session, env, tmp_path, prepared):
    point = _write(tmp_path / "slot" / "config.yaml", _point())
    ms.prepare_mlperf_round(point, env, str(tmp_path))
    ms.prepare_mlperf_round(point, env, str(tmp_path))

    envs = yaml.safe_load(point.read_text())["benchmark"]["envs"]
    config = SuiteConfig.from_document(yaml.safe_load(Path(envs[MLPERF_SERVER_CONFIG_ENV]).read_text()))
    assert not {MLPERF_SERVER_CONFIG_ENV, MLPERF_SERVER_PID_DIR_ENV, MLPERF_SUITE_PYTHON_ENV} & set(
        config.server.user.env_set
    )


def test_a_point_on_another_script_is_left_alone(tmp_path, env, prepared):
    document = _point()
    document["benchmark"]["benchmark_script"] = "benchmark_serving.sh"
    point = _write(tmp_path / "config.yaml", document)
    before = point.read_text()

    ms.prepare_mlperf_round(point, env, str(tmp_path))

    assert point.read_text() == before
    assert prepared == []


# --- the client script: inference_suite boots the server before the harness and reaps it after ---

_FAKE_SUITE = """#!/usr/bin/env bash
echo "suite $*" >> "$CALLS"
if [ "$3" = run ]; then exit "${BOOT_RC:-0}"; fi
"""
_FAKE_HARNESS = """#!/usr/bin/env bash
echo "harness $*" >> "$CALLS"
mkdir -p "$RESULTS_DIR/run"
echo '{}' > "$RESULTS_DIR/run/result_summary.json"
"""
_FAKE_MAPPER = """import sys
open(sys.argv[2], "w").write("{}")
"""


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _client(tmp_path: Path, boot_rc: int = 0, configured: bool = True) -> tuple[subprocess.CompletedProcess, list]:
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    shutil.copy(agentx_asset_dir() / "mlperf_agentic_client.sh", bench)
    (bench / "map_mlperf.py").write_text(_FAKE_MAPPER, encoding="utf-8")
    endpoints = tmp_path / "endpoints"
    _executable(endpoints / "utility" / "run_agentic.sh", _FAKE_HARNESS)
    calls = tmp_path / "calls"
    env = {
        "PATH": os.environ["PATH"],
        "CALLS": str(calls),
        "BOOT_RC": str(boot_rc),
        "RESULT_DIR": str(tmp_path / "result"),
        "MLPERF_ENDPOINTS_DIR": str(endpoints),
        "AGENTIC_DATASET_PATH": "/d",
        "MLPERF_TOKENIZER_DIR": "/t",
        "CONC": "16",
        "PORT": "30000",
        **{key: value for key, value in _CLIENT_ENVS.items() if key.startswith(("MLPERF_", "AGENTIC_"))},
    }
    if configured:
        env[MLPERF_SERVER_CONFIG_ENV] = "/slot/mlperf_server/config.yaml"
        env[MLPERF_SERVER_PID_DIR_ENV] = "/slot/mlperf_server/pid"
        env[MLPERF_SUITE_PYTHON_ENV] = str(_executable(tmp_path / "suite", _FAKE_SUITE))
    done = subprocess.run(
        ["bash", str(bench / "mlperf_agentic_client.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    return done, calls.read_text().splitlines() if calls.exists() else []


def test_the_client_runs_the_harness_between_the_boot_and_the_stop(tmp_path):
    done, calls = _client(tmp_path)

    assert done.returncode == 0, done.stderr
    run_dir = tmp_path / "result" / "mlperf_server_round"
    assert calls == [
        f"suite -m inference_suite run --config /slot/mlperf_server/config.yaml --run-dir {run_dir}",
        "harness smoke_test kimi-k3 mi355x perf",
        "suite -m inference_suite stop --pid-dir /slot/mlperf_server/pid",
    ]
    assert (tmp_path / "result" / "inferencex_result.json").is_file()


def test_a_server_that_does_not_boot_runs_no_harness_and_is_still_reaped(tmp_path):
    done, calls = _client(tmp_path, boot_rc=1)

    assert done.returncode == 3
    assert [call.split()[3] for call in calls] == ["run", "stop"]


def test_a_client_without_a_server_config_refuses_before_the_harness(tmp_path):
    done, calls = _client(tmp_path, configured=False)

    assert done.returncode != 0
    assert MLPERF_SERVER_CONFIG_ENV in done.stderr
    assert calls == []


def test_the_client_no_longer_speaks_magpies_run_phase():
    assert "MAGPIE_RUN_PHASE" not in (agentx_asset_dir() / "mlperf_agentic_client.sh").read_text()
