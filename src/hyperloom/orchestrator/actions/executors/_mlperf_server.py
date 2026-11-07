# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare an MLPerf AgentX round, whose server inference_suite boots and reaps.

The client script runs only the MLPerf harness. Before the round, this deploys the script, preflights the harness and
writes the config of an inference_suite ``boot`` round on the session's base beside the round's point YAML; it names
that config, the pid_dir and the interpreter that runs inference_suite in the point's environment. The script boots
the server with ``python -m inference_suite run`` and reaps it with ``stop`` on exit, so the server's pid is
inference_suite's. A point served off this node boots nothing: the script drives the server it is pointed at.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from hyperloom.common.agentx_workload import (
    MLPERF_CLIENT_SCRIPT,
    MLPERF_SERVED_MODEL,
    MLPERF_SERVER_CONFIG_ENV,
    MLPERF_SERVER_PID_DIR_ENV,
    MLPERF_SUITE_PYTHON_ENV,
    is_agentx_client_key,
)
from hyperloom.common.env_safety import valid_env_key
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
from hyperloom.inference_optimizer.agentx.preflight import check_mlperf_harness
from hyperloom.inference_optimizer.framework_registry import extra_args_env
from hyperloom.orchestrator.state.base_launch import frozen_session_base
from inference_suite.argv import remove_server_args
from inference_suite.contracts import GpuTelemetry, Lifecycle, LifecycleMode, RunPaths, SuiteConfig
from inference_suite.frameworks import Framework

from ._benchmark_interpreter import _resolve_probe_python
from ._subprocess_kill import resolve_benchmark_timeouts
from ._suite_config import RoundSettings, suite_config

#: The directory beside the round's point YAML that holds the boot round's config and pid_dir.
MLPERF_SERVER_DIR = "mlperf_server"
SERVER_CONFIG_FILE = "config.yaml"
SERVER_PID_DIR = "pid"
#: The config's run_dir is nominal: ``python -m inference_suite run`` takes the run_dir per invocation.
_NOMINAL_RUN_DIR = "run"
#: Where install.sh stages the pinned tools every inference_suite round verifies.
TOOL_CACHE_ENV = "INFERENCE_SUITE_TOOL_CACHE"
_STAGING_MODULE = "inference_suite.staging"
#: A point carrying this serves off this node; Hyperloom sets it for a multi-node client-only session.
_REMOTE_SERVER_ENV = "BENCHMARK_BASE_URL"
#: The point marks a boot-only round with this, which :func:`suite_config` reads.
_RUN_PHASE_ENV = "MAGPIE_RUN_PHASE"
_SERVER_ONLY_PHASE = "server"
#: The point's client knobs that never reach the server.
_CLIENT_ONLY_ENV = frozenset({"MODEL", "FRAMEWORK"})
_SERVED_MODEL_FLAG = "--served-model-name"
#: An agentic turn idles for minutes between tool results, so a server defaulting to a short keep-alive drops the
#: connection mid-trajectory.
_KEEP_ALIVE_SEC = "900"
_KEEP_ALIVE_ENV: Mapping[Framework, str] = MappingProxyType(
    {Framework.VLLM: "VLLM_HTTP_TIMEOUT_KEEP_ALIVE", Framework.SGLANG: "SGLANG_TIMEOUT_KEEP_ALIVE"}
)
#: The boot round ends at ready, before any GPU reading would mean anything.
_NO_TELEMETRY = GpuTelemetry(enabled=False, interval_sec=1.0)
_UTF8 = "utf-8"


class MlperfServerError(ValueError):
    """The round's server cannot be booted through inference_suite."""


def _manifest_sha256(tool_cache: str) -> str:
    """The sha256 of the staged tools' manifest, which inference_suite's own check prints once the tools verify."""
    done = subprocess.run(
        [sys.executable, "-m", _STAGING_MODULE, "check", "--tool-cache", tool_cache],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise MlperfServerError(f"the tools at {tool_cache} are not staged ({done.stderr.strip()}); run install.sh")
    return done.stdout.strip()


def _server_point(document: Mapping[str, Any]) -> dict[str, Any]:
    """The point as a boot-only round reads it: the server's own knobs, the keep-alive, inference_suite's model name."""
    bench = dict(document["benchmark"])
    envs = {str(key): str(value) for key, value in (bench.get("envs") or {}).items() if value is not None}
    envs = {key: value for key, value in envs.items() if key not in _CLIENT_ONLY_ENV}
    envs = {key: value for key, value in envs.items() if not is_agentx_client_key(key, envs)}
    framework = Framework(str(bench.get("framework") or "").strip().lower())
    extra = extra_args_env(framework.value)
    # The flag stays in the point for GEAK's handoff, which launches the builtin server script; the round's server
    # takes the name from its config.
    envs[extra] = remove_server_args(envs.get(extra, ""), [_SERVED_MODEL_FLAG])
    keep_alive = _KEEP_ALIVE_ENV.get(framework)
    if keep_alive is not None:
        envs.setdefault(keep_alive, _KEEP_ALIVE_SEC)
    envs[_RUN_PHASE_ENV] = _SERVER_ONLY_PHASE
    bench["envs"] = envs
    return {**document, "benchmark": bench}


def _inherited(env: Mapping[str, str]) -> tuple[str, ...]:
    """The benchmark process's environment the server sees, as Magpie's launcher let it; the config sets the GPUs."""
    return tuple(sorted(name for name in env if valid_env_key(name) and name not in GPU_MASK_ENV_NAMES))


def server_config(document: Mapping[str, Any], root: Path, env: Mapping[str, str]) -> SuiteConfig:
    """The config of the boot round that serves the MLPerf point ``document``, with its pid_dir under ``root``.

    Raises:
        MlperfServerError: The session froze no base, or no tools are staged.
        ConfigError: The point is not one a round serves.
    """
    frozen = frozen_session_base()
    if frozen is None:
        raise MlperfServerError("the session froze no base launch for the server; start a new session")
    tool_cache = str(env.get(TOOL_CACHE_ENV) or "").strip()
    if not tool_cache:
        raise MlperfServerError(f"{TOOL_CACHE_ENV} is not set; run install.sh to stage the tools")
    framework = str(document["benchmark"].get("framework") or "").strip().lower()
    settings = RoundSettings(
        base_path=str(frozen.path),
        paths=RunPaths(str(root / _NOMINAL_RUN_DIR), tool_cache, _manifest_sha256(tool_cache)),
        framework_python=_resolve_probe_python(framework, env=env),
        lifecycle=Lifecycle(LifecycleMode.BOOT, str(root / SERVER_PID_DIR)),
        telemetry=_NO_TELEMETRY,
        silence_sec=resolve_benchmark_timeouts(env)[0],
        env_inherit=_inherited(env),
    )
    config = suite_config(_server_point(document), frozen.launch, settings)
    return replace(config, server=replace(config.server, served_model_name=MLPERF_SERVED_MODEL))


def prepare_mlperf_round(config_path: Path, env: Mapping[str, str], inferencex_path: str) -> None:
    """Deploy the client, preflight its harness, and name the server's boot config in the point at ``config_path``.

    A point that runs another script is left as it is.

    Raises:
        AgentXPreflightError: The MLPerf harness cannot run.
        MlperfServerError: The server cannot be booted through inference_suite.
        ConfigError: The point is not one a round serves.
    """
    document = yaml.safe_load(config_path.read_text(encoding=_UTF8))
    bench = document["benchmark"]
    if Path(str(bench.get("benchmark_script") or "")).name != MLPERF_CLIENT_SCRIPT:
        return
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    check_mlperf_harness(env)
    envs = bench.setdefault("envs", {})
    for name in (MLPERF_SERVER_CONFIG_ENV, MLPERF_SERVER_PID_DIR_ENV, MLPERF_SUITE_PYTHON_ENV):
        envs.pop(name, None)
    if not str(envs.get(_REMOTE_SERVER_ENV) or "").strip():
        root = config_path.parent / MLPERF_SERVER_DIR
        config = server_config(document, root, env)
        root.mkdir(parents=True, exist_ok=True)
        (root / SERVER_CONFIG_FILE).write_text(yaml.safe_dump(config.to_document(), sort_keys=False), encoding=_UTF8)
        envs[MLPERF_SERVER_CONFIG_ENV] = str(root / SERVER_CONFIG_FILE)
        envs[MLPERF_SERVER_PID_DIR_ENV] = str(config.lifecycle.pid_dir)
        envs[MLPERF_SUITE_PYTHON_ENV] = sys.executable
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding=_UTF8)


__all__ = [
    "MLPERF_SERVER_DIR",
    "SERVER_CONFIG_FILE",
    "SERVER_PID_DIR",
    "TOOL_CACHE_ENV",
    "MlperfServerError",
    "prepare_mlperf_round",
    "server_config",
]
