# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize and prepare the workload through its owning benchmark contract.

An MLPerf AgentX round's server is inference_suite's. Its client script runs only the MLPerf harness: preparing the
round deploys the script, preflights the harness and writes the config of an inference_suite ``boot`` round on the
session's base beside the round's point YAML, naming that config, the pid_dir and the interpreter that runs
inference_suite in the point's environment. The script boots the server with ``python -m inference_suite run`` and
reaps it with ``stop`` on exit.
"""

from __future__ import annotations

import logging
import os
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
    is_agentx_client_script,
    is_mlperf_backend,
)
from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.common.env_safety import valid_env_key
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.framework_registry import extra_args_env
from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch, frozen_session_base
from inference_suite.argv import remove_server_args
from inference_suite.contracts import GpuTelemetry, Lifecycle, LifecycleMode, RunPaths, SuiteConfig
from inference_suite.frameworks import Framework

from ._benchmark_interpreter import _resolve_probe_python
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_RETURNCODE,
    BENCHMARK_TIMEOUT_ENV,
    PreparationFailure,
    default_benchmark_timeout_sec,
    resolve_benchmark_timeouts,
)
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
_UTF8 = "utf-8"
#: The boot round ends at ready, before any GPU reading would mean anything.
_NO_TELEMETRY = GpuTelemetry(enabled=False, interval_sec=1.0)


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
    """The point as a boot-only round reads it: the server's own knobs, the keep-alive, inference_suite's model name.

    The client script is the round's caller, not part of the server it boots.
    """
    bench = {key: value for key, value in document["benchmark"].items() if key != "benchmark_script"}
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


def mlperf_server_config(document: Mapping[str, Any], root: Path, env: Mapping[str, str]) -> SuiteConfig:
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
    # The default benchmark path never imports the agentx package.
    from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
    from hyperloom.inference_optimizer.agentx.preflight import check_mlperf_harness

    document = yaml.safe_load(config_path.read_text(encoding=_UTF8))
    bench = document["benchmark"]
    if Path(str(bench.get("benchmark_script") or "")).name != MLPERF_CLIENT_SCRIPT:
        return
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    check_mlperf_harness(env)
    envs = bench.setdefault("envs", {})
    for name in (MLPERF_SERVER_CONFIG_ENV, MLPERF_SERVER_PID_DIR_ENV, MLPERF_SUITE_PYTHON_ENV):
        envs.pop(name, None)
    root = config_path.parent / MLPERF_SERVER_DIR
    config = mlperf_server_config(document, root, env)
    root.mkdir(parents=True, exist_ok=True)
    (root / SERVER_CONFIG_FILE).write_text(yaml.safe_dump(config.to_document(), sort_keys=False), encoding=_UTF8)
    envs[MLPERF_SERVER_CONFIG_ENV] = str(root / SERVER_CONFIG_FILE)
    envs[MLPERF_SERVER_PID_DIR_ENV] = str(config.lifecycle.pid_dir)
    envs[MLPERF_SUITE_PYTHON_ENV] = sys.executable
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding=_UTF8)


def composed_benchmark_timeout_sec(config_path: Path) -> float:
    """The hard cap the rounds of one materialized config are granted, sized to the env that config launches with.

    The workload a round has to get through is named by the composed env, not by the
    variant spec: the pin that sizes it -- a sweep's ``CONC``, an agentic flow -- is as
    often inherited from the base YAML or from the recipe's base layer as it is set by
    the variant, and a variant may equally clear one the base pinned. Sizing the grant
    from the variant's own deltas alone would grant the time some other, usually
    smaller, run needs, so it is read back off the config the round is launched from.

    Only the size comes from that env. The hard cap's override is the operator's grant,
    read and validated once against the process env; a config's value for that name is
    the workload's own and is never parsed as a grant here. A round whose config clears
    the override runs on the size its workload needs. A config that names a workload pin
    this cannot read raises, as launching that round would: the caller owns the blast
    radius of one unusable config, and the grid runner ends that variant on it.
    """
    point = read_benchmark_point(config_path)
    envs = dict(os.environ)
    envs.update({str(k): str(v) for k, v in point.envs.items()})
    if point.agentx:
        # AgentX keeps the variant's deltas in its launch overrides and applies them over the
        # inherited env when it starts; the same order here, so the cap is sized to the env the
        # round runs under rather than to the one it inherited.
        for name in point.agentx_launch.unset_env:
            envs.pop(name, None)
        envs.update({str(k): str(v) for k, v in point.agentx_launch.env.items()})
    if BENCHMARK_TIMEOUT_ENV in envs and BENCHMARK_TIMEOUT_ENV in os.environ:
        return resolve_benchmark_timeouts()[1]
    return default_benchmark_timeout_sec(envs)


def prepare_mlperf_client(config_path: Path, env: dict[str, str]) -> PreparationFailure | None:
    """Prepare the MLPerf client a backend round at ``config_path`` runs, against the ``$INFERENCEX_PATH`` checkout."""
    error = prepare_agentx_runtime(
        env=env, inferencex_path=os.environ.get("INFERENCEX_PATH", "").strip(), config_path=config_path
    )
    if not error:
        return None
    logging.getLogger(__name__).error("%s; failing this benchmark", error)
    return PreparationFailure(AGENTX_PREFLIGHT_RETURNCODE, error)


def prepare_agentx_runtime(
    *,
    env: dict[str, str] | None = None,
    inferencex_path: str | None = None,
    config_path: Path | str | None = None,
    active: bool | None = None,
) -> str | None:
    """Deploy and preflight the MLPerf client, and write the config its server boots from."""
    runtime_env = env or os.environ
    if not is_mlperf_backend(runtime_env):
        return None
    if active is None:
        active = agentx_enabled(runtime_env)
        if not active and config_path:
            try:
                active = is_agentx_client_script(read_benchmark_point(Path(config_path)).benchmark_script)
            except (OSError, ValueError, yaml.YAMLError):
                active = False
    if not active:
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError

    try:
        prepare_mlperf_round(Path(config_path or ""), runtime_env, str(inferencex_path or ""))
    except AgentXPreflightError as exc:
        return f"AgentX preflight failed: {exc}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, BaseLaunchMismatch) as exc:
        return f"AgentX runtime preparation failed: {type(exc).__name__}: {exc}"
    return None
