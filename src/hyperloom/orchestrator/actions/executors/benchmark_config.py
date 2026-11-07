# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize and prepare the workload through its owning benchmark contract.

An MLPerf AgentX round runs on inference_suite, which serves the session's frozen base and runs the MLPerf harness as
the round's load; preparing the round preflights that harness.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from hyperloom.common.agentx_workload import is_agentx_client_script, is_mlperf_backend
from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch, frozen_session_base

from ._subprocess_kill import (
    BENCHMARK_TIMEOUT_ENV,
    default_benchmark_timeout_sec,
    resolve_benchmark_timeouts,
)


def check_mlperf_round(config_path: Path, env: Mapping[str, str]) -> None:
    """Refuse the MLPerf point at ``config_path`` unless its round can run; a point on another script is not checked.

    Raises:
        AgentXPreflightError: The session froze no base launch for the round's server, or the harness cannot run.
        BaseLaunchMismatch: The session's base launch is not the one it froze.
    """
    # The default benchmark path never imports the agentx package.
    from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError, check_mlperf_harness

    if not is_agentx_client_script(read_benchmark_point(config_path).benchmark_script):
        return
    if frozen_session_base() is None:
        raise AgentXPreflightError(
            "an MLPerf round runs on inference_suite, which serves the session's frozen base launch, and this session "
            "froze none; start a new session"
        )
    check_mlperf_harness(env)


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


def prepare_agentx_runtime(
    *,
    env: Mapping[str, str] | None = None,
    config_path: Path | str | None = None,
    active: bool | None = None,
) -> str | None:
    """Why the MLPerf round at ``config_path`` cannot run, or ``None`` when it can or is not an MLPerf round."""
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
        check_mlperf_round(Path(config_path or ""), runtime_env)
    except AgentXPreflightError as exc:
        return f"AgentX preflight failed: {exc}"
    except (OSError, ValueError, yaml.YAMLError, BaseLaunchMismatch) as exc:
        return f"AgentX runtime preparation failed: {type(exc).__name__}: {exc}"
    return None
