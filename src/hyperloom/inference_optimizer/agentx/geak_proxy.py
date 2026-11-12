# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seed GEAK's proposal workload from the server configuration the measured round's ``launch.json`` records."""

from __future__ import annotations

import copy
import shlex
from typing import Any

_PROXY_OWNED_OPTIONS = frozenset(
    {
        "--model",
        "--model-path",
        "--served-model-name",
        "--host",
        "--port",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--pp",
        "--pp-size",
        "--pipeline-parallel-size",
        "--ep",
        "--ep-size",
        "--expert-parallel-size",
        "--dp",
        "--dp-size",
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-size-local",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--enable-dp-attention",
        "--enable-expert-parallel",
        "--decode-context-parallel-size",
        "--prefill-context-parallel-size",
    }
)


def _tunable_flags(flags: str) -> str:
    retained: list[str] = []
    keep = False
    for token in shlex.split(flags):
        if token.startswith("--"):
            keep = token.split("=", 1)[0] not in _PROXY_OWNED_OPTIONS
        if keep:
            retained.append(token)
    return shlex.join(retained)


def seed_native_geak_proxy(handoff: dict[str, Any]) -> None:
    """Pass the measured round's tunables through GEAK's existing effective-config contract.

    GEAK owns its proxy endpoint, client, topology and profiling lifecycle. Its
    measurements remain incomparable even when all server tunables match.
    Only Hyperloom's subsequent canonical AgentX run can accept the product.

    Raises:
        ValueError: The measured round left no ``launch.json`` to take the tunables from.
    """
    spec = copy.deepcopy(handoff.get("baseline_env_spec") or {})
    if not spec.get("launch_json"):
        raise ValueError("Native AgentX GEAK proposals require the measured round's launch.json")
    config = spec["config"]
    envs = dict(config.get("extra_envs") or {})
    config.update(
        {
            "server_launch_flags": _tunable_flags(str(config.get("server_launch_flags") or "")),
            "extra_server_args": "",
            "extra_envs": envs,
        }
    )
    handoff["baseline_env_spec"] = spec
    handoff["accepted_flags"] = ""
    handoff["accepted_env"] = shlex.join(f"{name}={value}" for name, value in envs.items())
    handoff["raw_baseline_tput"] = 0.0
    handoff["orchestrator_best_tput_same_config"] = 0.0
    handoff["same_config_reference_status"] = "unverified"
    handoff["same_config_reference_verification_status"] = "unverified_workload"
