# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seed GEAK's proposal workload from an observed native server configuration."""

from __future__ import annotations

import copy
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import canonical_sha256

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


def _tunable_flags(argv: list[str], framework: str) -> str:
    if framework == "sglang" and argv[1:3] == ["-m", "sglang.launch_server"]:
        args = argv[3:]
    elif framework == "vllm" and argv[1:2] == ["serve"] and len(argv) >= 3:
        args = argv[3:]
    else:
        raise ValueError("Native AgentX launch evidence has an unsupported server entry point")
    retained: list[str] = []
    keep = False
    for token in args:
        if token.startswith("--"):
            keep = token.split("=", 1)[0] not in _PROXY_OWNED_OPTIONS
        if keep:
            retained.append(token)
    return shlex.join(retained)


def seed_native_geak_proxy(
    handoff: dict[str, Any], *, benchmark: Mapping[str, Any], measurement: Mapping[str, Any]
) -> None:
    """Pass observed tunables through GEAK's existing effective-config contract.

    GEAK owns its proxy endpoint, client, topology and profiling lifecycle. Its
    measurements remain incomparable even when all server tunables match.
    Only Hyperloom's subsequent canonical AgentX run can accept the product.
    """
    evidence = measurement.get("agentx_server_launch")
    if not isinstance(evidence, dict):
        workload = benchmark.get("workload_spec")
        evidence = workload.get("server_launch") if isinstance(workload, Mapping) else None
    if not isinstance(evidence, dict):
        raise ValueError("Native AgentX GEAK proposals require a verified server launch")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != canonical_sha256(unsigned):
        raise ValueError("Native AgentX GEAK server launch evidence changed")
    argv = evidence.get("effective_argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("Native AgentX GEAK server launch argv is missing")
    flags = _tunable_flags(argv, str(handoff.get("framework") or ""))
    runtime = dict(evidence.get("runtime_environment") or {})
    if argv and Path(argv[0]).is_absolute():
        executable_dir = str(Path(argv[0]).parent)
        runtime["PATH"] = ":".join(part for part in (executable_dir, runtime.get("PATH", "")) if part)
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        runtime.pop(name, None)
    spec = copy.deepcopy(handoff.get("baseline_env_spec") or {})
    config = spec.setdefault("config", {})
    config.update({"server_launch_flags": flags, "extra_server_args": "", "extra_envs": runtime})
    handoff["baseline_env_spec"] = spec
    handoff["accepted_flags"] = ""
    handoff["accepted_env"] = shlex.join(f"{name}={value}" for name, value in runtime.items())
    handoff["native_server_launch"] = copy.deepcopy(evidence)
    handoff["raw_baseline_tput"] = 0.0
    handoff["orchestrator_best_tput_same_config"] = 0.0
    handoff["same_config_reference_status"] = "unverified"
    handoff["same_config_reference_verification_status"] = "unverified_workload"
