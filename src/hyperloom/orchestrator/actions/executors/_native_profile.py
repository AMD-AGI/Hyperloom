# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reproduce an accepted native server in the separate diagnostic profile harness."""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256
from hyperloom.inference_optimizer.grid_server_args import merge_server_args, remove_server_args

from ._native_source import verify_native_source_imports


def project_native_profile(params: dict[str, Any], state: Any) -> None:
    """Use observed server flags/runtime while retaining diagnostic client ownership."""
    measurement = getattr(state, "current_best_measurement", None) or {}
    path = str(measurement.get("materialized_config") or getattr(state, "baseline_config_path", "") or "")
    if not path:
        raise ValueError("Native diagnostic profiling requires an accepted benchmark configuration")
    benchmark = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("benchmark") or {}
    evidence = measurement.get("agentx_server_launch")
    if not isinstance(evidence, Mapping):
        evidence = (benchmark.get("workload_spec") or {}).get("server_launch")
    if not isinstance(evidence, Mapping):
        raise ValueError("Native diagnostic profiling requires verified server launch evidence")
    if evidence.get("evidence_sha256") != canonical_sha256(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    ):
        raise ValueError("Native diagnostic profiling server launch evidence changed")
    argv = evidence.get("effective_argv") or []
    framework = str(benchmark.get("framework") or "").lower()
    if framework == "sglang" and argv[1:3] == ["-m", "sglang.launch_server"]:
        flags = argv[3:]
    elif framework == "vllm" and argv[1:2] == ["serve"] and len(argv) >= 3:
        flags = argv[3:]
    else:
        raise ValueError("Native diagnostic profiling cannot reproduce this server entrypoint")
    projected = remove_server_args(
        shlex.join(flags), ["--model", "--model-path", "--served-model-name", "--host", "--port"]
    )
    params["extra_server_args"] = merge_server_args(projected, str(params.get("extra_server_args") or ""))
    runtime = {str(k): str(v) for k, v in (evidence.get("runtime_environment") or {}).items()}
    safe_env, _ = filter_untrusted_env_mapping(runtime, allow_predicate=is_allowed_variant_env_key)
    params["extra_envs"] = {**safe_env, **dict(params.get("extra_envs") or {})}
    override = dict(params.get("runtime_override") or {})
    for env_key, field in (("PYTHONPATH", "pythonpath_prefixes"), ("LD_LIBRARY_PATH", "ld_library_path_prefix")):
        if runtime.get(env_key):
            override[field] = [entry for entry in runtime[env_key].split(":") if entry]
    executable = str(evidence.get("resolved_executable") or argv[0])
    executable = shutil.which(executable, path=runtime.get("PATH")) or executable
    if not Path(executable).is_absolute():
        raise ValueError("Native diagnostic profiling cannot resolve the accepted server executable")
    if framework == "sglang":
        override["runtime_python_exe"] = executable
    else:
        override["framework_bin"] = executable
        override["framework_python"] = str(Path(executable).parent / "python")
    params["runtime_override"] = override
    verify_native_source_imports(benchmark)
