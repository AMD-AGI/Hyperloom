# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reproduce an accepted native server in the separate diagnostic profile harness."""

from __future__ import annotations

import shlex
import shutil
import hashlib
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.inference_optimizer.grid_server_args import merge_server_args, remove_server_args

from ._native_source import verify_native_source_imports


def managed_profile_benchmark(benchmark: Mapping[str, Any]) -> bool:
    """Identify a resolved Magpie-owned diagnostic configuration."""
    agentx = benchmark.get("agentx") or {}
    return bool(
        isinstance(agentx, Mapping)
        and (
            (agentx.get("resolved") or {}).get("server-launch-spec")
            or (benchmark.get("workload_spec") or {}).get("profile_parent")
        )
        and ((benchmark.get("profiler") or {}).get("torch_profiler") or {}).get("enabled") is True
    )


def materialize_managed_profile(
    config: dict[str, Any],
    output_dir: Path,
    *,
    out_name: str,
    snapshot: Mapping[str, Any] | None,
    **candidate: Any,
) -> Path:
    """Derive a diagnostic without reapplying ambient workload or recipe defaults."""
    from ._native_candidate import install_native_launch_snapshot, update_native_candidate_file

    if snapshot is not None:
        install_native_launch_snapshot(config["benchmark"], snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / out_name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    update_native_candidate_file(path, **candidate)
    return path


def prepare_managed_profile(params: dict[str, Any], state: Any, output_dir: Path) -> None:
    """Profile the accepted native implementation without replaying candidate deltas."""
    from hyperloom.inference_optimizer.agentx.identity import validate_server_launch
    from ._native_candidate import apply_native_candidate

    measurement = getattr(state, "current_best_measurement", None) or {}
    accepted = str(measurement.get("materialized_config") or getattr(state, "baseline_config_path", "") or "")
    if not accepted:
        raise ValueError("Native diagnostic profiling requires an accepted benchmark configuration")
    config = yaml.safe_load(Path(accepted).read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    evidence = measurement.get("agentx_server_launch") or (benchmark.get("workload_spec") or {}).get("server_launch")
    errors = validate_server_launch(benchmark, evidence)
    if errors:
        raise ValueError("Accepted native server launch is not verified: " + "; ".join(errors))
    verify_native_source_imports(benchmark)
    workload = benchmark.setdefault("workload_spec", {})
    workload["profile_parent"] = {
        "materialized_config": str(Path(accepted).resolve()),
        "config_sha256": hashlib.sha256(Path(accepted).read_bytes()).hexdigest(),
        "workload_fingerprint": (workload.get("execution") or {}).get("workload_fingerprint"),
    }
    settings: dict[str, Any] = {"enabled": True, "detailed_annotations": True}
    template = params.get("config_path")
    if template and Path(template).resolve() != Path(accepted).resolve():
        template_benchmark = (yaml.safe_load(Path(template).read_text(encoding="utf-8")) or {}).get("benchmark") or {}
        settings.update((template_benchmark.get("profiler") or {}).get("torch_profiler") or {})
    settings.update((params.get("profiler") or {}).get("torch_profiler") or {})
    settings.update(params.get("torch_profiler") or {})
    for key in (
        "num_steps",
        "num_profiles",
        "start_seconds",
        "interval_seconds",
        "capture_timeout_seconds",
        "flush_timeout_seconds",
        "detailed_annotations",
    ):
        if key in params:
            settings[key] = params[key]
    settings["enabled"] = True
    benchmark["profiler"] = {
        "torch_profiler": settings,
        "system_profiler": {"enabled": False},
        "tracelens": {"enabled": False},
    }
    apply_native_candidate(
        benchmark,
        extra_server_args=str(params.get("extra_server_args") or ""),
        extra_envs=params.get("extra_envs"),
        remove_args=params.get("remove_args"),
        unset_envs=params.get("unset_envs"),
        args_mode=str(params.get("args_mode") or "append"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "managed_profile.source.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    params["config_path"] = str(destination)
    params["native_launch_overrides"] = deepcopy(benchmark["agentx"]["launch_overrides"])
    params["baseline_double_run"] = False
    params["disable_run_eval"] = True
    for key in ("extra_server_args", "extra_envs", "remove_args", "unset_envs", "args_mode"):
        params.pop(key, None)


def project_native_profile(params: dict[str, Any], state: Any) -> None:
    """Use observed server flags/runtime while retaining diagnostic client ownership."""
    from hyperloom.inference_optimizer.agentx.identity import canonical_sha256

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
