# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate optimizer candidates into Magpie's native server launch contract."""

from __future__ import annotations

import os
import shlex
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import BLOCKED_CHILD_ENV_NAMES, _ENV_KEY_RE
from hyperloom.inference_optimizer.agentx.identity import has_launch_contract
from ._native_args import compose_native_args
from ._native_source import source_file_hashes, verify_native_source_imports

_FIXED_ENVS = frozenset(
    {
        "MODEL",
        "MODEL_PATH",
        "FRAMEWORK",
        "TP",
        "EP",
        "PP",
        "PCP_SIZE",
        "DCP_SIZE",
        "DP",
        "CONC",
        "ISL",
        "OSL",
        "NUM_PROMPTS",
        "NUM_WARMUPS",
        "RANDOM_RANGE_RATIO",
        "RUN_EVAL",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "AIPERF_BIN",
        "WEKA_LOADER_OVERRIDE",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "path_prefix",
        "entrypoint_bin_dir",
        "pythonpath_prefix",
        "pythonpath_prefixes",
        "ld_library_path_prefix",
        "runtime_env",
        "framework_bin",
        "framework_python",
        "framework_venv_root",
        "runtime_python_exe",
    }
)


def install_native_launch_snapshot(benchmark: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
    """Restore the accepted server implementation without replaying its deltas."""
    if not has_launch_contract(benchmark) or not has_launch_contract({"agentx": {"launch_overrides": snapshot}}):
        raise ValueError("Native launch snapshots require an existing version 1 optimization contract")
    benchmark["agentx"]["launch_overrides"] = deepcopy(dict(snapshot))


def _fixed_env(name: str) -> bool:
    return name in _FIXED_ENVS or name.startswith(("AGENTX_", "AIPERF_", "INFERENCEX_"))


def _runtime_candidate(benchmark: Mapping[str, Any], override: Mapping[str, Any], env: dict[str, str]) -> str:
    """Resolve the exact server interpreter/entrypoint and runtime search paths."""
    unknown = set(override).difference(_RUNTIME_KEYS)
    if unknown:
        raise ValueError(f"Unsupported native runtime_override fields: {sorted(unknown)}")
    from ._grid_runner import _RUNTIME_ENV_RESERVED, apply_runtime_override

    runtime_env = override.get("runtime_env") or {}
    if not isinstance(runtime_env, Mapping):
        raise ValueError("Native runtime_env must be an environment mapping")
    for name in runtime_env:
        if (
            _fixed_env(str(name))
            or str(name) in BLOCKED_CHILD_ENV_NAMES | _RUNTIME_ENV_RESERVED
            or not _ENV_KEY_RE.fullmatch(str(name))
        ):
            raise ValueError(f"Native runtime cannot override protected environment: {name}")
    for key in (
        "path_prefix",
        "entrypoint_bin_dir",
        "pythonpath_prefix",
        "pythonpath_prefixes",
        "ld_library_path_prefix",
    ):
        raw = override.get(key)
        for entry in [raw] if isinstance(raw, str) else raw or []:
            path = Path(str(entry)).expanduser()
            if not path.is_absolute() or not path.is_dir() or ":" in str(path) or any(c in str(path) for c in "\n\r\0"):
                raise ValueError(f"Native runtime requires an existing absolute directory for {key}: {entry}")
    combined = {**{str(k): str(v) for k, v in (benchmark.get("envs") or {}).items()}, **env}
    before = dict(combined)
    apply_runtime_override(combined, dict(override))
    env.update({k: v for k, v in combined.items() if before.get(k) != v and not k.startswith("HYPERLOOM_FRAMEWORK_")})
    framework = str(benchmark.get("framework") or "").lower()
    python = str(override.get("runtime_python_exe") or override.get("framework_python") or "")
    bin_path = str(override.get("framework_bin") or "")
    venv = str(override.get("framework_venv_root") or "")
    if framework == "sglang":
        executable = python or (str(Path(venv) / "bin/python") if venv else "")
        if not executable and bin_path:
            root = Path(bin_path) if Path(bin_path).is_dir() else Path(bin_path).parent
            executable = str(root / "python")
    elif framework == "vllm":
        bin_root = (
            Path(bin_path) if bin_path else Path(python).parent if python else Path(venv) / "bin" if venv else None
        )
        executable = str(bin_root / "vllm") if bin_root and bin_root.is_dir() else str(bin_root or "")
    else:
        raise ValueError(f"Native runtime override is unsupported for framework {framework!r}")
    if executable:
        path = Path(executable).expanduser()
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Native server executable is not an absolute executable file: {executable}")
        return str(path)
    if python or bin_path or venv:
        raise ValueError("Native runtime does not identify a usable server executable")
    return ""


def _base_removals(benchmark: Mapping[str, Any], existing: Mapping[str, Any], requested: Any) -> list[str]:
    removed = to_str_list(existing.get("remove_args"))
    appended = {token.split("=", 1)[0] for token in existing.get("append_args", []) if token.startswith("--")}
    workload = benchmark.get("workload_spec") or {}
    observed = workload.get("server_launch") or {}
    base = observed.get("base_argv")
    base_flags = {token.split("=", 1)[0] for token in base or [] if token.startswith("--")}
    for raw in to_str_list(requested):
        tokens = shlex.split(raw)
        if len(tokens) != 1 or not tokens[0].startswith("--") or "=" in tokens[0]:
            raise ValueError(f"Native remove_args requires a flag name: {raw!r}")
        name = tokens[0]
        if name in appended and base is None:
            raise ValueError("Removing an inherited native candidate flag requires verified base_argv evidence")
        if name in appended and name not in base_flags:
            continue
        if name not in removed:
            removed.append(name)
    return removed


def apply_native_candidate(
    benchmark: dict[str, Any],
    *,
    extra_server_args: str = "",
    extra_envs: Mapping[str, Any] | None = None,
    remove_args: Any = None,
    unset_envs: Any = None,
    args_mode: str = "append",
    runtime_override: Mapping[str, Any] | None = None,
    overlay_pythonpath: str = "",
    source_files: Mapping[str, str] | None = None,
    absent_source_files: Sequence[str] = (),
) -> None:
    """Compose one candidate layer without changing the canonical benchmark inputs."""
    if not has_launch_contract(benchmark):
        raise ValueError("Native optimization requires an accepted launch_overrides version 1 contract")
    agentx = benchmark["agentx"]
    existing = dict(agentx["launch_overrides"])
    env = {str(k): str(v) for k, v in (existing.get("env") or {}).items()}
    fixed = benchmark.get("envs") or {}
    removed_env = to_str_list(existing.get("unset_env"))
    for name in to_str_list(unset_envs):
        if _fixed_env(name) or name in BLOCKED_CHILD_ENV_NAMES or not _ENV_KEY_RE.fullmatch(name):
            raise ValueError(f"Native candidate cannot unset protected environment: {name}")
        env.pop(name, None)
        if name not in removed_env:
            removed_env.append(name)
    for key, raw in (extra_envs or {}).items():
        name, value = str(key), str(raw)
        if _fixed_env(name):
            if name in fixed and str(fixed[name]) == value:
                continue
            raise ValueError(f"Native candidate cannot change canonical workload environment: {name}")
        if name in BLOCKED_CHILD_ENV_NAMES or not _ENV_KEY_RE.fullmatch(name):
            raise ValueError(f"Native candidate environment is not allowed: {name}")
        env[name] = value
        if name in removed_env:
            removed_env.remove(name)
    replace = str(args_mode).lower() == "replace"
    removed_args = _base_removals(benchmark, existing, remove_args)
    args = compose_native_args(
        existing.get("append_args") or [],
        str(extra_server_args or ""),
        remove=to_str_list(remove_args),
        replace=replace,
    )
    files = dict(existing.get("source_files") or {})
    if overlay_pythonpath:
        overlay = Path(overlay_pythonpath).expanduser()
        if not overlay.is_absolute() or not overlay.is_dir() or not (overlay / "sitecustomize.py").is_file():
            raise ValueError("Native kernel overlay must be an existing absolute directory with sitecustomize.py")
        if ":" in str(overlay) or any(c in str(overlay) for c in "\n\r\0"):
            raise ValueError("Native kernel overlay must be one directory")
        inherited = env.get("PYTHONPATH", str(fixed.get("PYTHONPATH") or ""))
        env["PYTHONPATH"] = ":".join(
            [str(overlay), *[part for part in inherited.split(":") if part and part != str(overlay)]]
        )
        files.update(
            source_file_hashes(sorted(p for p in overlay.rglob("*") if p.is_file() and "__pycache__" not in p.parts))
        )
    executable = str(existing.get("executable") or "")
    if runtime_override:
        executable = _runtime_candidate(benchmark, runtime_override, env) or executable
    files.update(dict(source_files or {}))
    absent = set(existing.get("absent_source_files") or []).difference(source_files or {})
    absent.update(absent_source_files)
    for path in absent_source_files:
        files.pop(path, None)
    updated: dict[str, Any] = {
        **existing,
        "version": 1,
        "append_args": args,
        "remove_args": removed_args,
        "replace_args": bool(existing.get("replace_args")) or replace,
        "env": dict(sorted(env.items())),
        "unset_env": sorted(removed_env),
        "source_files": dict(sorted(files.items())),
        "absent_source_files": sorted(absent),
    }
    if executable:
        updated["executable"] = executable
    agentx["launch_overrides"] = updated


def update_native_candidate_file(config_path: Path, **candidate: Any) -> None:
    """Attach post-apply source/runtime evidence to the YAML consumed by Magpie."""
    from hyperloom.inference_optimizer.agentx.native import resolve_native_recipe

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    apply_native_candidate(benchmark, **candidate)
    topology = (benchmark.get("workload_spec") or {}).get("resolved_topology") or {}
    resolve_native_recipe(
        benchmark,
        inferencex_path=str(benchmark.get("inferencex_path") or os.environ.get("INFERENCEX_PATH") or ""),
        expected_gpu_count=int(os.environ.get("HYPERLOOM_AGENTX_GPU_COUNT") or topology.get("gpu_count") or 0),
    )
    verify_native_source_imports(benchmark)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def record_launch_evidence(config_path: Path, measurement: Mapping[str, Any]) -> None:
    """Retain verified base argv for composing later candidates against this recipe."""
    if measurement.get("valid_measurement") is not True or measurement.get("agentx_launch_contract") != 1:
        return
    evidence = measurement.get("agentx_server_launch")
    if not isinstance(evidence, Mapping) or measurement.get("agentx_candidate_fingerprint") != evidence.get(
        "evidence_sha256"
    ):
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    if not has_launch_contract(benchmark):
        return
    benchmark.setdefault("workload_spec", {})["server_launch"] = dict(evidence)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
