# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate Hyperloom candidates into Magpie's public AgentX launch contract."""

from __future__ import annotations

import copy
import hashlib
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.unified_diff import touched_paths


_ENCODING = "utf-8"


@dataclass
class LaunchChanges:
    """One candidate relative to the session's accepted launch."""

    args: str = ""
    env: Mapping[str, Any] = field(default_factory=dict)
    remove_args: list[str] = field(default_factory=list)
    unset_env: list[str] = field(default_factory=list)
    replace_args: bool = False


REPLACE = "replace"


def load_benchmark(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding=_ENCODING))["benchmark"])


def native_config(path: Path) -> bool:
    return is_magpie_agentx(load_benchmark(path))


def apply_launch_changes(benchmark: dict[str, Any], changes: LaunchChanges) -> None:
    """Compose a candidate; Magpie validates protected args and environment."""
    from hyperloom.inference_optimizer.grid_server_args import compose_server_args, dedupe_extra_server_args

    agentx = benchmark["agentx"]
    if not isinstance(agentx, dict):
        agentx = {"enabled": True}
        benchmark["agentx"] = agentx
    launch = agentx.setdefault("launch_overrides", {"version": 1})
    inherited = shlex.join(launch.get("append_args", []))
    launch["append_args"] = shlex.split(
        dedupe_extra_server_args(
            compose_server_args(
                inherited_args=inherited,
                variant_extra_args=changes.args,
                remove_args=changes.remove_args,
                args_mode=REPLACE if changes.replace_args else "append",
            )
        )
    )
    launch["replace_args"] = changes.replace_args or launch.get("replace_args", False)
    launch["remove_args"] = sorted(set(launch.get("remove_args", [])) | set(changes.remove_args))
    env = launch.setdefault("env", {})
    for name in changes.unset_env:
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in changes.env.items()})
    launch["unset_env"] = sorted((set(launch.get("unset_env", [])) | set(changes.unset_env)) - env.keys())


def materialize_agentx(
    config: dict[str, Any],
    output: Path,
    changes: LaunchChanges,
    *,
    model_path: str | None = None,
    inferencex_path: str | None = None,
) -> Path:
    """Keep the workload YAML intact while changing only the server candidate."""
    config = copy.deepcopy(config)
    bench = config["benchmark"]
    envs = bench.setdefault("envs", {})
    if model_path and Path(model_path).is_dir():
        envs["MODEL_PATH"] = model_path
    if inferencex_path:
        bench["inferencex_path"] = inferencex_path
    apply_launch_changes(bench, changes)
    envs["RUN_EVAL"] = "false"
    bench.pop("server_lifecycle", None)
    bench.pop("benchmark_script", None)
    bench.pop("workload_spec", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)
    return output


def bind_source_evidence(config_path: Path, patches: list[dict[str, Any]]) -> None:
    """Hash applied patch files immediately before Magpie launches the server."""
    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    if not is_magpie_agentx(bench):
        return
    apply_launch_changes(bench, LaunchChanges())
    launch = bench["agentx"]["launch_overrides"]
    files = launch.setdefault("source_files", {})
    absent = set(launch.get("absent_source_files", []))
    for patch in patches:
        root = Path(patch["framework_root"]).resolve()
        text = Path(patch["patch_file"]).read_text(encoding=_ENCODING)
        for relative in touched_paths(text):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Patch path escapes framework root: {relative}")
            name = str(path)
            if path.is_file():
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                absent.discard(name)
            else:
                files.pop(name, None)
                absent.add(name)
    launch["absent_source_files"] = sorted(absent)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)


def enable_profile(config_path: Path) -> None:
    """Derive the diagnostic run without modifying the accepted YAML."""
    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    profile = config["benchmark"].setdefault("profiler", {}).setdefault("torch_profiler", {})
    profile["enabled"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)


def runtime_executable(runtime: Mapping[str, Any]) -> str:
    """Resolve Python for Magpie's SGLang and vLLM module entry points."""
    executable = str(runtime.get("runtime_python_exe") or runtime.get("framework_python") or "")
    root = str(runtime.get("framework_bin") or runtime.get("path_prefix") or "")
    if not root and runtime.get("framework_venv_root"):
        root = str(Path(runtime["framework_venv_root"]) / "bin")
    if root and not executable:
        directory = Path(root).parent if Path(root).is_file() else Path(root)
        executable = str(directory / "python")
    if executable and (not Path(executable).is_absolute() or not os.access(executable, os.X_OK)):
        raise ValueError(f"AgentX server executable must be an absolute executable file: {executable}")
    return executable


def apply_native_runtime(config_path: Path, runtime: Mapping[str, Any], overlay: str = "") -> None:
    """Activate the attempt runtime in Magpie's managed server, never the replay client."""
    from hyperloom.inference_optimizer.runtime_environment import apply_runtime_override

    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    if not is_magpie_agentx(bench):
        return
    apply_launch_changes(bench, LaunchChanges())
    launch = bench["agentx"]["launch_overrides"]
    environment = dict(launch["env"])
    apply_runtime_override(environment, dict(runtime))
    launch["env"] = {key: value for key, value in environment.items() if not key.startswith("HYPERLOOM_FRAMEWORK_")}
    if overlay:
        root = Path(overlay)
        if not root.is_absolute() or not root.is_dir() or ":" in overlay:
            raise ValueError("AgentX Python overlay must be one existing absolute directory")
        for path in root.rglob("*.py"):
            launch.setdefault("source_files", {})[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        inherited = launch["env"].get("PYTHONPATH", "")
        launch["env"]["PYTHONPATH"] = overlay + (":" + inherited if inherited else "")
    executable = runtime_executable(runtime)
    if executable:
        launch["executable"] = executable
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)


def activate_runtime(config_path: Path, runtime: Mapping[str, str]) -> None:
    """Activate a candidate in the boundary owned by its benchmark backend."""
    if not runtime:
        return
    if native_config(config_path):
        apply_native_runtime(config_path, runtime)
        return
    from hyperloom.inference_optimizer.runtime_environment import apply_runtime_override

    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    apply_runtime_override(config["benchmark"].setdefault("envs", {}), dict(runtime))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)


def materialize_variant(
    config: dict[str, Any],
    variant: Any,
    output: Path,
    *,
    model_path: str | None,
    base_args: str,
    base_env: Mapping[str, Any] | None,
    base_remove: list[str] | None,
    base_unset: list[str] | None,
    base_args_mode: str,
) -> Path:
    """Compose the accepted launch and candidate, then bind its applied source."""
    apply_launch_changes(
        config["benchmark"],
        LaunchChanges(
            base_args,
            base_env or {},
            base_remove or [],
            base_unset or [],
            base_args_mode == REPLACE,
        ),
    )
    path = materialize_agentx(
        config,
        output / "config.yaml",
        LaunchChanges(
            variant.extra_server_args,
            variant.extra_envs,
            variant.remove_args,
            variant.unset_envs,
            variant.args_mode == REPLACE,
        ),
        model_path=model_path,
    )
    bind_source_evidence(path, variant.source_patches)
    apply_native_runtime(path, variant.runtime_override, variant.overlay_pythonpath)
    return path
