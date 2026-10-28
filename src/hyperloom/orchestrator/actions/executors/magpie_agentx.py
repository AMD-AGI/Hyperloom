# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate Hyperloom candidates into Magpie's public AgentX launch contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Mapping

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.unified_diff import touched_paths
from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID


_ENCODING = "utf-8"
#: vLLM's per-step timeout, at the value the InferenceX vLLM configs set (``assets/configs/baseline_vllm.yaml``).
VLLM_STEP_TIMEOUT_ENV = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
VLLM_STEP_TIMEOUT_SEC = 1200


class ServerArgvInvalidError(ValueError):
    """Composed server args that cannot become an argv without corrupting a value."""

    error_class = ARGV_INVALID


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


def recipe_option_names(config_path: Path) -> set[str]:
    """Canonical options from this config's verified Magpie launch, when available."""
    receipt_path = config_path.parent / "launch_evidence.json"
    if not receipt_path.is_file():
        return set()
    receipt = json.loads(receipt_path.read_text(encoding=_ENCODING))
    digest = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    if receipt.get("recipe_digest") != digest:
        return set()
    return {
        token.partition("=")[0]
        for token in receipt.get("magpie_server_launch", {}).get("base_argv", [])
        if token.startswith("--")
    }


def _inherited_server_args(argv: list[str]) -> str:
    """Join the accepted argv with single spaces, which :func:`_server_argv` splits back into the same tokens.

    Every token :func:`_server_argv` returns is one argv token on its own (non-empty, no whitespace, not
    quote-wrapped), which is what makes repeated composition of the same launch idempotent.

    Raises:
        ServerArgvInvalidError: When a token, such as one written by hand in the workload YAML, is not one argv token.
    """
    from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

    for token in argv:
        split = tokenize_server_args_preserving_json(token)
        if split is None or len(split[1]) != 1:
            raise ServerArgvInvalidError(
                f"{ARGV_INVALID}: launch_overrides.append_args token {token!r} is not one argv token: it is empty, "
                "contains whitespace, is quote-wrapped, or has an unbalanced quote or bracket"
            )
    return " ".join(argv)


def _server_argv(server_args: str, requested: str) -> list[str]:
    """Split server args the way Hyperloom writes them: a JSON value is one token that keeps its quotes.

    Raises:
        ServerArgvInvalidError: When a token boundary is ambiguous or a JSON-looking value does not parse; the
            message quotes ``requested``, the caller's own args, since ``server_args`` is already normalized.
    """
    from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

    split = tokenize_server_args_preserving_json(server_args)
    if split is None:
        raise ServerArgvInvalidError(
            f"{ARGV_INVALID}: server args do not split into argv tokens (a value is empty, contains whitespace, "
            f"or has an unbalanced quote or bracket): {requested}"
        )
    for token in split[1]:
        if token.endswith("\\"):
            # A shell would join ``my\ model`` into one value; argv splitting keeps two, so refuse it here
            # rather than let the framework reject a stray positional at launch.
            raise ServerArgvInvalidError(
                f"{ARGV_INVALID}: a server arg value ends with a backslash escape, which is not shell-parsed: "
                f"{requested}"
            )
        value = token.partition("=")[2] if token.startswith("-") else token
        if value.startswith(("{", "[")):
            try:
                json.loads(value)
            except ValueError as exc:
                raise ServerArgvInvalidError(f"{ARGV_INVALID}: server arg value is not valid JSON: {value}") from exc
    return split[1]


def apply_launch_changes(
    benchmark: dict[str, Any], changes: LaunchChanges, *, recipe_options: AbstractSet[str] = frozenset()
) -> None:
    """Compose a candidate; Magpie validates protected args and environment.

    Raises:
        ServerArgvInvalidError: When the composed args cannot become an argv intact.
    """
    from hyperloom.inference_optimizer.grid_server_args import compose_server_args, dedupe_extra_server_args

    agentx = benchmark["agentx"]
    if not isinstance(agentx, dict):
        agentx = {"enabled": True}
        benchmark["agentx"] = agentx
    launch = agentx.setdefault("launch_overrides", {"version": 1})
    appended_options = {token.partition("=")[0] for token in launch.get("append_args", []) if token.startswith("--")}
    launch["append_args"] = _server_argv(
        dedupe_extra_server_args(
            compose_server_args(
                inherited_args=_inherited_server_args(launch.get("append_args", [])),
                variant_extra_args=changes.args,
                remove_args=changes.remove_args,
                args_mode=REPLACE if changes.replace_args else "append",
            )
        ),
        changes.args,
    )
    launch["replace_args"] = changes.replace_args or launch.get("replace_args", False)
    # Magpie matches removals against the recipe, before appending our options. A local-only ablation must not
    # become a second removal there; options present in both layers must still be removed from both.
    recipe_removals = set(changes.remove_args) - (appended_options - recipe_options)
    launch["remove_args"] = sorted(set(launch.get("remove_args", [])) | recipe_removals)
    env = launch.setdefault("env", {})
    for name in changes.unset_env:
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in changes.env.items()})
    launch["unset_env"] = sorted((set(launch.get("unset_env", [])) | set(changes.unset_env)) - env.keys())


def materialize_agentx(
    config: dict[str, Any],
    output: Path,
    *,
    model_path: str | None = None,
    recipe_options: AbstractSet[str] = frozenset(),
) -> Path:
    """Keep the workload YAML intact while changing only the server candidate."""
    config = copy.deepcopy(config)
    bench = config["benchmark"]
    _default_step_timeouts(bench, recipe_options)
    envs = bench.setdefault("envs", {})
    if model_path and Path(model_path).is_dir():
        envs["MODEL_PATH"] = model_path
    envs["RUN_EVAL"] = "false"
    bench.pop("server_lifecycle", None)
    bench.pop("benchmark_script", None)
    bench.pop("workload_spec", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)
    return output


def _default_step_timeouts(bench: dict[str, Any], recipe_options: AbstractSet[str]) -> None:
    """Give the launch the per-step timeout the InferenceX launch gets, unless the recipe or the candidate sets one: a
    first prefill that builds an aiter JIT module outlasts sglang's 300 s scheduler watchdog and vLLM's own step
    timeout, either of which kills a healthy server. A candidate that unsets vLLM's variable keeps it unset."""
    from hyperloom.inference_optimizer.grid_server_args import SGLANG_WATCHDOG_FLAG, inject_sglang_watchdog_timeout

    launch = bench["agentx"].setdefault("launch_overrides", {"version": 1})
    if str(bench.get("framework") or "").lower() == "vllm":
        if VLLM_STEP_TIMEOUT_ENV not in launch.get("unset_env", []):
            launch.setdefault("env", {}).setdefault(VLLM_STEP_TIMEOUT_ENV, str(VLLM_STEP_TIMEOUT_SEC))
        return
    if SGLANG_WATCHDOG_FLAG in recipe_options:
        return
    guarded = inject_sglang_watchdog_timeout(
        _inherited_server_args(launch.get("append_args", [])), bench.get("framework")
    )
    launch["append_args"] = _server_argv(guarded, guarded)


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
    """Prepare TraceLens instrumentation and fill any missing AgentX capture settings.

    An explicit ``num_steps``, ``num_profiles``, ``start_seconds``,
    ``interval_seconds``, or ``detailed_annotations`` already present in the
    profile copy is left unchanged. A missing ``detailed_annotations`` follows the patch
    outcome, as the fixed-sequence request body does: Magpie probes the server whenever the
    field is true and raises when the server lacks it, so claiming it on a runtime the patch
    did not reach ends the round before the server starts.
    """
    from ._profile_setup import (
        annotations_supported,
        apply_sglang_shape_tool,
        patch_for_profile,
        shape_discovery_enabled,
    )

    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    framework = str(bench["framework"]).lower()
    _, sitecustomize = patch_for_profile(bench.setdefault("envs", {}), framework)
    apply_launch_changes(bench, LaunchChanges())
    launch = bench["agentx"]["launch_overrides"]
    server_env = launch["env"]
    profile_env: dict[str, str] = {}
    if sitecustomize:
        if "PYTHONPATH" in server_env:
            profile_env["PYTHONPATH"] = str(server_env["PYTHONPATH"])
        if "PYTHONPATH" in launch["unset_env"]:
            profile_env["PYTHONPATH"] = ""
        apply_sglang_shape_tool(profile_env, shape_discovery=shape_discovery_enabled(bench))
    if framework == "vllm":
        default = str(VLLM_STEP_TIMEOUT_SEC)
        inherited = (
            default if VLLM_STEP_TIMEOUT_ENV in launch["unset_env"] else os.environ.get(VLLM_STEP_TIMEOUT_ENV, default)
        )
        profile_env[VLLM_STEP_TIMEOUT_ENV] = str(
            max(VLLM_STEP_TIMEOUT_SEC, int(server_env.get(VLLM_STEP_TIMEOUT_ENV, inherited)))
        )
    apply_launch_changes(bench, LaunchChanges(env=profile_env))
    torch_profiler = bench.setdefault("profiler", {}).setdefault("torch_profiler", {})
    torch_profiler["enabled"] = True
    torch_profiler.setdefault("num_steps", 256)
    torch_profiler.setdefault("num_profiles", 1)
    torch_profiler.setdefault("start_seconds", 300)
    torch_profiler.setdefault("interval_seconds", 900)
    torch_profiler.setdefault("detailed_annotations", annotations_supported(bench["envs"]))
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
    config_path: Path,
    variant: Any,
    output: Path,
    *,
    model_path: str | None,
    base: LaunchChanges,
) -> Path:
    """Compose the accepted launch and candidate, then bind its applied source."""
    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    recipe_options = recipe_option_names(config_path)
    apply_launch_changes(config["benchmark"], base, recipe_options=recipe_options)
    apply_launch_changes(
        config["benchmark"],
        LaunchChanges(
            variant.extra_server_args,
            variant.extra_envs,
            variant.remove_args,
            variant.unset_envs,
            variant.args_mode == REPLACE,
        ),
        recipe_options=recipe_options,
    )
    path = materialize_agentx(config, output / "config.yaml", model_path=model_path, recipe_options=recipe_options)
    bind_source_evidence(path, variant.source_patches)
    apply_native_runtime(path, variant.runtime_override, variant.overlay_pythonpath)
    return path
