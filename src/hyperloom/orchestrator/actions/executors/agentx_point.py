# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose Hyperloom candidates into an AgentX point's ``agentx.launch_overrides``, the launch its round serves."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.common.benchmark_config import BenchmarkPoint, is_agentx, read_benchmark_point
from hyperloom.common.outside_facts import AITER_LOG_TUNED_CONFIG_ENV
from hyperloom_common.unified_diff import touched_paths
from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID
from hyperloom.orchestrator.state.base_launch import ServerBase
from inference_suite.argv import removal_option_names, tokenize_server_args_preserving_json


_ENCODING = "utf-8"


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


def native_config(path: Path) -> bool:
    return read_benchmark_point(path).agentx


def _inherited_server_args(argv: list[str]) -> str:
    """Join the accepted argv with single spaces, which :func:`_server_argv` splits back into the same tokens.

    Every token :func:`_server_argv` returns is one argv token on its own (non-empty, no whitespace, not
    quote-wrapped), which is what makes repeated composition of the same launch idempotent.

    Raises:
        ServerArgvInvalidError: When a token, such as one written by hand in the workload YAML, is not one argv token.
    """
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


def _launch_server_args(launch: Mapping[str, Any]) -> str:
    """The server args this launch appends over the session's base."""
    return " ".join(launch.get("append_args", []))


def _launch_envs(benchmark: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, Any]:
    """The environment this launch runs with: a round layers the launch's own environment over the workload's."""
    unset = set(launch.get("unset_env", []))
    envs = {**(benchmark.get("envs") or {}), **(launch.get("env") or {})}
    return {name: value for name, value in envs.items() if name not in unset}


def apply_launch_changes(
    benchmark: dict[str, Any],
    changes: LaunchChanges,
    *,
    accepted: bool = False,
) -> None:
    """Compose a candidate onto the point's launch.

    This is where every AgentX launch is composed, so it is where a candidate meets the speculative-decoding pin the
    env-composed launches meet as they compose, before a round runs an unpinned launch.
    ``accepted`` marks a layer that is the session's own accepted launch rather than a candidate, which is what the
    pin is measured against, the way the base layer is on the env-composed path.

    Raises:
        ServerArgvInvalidError: When the composed args cannot become an argv intact.
        SpeculativeDecodingPinnedError: When a candidate changes the draft or the simulated acceptance its accepted
            launch pins.
    """
    from hyperloom.inference_optimizer.grid_server_args import compose_server_args, dedupe_extra_server_args

    from ._spec_decode_pin import refuse_unpinned_speculative_change

    agentx = benchmark["agentx"]
    if not isinstance(agentx, dict):
        agentx = {"enabled": True}
        benchmark["agentx"] = agentx
    launch = agentx.setdefault("launch_overrides", {"version": 1})
    accepted_args = _launch_server_args(launch)
    accepted_envs = _launch_envs(benchmark, launch)
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
    # A removal is matched against the base launch, before our options are appended, and is one bare long option
    # name per entry -- so the value a spec is scoped to is dropped here, at the boundary that asks for it, and
    # never upstream: compose_server_args above still needs the scope to keep an option the variant re-adds.
    # A local-only ablation must not become a second removal there; options present in both layers must still
    # be removed from both.
    removals = set(removal_option_names(changes.remove_args)) - appended_options
    launch["remove_args"] = sorted(set(launch.get("remove_args", [])) | removals)
    env = launch.setdefault("env", {})
    for name in changes.unset_env:
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in changes.env.items()})
    launch["unset_env"] = sorted((set(launch.get("unset_env", [])) | set(changes.unset_env)) - env.keys())
    if not accepted:
        refuse_unpinned_speculative_change(
            accepted_args, accepted_envs, _launch_server_args(launch), _launch_envs(benchmark, launch)
        )


def seed_session_base(benchmark: dict[str, Any], base: ServerBase) -> None:
    """Make the session's frozen base the launch a round serves, in place of the argv its own recipe resolves.

    Only a workload YAML no layer was composed on yet takes it; seeding a composed one would replace its layers.
    """
    if BenchmarkPoint.from_document({"benchmark": benchmark}).agentx_launch.composed or not (base.args or base.env):
        return
    apply_launch_changes(benchmark, LaunchChanges(base.args, dict(base.env), replace_args=True), accepted=True)


def _refuse_unrunnable_speculative_launch(benchmark: Mapping[str, Any], *, model_path: str | None) -> None:
    """Refuse a composed AgentX launch the pinned stack has no draft kernel for.

    The native path composes its launch here rather than in ``benchmark.envs``, so this is where it meets the same
    check the env-composed launches get.

    Raises:
        SpeculativeCapabilityUnsupportedError: When the launch drafts with a kernel the stack does not ship, or outside
            the draft bounds the recipe declares.
    """
    from hyperloom.inference_optimizer.speculative_capability import refuse_unrunnable_speculative_stack

    launch = (benchmark.get("agentx") or {}).get("launch_overrides") or {}
    refuse_unrunnable_speculative_stack(
        _inherited_server_args(list(launch.get("append_args", []))),
        benchmark,
        # launch_server.sh reads TP from the launch environment, which a round layers over the workload's.
        envs=_launch_envs(benchmark, launch),
        model_path=model_path or "",
    )


def materialize_agentx(
    config: dict[str, Any],
    output: Path,
    *,
    model_path: str | None = None,
) -> Path:
    """Keep the workload YAML intact while changing only the server candidate."""
    config = copy.deepcopy(config)
    bench = config["benchmark"]
    _log_tuned_config_hits(bench)
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


def _log_tuned_config_hits(bench: dict[str, Any]) -> None:
    """Have the server log aiter's tuned-config hits unless the candidate unsets it; apply verification reads them."""
    agentx = bench.get("agentx")
    if not isinstance(agentx, dict):
        return
    launch = agentx.setdefault("launch_overrides", {"version": 1})
    if AITER_LOG_TUNED_CONFIG_ENV not in launch.get("unset_env", []):
        launch.setdefault("env", {}).setdefault(AITER_LOG_TUNED_CONFIG_ENV, "1")


def bind_source_evidence(config_path: Path, patches: list[dict[str, Any]]) -> None:
    """Hash applied patch files immediately before a round launches the server."""
    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    if not is_agentx(bench):
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
    outcome, as the fixed-sequence request body does: claiming it on a runtime the patch did not
    reach fails the round, whose profile start asks the server for annotations it cannot give.
    """
    from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path

    from ._profile_setup import (
        apply_sglang_shape_tool,
        patch_for_profile,
        shape_discovery_enabled,
    )

    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    framework = str(bench["framework"]).lower()
    patch = patch_for_profile(bench.setdefault("envs", {}), framework)
    apply_launch_changes(bench, LaunchChanges())
    launch = bench["agentx"]["launch_overrides"]
    server_env = launch["env"]
    profile_env: dict[str, str] = {}
    if patch.sitecustomize or patch.site is not None:
        if "PYTHONPATH" in server_env:
            profile_env["PYTHONPATH"] = str(server_env["PYTHONPATH"])
        if "PYTHONPATH" in launch["unset_env"]:
            profile_env["PYTHONPATH"] = ""
    if patch.site is not None:
        prepend_launch_path(profile_env, "PYTHONPATH", str(patch.site))
    if patch.sitecustomize:
        apply_sglang_shape_tool(profile_env, shape_discovery=shape_discovery_enabled(bench))
    if framework == "vllm":
        name = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        inherited = "1200" if name in launch["unset_env"] else os.environ.get(name, "1200")
        profile_env[name] = str(max(1200, int(server_env.get(name, inherited))))
    apply_launch_changes(bench, LaunchChanges(env=profile_env))
    torch_profiler = bench.setdefault("profiler", {}).setdefault("torch_profiler", {})
    torch_profiler["enabled"] = True
    torch_profiler.setdefault("num_steps", 256)
    torch_profiler.setdefault("num_profiles", 1)
    torch_profiler.setdefault("start_seconds", 300)
    torch_profiler.setdefault("interval_seconds", 900)
    torch_profiler.setdefault("detailed_annotations", patch.annotations)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_ENCODING)


def runtime_executable(runtime: Mapping[str, Any]) -> str:
    """Resolve the Python that runs SGLang's and vLLM's module entry points."""
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
    """Activate the attempt runtime in the round's server, never the replay client."""
    from hyperloom.inference_optimizer.runtime_environment import apply_runtime_override

    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    bench = config["benchmark"]
    if not is_agentx(bench):
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
    """Compose the accepted launch and candidate, then bind its applied source.

    Raises:
        SpeculativeCapabilityUnsupportedError: When the composed launch drafts with a kernel the pinned stack
            does not ship.
        SpeculativeDecodingPinnedError: When the candidate changes the draft or the simulated acceptance the
            recipe and the accepted launch pin.
    """
    config = yaml.safe_load(config_path.read_text(encoding=_ENCODING))
    apply_launch_changes(config["benchmark"], base, accepted=True)
    apply_launch_changes(
        config["benchmark"],
        LaunchChanges(
            variant.extra_server_args,
            variant.extra_envs,
            variant.remove_args,
            variant.unset_envs,
            variant.args_mode == REPLACE,
        ),
    )
    _refuse_unrunnable_speculative_launch(config["benchmark"], model_path=model_path)
    path = materialize_agentx(config, output / "config.yaml", model_path=model_path)
    bind_source_evidence(path, variant.source_patches)
    apply_native_runtime(path, variant.runtime_override, variant.overlay_pythonpath)
    return path


def materialize_grid_variant(
    config_path: Path, variant: Any, output: Path, *, model_path: str | None, base: Any
) -> Path:
    """Materialize a grid variant whose accepted launch is the grid recipe's base layer."""
    return materialize_variant(
        config_path,
        variant,
        output,
        model_path=model_path,
        base=LaunchChanges(
            base.extra_server_args,
            base.extra_envs or {},
            base.remove_args or [],
            base.unset_envs or [],
            base.args_mode == "replace",
        ),
    )
