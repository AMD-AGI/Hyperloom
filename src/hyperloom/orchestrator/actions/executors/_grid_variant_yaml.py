# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize one grid variant's Magpie YAML from the run's base recipe."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import BLOCKED_EXTERNAL_ENV_NAMES
from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    moe_runner_requires_aiter,
    server_args_env_name,
    _shell_safe_dedupe as _shell_safe_dedupe,
    _SGLANG_MOE_RUNNER_BACKEND_RE as _SGLANG_MOE_RUNNER_BACKEND_RE,
)
from hyperloom.inference_optimizer.runtime_environment import apply_runtime_override

from ._benchmark_runtime import apply_runtime_benchmark_overrides
from ._grid_base import GridVariant
from ._recipe_script import launcher_overwritten_envs
from ._server_argv import seal_server_argv
from ._spec_decode_pin import refuse_unpinned_speculative_change
from .magpie_agentx import LaunchChanges, materialize_variant

log = logging.getLogger(__name__)


def _apply_variant_env(envs, variant, base_extra_envs, base_unset_envs):
    """Apply accepted environment edits before the candidate's set/unset operations."""
    # Composed base-then-variant, so a variant unsetting a key the base sets
    # removes it: the last layer to name a key is the one that decides it.
    for k in to_str_list(base_unset_envs):
        if k.strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            continue
        envs.pop(k, None)
    for k, v in (base_extra_envs or {}).items():
        envs[str(k)] = str(v)
    base_envs = dict(envs)
    for k in getattr(variant, "unset_envs", []) or []:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(k).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("grid: refusing to unset pinned env %s for variant %s", k, variant.name)
            continue
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    return base_envs


def _runnable_moe_backend_args(combined: str, *, args_env: str, model_path: str | None, variant: GridVariant) -> str:
    """``combined`` with a MoE runner backend this checkpoint cannot run dropped.

    A grid variant never injects one itself, but it inherits one from the baseline recipe it
    was seeded with or from an explicitly authored variant, and a quant scheme implemented
    only on the aiter runner crashes on the first forward pass under any other.
    """
    if not combined or not _SGLANG_MOE_RUNNER_BACKEND_RE.search(combined):
        return combined
    if args_env != "EXTRA_SGLANG_ARGS" or not moe_runner_requires_aiter(combined, model_path):
        return combined
    from ._workload_envs import _remove_moe_runner_backend_arg

    log.warning(
        "grid: dropping inherited --moe-runner-backend for variant %s: "
        "this checkpoint's MoE quant scheme is only implemented on the "
        "aiter runner and would crash on the first forward pass.",
        variant.name,
    )
    return _remove_moe_runner_backend_arg(combined)


def _importable_overlay_dir(variant: GridVariant) -> str:
    """This variant's overlay directory, or ``""`` when it names none the server may import.

    A legitimate authored-kernel overlay is a single existing directory: never a
    ``:``-joined list, a ``..`` traversal, or a path carrying control characters.
    """
    overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if not overlay:
        return ""
    contained = ":" not in overlay and ".." not in Path(overlay).parts
    if contained and not any(c in overlay for c in ("\n", "\r", "\x00")) and Path(overlay).is_dir():
        return overlay
    log.warning(
        "grid: dropping unsafe overlay_pythonpath %r (not a single existing directory / contains separator or traversal)",
        overlay,
    )
    return ""


def _pin_xdit_tool_path(envs: dict[str, Any], framework: Any) -> None:
    """Force the xdit wrapper's own tool directories onto ``PATH``.

    It needs both ``/venv/bin`` (the ``xdit`` console script) and ``/opt/rocm/bin``
    (``hipcc``), and an LLM-supplied PATH must not be able to drop either.
    """
    if str(framework or "").strip().lower() != "xdit":
        return
    parts = [p for p in str(envs.get("PATH", "") or "").split(":") if p]
    for essential in ("/opt/rocm/bin", "/venv/bin"):
        if essential not in parts:
            parts.insert(0, essential)
    envs["PATH"] = ":".join(parts)


def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    server_lifecycle: dict[str, Any] | None = None,
    base_args_mode: str = "append",
    base_extra_envs: dict[str, str] | None = None,
    base_remove_args: list[str] | None = None,
    base_unset_envs: list[str] | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk."""
    cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    bench = cfg.setdefault("benchmark", {})
    if is_magpie_agentx(bench):
        return materialize_variant(
            base_yaml_path,
            variant,
            output_subdir,
            model_path=model_path,
            base=LaunchChanges(
                base_extra_args,
                base_extra_envs or {},
                base_remove_args or [],
                base_unset_envs or [],
                base_args_mode == "replace",
            ),
        )
    replacing = str(base_args_mode).strip().lower() == "replace"
    envs = apply_runtime_benchmark_overrides(
        bench,
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
        conc=variant_conc(variant),
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    # A replacing base drops the inherited string wholesale, so its own removals have nothing left to strip.
    inherited, base_remove = ("", []) if replacing else (str(envs.get(extra_args_env, "")), base_remove_args)
    base_args = compose_server_args(inherited_args=inherited, base_extra_args=base_extra_args, remove_args=base_remove)
    combined = compose_server_args(
        inherited_args=inherited,
        base_extra_args=base_extra_args,
        variant_extra_args=variant.extra_server_args,
        remove_args=to_str_list(base_remove) + to_str_list(getattr(variant, "remove_args", [])),
        args_mode=getattr(variant, "args_mode", "append"),
    )
    combined = _runnable_moe_backend_args(combined, args_env=extra_args_env, model_path=model_path, variant=variant)
    if combined:
        envs[extra_args_env] = _shell_safe_dedupe(combined)
    elif extra_args_env in envs or variant.args_mode == "replace" or base_args_mode == "replace":
        envs[extra_args_env] = ""
    base_envs = _apply_variant_env(envs, variant, base_extra_envs, base_unset_envs)
    from ._workload_envs import pin_mlperf_round_concurrency
    from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path

    pin_mlperf_round_concurrency(envs)
    # The launcher re-exports these unconditionally, so a value carried here is
    # one the run never used.
    for k in launcher_overwritten_envs(bench) & envs.keys():
        log.warning("grid: dropping %s for variant %s; the recipe overwrites it", k, variant.name)
        envs.pop(k, None)
    # The three AgentX bounds took this rung's CONC through ``variant_conc`` above, not through this merge: raising
    # the client's grace alone would make the round wait inside a cap that did not move with it.
    _overlay = _importable_overlay_dir(variant)
    if _overlay:
        prepend_launch_path(envs, "PYTHONPATH", _overlay)

    # Attempt runtime override: inject path_prefix/pythonpath_prefix/framework_bin etc. into benchmark.envs so the
    # server subprocess resolves the attempt runtime.
    _rt_override = getattr(variant, "runtime_override", None) or {}
    if _rt_override:
        apply_runtime_override(envs, _rt_override)
    refuse_unpinned_speculative_change(base_args, base_envs, combined, envs)

    _pin_xdit_tool_path(envs, bench.get("framework"))

    if server_lifecycle is not None:
        from ._server_lifecycle import inject_lifecycle

        inject_lifecycle(
            bench,
            cleanup=bool(server_lifecycle.get("cleanup", True)),
            pid_dir=server_lifecycle["pid_dir"],
            port=int(server_lifecycle["port"]),
        )

    # The final write to the argument env; nothing below may touch it.
    seal_server_argv(envs, bench.get("framework"))
    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def variant_conc(variant: Any) -> int | None:
    """The concurrency a variant will run at, or ``None`` when it names none."""
    raw = (getattr(variant, "extra_envs", None) or {}).get("CONC")
    try:
        conc = int(raw)
    except (TypeError, ValueError):
        return None
    return conc if conc > 0 else None
