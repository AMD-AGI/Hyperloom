# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens profile setup shared by fixed-sequence and native AgentX runs.

The capture window, runtime patch, and kernel-shape tool are decided here once.
Fixed-sequence materialization writes that decision into the Magpie shell env.
Native AgentX publishes the same decision through Magpie's torch-profiler fields
and ``launch_overrides``, which is what the native server actually reads.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
from typing import Any, NamedTuple

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.env import env_bool, env_flag
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.model_config_utils import _load_model_config_dict, _model_is_gemma2
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from hyperloom.inference_optimizer.session.paths import asset_root

log = logging.getLogger(__name__)

# Shipped profile YAML keys the AgentX copy does not carry. Fixed-sequence runs
# already load that YAML, so seeding is a no-op there.
_SEEDED_PROFILE_ENVS = (
    "PROFILE",
    "EXTRA_VLLM_ARGS",
    "EXTRA_SGLANG_ARGS",
    "EXTRA_ATOM_ARGS",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
    "SGLANG_PROFILE_WITH_STACK",
    "SGLANG_PROFILE_RECORD_SHAPE",
    "SGLANG_GRAPH_BATCH_CAPTURE",
    "PROFILE_EXTRA_BODY",
)
# Server process env. ``benchmark.envs`` on a native AgentX run reaches the client,
# not ``python -m vllm`` / ``python -m sglang``.
_SERVER_ENV_KEYS = (
    "PYTHONPATH",
    "TRACELENS_SHAPE_DISCOVERY",
    "SGLANG_PROFILE_WITH_STACK",
    "SGLANG_PROFILE_RECORD_SHAPE",
    "SGLANG_GRAPH_BATCH_CAPTURE",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
    "ATOM_ENABLE_DETAILED_ANNOTATION",
    "ATOM_PROFILER_MORE",
)


class ProfileInstrumentation(NamedTuple):
    """What one profile setup wrote, so later steps can restore it."""

    num_prompts: int
    atom_caps: Any
    vllm_flags: list[tuple[str, str]]
    vllm_cap: int
    max_iters: int
    detailed: bool


def prepare_profile(bench: dict[str, Any], envs: dict[str, Any], *, write_workload: bool) -> ProfileInstrumentation:
    """Apply the fixed-sequence TraceLens profile setup to ``bench``.

    ``write_workload`` records the profile output length on the shell env. Native
    AgentX leaves the replay workload alone and publishes the capture through
    Magpie's profiler schema instead.
    """
    if is_magpie_agentx(bench):
        _seed_shipped_profile_env(bench, envs)
    delay_iters, max_iters, osl_val, num_prompts = _capture_plan(envs)
    if write_workload:
        envs["OSL"] = osl_val
    outcome = _apply_framework_profile(
        bench,
        envs,
        delay_iters=delay_iters,
        max_iters=max_iters,
        num_prompts=num_prompts,
    )
    if is_magpie_agentx(bench):
        _publish_agentx_profile(bench, envs, outcome)
    return outcome


def apply_atom_profile_caps(envs: dict[str, Any], caps: Any) -> None:
    """Write the ATOM TraceLens knobs the installed package accepts."""
    if caps is None:
        return
    if caps.detailed_annotation:
        envs["ATOM_ENABLE_DETAILED_ANNOTATION"] = "1"
    if caps.profiler_more:
        envs["ATOM_PROFILER_MORE"] = "1"
    if caps.mark_trace:
        extra = str(envs.get("EXTRA_ATOM_ARGS", "")).strip()
        if "--mark-trace" not in extra:
            envs["EXTRA_ATOM_ARGS"] = f"{extra} --mark-trace".strip()


def _capture_plan(envs: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return delay steps, captured steps, profile output length, and prompt count."""
    from hyperloom.orchestrator.actions.executors import _workload_envs as workload

    try:
        ratio = float(envs.get("RANDOM_RANGE_RATIO", 1.0))
    except (TypeError, ValueError):
        ratio = 1.0
    defaults = workload.cli_workload_defaults()
    osl_val = int(envs.get("OSL") or defaults[1])
    conc_val = int(envs.get("CONC") or defaults[2])
    safe_conc = max(conc_val, 1)
    cap = workload._profile_steps_cap(os.environ)
    profile_osl_raw = os.environ.get("PROFILE_OSL", "").strip()
    profile_osl_explicit = profile_osl_raw.isdigit() and int(profile_osl_raw) > 0
    if profile_osl_explicit:
        osl_val = int(profile_osl_raw)
    else:
        osl_val = min(osl_val, workload._PROFILE_DEFAULT_OSL)
    safe_osl = max(osl_val, 1)
    steady_floor = math.ceil(safe_osl * (1.0 + ratio) / (2.0 * safe_conc))
    if steady_floor > cap:
        osl_val, safe_osl, steady_floor = _fit_profile_osl(
            osl_val,
            safe_osl,
            steady_floor,
            cap=cap,
            ratio=ratio,
            safe_conc=safe_conc,
            profile_osl_explicit=profile_osl_explicit,
        )
    delay_iters, max_iters = workload._profile_capture_window(
        os.environ,
        cap=cap,
        delay_iters=max(0, int(osl_val * (ratio + 1) * 3 - cap / 2)),
        steady_floor=steady_floor,
    )
    required_iters = delay_iters + max_iters
    iters_to_prompts = max(1, (required_iters * safe_conc + safe_osl - 1) // safe_osl)
    return delay_iters, max_iters, osl_val, max(safe_conc, iters_to_prompts * 2)


def _fit_profile_osl(
    osl_val: int,
    safe_osl: int,
    steady_floor: int,
    *,
    cap: int,
    ratio: float,
    safe_conc: int,
    profile_osl_explicit: bool,
) -> tuple[int, int, int]:
    """Keep an explicit profile length, or shorten the automatic one until it fits."""
    if profile_osl_explicit:
        log.warning(
            "PROFILE_OSL=%d needs %d captured steps to reach steady "
            "state, above the profile cap of %d; the trace may lack a "
            "steady-state window (trace_split_no_steady_state). Lower "
            "--profile-osl or raise HYPERLOOM_PROFILE_MAX_STEPS_CAP.",
            osl_val,
            steady_floor,
            cap,
        )
        return osl_val, safe_osl, steady_floor
    fitted_osl = max(1, int(cap * 2 * safe_conc / (1.0 + ratio)))
    log.warning(
        "profile OSL %d would need %d captured steps to reach "
        "steady state (> cap %d); lowering profile OSL to %d so the "
        "capture stays serializable. Baseline/optimize unaffected.",
        osl_val,
        steady_floor,
        cap,
        fitted_osl,
    )
    osl_val = fitted_osl
    safe_osl = max(osl_val, 1)
    steady_floor = math.ceil(safe_osl * (1.0 + ratio) / (2.0 * safe_conc))
    return osl_val, safe_osl, steady_floor


def _apply_framework_profile(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    delay_iters: int,
    max_iters: int,
    num_prompts: int,
) -> ProfileInstrumentation:
    from hyperloom.orchestrator.actions.executors import _workload_envs as workload

    framework = str(bench.get("framework") or "").lower()
    patch_ok, sitecustomize = _patch_for_profile(envs, framework)
    if "atom" in framework:
        caps = workload._atom_tracelens_caps()
        conc = int(envs.get("CONC") or workload.cli_workload_defaults()[2])
        return ProfileInstrumentation(conc, caps, [], 0, max_iters, bool(caps.detailed_annotation))
    if "vllm" in framework:
        flags, detailed = _apply_vllm_profile(envs, delay_iters=delay_iters, max_iters=max_iters, patch_ok=patch_ok)
        return ProfileInstrumentation(num_prompts, None, flags, max_iters, max_iters, detailed)
    detailed = _apply_sglang_profile(
        bench,
        envs,
        delay_iters=delay_iters,
        max_iters=max_iters,
        sitecustomize=sitecustomize,
        patch_ok=patch_ok,
    )
    return ProfileInstrumentation(num_prompts, None, [], 0, max_iters, detailed)


def _patch_for_profile(envs: dict[str, Any], framework: str) -> tuple[bool, bool]:
    """Patch the installed server the way the fixed-sequence profile does."""
    from hyperloom.orchestrator.actions.executors import _workload_envs as workload
    from hyperloom.orchestrator.actions.executors._server_patcher import (
        ensure_sglang_patched_for_tracelens,
        ensure_vllm_patched_for_tracelens,
        resolve_sglang_shape_mode,
    )

    is_atom = "atom" in framework
    is_sglang = "sglang" in framework
    sitecustomize = is_sglang and resolve_sglang_shape_mode() == "sitecustomize"
    patch_attempted = workload._tracelens_patch_enabled() and not is_atom and not sitecustomize
    envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "not_attempted"
    if patch_attempted:
        if "vllm" in framework:
            patch_ok = ensure_vllm_patched_for_tracelens()
        else:
            patch_ok = ensure_sglang_patched_for_tracelens()
        envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "ok" if patch_ok else "unavailable"
        if not patch_ok:
            envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] = workload._TRACELENS_PATCH_UNAVAILABLE
            log.warning(
                "TraceLens runtime patch unavailable for framework=%s; "
                "profile will omit annotation-only flags and roofline "
                "analysis may be degraded.",
                framework or "<unset>",
            )
        return patch_ok, sitecustomize
    if sitecustomize and workload._tracelens_patch_enabled():
        # Shapes come from kernel_shape_tool, but the per-batch-size capture trace still needs
        # sglang_gc_patch or it IndexErrors. Only profiling captures traces, so patch only here.
        if not ensure_sglang_patched_for_tracelens():
            log.warning(
                "SGLang graph-capture patch (sglang_gc_patch) not applied; profiling "
                "continues on the unpatched server, where per-batch-size CUDA-graph "
                "capture can IndexError on multi-variant models (DSA dense+sparse). "
                "It needs TRACELENS_ROOT with an exact "
                "examples/custom_workflows/inference_analysis/sglang_gc_patch/sglang_<X_Y_Z> dir."
            )
    return False, sitecustomize


def _apply_vllm_profile(
    envs: dict[str, Any],
    *,
    delay_iters: int,
    max_iters: int,
    patch_ok: bool,
) -> tuple[list[tuple[str, str]], bool]:
    from hyperloom.orchestrator.actions.executors import _workload_envs as workload

    existing = str(envs.get("EXTRA_VLLM_ARGS", ""))
    flags = [
        ("delay_iterations", f"--profiler-config.delay_iterations {delay_iters}"),
        ("max_iterations", f"--profiler-config.max_iterations {max_iters}"),
    ]
    # Magpie's launcher puts ``profiler=torch`` before EXTRA_VLLM_ARGS. The argv
    # probe only sees EXTRA_VLLM_ARGS, and vLLM rejects the bounds without it.
    # A dir here would override Magpie's ``<workspace>/torch_trace`` by last-wins.
    if workload._profiler_flag_value(existing, "profiler") is None:
        flags.append(("profiler", "--profiler-config.profiler torch"))
    if patch_ok:
        flags.append(("capture_torch_profiler", "--profiler-config.capture_torch_profiler True"))
        flags.append(("detailed_trace_annotation", "--profiler-config.detailed_trace_annotation True"))
    flags.append(("ignore_frontend", "--profiler-config.ignore_frontend True"))
    flags.append(("torch_profiler_dump_cuda_time_total", "--profiler-config.torch_profiler_dump_cuda_time_total False"))
    # Repeated flags last-win, so the computed cap replaces a YAML pin.
    # ``ignore_frontend`` is already in the shipped YAML; vLLM warns on the duplicate.
    profiler_args = " ".join(
        flag for sentinel, flag in flags if sentinel != "ignore_frontend" or "ignore_frontend" not in existing
    )
    if "delay_iterations" not in existing:
        envs["EXTRA_VLLM_ARGS"] = f"{existing} {profiler_args}".strip()
    return flags, patch_ok


def _apply_sglang_profile(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    delay_iters: int,
    max_iters: int,
    sitecustomize: bool,
    patch_ok: bool,
) -> bool:
    from hyperloom.orchestrator.actions.executors import _workload_envs as workload

    extra_body = _profile_extra_body(envs)
    extra_body["start_step"] = delay_iters
    extra_body["num_steps"] = max_iters
    shape_discovery = _shape_discovery_enabled(bench)
    if sitecustomize:
        return _apply_sglang_shape_tool(envs, extra_body, shape_discovery=shape_discovery)
    degraded = envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") == workload._TRACELENS_PATCH_UNAVAILABLE
    if degraded:
        shape_discovery = False
    extra_body["shape_discovery"] = shape_discovery
    if degraded:
        extra_body["detailed_annotations"] = False
    else:
        extra_body.setdefault("detailed_annotations", True)
    envs["PROFILE_EXTRA_BODY"] = json.dumps(extra_body)
    if patch_ok and shape_discovery:
        existing = str(envs.get("EXTRA_SGLANG_ARGS", ""))
        if "shape-discovery-for-cuda-graph-profile" not in existing:
            envs["EXTRA_SGLANG_ARGS"] = f"{existing} --enable-shape-discovery-for-cuda-graph-profile".strip()
    return bool(extra_body.get("detailed_annotations"))


def _apply_sglang_shape_tool(envs: dict[str, Any], extra_body: dict[str, Any], *, shape_discovery: bool) -> bool:
    """Shapes come from kernel_shape_tool. Unpatched SGLang rejects the request flag and the graph arg."""
    from hyperloom.orchestrator.actions.executors._server_patcher import kernel_shape_tool_dir

    extra_body.pop("shape_discovery", None)
    extra_body.setdefault("detailed_annotations", True)
    tool_dir = kernel_shape_tool_dir()
    if shape_discovery and tool_dir is not None:
        prepend_launch_path(envs, "PYTHONPATH", str(tool_dir))
        envs["TRACELENS_SHAPE_DISCOVERY"] = "1"
    else:
        envs["TRACELENS_SHAPE_DISCOVERY"] = "0"
        if shape_discovery and tool_dir is None:
            log.warning(
                "SGLang shape mode=sitecustomize but kernel_shape_tool "
                "not found under TRACELENS_ROOT; shapes will be absent "
                "(set TRACELENS_ROOT to an NFS path visible to the server).",
            )
    envs["PROFILE_EXTRA_BODY"] = json.dumps(extra_body)
    return bool(extra_body.get("detailed_annotations"))


def _shape_discovery_enabled(bench: dict[str, Any]) -> bool:
    enabled = env_flag("HYPERLOOM_PROFILE_SHAPE_DISCOVERY", default=True)
    forced = env_bool("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE")
    model = str(bench.get("model") or "")
    if enabled and not forced and _model_is_gemma2(model):
        enabled = False
        log.info(
            "Gemma2 roofline: disabling shape-discovery to avoid "
            "CUDA-graph capture crash (hipErrorStreamCapture"
            "Unsupported); CUDA graph + profiling kept. Set "
            "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
        )
        if _load_model_config_dict(model) is None:
            log.warning(
                "Gemma2 detected via path heuristic (no readable config.json at %r); shape-discovery skip may be imprecise.",
                model,
            )
    return enabled


def _profile_extra_body(envs: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
    except (ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _seed_shipped_profile_env(bench: dict[str, Any], envs: dict[str, Any]) -> None:
    framework = str(bench.get("framework") or "").strip().lower()
    if framework not in framework_registry.names():
        return
    path = asset_root() / "assets" / "configs" / framework_registry.shipped_config_name("profile", framework)
    shipped = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    shipped_envs = (shipped.get("benchmark") or {}).get("envs") or {}
    for key in _SEEDED_PROFILE_ENVS:
        if key in shipped_envs and key not in envs:
            envs[key] = shipped_envs[key]


def _publish_agentx_profile(bench: dict[str, Any], envs: dict[str, Any], outcome: ProfileInstrumentation) -> None:
    """Project the shared decision onto the fields Magpie's native server reads."""
    torch_profiler = bench.setdefault("profiler", {}).setdefault("torch_profiler", {})
    torch_profiler["enabled"] = True
    torch_profiler["num_steps"] = outcome.max_iters
    torch_profiler["num_profiles"] = 1
    torch_profiler["start_seconds"] = 0
    torch_profiler["detailed_annotations"] = outcome.detailed
    apply_atom_profile_caps(envs, outcome.atom_caps)
    agentx = bench.get("agentx")
    if not isinstance(agentx, dict):
        agentx = {"enabled": True}
        bench["agentx"] = agentx
    launch = agentx.get("launch_overrides")
    if not isinstance(launch, dict):
        launch = {"version": 1}
        agentx["launch_overrides"] = launch
    launch.setdefault("version", 1)
    server_env = launch.setdefault("env", {})
    for key in _SERVER_ENV_KEYS:
        if key in envs:
            server_env[key] = str(envs[key])
    for env_name in ("EXTRA_SGLANG_ARGS", "EXTRA_ATOM_ARGS"):
        _append_server_flags(launch, str(envs.get(env_name) or ""))


def _append_server_flags(launch: dict[str, Any], text: str) -> None:
    """Add shell server flags to the native argv. ``--profiler-config`` stays Magpie's."""
    current = [str(token) for token in launch.get("append_args") or []]
    skip_value = False
    for token in shlex.split(text):
        if token.startswith("--profiler-config"):
            skip_value = True
            continue
        if skip_value and not token.startswith("-"):
            skip_value = False
            continue
        skip_value = False
        if token not in current:
            current.append(token)
    if current:
        launch["append_args"] = current
