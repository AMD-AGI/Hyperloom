# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens trace analysis: argv assembly and the trace_analyze handler."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._kernel_agent_tool import (
    HandlerResult,
    _run_subprocess,
    _shape_tool_result,
)
from hyperloom_common.coerce import first_text
from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.inference_optimizer.session import paths
from hyperloom.inference_optimizer.trace.trajectory_trace import inherited_scope_fields
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute


log = logging.getLogger(__name__)

#: The payload key that marks one PD engine role's trace: no LLM phase hints, same-stream overlaps clamped.
PD_ROLE_TRACE_KEY = "pd_role_trace"

#: Payload key: the physical cards the caller holds for the arch microbenchmark (``--arch-gpus``); an empty list lets it
#: run on none. Absent, the analysis picks among the visible GPUs.
ARCH_GPUS_PARAM = "arch_gpus"


def _tracelens_root_error(root: Path) -> str | None:
    """Validate that the resolved TraceLens root is a usable git checkout.

    A directory that exists but lacks ``.git`` is not usable and must be reported
    so a non-default override fails fast and a default path is self-healed.

    Returns:
        str | None: A human-readable error when the checkout is missing or
            incomplete, or ``None`` when it is a usable git checkout.
    """
    if not root.is_dir():
        return (
            f"TraceLens root not found: {root}; run "
            "src/hyperloom/inference_optimizer/assets/install.sh "
            "or set TRACELENS_ROOT to an existing checkout"
        )
    if not (root / ".git").exists():
        return (
            f"TraceLens root incomplete (not a git checkout): {root}; "
            "run src/hyperloom/inference_optimizer/assets/install.sh "
            "or set TRACELENS_ROOT to a valid checkout"
        )
    return None


def _maybe_selfheal_tracelens_root(root: Path, *, log: Any = None) -> None:
    """Rebuild the pod-local TraceLens checkout if it vanished mid-run.

    Only the installer-managed default path is healed; an explicit
    ``TRACELENS_ROOT`` override must fail fast when missing. A heal that cannot
    load or fails is logged and the caller's validation reports the error.
    """
    # The installer-managed checkout is <deps_cache_root>/TraceLens or the
    # per-revision <deps_cache_root>/TraceLens@<sha>; both are healable. An
    # explicit override elsewhere must fail fast (never auto-clone).
    try:
        cache_root = paths.deps_cache_root().resolve()
        root_resolved = Path(root).resolve()
    except OSError:
        return
    is_default = root_resolved.parent == cache_root and (
        root_resolved.name == "TraceLens" or root_resolved.name.startswith("TraceLens@")
    )
    if not is_default:
        return  # explicit non-default override: never auto-clone
    try:
        from hyperloom.orchestrator.trace_analysis import tracelens_analysis as _tla

        heal_log = getattr(log, "warning", None) or (lambda *_a, **_k: None)
        heal_log("trace_analyze: TraceLens root %s missing; attempting self-heal", root)
        _tla._ensure_tracelens_checkout(root, log_path=Path(os.devnull))
    except (ImportError, OSError, subprocess.TimeoutExpired) as exc:
        # tracelens_analysis imports TraceLens at top level, so a vanished checkout fails the import.
        _log = getattr(log, "warning", None)
        if _log:
            _log("trace_analyze: TraceLens self-heal failed: %s", exc)


def _coerce_runtime_value(value: Any) -> Any:
    """Best-effort coercion of a string runtime value to ``int`` or ``float``.

    Integer-looking strings become ``int``; strings containing ``.`` that
    parse as a float become ``float``. Anything else (including unparseable
    strings and non-string inputs) is returned unchanged.

    Args:
        value (Any): The raw value to coerce.

    Returns:
        Any: The coerced numeric value, or the original value when no safe
            numeric coercion applies.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        try:
            return float(stripped) if "." in stripped else value
        except ValueError:
            return value
    return value


def _materialized_workload(config_path: str) -> dict[str, Any]:
    """The split hints (``conc``, ``osl``, ``random_range_ratio``) of a materialized point YAML, coerced to numbers.

    Args:
        config_path (str): Path to the materialized workload YAML config.

    Returns:
        dict[str, Any]: The hints the config sets, or an empty dict when the path is missing or unreadable.
    """
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    import yaml

    try:
        envs = read_benchmark_point(path).envs
    except (OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("failed to read materialized workload config %s: %s", path, exc)
        return {}
    return {
        out_key: _coerce_runtime_value(envs[src_key])
        for out_key, src_key in (("conc", "CONC"), ("osl", "OSL"), ("random_range_ratio", "RANDOM_RANGE_RATIO"))
        if src_key in envs
    }


_TRACELENS_ANALYSIS_MODULE = "hyperloom.orchestrator.trace_analysis.tracelens_analysis"
_ANALYSIS_TOOL = "tracelens"


def _optional_flags(*pairs: tuple[str, Any]) -> list[str]:
    """The ``flag value`` argv tokens of every pair whose value is set, in order."""
    tokens: list[str] = []
    for flag, value in pairs:
        if value:
            tokens += [flag, str(value)]
    return tokens


@dataclass(frozen=True)
class _AnalysisRun:
    """What the handler resolved for one analysis run: the trace, workspace, tool and the workload it describes."""

    trace_input: Any
    workspace_path: str
    tracelens_root: Path
    route: AnalysisRoute
    scriptable: bool
    model_name: str
    framework: str
    target_platform: str


def _split_llm_flags(payload: dict, workload: dict) -> list[str]:
    flags = ["--split-llm-inference"]
    for flag, payload_key, workload_key in (
        ("--split-conc", "split_conc", "conc"),
        ("--split-osl", "split_osl", "osl"),
        ("--split-r", "split_r", "random_range_ratio"),
    ):
        hint = payload.get(payload_key) or workload.get(workload_key)
        if hint not in (None, ""):
            flags += [flag, str(hint).strip()]
    return flags


def _build_trace_analyze_cmd(
    payload: dict,
    run: _AnalysisRun,
    *,
    session_dir: Path,
    workload: dict,
) -> "tuple[list[str], str]":
    """Assemble the trace-analysis tool argv; returns
    ``(cmd, steady_state_mode)`` so the caller can record discovery provenance."""
    scriptable = run.scriptable
    cmd = [
        sys.executable,
        "-m",
        _TRACELENS_ANALYSIS_MODULE,
        "--trace-input",
        str(run.trace_input),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--workspace-path",
        run.workspace_path,
        "--route",
        run.route.value,
        # Pass the resolved root explicitly so the tool never relies on inherited env.
        "--tracelens-root",
        str(run.tracelens_root),
        "--target-platform",
        run.target_platform,
    ]
    # The tool's SDK run books its model requests onto this session's trajectory ledger.
    for key, value in inherited_scope_fields().items():
        cmd += [f"--trajectory-{key.replace('_', '-')}", str(value)]

    cmd += _optional_flags(
        ("--model-name", run.model_name),
        ("--framework", run.framework),
    )
    arch_gpus = payload.get(ARCH_GPUS_PARAM)
    if arch_gpus is not None:
        cmd += ["--arch-gpus", ",".join(str(int(gpu)) for gpu in arch_gpus)]

    capture_folder = (
        payload.get("capture_folder") or payload.get("graph_capture_path") or payload.get("capture_folder_path")
    )
    pd_role_trace = bool(payload.get(PD_ROLE_TRACE_KEY))
    if scriptable:
        steady_state_mode = "generic"
    else:
        steady_state_mode = str(
            payload.get("steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
        ).strip()
        if not pd_role_trace:
            cmd += _split_llm_flags(payload, workload)
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    cmd += _optional_flags(
        ("--capture-folder", capture_folder),
        ("--steady-state-mode", steady_state_mode),
        ("--roofline-output-name", roofline_output_name),
    )
    if pd_role_trace:
        cmd += ["--clamp-stream-overlaps"]
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    return cmd, steady_state_mode


def _analysis_steady_state(result: dict[str, Any], *, requested_mode: str) -> dict[str, Any]:
    """Describe the steady-state window the run analysed, from the tool's split selection.

    Args:
        result: The analysis tool's result dict.
        requested_mode: The steady-state mode asked of the tool.

    Returns:
        A dict naming the requested mode, the raw selection and whether the tool
        fell back to the full trace.
    """
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    selection = run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {}
    return {
        "requested_mode": str(requested_mode or ""),
        "selected": selection,
        "fell_back_to_full_trace": bool(selection.get("fell_back_to_full_trace")),
    }


def _build_analysis_meta(
    result: dict[str, Any],
    *,
    route: AnalysisRoute,
    requested_mode: str,
    trace_input: str,
    duration_sec: float,
) -> dict[str, Any]:
    """Assemble the per-run analysis metadata the roofline timeline event carries.

    ``route`` records the routing policy (``agent`` / ``non-llm``) and ``tool`` the
    implementation that ran, which is TraceLens on both. Tool-specific analysis
    output lands under ``route_ext`` rather than widening the shared envelope.

    Args:
        result: The analysis tool's result dict.
        route: The analysis route the run took.
        requested_mode: The steady-state mode asked of the tool.
        trace_input: The trace the run analyzed.
        duration_sec: Wall-clock seconds the subprocess took.

    Returns:
        The analysis metadata dict.
    """
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    steps = run_meta.get("steps")
    return {
        "route": route.value,
        "tool": _ANALYSIS_TOOL,
        "steady_state_mode": str(requested_mode or ""),
        "trace_input": str(trace_input or ""),
        "duration_sec": duration_sec,
        "steady_state": _analysis_steady_state(result, requested_mode=requested_mode),
        "preflight": run_meta.get("preflight") if isinstance(run_meta.get("preflight"), dict) else {},
        "split": run_meta.get("split") if isinstance(run_meta.get("split"), dict) else {},
        "selection": run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {},
        "steps": [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else [],
        "route_ext": run_meta.get("route_ext") if isinstance(run_meta.get("route_ext"), dict) else {},
    }


def _record_tool_version(session_dir: Path) -> None:
    # This run is the only place the TraceLens build that produced the session's hot
    # kernels is in scope. Nothing downstream can recover it, so it is recorded here even
    # though the rest of the discovery run is already on the roofline event.
    from hyperloom.inference_optimizer.breakdown.recorder import recorder_warnings, tool_versions
    from hyperloom.inference_optimizer.breakdown.recorder.trace import trace_skip

    try:
        tool_versions.record_tool_version(session_dir, tool=_ANALYSIS_TOOL)
    except recorder_warnings.RECORDING_ERRORS as exc:
        trace_skip(reason="caller raised before the recorder", section="versions", entity=_ANALYSIS_TOOL, error=exc)


def _framework_for_analysis(payload: dict, state: Any) -> tuple[str, list[dict[str, Any]]]:
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    state_framework = first_text(state.framework)
    payload_framework = first_text(payload.get("framework"))
    # Payload metadata remains authoritative for ordinary serving frameworks.
    # The exception is a scriptable session receiving a stale non-scriptable
    # default (commonly ``sglang``): that would make xDiT follow the LLM trace
    # splitter, which discards its raw diffusion GPU kernels.
    if not (payload_framework and is_scriptable(state_framework) and not is_scriptable(payload_framework)):
        return payload_framework or state_framework, []
    log.warning(
        "trace_analyze: overriding payload framework %r with session scriptable framework %r to preserve the raw trace",
        payload_framework,
        state_framework,
    )
    return state_framework, [
        {
            "code": "stale_framework_overridden",
            "severity": "warning",
            "message": (
                f"overrode non-scriptable payload framework {payload_framework!r} "
                f"with scriptable session framework {state_framework!r} "
                "to preserve the raw trace"
            ),
            "payload_framework": payload_framework,
            "session_framework": state_framework,
        }
    ]


def _requested_route(payload: dict) -> str:
    # Only an absent or blank payload value defers to the env var. A non-blank
    # value is kept even when unrecognized, so it reaches the caller's check rather
    # than silently overriding the env with the ``agent`` default.
    raw_route = payload.get("analysis_route")
    route_text = "" if raw_route is None else str(raw_route).strip()
    return (route_text or os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "").strip()).lower()


def _surface_artifact_paths(result: dict[str, Any]) -> None:
    artifacts = result.get("artifact_paths")
    if not isinstance(artifacts, dict):
        artifacts = {}
    if artifacts.get("kernel_candidates"):
        result["candidates_path"] = artifacts["kernel_candidates"]
    report_path = str(result.get("trace_report_path") or artifacts.get("trace_report_path") or "")
    if report_path:
        result["trace_report_path"] = report_path
    # Surface the reusable-vs-skipped audit sidecar.
    if artifacts.get("tracelens_summary"):
        result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
    if artifacts.get("kernel_roofline"):
        result["kernel_roofline_path"] = str(artifacts["kernel_roofline"])


def _record_analysis_failure(result: dict[str, Any]) -> None:
    failure_warning: dict[str, Any] = {
        "code": "tracelens_analysis_failed",
        "severity": "warning",
        "message": (
            "TraceLens analysis failed; refusing to treat this as a "
            "successful empty-kernel result. See ``stderr_tail`` / "
            "``error`` for the upstream failure."
        ),
    }
    for key in ("returncode", "rc", "error", "stderr_tail", "raw_stdout_tail"):
        if key in result and result[key] not in (None, ""):
            failure_warning[key] = result[key]
    health = list(result.get("trace_health_warnings") or [])
    health.append(failure_warning)
    result["trace_health_warnings"] = health
    result["hot_kernels"] = []
    result.setdefault("orchestrator_error", failure_warning.get("error", ""))


def _invalid_route_result(requested_route: str) -> dict[str, Any]:
    """Log and return the failure for an explicit analysis route that names no known route."""
    valid_routes = [route.value for route in AnalysisRoute]
    message = (
        f"unknown analysis_route {requested_route!r} (expected one of {valid_routes}); "
        "refusing to fall back to 'agent' because that may start an LLM session. "
        "Use 'non-llm' for no-LLM trace analysis."
    )
    log.error("trace_analyze: %s", message)
    return {
        "status": "failed",
        "error_class": "invalid_analysis_route",
        "error": message,
        "requested_route": requested_route,
        "valid_routes": valid_routes,
    }


async def _run_analysis_tool(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    """Run the analysis tool and shape its result; a timeout becomes a failed result."""
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired:
        cmd_repr = " ".join(str(c) for c in cmd)
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
        }
    return result


def _settle_tool_result(result: dict[str, Any], framework_warnings: list[dict[str, Any]]) -> None:
    """Surface the result's artifacts, record a failed run, and prepend the handler's own warnings."""
    _surface_artifact_paths(result)

    # A failed TraceLens run is a hard failure, not "empty candidates".
    if result.get("status") == "failed" and "trace_split_no_steady_state" not in str(result.get("error")):
        _record_analysis_failure(result)

    # Prepend handler validation warnings so they reach the LLM.
    result["trace_health_warnings"] = framework_warnings + list(result.get("trace_health_warnings") or [])


async def trace_analyze_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run Hyperloom/kernel-agent's tracelens_analysis.py on a trace dir.

    The explicit payload framework normally takes precedence over the persisted
    session value.  A scriptable session overrides a conflicting non-scriptable
    payload framework so a diffusion trace is not sent through the LLM
    prefill/decode splitter.

    Args:
        payload (dict): Request payload (see ``Required payload`` /
            ``Optional payload`` below for the recognized keys).
        session_dir (Path): Session root used for resolving inputs and writing
            the analysis outputs.

    Required payload:
        trace_input: path to a torch_trace dir or single .trace.json.gz file.

    Returns the tool's result dict with ``status``, surfaced artifact paths, and
    ``trace_health_warnings``; on failure, ``returncode`` / ``error`` and empty ``hot_kernels``.
    """
    trace_input = payload.get("trace_input") or payload.get("trace_dir")
    if not trace_input:
        return {"status": "failed", "error": "missing 'trace_input' in payload"}
    # Whatever the caller's payload leaves out of the workload context comes from SharedState.
    from ...state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    framework, framework_warnings = _framework_for_analysis(payload, state)
    target_platform = first_text(payload.get("target_platform"), state.gpu_type)
    if not target_platform:
        return {
            "status": "failed",
            "error_class": "gpu_label_missing",
            "error": "no GPU label: the payload names no target_platform and the session recorded no gpu_type",
        }
    model_name = first_text(payload.get("model_name"), state.model_name, state.model_path)

    # Analysis route: default ``agent`` (TraceLens SDK/LLM); ``non-llm`` runs TraceLens alone,
    # selected via payload ``analysis_route`` / ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``.
    requested_route = _requested_route(payload)
    # An explicit unknown route is a configuration error. Falling back to
    # ``agent`` could turn a no-LLM request into a paid model session.
    try:
        analysis_route = AnalysisRoute(requested_route) if requested_route else AnalysisRoute.AGENT
    except ValueError:
        return _invalid_route_result(requested_route)
    tracelens_root = paths.tracelens_root()
    # Self-heal when the checkout is missing or incomplete (no .git).
    if not (tracelens_root / ".git").exists():
        _maybe_selfheal_tracelens_root(tracelens_root, log=log)
    tl_err = _tracelens_root_error(tracelens_root)
    if tl_err:
        return {"status": "failed", "error_class": "tracelens_root_missing", "error": tl_err}

    # Pass the session root so artefacts settle under ``<session_dir>/kernel-agent/runs/...``.
    workspace_path = payload.get("workspace_path") or str(session_dir)
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    # Scriptable frameworks (xDiT) have no decode steady-state window, so feed the
    # raw trace and drop the --split-* hints.
    scriptable = is_scriptable(framework)

    workload = _materialized_workload(state.baseline_config_path)

    cmd, steady_state_mode = _build_trace_analyze_cmd(
        payload,
        _AnalysisRun(
            trace_input=trace_input,
            workspace_path=workspace_path,
            tracelens_root=tracelens_root,
            route=analysis_route,
            scriptable=scriptable,
            model_name=model_name,
            framework=framework,
            target_platform=target_platform,
        ),
        session_dir=session_dir,
        workload=workload,
    )
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
    result = await _run_analysis_tool(cmd, timeout_sec)
    _disc_duration_sec = round(time.monotonic() - _disc_started, 3)
    _settle_tool_result(result, framework_warnings)

    # Surfaced for the caller's SBD V6 roofline event, which records the run
    # as it happens rather than re-deriving it at export time.
    result["analysis_meta"] = _build_analysis_meta(
        result,
        route=analysis_route,
        requested_mode=steady_state_mode,
        trace_input=str(trace_input),
        duration_sec=_disc_duration_sec,
    )
    _record_tool_version(session_dir)
    return result
