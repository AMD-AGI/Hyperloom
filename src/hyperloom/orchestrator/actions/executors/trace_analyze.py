# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens/bypass trace analysis: candidate env enrichment, argv assembly, and the trace_analyze handler."""

from __future__ import annotations

import json
import logging
import os
import shlex
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
from hyperloom.common.coerce import first_text, optional_positive_int
from hyperloom.inference_optimizer.session import paths
from hyperloom.inference_optimizer.trace.trajectory_trace import inherited_scope_fields

log = logging.getLogger(__name__)

# Recognized trace-analysis routes. Only an omitted value defaults to ``agent``;
# an explicit unknown value fails before dispatch so it cannot start an LLM.
_VALID_ANALYSIS_ROUTES = frozenset({"bypass", "agent"})


_CANDIDATE_ENV_KEYS = {
    "CONC",
    "ISL",
    "OSL",
    "TP",
    "NUM_PROMPTS",
    "NUM_WARMUPS",
    "MAX_MODEL_LEN",
    "RANDOM_RANGE_RATIO",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
}


_CANDIDATE_ENV_PREFIXES = (
    "SGLANG_",
    "VLLM_",
    "AITER_",
    "TRITON_",
    "FLYDSL_",
    "HIPBLASLT_",
    "PYTORCH_TUNABLEOP_",
)


_SENSITIVE_ENV_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


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


def _candidate_env_allowed(key: str) -> bool:
    """Decide whether an env var may be forwarded as candidate metadata.

    Rejects anything that looks sensitive (keys, tokens, secrets, passwords,
    credentials); otherwise allows the key if it is in the explicit allowlist
    or starts with a known safe prefix (e.g. ``SGLANG_``, ``VLLM_``).

    Args:
        key (str): Environment variable name to test.

    Returns:
        bool: ``True`` if the env var is safe to surface, ``False`` otherwise.
    """
    upper = key.upper()
    if any(part in upper for part in _SENSITIVE_ENV_PARTS):
        return False
    return key in _CANDIDATE_ENV_KEYS or any(key.startswith(prefix) for prefix in _CANDIDATE_ENV_PREFIXES)


def _split_server_args(raw: str) -> list[str]:
    """Tokenize a raw server-args string into an argv list.

    Args:
        raw (str): Raw shell-style server argument string.

    Returns:
        list[str]: The parsed argv tokens, or an empty list when ``raw`` is
            falsy or cannot be parsed (a warning is logged on parse failure).
    """
    try:
        return shlex.split(raw) if raw else []
    except ValueError:
        log.warning("failed to parse materialized server args; preserving raw string")
        return []


def _load_materialized_workload_metadata(config_path: str) -> dict[str, Any]:
    """Extract runtime workload context from a materialized Magpie YAML config.

    Reads the config's ``benchmark`` block and derives the per-framework
    server-args env name, the allowed candidate env vars, and a normalized
    ``runtime_args`` view (framework, model, precision, server args, and the
    coerced workload knobs such as ``tp`` / ``conc`` / ``isl`` / ``osl``).

    Args:
        config_path (str): Path to the materialized workload YAML config.

    Returns:
        dict[str, Any]: A dict with ``env_vars`` and ``runtime_args`` keys, or
            an empty dict when the path is missing/unreadable. Empty/``None``
            ``runtime_args`` entries are dropped.
    """
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    import yaml  # type: ignore[import-untyped]

    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("failed to read materialized workload config %s: %s", path, exc)
        return {}
    bench = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    framework = str(bench.get("framework") or "").strip().lower()
    # Per-framework env-name source of truth (e.g. atom reads ``EXTRA_ATOM_ARGS``).
    from hyperloom.inference_optimizer.framework_registry import server_args_env_name

    server_key = server_args_env_name(framework)
    server_args = str(envs.get(server_key) or "").strip()
    workload = {
        out_key: _coerce_runtime_value(envs[src_key])
        for out_key, src_key in (
            ("tp", "TP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("num_prompts", "NUM_PROMPTS"),
            ("num_warmups", "NUM_WARMUPS"),
            ("max_model_len", "MAX_MODEL_LEN"),
            ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        )
        if src_key in envs
    }
    runtime_args = {
        "materialized_config": str(path),
        "framework": framework or None,
        "model": bench.get("model"),
        "precision": bench.get("precision"),
        "server_args": server_args,
        "server_args_argv": _split_server_args(server_args),
        "workload": workload,
    }
    return {
        "env_vars": {str(key): str(value) for key, value in envs.items() if _candidate_env_allowed(str(key))},
        "runtime_args": {key: value for key, value in runtime_args.items() if value not in (None, "", {})},
    }


def _enrich_candidate_runtime_metadata(
    candidates: Any,
    metadata: dict[str, Any],
) -> None:
    """Backfill runtime env/args metadata onto each candidate kernel in place.

    For every dict candidate, sets default ``env_vars`` and ``runtime_args``
    entries from ``metadata`` without overwriting values the candidate already
    carries (uses ``setdefault`` semantics).

    Args:
        candidates (Any): Expected to be a list of candidate dicts; ignored if
            not a list.
        metadata (dict[str, Any]): Metadata with ``env_vars`` / ``runtime_args``
            sub-dicts as produced by
            :func:`_load_materialized_workload_metadata`.

    Returns:
        None: The ``candidates`` list is mutated in place.
    """
    if not isinstance(candidates, list) or not metadata:
        return
    env_vars = metadata.get("env_vars") if isinstance(metadata.get("env_vars"), dict) else {}
    runtime_args = metadata.get("runtime_args") if isinstance(metadata.get("runtime_args"), dict) else {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_env = item.setdefault("env_vars", {})
        if isinstance(item_env, dict):
            for key, value in env_vars.items():
                item_env.setdefault(key, value)
        item_args = item.setdefault("runtime_args", {})
        if isinstance(item_args, dict):
            for key, value in runtime_args.items():
                item_args.setdefault(key, value)


def _enrich_candidate_trace_report(candidates: Any, report_path: str) -> None:
    """Stamp the TraceLens report path onto each candidate kernel in place.

    Args:
        candidates (Any): Expected to be a list of candidate dicts; ignored if
            not a list.
        report_path (str): Path to the TraceLens ``analysis.md`` report; ignored
            if empty.

    Returns:
        None: Each dict candidate gains a default ``trace_report_path`` entry.
    """
    if not isinstance(candidates, list) or not report_path:
        return
    for item in candidates:
        if isinstance(item, dict):
            item.setdefault("trace_report_path", report_path)


def _enrich_candidates_artifact(
    candidates_path: str,
    metadata: dict[str, Any],
    *,
    trace_report_path: str = "",
) -> None:
    """Rewrite the on-disk candidates artifact with enriched metadata.

    Loads the ``candidates_path`` JSON, enriches its ``hot_kernels`` and
    ``hot_kernels_top15`` lists with runtime metadata and (optionally) the
    TraceLens report path, then writes the artifact back out (pretty-printed,
    key-sorted). No-op when the path is missing or unreadable.

    Args:
        candidates_path (str): Path to the candidates JSON artifact to update.
        metadata (dict[str, Any]): Runtime metadata to merge into each kernel.
        trace_report_path (str): Optional TraceLens report path to record at
            both the top level and on each kernel entry.

    Returns:
        None: The artifact file is rewritten in place when changes apply.
    """
    if not candidates_path:
        return
    path = Path(candidates_path)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("failed to read candidates artifact %s: %s", path, exc)
        return
    if not isinstance(data, dict):
        return
    if metadata:
        _enrich_candidate_runtime_metadata(data.get("hot_kernels"), metadata)
        _enrich_candidate_runtime_metadata(data.get("hot_kernels_top15"), metadata)
    if trace_report_path:
        data.setdefault("trace_report_path", trace_report_path)
        artifact_paths = data.setdefault("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            artifact_paths.setdefault("trace_report_path", trace_report_path)
        _enrich_candidate_trace_report(data.get("hot_kernels"), trace_report_path)
        _enrich_candidate_trace_report(
            data.get("hot_kernels_top15"),
            trace_report_path,
        )
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_TRACELENS_ANALYSIS_MODULE = "hyperloom.orchestrator.trace_analysis.tracelens_analysis"
_BYPASS_TRACE_ANALYSIS_MODULE = "hyperloom.orchestrator.trace_analysis.bypass_trace_analysis"


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
    tracelens_root: "Path | None"
    is_bypass: bool
    scriptable: bool
    model_name: str
    framework: str
    target_platform: str
    analysis_mode: str


def _bypass_flags(state: Any, analysis_mode: str) -> list[str]:
    """Flags only the bypass route reads; the analysis-mode hint retires with the bypass route."""
    flags = _optional_flags(("--analysis-mode", analysis_mode))
    if str(getattr(state, "benchmark_mode", "") or "").strip().lower() == "agentx":
        flags += ["--require-single-rank"]
        flags += _optional_flags(("--tensor-parallel-size", optional_positive_int(getattr(state, "tp", None))))
    return flags


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
    state: Any,
    workload: dict,
) -> "tuple[list[str], str]":
    """Assemble the trace-analysis tool argv (TraceLens or bypass); returns
    ``(cmd, steady_state_mode)`` so the caller can record discovery provenance."""
    is_bypass = run.is_bypass
    scriptable = run.scriptable
    # Both tools share the CLI surface below except ``--tracelens-root``.
    cmd = [
        sys.executable,
        "-m",
        _BYPASS_TRACE_ANALYSIS_MODULE if is_bypass else _TRACELENS_ANALYSIS_MODULE,
        "--trace-input",
        str(run.trace_input),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--workspace-path",
        run.workspace_path,
    ]
    if not is_bypass:
        # Pass the resolved root explicitly so the tool never relies on inherited env.
        cmd += ["--tracelens-root", str(run.tracelens_root)]
        # The tool's SDK run books its model requests onto this session's trajectory ledger.
        for key, value in inherited_scope_fields().items():
            cmd += [f"--trajectory-{key.replace('_', '-')}", str(value)]
    else:
        cmd += _bypass_flags(state, run.analysis_mode)

    model_path = first_text(payload.get("model_path"), getattr(state, "model_path", ""), os.environ.get("MODEL_PATH"))
    precision = first_text(payload.get("precision"), getattr(state, "precision", ""))
    cmd += _optional_flags(
        ("--model-name", run.model_name),
        ("--framework", run.framework),
        ("--target-platform", run.target_platform),
        ("--model-path", model_path),
        ("--precision", precision),
    )

    if scriptable:
        num_denoise = optional_positive_int(payload.get("num_denoise_steps"))
        cmd += _optional_flags(("--num-denoise-steps", num_denoise))
    elif not is_bypass:
        cmd += _split_llm_flags(payload, workload)

    capture_folder = (
        payload.get("capture_folder") or payload.get("graph_capture_path") or payload.get("capture_folder_path")
    )
    if scriptable and not is_bypass:
        steady_state_mode = "generic"
    else:
        steady_state_mode = str(
            payload.get("steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
        ).strip()
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    cmd += _optional_flags(
        ("--capture-folder", capture_folder),
        ("--steady-state-mode", steady_state_mode),
        ("--roofline-output-name", roofline_output_name),
    )
    if payload.get("dry_run"):
        cmd += ["--dry-run"]
    return cmd, steady_state_mode


# TraceLens picks its steady-state window by writing split chunks and selecting
# one file; the TraceLens-free reader picks a window in memory and never writes
# chunks. Both answer "is the window this analysis rests on trustworthy", so the
# event normalizes them onto one shape and keeps the raw form under ``selected``.
_STEADY_SOURCE_SPLIT_CHUNK = "split_chunk"
_STEADY_SOURCE_READER_WINDOW = "in_reader_window"


def _analysis_steady_state(
    result: dict[str, Any],
    *,
    requested_mode: str,
    tool: str,
) -> dict[str, Any]:
    """Normalize the steady-state window across analysis tools.

    Args:
        result: The analysis tool's result dict.
        requested_mode: The steady-state mode asked of the tool.
        tool: ``tracelens`` or ``bypass``.

    Returns:
        A dict naming the requested mode, how the window was picked, the raw
        selection, whether the tool fell back to the full trace, and the
        aggregation scope the shares are anchored to.
    """
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    scope = str(result.get("aggregation_scope") or run_meta.get("aggregation_scope") or "")
    if tool == "bypass":
        selected = result.get("steady_window") or {}
        fell_back = bool(result.get("estimated")) or (bool(scope) and scope != "steady_state")
        source = _STEADY_SOURCE_READER_WINDOW
    else:
        selection = run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {}
        selected = selection or {}
        fell_back = bool(selection.get("fell_back_to_full_trace"))
        source = _STEADY_SOURCE_SPLIT_CHUNK
    return {
        "requested_mode": str(requested_mode or ""),
        "source": source,
        "selected": selected if isinstance(selected, dict) else {"value": selected},
        "fell_back_to_full_trace": fell_back,
        "aggregation_scope": scope,
    }


def _analysis_tool(route: str) -> str:
    """The tool a route runs: agent runs TraceLens, bypass its standalone reader."""
    return "bypass" if route == "bypass" else "tracelens"


def _build_analysis_meta(
    result: dict[str, Any],
    *,
    route: str,
    requested_mode: str,
    trace_input: str,
    duration_sec: float,
) -> dict[str, Any]:
    """Assemble the per-run analysis metadata the roofline timeline event carries.

    The TraceLens agent and TraceLens-free reader share this envelope. ``route``
    records the routing policy (``agent`` / ``bypass``), while ``tool`` records
    the implementation that ran (``tracelens`` / ``bypass``). Tool-specific
    analysis output lands under ``route_ext`` rather than widening the shared
    envelope.

    Args:
        result: The analysis tool's result dict.
        route: The requested analysis route (``agent`` / ``bypass``); it names
            the tool that ran (``tracelens`` / ``bypass``).
        requested_mode: The steady-state mode asked of the tool.
        trace_input: The trace the run analyzed.
        duration_sec: Wall-clock seconds the subprocess took.

    Returns:
        The analysis metadata dict.
    """
    tool = _analysis_tool(route)
    run_meta = result.get("run_meta") if isinstance(result.get("run_meta"), dict) else {}
    steps = run_meta.get("steps")
    return {
        "route": str(route or ""),
        "tool": str(tool or ""),
        "steady_state_mode": str(requested_mode or ""),
        "trace_input": str(trace_input or ""),
        "duration_sec": duration_sec,
        "steady_state": _analysis_steady_state(result, requested_mode=requested_mode, tool=tool),
        "preflight": run_meta.get("preflight") if isinstance(run_meta.get("preflight"), dict) else {},
        "split": run_meta.get("split") if isinstance(run_meta.get("split"), dict) else {},
        "selection": run_meta.get("selection") if isinstance(run_meta.get("selection"), dict) else {},
        "steps": [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else [],
        "route_ext": run_meta.get("route_ext") if isinstance(run_meta.get("route_ext"), dict) else {},
    }


def _record_tool_version(session_dir: Path, tool: str) -> None:
    # This run is the only place the build of the reader that produced the session's hot
    # kernels is in scope. Nothing downstream can recover it, so it is recorded here even
    # though the rest of the discovery run is already on the roofline event.
    from hyperloom.inference_optimizer.breakdown.recorder import recorder_warnings, tool_versions
    from hyperloom.inference_optimizer.breakdown.recorder.trace import trace_skip

    try:
        tool_versions.record_tool_version(session_dir, tool=tool)
    except recorder_warnings.RECORDING_ERRORS as exc:
        trace_skip(reason="caller raised before the recorder", section="versions", entity=tool, error=exc)


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


def _surface_artifact_paths(result: dict[str, Any]) -> str:
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
    return report_path


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
    # Backfill workload context from SharedState when Orchestration omits it.
    from ...state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    framework, framework_warnings = _framework_for_analysis(payload, state)
    target_platform = first_text(payload.get("target_platform"), state.gpu_type)
    model_name = first_text(payload.get("model_name"), state.model_name, state.model_path)
    analysis_mode = first_text(payload.get("analysis_mode"))
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    # Analysis route: default ``agent`` (TraceLens SDK/LLM); ``bypass`` (no-LLM,
    # but still imports TraceLens for source mapping) is the explicit route via
    # payload ``analysis_route`` / ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``.
    explicit_route = _requested_route(payload)
    # An explicit unknown route is a configuration error. Falling back to
    # ``agent`` could turn a no-LLM request into a paid model session.
    if explicit_route and explicit_route not in _VALID_ANALYSIS_ROUTES:
        valid_routes = sorted(_VALID_ANALYSIS_ROUTES)
        message = (
            f"unknown analysis_route {explicit_route!r} (expected one of {valid_routes}); "
            "refusing to fall back to 'agent' because that may start an LLM session. "
            "Use 'bypass' for no-LLM trace analysis."
        )
        log.error("trace_analyze: %s", message)
        return {
            "status": "failed",
            "error_class": "invalid_analysis_route",
            "error": message,
            "requested_route": explicit_route,
            "valid_routes": valid_routes,
        }
    analysis_route = explicit_route or "agent"
    is_bypass = analysis_route == "bypass"
    # Both routes need an importable checkout: bypass transitively imports
    # TraceLens for source-path mapping.
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

    # Load materialized baseline workload metadata once.
    metadata = _load_materialized_workload_metadata(state.baseline_config_path)
    workload = metadata.get("runtime_args", {}).get("workload", {})

    cmd, steady_state_mode = _build_trace_analyze_cmd(
        payload,
        _AnalysisRun(
            trace_input=trace_input,
            workspace_path=workspace_path,
            tracelens_root=tracelens_root,
            is_bypass=is_bypass,
            scriptable=scriptable,
            model_name=model_name,
            framework=framework,
            target_platform=target_platform,
            analysis_mode=analysis_mode,
        ),
        session_dir=session_dir,
        state=state,
        workload=workload,
    )
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
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
    _disc_duration_sec = round(time.monotonic() - _disc_started, 3)
    report_path = _surface_artifact_paths(result)
    if report_path:
        _enrich_candidate_trace_report(result.get("hot_kernels"), report_path)

    # A failed TraceLens run is a hard failure, not "empty candidates".
    if result.get("status") == "failed" and "trace_split_no_steady_state" not in str(result.get("error")):
        _record_analysis_failure(result)

    # Prepend handler validation warnings so they reach the LLM.
    result["trace_health_warnings"] = framework_warnings + list(result.get("trace_health_warnings") or [])

    _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
    candidates_path = result.get("candidates_path")
    if isinstance(candidates_path, str):
        _enrich_candidates_artifact(candidates_path, metadata, trace_report_path=report_path)

    # Route and tool are one-to-one after the no-LLM TraceLens route was
    # removed: agent runs TraceLens, while bypass runs its standalone reader.
    _disc_tool = _analysis_tool(analysis_route)
    # Surfaced for the caller's SBD V6 roofline event, which records the run
    # as it happens rather than re-deriving it at export time.
    result["analysis_meta"] = _build_analysis_meta(
        result,
        route=analysis_route,
        requested_mode=steady_state_mode,
        trace_input=str(trace_input),
        duration_sec=_disc_duration_sec,
    )
    _record_tool_version(session_dir, _disc_tool)
    return result
