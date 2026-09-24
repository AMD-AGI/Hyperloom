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
from pathlib import Path
from typing import Any

from ._kernel_agent_tool import (
    HandlerResult,
    _kernel_agent_root_error,
    _kernel_agent_tool_path,
    _run_subprocess,
    _shape_tool_result,
)
from .._recorder_trace import trace_recording_skipped

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


def _resolve_tracelens_root() -> Path:
    """Resolve the TraceLens checkout, independent of inherited env.

    Falls back to the install-script-derived pod-local path so trace analysis
    works even when the coordinator process did not source kernel-agent.env.sh.

    Returns:
        Path: The resolved TraceLens root (may not exist yet; callers validate).
    """
    from hyperloom.inference_optimizer.session import paths

    return paths.tracelens_root()


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
            "src/hyperloom/agents/kernel/scripts/install.sh "
            "or set TRACELENS_ROOT to an existing checkout"
        )
    if not (root / ".git").exists():
        return (
            f"TraceLens root incomplete (not a git checkout): {root}; "
            "run src/hyperloom/agents/kernel/scripts/install.sh "
            "or set TRACELENS_ROOT to a valid checkout"
        )
    return None


def _maybe_selfheal_tracelens_root(root: Path, *, log: Any = None) -> None:
    """Rebuild the pod-local TraceLens checkout if it vanished mid-run.

    Only the installer-managed default path is healed; an explicit
    ``TRACELENS_ROOT`` override must fail fast when missing. Best-effort: any
    failure is swallowed so the caller's validation produces the error.
    """
    from hyperloom.inference_optimizer.session import paths

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
        tool = _kernel_agent_tool_path("tracelens_analysis.py")
        tools_dir = str(tool.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import tracelens_analysis as _tla  # type: ignore[import-not-found]

        heal_log = getattr(log, "warning", None) or (lambda *_a, **_k: None)
        heal_log("trace_analyze: TraceLens root %s missing; attempting self-heal", root)
        _tla._ensure_tracelens_checkout(root, log_path=Path(os.devnull))
    except Exception as exc:  # noqa: BLE001  # heal is best-effort; validation reports the real error
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
    try:
        import yaml  # type: ignore[import-untyped]

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read materialized workload config %s: %s", path, exc)
        return {}
    bench = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    framework = str(bench.get("framework") or "").strip().lower()
    # Per-framework env-name source of truth (e.g. atom reads ``EXTRA_ATOM_ARGS``).
    from ._grid_runner import server_args_env_name

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
    except Exception as exc:  # noqa: BLE001
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


def _build_trace_analyze_cmd(
    payload: dict,
    *,
    session_dir: Path,
    state: Any,
    workspace_path: str,
    trace_input: Any,
    tracelens_root: "Path | None",
    is_bypass: bool,
    scriptable: bool,
    workload: dict,
    model_name: str,
    framework: str,
    target_platform: str,
    analysis_mode: str,
) -> "tuple[list[str], str]":
    """Assemble the trace-analysis tool argv (TraceLens or bypass); returns
    ``(cmd, steady_state_mode)`` so the caller can record discovery provenance."""
    # Both tools share the CLI surface below except ``--tracelens-root``.
    tool_name = "bypass_trace_analysis.py" if is_bypass else "tracelens_analysis.py"
    cmd = [
        "python3" if is_bypass else sys.executable,
        str(_kernel_agent_tool_path(tool_name)),
        "--trace-input",
        str(trace_input),
        "--session-id",
        str(payload.get("session_id") or session_dir.name),
        "--workspace-path",
        workspace_path,
    ]
    if not is_bypass:
        # Pass the resolved root explicitly so the tool never relies on inherited env.
        cmd += ["--tracelens-root", str(tracelens_root)]
    elif str(getattr(state, "benchmark_mode", "") or "").strip().lower() == "agentx":
        cmd += ["--require-single-rank"]
        try:
            state_tp = int(getattr(state, "tp", 0) or 0)
        except (TypeError, ValueError):
            state_tp = 0
        if state_tp > 0:
            cmd += ["--tensor-parallel-size", str(state_tp)]
    if model_name:
        cmd += ["--model-name", str(model_name)]
    if framework:
        cmd += ["--framework", str(framework)]
    if target_platform:
        cmd += ["--target-platform", str(target_platform)]
    if analysis_mode:
        cmd += ["--analysis-mode", str(analysis_mode)]

    # Model identity informs source resolution for every framework, not only the
    # diffusion roofline. Keep the standard payload > state > environment
    # precedence so ordinary sglang/vLLM production requests carry config.json
    # selectors into the bounded model context.
    model_path = str(
        payload.get("model_path") or getattr(state, "model_path", "") or os.environ.get("MODEL_PATH") or ""
    ).strip()
    if model_path:
        cmd += ["--model-path", model_path]
    precision = str(
        payload.get("precision") or getattr(state, "precision", "") or workload.get("precision") or ""
    ).strip()
    if precision:
        cmd += ["--precision", precision]
    runtime_config = str(payload.get("runtime_config") or getattr(state, "baseline_config_path", "") or "").strip()
    if runtime_config and not is_bypass:
        cmd += ["--runtime-config", runtime_config]

    if scriptable:
        # --skip-split is TraceLens-only; the bypass backend has its own windowing.
        if not is_bypass:
            cmd += ["--skip-split"]
        # Forward the denoise-step count for per-step roofline timings.
        # Priority: payload override > baseline workload metadata.
        num_denoise = payload.get("num_denoise_steps") or workload.get("num_inference_steps")
        if num_denoise not in (None, ""):
            try:
                if int(num_denoise) > 0:
                    cmd += ["--num-denoise-steps", str(int(num_denoise))]
            except (TypeError, ValueError):
                pass
    else:
        # Splitter workload hints. Priority: payload override > baseline metadata
        # > drop the flag.
        split_conc = payload.get("split_conc") or workload.get("conc")
        if split_conc not in (None, ""):
            cmd += ["--split-conc", str(split_conc).strip()]
        split_osl = payload.get("split_osl") or workload.get("osl")
        if split_osl not in (None, ""):
            cmd += ["--split-osl", str(split_osl).strip()]
        split_r = payload.get("split_r") or workload.get("random_range_ratio")
        if split_r not in (None, ""):
            cmd += ["--split-r", str(split_r).strip()]

    capture_folder = (
        payload.get("capture_folder") or payload.get("graph_capture_path") or payload.get("capture_folder_path")
    )
    if capture_folder:
        cmd += ["--capture-folder", str(capture_folder)]
    # Forward TraceLens splitter steady-state mode via payload or env.
    steady_state_mode = payload.get("steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "")
    steady_state_mode = str(steady_state_mode).strip()
    if steady_state_mode:
        cmd += ["--steady-state-mode", steady_state_mode]
    # Post-kernel-opt roofline writes a separate report so it never overwrites
    # the baseline kernel_roofline.json.
    roofline_output_name = str(payload.get("roofline_output_name") or "").strip()
    if roofline_output_name:
        cmd += ["--roofline-output-name", roofline_output_name]
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


def _build_analysis_meta(
    result: dict[str, Any],
    *,
    route: str,
    tool: str,
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
        route: The requested analysis route (``agent`` / ``bypass``).
        tool: The tool that actually ran (``tracelens`` / ``bypass``).
        requested_mode: The steady-state mode asked of the tool.
        trace_input: The trace the run analyzed.
        duration_sec: Wall-clock seconds the subprocess took.

    Returns:
        The analysis metadata dict.
    """
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
    root_err = _kernel_agent_root_error()
    if root_err:
        return {"status": "failed", "error_class": "kernel_agent_root_missing", "error": root_err}
    # Backfill workload context from SharedState when Orchestration omits it.
    from ...state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state_framework = str(state.framework or "").strip()
    payload_framework = str(payload.get("framework") or "").strip()
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    # Payload metadata remains authoritative for ordinary serving frameworks.
    # The exception is a scriptable session receiving a stale non-scriptable
    # default (commonly ``sglang``): that would make xDiT follow the LLM trace
    # splitter, which discards its raw diffusion GPU kernels.
    framework = payload_framework or state_framework
    framework_warnings: list[dict[str, Any]] = []
    if payload_framework and is_scriptable(state_framework) and not is_scriptable(payload_framework):
        framework = state_framework
        framework_warnings.append(
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
        )
        log.warning(
            "trace_analyze: overriding payload framework %r with session "
            "scriptable framework %r to preserve the raw trace",
            payload_framework,
            state_framework,
        )
    target_platform = (payload.get("target_platform") or state.gpu_type or "").strip()
    model_name = (payload.get("model_name") or state.model_name or state.model_path or "").strip()
    analysis_mode = (payload.get("analysis_mode") or "").strip()
    if not analysis_mode and framework.lower() in {"vllm", "sglang"}:
        analysis_mode = "inference"

    # Analysis route: default ``agent`` (TraceLens); ``bypass`` (TraceLens-free)
    # is the explicit route via payload ``analysis_route`` /
    # ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``. Coerce to str.
    # Only an absent or blank payload value defers to the env var. A non-blank
    # value is kept even when unrecognized, so it reaches the check below rather
    # than silently overriding the env with the ``agent`` default.
    raw_route = payload.get("analysis_route")
    route_text = "" if raw_route is None else str(raw_route).strip()
    if not route_text:
        route_text = os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "").strip()
    explicit_route = route_text.lower()
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
    # Resolve TraceLens root independently of inherited env, self-healing a
    # vanished checkout before validation. Skipped on bypass.
    tracelens_root: Path | None = None
    if not is_bypass:
        tracelens_root = _resolve_tracelens_root()
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
    workload = metadata.get("runtime_args", {}).get("workload", {}) if isinstance(metadata, dict) else {}

    cmd, steady_state_mode = _build_trace_analyze_cmd(
        payload,
        session_dir=session_dir,
        state=state,
        workspace_path=workspace_path,
        trace_input=trace_input,
        tracelens_root=tracelens_root,
        is_bypass=is_bypass,
        scriptable=scriptable,
        workload=workload,
        model_name=model_name,
        framework=framework,
        target_platform=target_platform,
        analysis_mode=analysis_mode,
    )
    timeout_sec = int(payload.get("budget_minutes", 60)) * 60

    _disc_started = time.monotonic()
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
        result = _shape_tool_result(rc, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        result = {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}",
        }
    _disc_duration_sec = round(time.monotonic() - _disc_started, 3)
    artifacts = result.get("artifact_paths") if isinstance(result, dict) else None
    if isinstance(artifacts, dict) and artifacts.get("kernel_candidates"):
        result["candidates_path"] = artifacts["kernel_candidates"]
    # Surface analysis.md path at the handler boundary for the Coordinator.
    if isinstance(result, dict):
        report_path = result.get("trace_report_path")
        if not report_path and isinstance(artifacts, dict):
            report_path = artifacts.get("trace_report_path")
        if report_path:
            result["trace_report_path"] = str(report_path)
            _enrich_candidate_trace_report(
                result.get("hot_kernels"),
                str(report_path),
            )
        # Surface the reusable-vs-skipped audit sidecar.
        if isinstance(artifacts, dict) and artifacts.get("tracelens_summary"):
            result["tracelens_summary_path"] = str(artifacts["tracelens_summary"])
        if isinstance(artifacts, dict) and artifacts.get("kernel_roofline"):
            result["kernel_roofline_path"] = str(artifacts["kernel_roofline"])

        # A failed TraceLens run is a hard failure, not "empty candidates".
        if result.get("status") == "failed" and "trace_split_no_steady_state" not in str(result.get("error") or ""):
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

        # Prepend handler validation warnings so they reach the LLM.
        result["trace_health_warnings"] = framework_warnings + list(result.get("trace_health_warnings") or [])

        _enrich_candidate_runtime_metadata(result.get("hot_kernels"), metadata)
        candidates_path = result.get("candidates_path")
        if isinstance(candidates_path, str):
            _enrich_candidates_artifact(
                candidates_path,
                metadata,
                trace_report_path=str(report_path or ""),
            )

        # Route and tool are one-to-one after the no-LLM TraceLens route was
        # removed: agent runs TraceLens, while bypass runs its standalone reader.
        _disc_route = analysis_route
        _disc_tool = "bypass" if is_bypass else "tracelens"
        # Surfaced for the caller's SBD V6 roofline event, which records the run
        # as it happens rather than re-deriving it at export time.
        result["analysis_meta"] = _build_analysis_meta(
            result,
            route=_disc_route,
            tool=_disc_tool,
            requested_mode=steady_state_mode,
            trace_input=str(trace_input),
            duration_sec=_disc_duration_sec,
        )
        # This run is the only place the build of the reader that produced the
        # session's hot kernels is in scope. Nothing downstream can recover it,
        # so it is recorded here even though the rest of the discovery run is
        # already on the roofline event.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import tool_versions

            tool_versions.record_tool_version(session_dir, tool=_disc_tool)
        except Exception as exc:  # noqa: BLE001
            trace_recording_skipped(
                "versions",
                reason="caller raised before the recorder",
                entity=_disc_tool,
                error=exc,
            )

    return result
