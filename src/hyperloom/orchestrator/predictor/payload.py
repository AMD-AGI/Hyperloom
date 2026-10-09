# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the predictor request from session state.

The body carries exactly the fields the PrimaTune service reads, in Hyperloom's
own spelling; the service owns the rename into its prompt, so a drift shows up
as one missing key on the side that renders it. A block the session cannot fill
completely is omitted rather than half-filled. See
``docs/reference/primatune-predictor.md`` for the wire contract.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Mapping

from hyperloom.orchestrator.phases.machine_state import phase_elapsed_seconds, resolve_keep_threshold
from hyperloom.orchestrator.predictor import evidence
from hyperloom.orchestrator.predictor.config import PHASE_LABEL

REQUEST_SCHEMA = "hyperloom.predictor_request.v1"

#: Hot kernels sent; the service keeps its own top five of these.
HOT_KERNEL_TOP_N = 8

#: Architecture facts the service reads off ``identification.model_info``.
_MODEL_INFO_KEYS = ("model_type", "attention_type", "num_hidden_layers", "num_experts", "hidden_size", "head_dim")

#: Fields copied as-is from a ``hot_kernels_top15`` row.
_KERNEL_KEYS = ("name", "gpu_pct", "efficiency_percent", "arithmetic_intensity", "kernel_category")


def _num(value: Any) -> float | None:
    """A finite float, or ``None`` for anything else, bools included."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive(value: Any) -> float | None:
    out = _num(value)
    return out if out is not None and out > 0 else None


def _text(value: Any) -> str | None:
    return str(value or "").strip() or None


def _bound(value: Any) -> str | None:
    """A bound label, lower-cased; ``unknown`` is dropped because the predictor renders ``<label>-bound``."""
    text = str(value or "").strip().lower()
    return text if text and text != "unknown" else None


def _profile_age_sec(trace: dict[str, Any]) -> int | None:
    raw = str(trace.get("ts") or "").strip()
    try:
        stamped = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=_dt.timezone.utc)
    return max(0, int((_dt.datetime.now(_dt.timezone.utc) - stamped).total_seconds()))


def _roofline(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Roofline ceilings plus what the perf model added; ``None`` without either ceiling."""
    mem = _positive(snapshot.get("roofline_mem_ceiling_tok_per_sec"))
    cmp_ = _positive(snapshot.get("roofline_cmp_ceiling_tok_per_sec"))
    if mem is None and cmp_ is None:
        return None
    perfmodel = snapshot.get("perfmodel_breakdown")
    perfmodel = perfmodel if isinstance(perfmodel, dict) else {}
    ops = [op for op in perfmodel.get("ops") or [] if isinstance(op, dict)]
    return {
        "roofline_mem_ceiling_tok_per_sec": mem,
        "roofline_cmp_ceiling_tok_per_sec": cmp_,
        "roofline_bound_kind": _bound(snapshot.get("roofline_bound_kind")),
        "achieved_tok_per_sec": _positive(snapshot.get("achieved_tok_per_sec")),
        "gap_to_roofline_pct": _num(snapshot.get("gap_to_roofline_pct")),
        "hbm_bw_gbps": _positive(perfmodel.get("hbm_bw_gbps")),
        "peak_achievable_tflops": _positive(perfmodel.get("peak_achievable_tflops")),
        "n_ops_total": len(ops) if ops else None,
        "n_ops_memory_bound": sum(str(op.get("bound") or "").lower() == "memory" for op in ops) if ops else None,
    }


def _hot_kernels(trace: dict[str, Any], sites: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Top hot kernels with the operand args, call count and source location the summary rows lack."""
    rows = trace.get("hot_kernels_top15")
    if not isinstance(rows, list) or not rows:
        return None
    report = str(trace.get("analysis_md_text") or "")
    p_items = evidence.p_item_index(report) if report else {}
    out: list[dict[str, Any]] = []
    for row in rows[:HOT_KERNEL_TOP_N]:
        if not isinstance(row, dict):
            continue
        keys = (str(row.get("kernel_id") or "").strip(), str(row.get("name") or "").strip())
        site = next((sites[key] for key in keys if key and key in sites), {})
        p_item = p_items.get(keys[1]) or {}
        kernel = {key: row.get(key) for key in _KERNEL_KEYS}
        kernel.update(
            bound_type=_bound(row.get("bound_type")),
            source_file=site.get("source_file", row.get("source_file")),
            source_line=site.get("source_line", row.get("source_line")),
            source_function=site.get("source_function", row.get("source_function")),
            time_us=p_item.get("time_us"),
            args=p_item.get("args"),
            call_count=p_item.get("call_count"),
        )
        out.append(kernel)
    return out or None


def _evidence(state: Any, sites: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    trace = state.last_trace_analyze if isinstance(state.last_trace_analyze, dict) else {}
    # Hot kernels alone count: a trace whose quality gate withheld analysis.md still names where device time went.
    if not (trace.get("analysis_md_text") or trace.get("hot_kernels_top15")):
        return {"profile_available": False}
    snapshots = state.roofline_snapshots if isinstance(state.roofline_snapshots, list) else []
    snapshot = snapshots[-1] if snapshots and isinstance(snapshots[-1], dict) else {}
    report = str(trace.get("analysis_md_text") or "")
    block = {
        "profile_available": True,
        "profile_age_sec": _profile_age_sec(trace),
        "roofline": _roofline(snapshot),
        "window": evidence.parse_window(report) if report else None,
        "operators": evidence.parse_operators(report) if report else None,
        "hot_kernels": _hot_kernels(trace, sites),
    }
    return {key: value for key, value in block.items() if value is not None}


def _stack(state: Any) -> list[dict[str, Any]]:
    """The applied stack, one row per KEEP with that step's own contribution rather than the accumulation."""
    out: list[dict[str, Any]] = []
    for step in state.optimization_stack or []:
        if not isinstance(step, dict):
            continue
        envs = step.get("extra_envs")
        out.append(
            {
                "candidate_extra_server_args": _text(step.get("candidate_extra_server_args")),
                "extra_envs": dict(envs) if isinstance(envs, dict) and envs else None,
                "tput": _positive(step.get("tput")),
            }
        )
    return out


def build_request(state: Any, *, session_id: str, sites: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the request body for the current decision point.

    ``sites`` is :func:`~hyperloom.orchestrator.predictor.source_sites.load_source_sites` for the
    session's analysis run, read by the caller off the event loop.
    """
    info = state.model_info if isinstance(state.model_info, dict) else {}
    history = state.phase_history if isinstance(state.phase_history, list) else []
    last_transition = history[-1] if history and isinstance(history[-1], dict) else {}
    best = state.current_best if isinstance(state.current_best, dict) else {}
    return {
        "schema": REQUEST_SCHEMA,
        "session_id": session_id,
        "identification": {
            "model_name": _text(state.model_name),
            "model_class": _text(state.model_class),
            "gpu_type": _text(state.gpu_type),
            "framework": _text(state.framework),
            "framework_version": _text(state.framework_version),
            "precision": _text(state.precision),
            "tp": state.tp,
            "ep": state.ep,
            "model_info": {key: info.get(key) for key in _MODEL_INFO_KEYS},
        },
        "workload": {
            "isl": state.isl,
            "osl": state.osl,
            "conc": state.conc,
            "max_model_len": state.max_model_len,
        },
        "phase": {
            "phase": PHASE_LABEL,
            "phase_reason": _text(last_transition.get("reason")),
            "phase_elapsed_seconds": round(phase_elapsed_seconds(state), 1),
        },
        "performance": {
            "baseline_tput": _positive(state.baseline_tput),
            # Candidates are graded against the champion once the stack is non-empty.
            "current_best_tput": _positive(best.get("tput")),
            "cumulative_gain_validated": _num(state.cumulative_gain_validated),
            # Decays with the macro-cycle and doubles on multi-node.
            "keep_threshold_pct": resolve_keep_threshold(state),
            "optimization_stack": _stack(state),
        },
        "evidence": _evidence(state, sites),
    }
