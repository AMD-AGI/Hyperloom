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

import time
from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.coerce import text_or_none, to_float, to_unix
from hyperloom.orchestrator.predictor import evidence
from hyperloom.common.predictor_config import PHASE_LABEL
from hyperloom.orchestrator.predictor.view import SessionView


@dataclass(frozen=True)
class PhaseBudget:
    """The FRAMEWORK clock a request is conditioned on, as the phase reads it when the predictor asks."""

    #: Seconds the phase can still spend: the tighter of this entry's budget and the cap across entries.
    headroom_sec: float | None
    #: Seconds since this FRAMEWORK entry began.
    elapsed_sec: float
    #: Gain a variant needs to KEEP in this macro-cycle.
    keep_threshold_pct: float


REQUEST_SCHEMA = "hyperloom.predictor_request.v1"

#: Hot kernels sent; the service keeps its own top five of these.
HOT_KERNEL_TOP_N = 8

#: Architecture facts the service reads off ``identification.model_info``.
_MODEL_INFO_KEYS = ("model_type", "attention_type", "num_hidden_layers", "num_experts", "hidden_size", "head_dim")

#: Fields copied as-is from a ``hot_kernels_top15`` row.
_KERNEL_KEYS = ("name", "gpu_pct", "efficiency_percent", "arithmetic_intensity", "kernel_category")


def _positive(value: Any) -> float | None:
    """A rate or a ceiling, which is absent unless positive."""
    out = to_float(value)
    return out if out is not None and out > 0 else None


def _bound(value: Any) -> str | None:
    """A bound label, lower-cased; ``unknown`` is dropped because the predictor renders ``<label>-bound``."""
    text = str(value or "").strip().lower()
    return text if text and text != "unknown" else None


def _profile_age_sec(trace: dict[str, Any]) -> int | None:
    stamped = to_unix(trace.get("ts"))
    return None if stamped is None else max(0, int(time.time() - stamped))


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
        "gap_to_roofline_pct": to_float(snapshot.get("gap_to_roofline_pct")),
        "hbm_bw_gbps": _positive(perfmodel.get("hbm_bw_gbps")),
        "peak_achievable_tflops": _positive(perfmodel.get("peak_achievable_tflops")),
        "n_ops_total": len(ops) if ops else None,
        "n_ops_memory_bound": sum(str(op.get("bound") or "").lower() == "memory" for op in ops) if ops else None,
    }


def _hot_kernels(
    trace: dict[str, Any], sites: Mapping[str, dict[str, Any]], kernels: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Top hot kernels, with the operand args, time and call count read from the report or the kernel stats."""
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
        stats = next((kernels[key] for key in keys if key and key in kernels), {})
        p_item = p_items.get(keys[1]) or {}
        kernel = {key: row.get(key) for key in _KERNEL_KEYS}
        kernel.update(
            bound_type=_bound(row.get("bound_type")),
            source_file=site.get("source_file", row.get("source_file")),
            source_line=site.get("source_line", row.get("source_line")),
            source_function=site.get("source_function", row.get("source_function")),
            time_us=_first(p_item.get("time_us"), stats.get("time_us")),
            args=p_item.get("args"),
            call_count=_first(p_item.get("call_count"), stats.get("call_count")),
        )
        out.append(kernel)
    return out or None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _evidence(state: SessionView, sites: Mapping[str, dict[str, Any]], sidecars: Mapping[str, Any]) -> dict[str, Any]:
    trace = state.last_trace_analyze
    # Hot kernels alone count: a trace whose quality gate withheld analysis.md still names where device time went.
    if not (trace.get("analysis_md_text") or trace.get("hot_kernels_top15")):
        return {"profile_available": False}
    snapshot = state.roofline_snapshots[-1] if state.roofline_snapshots else {}
    report = str(trace.get("analysis_md_text") or "")
    block = {
        "profile_available": True,
        "profile_age_sec": _profile_age_sec(trace),
        "roofline": _roofline(snapshot),
        # The bypass layout parses into what the predictor was trained on; the TraceLens route's report does not.
        "window": _first(evidence.parse_window(report) if report else None, sidecars.get("window")),
        "operators": _first(evidence.parse_operators(report) if report else None, sidecars.get("operators")),
        "hot_kernels": _hot_kernels(trace, sites, sidecars.get("kernels") or {}),
    }
    return {key: value for key, value in block.items() if value is not None}


def _stack(state: SessionView) -> list[dict[str, Any]]:
    """The applied stack, one row per KEEP with that step's own contribution rather than the accumulation."""
    out: list[dict[str, Any]] = []
    for step in state.optimization_stack:
        envs = step.get("extra_envs")
        out.append(
            {
                "candidate_extra_server_args": text_or_none(step.get("candidate_extra_server_args")),
                "extra_envs": dict(envs) if isinstance(envs, dict) and envs else None,
                "tput": _positive(step.get("tput")),
            }
        )
    return out


def build_request(
    state: SessionView,
    *,
    sites: Mapping[str, dict[str, Any]],
    sidecars: Mapping[str, Any],
    budget: PhaseBudget,
) -> dict[str, Any]:
    """Build the request body for the current decision point.

    ``sites`` and ``sidecars`` are :func:`~hyperloom.orchestrator.predictor.source_sites.load_source_sites`
    and :func:`~hyperloom.orchestrator.predictor.sidecars.load_sidecars` for the session's analysis
    run, read by the caller off the event loop. ``budget`` is the phase's clock and KEEP bar as
    FRAMEWORK reads them when it asks.
    """
    info = state.model_info
    last_transition = state.phase_history[-1] if state.phase_history else {}
    best = state.current_best
    return {
        "schema": REQUEST_SCHEMA,
        "session_id": str(state.session_id or ""),
        "identification": {
            "model_name": text_or_none(state.model_name),
            "model_class": text_or_none(state.model_class),
            "gpu_type": text_or_none(state.gpu_type),
            "framework": text_or_none(state.framework),
            "framework_version": text_or_none(state.framework_version),
            "precision": text_or_none(state.precision),
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
            "phase_reason": text_or_none(last_transition.get("reason")),
            "phase_elapsed_seconds": round(budget.elapsed_sec, 1),
        },
        "performance": {
            "baseline_tput": _positive(state.baseline_tput),
            # Candidates are graded against the champion once the stack is non-empty.
            "current_best_tput": _positive(best.get("tput")),
            "cumulative_gain_validated": to_float(state.cumulative_gain_validated),
            # Decays with the macro-cycle and doubles on multi-node.
            "keep_threshold_pct": budget.keep_threshold_pct,
            "optimization_stack": _stack(state),
        },
        "evidence": _evidence(state, sites, sidecars),
    }
