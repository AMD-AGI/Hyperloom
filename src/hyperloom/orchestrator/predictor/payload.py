# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assemble the predictor request from session state and trace artifacts.

The body uses Hyperloom's own field names. Renaming them here would duplicate a
mapping the consumer already maintains, and the failure mode is silent: a
renderer that reads a key nobody sent drops the sentence without complaining.
Keeping our vocabulary means a drift surfaces as one missing key on the side
that owns the renderer.

See ``docs/reference/primatune-predictor.md`` for the wire contract.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from hyperloom.orchestrator.phases.machine_state import (
    phase_elapsed_seconds,
    resolve_keep_threshold,
)
from hyperloom.orchestrator.predictor import evidence as ev
from hyperloom.orchestrator.predictor import source_sites as ss

log = logging.getLogger(__name__)

#: Wire version. Bumped on any field removal or meaning change.
REQUEST_SCHEMA = "hyperloom.predictor_request.v1"

#: Hot kernels forwarded. Matches the slice the specialist prompts already take
#: (``phases/explore.py``) and the consumer's own top-K.
HOT_KERNEL_TOP_N = 8

#: Fields projected from a ``hot_kernels_top15`` row, Hyperloom's spelling.
_KERNEL_FIELDS = (
    "name",
    "gpu_pct",
    "efficiency_percent",
    "arithmetic_intensity",
    "bound_type",
    "kernel_category",
    "source_file",
)

#: Fields the hot-kernel projection lacks; the P-item tables carry them.
_KERNEL_FROM_P_ITEM = ("time_us", "args", "call_count")


def _num(value: Any) -> float | None:
    """Return a finite float, or ``None`` for anything else (including bools)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _positive(value: Any) -> float | None:
    """Return a strictly positive float, else ``None``."""
    out = _num(value)
    return out if out is not None and out > 0 else None


def _bound_type(value: Any) -> str | None:
    """Normalise a bottleneck label, dropping the unclassified sentinel.

    TraceLens writes ``unknown`` when it could not classify an operation. A
    consumer renders a bound label by appending ``-bound`` to whatever arrives,
    turning that into ``unknown-bound`` — a category no corpus contains. An
    absent label costs one clause; a fabricated one costs the sentence's
    credibility.

    Args:
        value (Any): The raw ``bound_type`` / ``roofline_bound_kind``.

    Returns:
        str | None: The lower-cased label, or ``None`` when absent or unknown.
    """
    text = str(value or "").strip().lower()
    return text or None if text != "unknown" else None


def _profile_age_sec(last_trace_analyze: dict[str, Any], *, now: _dt.datetime | None = None) -> int | None:
    """Seconds since the trace analysis was recorded, from its ISO ``ts``."""
    raw = str(last_trace_analyze.get("ts") or "").strip()
    if not raw:
        return None
    try:
        stamped = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=_dt.timezone.utc)
    current = now or _dt.datetime.now(_dt.timezone.utc)
    return max(0, int((current - stamped).total_seconds()))


def _roofline(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Roofline ceilings plus whatever the optional perf model added.

    The three ceilings are the block: without them there is nothing to compare
    achieved throughput against. ``perfmodel_breakdown`` is attached
    best-effort, so its four derived fields stay optional — their absence costs
    the consumer two sentences, not the block.

    ``bound_kind`` of ``"unknown"`` becomes ``None``. The string is a real value
    downstream (it means the model could not classify), but sending it invites a
    consumer to match on a label it has no rule for; absence is what it is.

    Args:
        snapshot (dict[str, Any]): ``roofline_snapshots[-1]``.

    Returns:
        dict[str, Any] | None: The block, or ``None`` when both ceilings are
            missing.
    """
    mem = _positive(snapshot.get("roofline_mem_ceiling_tok_per_sec"))
    cmp_ = _positive(snapshot.get("roofline_cmp_ceiling_tok_per_sec"))
    if mem is None and cmp_ is None:
        log.debug("predictor_payload: dropping roofline block, no ceilings")
        return None

    perfmodel = snapshot.get("perfmodel_breakdown")
    perfmodel = perfmodel if isinstance(perfmodel, dict) else {}
    ops = perfmodel.get("ops")
    ops = ops if isinstance(ops, list) else []

    block: dict[str, Any] = {
        "roofline_mem_ceiling_tok_per_sec": mem,
        "roofline_cmp_ceiling_tok_per_sec": cmp_,
        "roofline_bound_kind": _bound_type(snapshot.get("roofline_bound_kind")),
        "achieved_tok_per_sec": _positive(snapshot.get("achieved_tok_per_sec")),
        "gap_to_roofline_pct": _num(snapshot.get("gap_to_roofline_pct")),
        "hbm_bw_gbps": _positive(perfmodel.get("hbm_bw_gbps")),
        "peak_achievable_tflops": _positive(perfmodel.get("peak_achievable_tflops")),
    }
    if ops:
        block["n_ops_total"] = len(ops)
        block["n_ops_memory_bound"] = sum(
            1 for op in ops if isinstance(op, dict) and str(op.get("bound") or "").lower() == "memory"
        )
    else:
        block["n_ops_total"] = None
        block["n_ops_memory_bound"] = None
    return block


def _hot_kernels(last_trace_analyze: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Top hot kernels, enriched with operand args, counts and source frames."""
    rows = last_trace_analyze.get("hot_kernels_top15")
    if not isinstance(rows, list) or not rows:
        return None

    analysis_md = str(last_trace_analyze.get("analysis_md_text") or "")
    p_items = ev.p_item_index(analysis_md) if analysis_md else {}
    sites = ss.load_source_sites(last_trace_analyze.get("analysis_md_path"))
    enriched = ss.attach_sites(rows[:HOT_KERNEL_TOP_N], sites)

    out: list[dict[str, Any]] = []
    for row in enriched:
        if not isinstance(row, dict):
            continue
        projected: dict[str, Any] = {key: row.get(key) for key in _KERNEL_FIELDS}
        # Same rule as roofline_bound_kind: a consumer appends "-bound" to
        # whatever it is given, so "unknown" renders as "unknown-bound" — a
        # classification the model never saw. Absence is the honest value.
        projected["bound_type"] = _bound_type(row.get("bound_type"))
        projected["source_line"] = row.get("source_line")
        projected["source_function"] = row.get("source_function")
        p_row = p_items.get(str(row.get("name") or "")) or {}
        for key in _KERNEL_FROM_P_ITEM:
            projected[key] = p_row.get(key)
        out.append(projected)
    return out or None


def _evidence(state: Any) -> dict[str, Any]:
    """The evidence section, with each sub-block present only when complete."""
    last_ta = getattr(state, "last_trace_analyze", None)
    last_ta = last_ta if isinstance(last_ta, dict) else {}

    # Hyperloom's own test for "is there evidence at all" (phases/explore.py):
    # hot kernels alone count, because a trace whose quality gate withheld
    # analysis.md still names where device time went.
    available = bool(last_ta.get("analysis_md_text") or last_ta.get("hot_kernels_top15"))
    if not available:
        return {"profile_available": False}

    snapshots = getattr(state, "roofline_snapshots", None)
    snapshot = snapshots[-1] if isinstance(snapshots, list) and snapshots else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}

    analysis_md = str(last_ta.get("analysis_md_text") or "")
    block: dict[str, Any] = {
        "profile_available": True,
        "profile_age_sec": _profile_age_sec(last_ta),
        "roofline": _roofline(snapshot),
        "window": ev.parse_window(analysis_md) if analysis_md else None,
        "operators": ev.parse_operators(analysis_md) if analysis_md else None,
        "hot_kernels": _hot_kernels(last_ta),
    }
    return {key: value for key, value in block.items() if value is not None or key == "profile_available"}


def _stack(state: Any) -> list[dict[str, Any]]:
    """The applied stack, one entry per accepted step.

    ``candidate_extra_server_args`` is that step's own contribution;
    ``extra_server_args`` on the same row is the accumulation up to it. Sending
    the latter would make every row repeat all preceding flags, which reads as a
    stack that re-applies itself.
    """
    stack = getattr(state, "optimization_stack", None)
    out: list[dict[str, Any]] = []
    for step in stack if isinstance(stack, list) else []:
        if not isinstance(step, dict):
            continue
        envs = step.get("extra_envs")
        out.append(
            {
                "candidate_extra_server_args": str(step.get("candidate_extra_server_args") or "").strip() or None,
                "extra_envs": dict(envs) if isinstance(envs, dict) and envs else None,
                "tput": _positive(step.get("tput")),
            }
        )
    return out


def build_request(state: Any, *, session_id: str = "", phase_label: str = "EXPLORE") -> dict[str, Any]:
    """Build the predictor request body from live session state.

    Args:
        state (Any): The ``SharedState``.
        session_id (str): Session id, for correlating logs across the two sides.
        phase_label (str): Value sent as ``phase.phase``. Defaults to
            ``EXPLORE`` because the pump feeds the configuration arm, which was
            its own phase under that name before the merge; the live phase is
            still available to the consumer through the session breakdown.

    Returns:
        dict[str, Any]: The JSON-serialisable request body.
    """
    history = getattr(state, "phase_history", None)
    last_transition = history[-1] if isinstance(history, list) and history else {}
    last_transition = last_transition if isinstance(last_transition, dict) else {}

    model_info = getattr(state, "model_info", None)

    return {
        "schema": REQUEST_SCHEMA,
        "session_id": str(session_id or ""),
        "identification": {
            "model_name": str(getattr(state, "model_name", "") or "") or None,
            "model_class": str(getattr(state, "model_class", "") or "") or None,
            "gpu_type": str(getattr(state, "gpu_type", "") or "") or None,
            "framework": str(getattr(state, "framework", "") or "") or None,
            "framework_version": str(getattr(state, "framework_version", "") or "") or None,
            "precision": str(getattr(state, "precision", "") or "") or None,
            "tp": getattr(state, "tp", None),
            "ep": getattr(state, "ep", None),
            "nodes": getattr(state, "nodes", None),
            "model_info": dict(model_info) if isinstance(model_info, dict) else {},
        },
        "workload": {
            "isl": getattr(state, "isl", None),
            "osl": getattr(state, "osl", None),
            "conc": getattr(state, "conc", None),
            "max_model_len": getattr(state, "max_model_len", None),
        },
        "phase": {
            "phase": str(phase_label or "").strip() or None,
            "phase_reason": str(last_transition.get("reason") or "").strip() or None,
            "phase_elapsed_seconds": round(phase_elapsed_seconds(state), 1),
            "macro_cycle": int(getattr(state, "macro_cycle", 0) or 0),
        },
        "performance": {
            "baseline_tput": _positive(getattr(state, "baseline_tput", None)),
            # Also the grading anchor: once the stack is non-empty, candidates
            # are scored against the reigning champion, not the baseline.
            "current_best_tput": _positive((getattr(state, "current_best", None) or {}).get("tput")),
            "cumulative_gain_validated": _num(getattr(state, "cumulative_gain_validated", None)),
            # Decays with the macro-cycle and doubles on multi-node, so this is
            # never a constant even though cycle 1 single-node happens to be 1.0.
            "keep_threshold_pct": resolve_keep_threshold(state),
            "optimization_stack": _stack(state),
        },
        "evidence": _evidence(state),
    }
