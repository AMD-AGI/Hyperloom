# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Analysis failures that recur on the same input, remembered against that input so it is not re-run.

A roofline holds the benchmark, profile and server lanes for up to an hour. When it fails in a way the same input
reproduces, running it again on that input buys the same failure. Its input is the recipe it profiled, identified by
``working_recipe_generation`` (which every KEEP bumps) and stamped on the task when it is enqueued, so a KEEP re-arms
it. Any other failure is transient and stays with ``roofline_failure_streak``. A trace analysis's input is the trace
it read, and for a timeout also the budget it ran under, so a request granting a different budget runs.

The records live in ``SharedState.analysis_failures``: ``"roofline:<recipe>"`` or ``"trace:<trace_input>"`` ->
``{"failure", "error", "ts"}``, a trace record also carrying ``budget_minutes``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)

#: Task param carrying the identity of the recipe a roofline profiles.
RECIPE_PARAM = "recipe_generation"

_TRACE_SPLIT = "trace_split_no_steady_state"
_ANALYSIS_TIMEOUT_CLASS = "subprocess_timeout"
_TIMEOUT_FAILURE = "analysis_timeout"
_ERROR_EXCERPT_CHARS = 500


def recurring_failure_class(result: Mapping[str, Any]) -> str:
    """The class of a failure that recurs on the same input, or ``""`` for one that may not.

    Reads a roofline result and the sub-step result it wraps, so the cause survives the roofline's own failure class.
    """
    sub = result.get("sub_result")
    layers = [result, sub] if isinstance(sub, Mapping) else [result]
    classes = {str(layer.get(key) or "") for layer in layers for key in ("error_class", "phase")}
    if _TRACE_SPLIT in classes or any(_TRACE_SPLIT in str(layer.get("error") or "") for layer in layers):
        return _TRACE_SPLIT
    if _ANALYSIS_TIMEOUT_CLASS in classes:
        return _TIMEOUT_FAILURE
    return ""


def roofline_recipe(state: Any, reason: str) -> str:
    """Identity of the launch configuration a roofline enqueued for ``reason`` profiles.

    PRELUDE profiles the baseline's own arguments, which are current_best's only until the first KEEP.
    """
    current_best = state.current_best if isinstance(state.current_best, dict) else {}
    if reason == "prelude_initial" and current_best.get("action") != "baseline":
        return "baseline-arm"
    return f"recipe-{int(state.working_recipe_generation or 0)}"


def repeated_roofline_failure(state: Any, recipe: str) -> dict[str, Any] | None:
    """The remembered recurring failure of a roofline on ``recipe``, or ``None`` when it may still succeed."""
    return (state.analysis_failures or {}).get(f"roofline:{recipe}")


def remember_roofline_failure(state: Any, params: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Remember a failed roofline whose failure recurs on the recipe it profiled; return that failure's class.

    Returns ``""``, remembering nothing, for a result that did not fail, a failure that may not recur, or a task
    enqueued without a recipe identity.
    """
    recipe = str(params.get(RECIPE_PARAM) or "")
    failure = _remember(state, f"roofline:{recipe}" if recipe else "", result)
    if failure:
        log.warning(
            "roofline on %s failed with %s, which recurs on the same input; not re-armed until a KEEP changes the "
            "recipe",
            recipe,
            failure,
        )
    return failure


def remember_trace_failure(state: Any, payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Remember a failed analysis of ``payload``'s trace that the same trace reproduces; return the failure's class."""
    trace = _trace_input(payload)
    return _remember(state, f"trace:{trace}" if trace else "", result, budget_minutes=payload.get("budget_minutes"))


def remembered_trace_failure(state: Any, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The failed result to serve for a trace analysis the same input already failed, or ``None`` to run it."""
    trace = _trace_input(payload)
    record = (state.analysis_failures or {}).get(f"trace:{trace}") if trace else None
    if record is None:
        return None
    if record["failure"] == _TIMEOUT_FAILURE and record.get("budget_minutes") != payload.get("budget_minutes"):
        return None
    log.warning("trace_analyze on %s not re-run: it already failed with %s", trace, record["failure"])
    return {
        "status": "failed",
        "error_class": "trace_analysis_failure_remembered",
        "failure": record["failure"],
        "error": (
            f"not re-run: analysing this trace already failed with {record['failure']} at {record['ts']}, which "
            f"the same trace reproduces: {record['error']}"
        ),
        "trace_input": trace,
        "hot_kernels": [],
    }


def analysis_skip_reason(state: Any) -> str:
    """Why ``internal_analysis_params`` declined to build an analysis, for the record of the skipped run."""
    return "gpu_trace_unsupported" if state.gpu_trace_unsupported_reason else "repeated_failure"


def _trace_input(payload: Mapping[str, Any]) -> str:
    return str(payload.get("trace_input") or payload.get("trace_dir") or "")


def _remember(state: Any, key: str, result: Mapping[str, Any], **extra: Any) -> str:
    """Record ``result`` under ``key`` when it failed in a way that recurs; return the failure's class or ``""``."""
    failure = recurring_failure_class(result) if result.get("status") == "failed" else ""
    if not (failure and key):
        return ""
    state.analysis_failures = {
        **(state.analysis_failures or {}),
        key: {
            "failure": failure,
            "error": str(result.get("error") or "")[:_ERROR_EXCERPT_CHARS],
            "ts": now_iso(),
            **extra,
        },
    }
    return failure
