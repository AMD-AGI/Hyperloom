# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Grading the accuracy of a kernel patch's end-to-end re-baseline.

A patch only KEEPs if it also holds accuracy, and the verdict is read back from
the re-baseline's own staged eval rather than measured again. A missing verdict
is not the same as a measured drop, and neither is the same as a serving
configuration that cannot answer an eval request at all; this is where those
three are told apart.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

log = logging.getLogger(__name__)

# Mirrors the completion ceiling the inferencex eval shim installs; kept in sync
# so the feasibility check reasons about the budget the eval will really ask for.
_EVAL_DEFAULT_MAX_TOKENS = 4096


def _eval_generation_budget() -> int:
    """Completion tokens the eval harness reserves per sample.

    Mirrors the clamp installed by the inferencex shim: ``HYPERLOOM_EVAL_MAX_TOKENS`` when it parses as a
    positive integer, else the shim's own default.
    """
    raw = (os.environ.get("HYPERLOOM_EVAL_MAX_TOKENS") or "").strip()
    if not raw:
        return _EVAL_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _EVAL_DEFAULT_MAX_TOKENS
    return value if value >= 0 else _EVAL_DEFAULT_MAX_TOKENS


class _EvalScore(NamedTuple):
    """One serving-eval score and the task, metric and file it was read from."""

    accuracy: float | None
    task: str
    metric: str
    source_file: str


def _integrate_eval_score(bench_result: dict[str, Any], workspace: Path) -> _EvalScore:
    """Read back the serving-eval score of a re-baseline."""
    from ..actions.executors._accuracy_gate import parse_eval_results

    measured = bench_result.get("accuracy")
    attached = _EvalScore(
        float(measured) if isinstance(measured, (int, float)) else None,
        str(bench_result.get("accuracy_task") or ""),
        str(bench_result.get("accuracy_metric") or ""),
        str(bench_result.get("accuracy_source") or ""),
    )
    if attached.accuracy is not None:
        return attached
    eval_out = parse_eval_results(workspace, framework=os.environ.get("FRAMEWORK") or None)
    parsed = eval_out.get("accuracy")
    if not isinstance(parsed, (int, float)):
        return attached
    return _EvalScore(
        float(parsed),
        str(eval_out.get("task") or ""),
        str(eval_out.get("metric") or ""),
        str(eval_out.get("source_file") or ""),
    )


def _grade_integrate_accuracy(
    bench_result: dict[str, Any],
    *,
    baseline_accuracy: float,
    workspace: Path,
    strict: bool = False,
    server_args: str = "",
) -> dict[str, Any]:
    """Grade a kernel re-baseline's accuracy against the session baseline.

    The staged re-baseline already ran the serving eval after hot throughput passed, so the score is read
    back rather than re-measured.
    """
    from ..actions.executors._accuracy_gate import (
        accuracy_keep_block,
        accuracy_passed,
        require_kernel_accuracy_default,
        resolve_served_context,
        served_context_hosts_eval,
    )

    scored = _integrate_eval_score(bench_result, workspace)
    new_accuracy = scored.accuracy

    accuracy_pass: bool | None = None
    if new_accuracy is not None and baseline_accuracy > 0:
        accuracy_pass = accuracy_passed(baseline_accuracy, new_accuracy)

    blocked, reason, degraded = accuracy_keep_block(
        accuracy_pass,
        required=True if strict else require_kernel_accuracy_default(),
        baseline_accuracy=baseline_accuracy,
    )
    if strict and degraded:
        blocked = True
        reason = "accuracy gate produced no eval result and this artifact has no other end-to-end correctness evidence"
    # A verdict can be missing because the eval broke, or because the serving
    # configuration cannot answer an eval request at all. Only the first says
    # anything about the patch. The second reproduces on every retry, so
    # charging it to the patch discards a kernel over a configuration choice.
    infeasible = False
    if accuracy_pass is None:
        fits, why = served_context_hosts_eval(
            served_max_model_len=resolve_served_context(
                server_args=server_args,
                env_max_model_len=os.environ.get("MAX_MODEL_LEN", 0),
            ),
            eval_max_tokens=_eval_generation_budget(),
        )
        if not fits:
            infeasible = True
            reason = why
            log.warning(
                "integrate_handler: the accuracy gate cannot run under this "
                "serving configuration, so no kernel can clear it until the "
                "configuration changes: %s",
                why,
            )
    log.info(
        "integrate_handler: accuracy gate pass=%s blocked=%s degraded=%s new=%s baseline=%.4f source=%s",
        accuracy_pass,
        blocked,
        degraded,
        "n/a" if new_accuracy is None else f"{new_accuracy:.4f}",
        baseline_accuracy,
        scored.source_file or "none",
    )
    return {
        "blocked": blocked,
        "accuracy_pass": accuracy_pass,
        "reason": reason,
        "degraded": degraded,
        "infeasible": infeasible,
        "accuracy": new_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "task": scored.task,
        "metric": scored.metric,
        "source_file": scored.source_file,
    }
