# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why an integration is refused before anything is measured."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float, to_str_list
from hyperloom.common.gain_math import incremental_gain_pct
from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_OUTPUT,
    constraint_veto_reason,
    holds_within_band,
    incomparability_reason,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
)
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.orchestrator.measurement.runtime_findings import (
    CORRECTNESS_FIX_MAX_DROP_PCT,
    correctness_fix_refusal,
    load_runtime_findings,
)

from ...enablement.lane import takes_enablement_round
from ...policy.gate import INTEGRATE_PATCH_PERMISSIVE_VERDICTS
from ...specialists.patch_safety import patch_targets_missing


def _derive_lane(params: dict[str, Any]) -> str:
    """Derive the retry lane name from integrate_patch params.

    Returns:
        ``"enablement"``, ``"perf_framework"``, or ``"perf_explore"``.
    """
    if params.get("enablement"):
        return "enablement"
    if params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"):
        return "perf_framework"
    return "perf_explore"


def _accuracy_delta_pct(measured: Any, baseline: Any) -> float | None:
    """Percent accuracy change of ``measured`` against ``baseline``.

    Returns ``None`` when either side is missing or the baseline is not
    positive, so callers fall back to their existing value.
    """
    m = to_float(measured)
    b = to_float(baseline)
    if m is None or b is None:
        return None
    return incremental_gain_pct(m, b)


def _measured_against(params: Mapping[str, Any], *, base_tput: float) -> dict[str, Any]:
    """The configuration this patch was measured on top of.

    Only the executor can answer this. The task's own params hold the stack as
    of dispatch, and a task queued before another KEEP landed is rebound onto
    the live stack top before it runs; ``base_tput`` is the drift-resolved
    anchor it was actually graded against. Carried on the result for the reason
    the configuration arm carries it -- what the session serves once this
    returns is no longer what the patch was judged against, so a reader that
    reconstructs the stack from the current config gets the wrong one.
    """
    return {
        "throughput": base_tput or None,
        "accuracy": float(params.get("accuracy_baseline") or 0.0) or None,
        "extra_server_args": str(params.get("base_extra_args") or "").strip(),
        "extra_envs": dict(params.get("base_extra_envs") or {}),
        "remove_args": to_str_list(params.get("base_remove_args")),
        "unset_envs": to_str_list(params.get("base_unset_envs")),
        "args_mode": str(params.get("base_args_mode") or "append"),
    }


def _preflight_missing_targets(
    framework_root: Path,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Return per-patch records for patches whose modify/delete targets are
    absent from ``framework_root`` at every ``-p`` strip level.

    A hallucinated-layout patch (e.g. modifying a CUDA-only file on a ROCm
    build) can never apply; flagging it here yields an actionable advisory
    instead of an opaque ``git_apply_failed`` after a wasted apply attempt.
    Patches supplied directly via ``params.patches`` bypass the
    authoring-time ``specialists.patch_safety`` vetting gate
    (:func:`vet_patches`), so they are checked here.

    Args:
        framework_root: The git checkout the patches target.
        patch_paths: The patch files to preflight.

    Returns:
        A list of per-patch records (``patch`` + ``missing_targets``) for
        patches whose targets are absent at every strip level.
    """
    records: list[dict[str, Any]] = []
    for patch in patch_paths:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        missing = patch_targets_missing(text, framework_root)
        if missing:
            records.append({"patch": str(patch), "missing_targets": missing})
    return records


_FIX_ACCURACY_UNAVAILABLE = "accuracy result unavailable"


def _fix_axis(graded: Any) -> str:
    return "interactivity p50" if graded.graded_on_intvty else "throughput"


def _intvty_guard_refusals(shared_state: Any, bench_result: dict[str, Any]) -> list[str]:
    """The interactivity KEEP guards a correctness fix still owes; only the p50 gain is relaxed."""
    ref_perf, _ = resolve_grading_anchor_perf(shared_state)
    cand_perf = perf_snapshot_from_mapping(bench_result)
    _, noise_pct = resolved_grading(shared_state)
    band = f"{noise_pct:.1f}% noise band" if noise_pct is not None else "noise band"
    refusals = [
        f"{key} fell outside the {band}"
        for key in (GRADED_INTVTY, GRADED_OUTPUT)
        if not holds_within_band(cand_perf, ref_perf, key, noise_pct=noise_pct)
    ]
    incomparable = incomparability_reason(cand_perf, ref_perf)
    if incomparable:
        refusals.append(f"rounds are not comparable ({incomparable})")
    return refusals


def _correctness_fix_refusals(
    resolves_finding: str,
    *,
    shared_state: Any,
    bench_result: dict[str, Any],
    graded: Any,
    delta_pct: float | None,
) -> list[str]:
    """Performance and finding reasons a declared correctness fix may not KEEP below the graded-axis threshold."""
    refusals: list[str] = []
    finding_refusal = correctness_fix_refusal(
        load_runtime_findings(getattr(shared_state, "current_best_measurement", None) or {}),
        load_runtime_findings(bench_result),
        resolves_finding,
    )
    if finding_refusal:
        refusals.append(finding_refusal)
    # The resolver only vetoes a KEEP verdict, and this route runs after a REVERT, so the budgets are read here.
    veto = constraint_veto_reason(bench_result, shared_state)
    if not graded.comparable:
        refusals.append(f"performance comparison unavailable: {graded.degrade_reason}")
    elif veto:
        refusals.append(veto)
    elif delta_pct is None:
        refusals.append(f"no measurable {_fix_axis(graded)}")
    elif delta_pct < -CORRECTNESS_FIX_MAX_DROP_PCT:
        refusals.append(
            f"{_fix_axis(graded)} delta {delta_pct:+.2f}% exceeds the "
            f"{CORRECTNESS_FIX_MAX_DROP_PCT:.1f}% correctness-fix allowance"
        )
    if graded.comparable and graded.graded_on_intvty:
        refusals.extend(_intvty_guard_refusals(shared_state, bench_result))
    return refusals


def _fix_accuracy_refusals(accuracy_pass: bool | None) -> list[str]:
    """The accuracy reason a declared correctness fix may not KEEP; empty when its eval passed."""
    if accuracy_pass is None:
        return [_FIX_ACCURACY_UNAVAILABLE]
    if accuracy_pass is not True:
        return ["accuracy eval failed"]
    return []


def _enforce_critic_gate(
    shared_state: Any,
    subject: str,
) -> "dict[str, Any] | None":
    """Enforce a permissive Critic verdict on ``subject`` before any side
    effect; returns a ``rejected_by_critic`` dict on failure, else ``None``
    when no SharedState is available or the verdict is permissive."""
    if shared_state is None:
        return None
    try:
        recorded = shared_state.get_specialist_patch_verdict(subject)
    except AttributeError:
        recorded = ""
    if (recorded or "").lower() not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
        _detail = f"verdict {recorded!r}" if recorded else "no Critic verdict on record"
        # No side effect has occurred yet; reject cleanly (nothing to revert).
        return {
            "status": "rejected_by_critic",
            "specialist_task_id": subject,
            "patches_applied": [],
            "patches_reverted": [],
            "reason": (
                f"integrate_patch requires a permissive Critic verdict "
                f"(approve/advise) for {subject!r}; {_detail}. Refusing to run."
            ),
        }
    return None


def _enablement_lane_closed_skip(params: Mapping[str, Any], shared_state: Any) -> "dict[str, Any] | None":
    """The skip for an enablement round the lane no longer takes; ``None`` for any other integrate."""
    if not params.get("enablement") or shared_state is None or takes_enablement_round(shared_state):
        return None
    return {"status": "skipped", "reason": f"enablement lane takes no rounds in phase {shared_state.phase}"}


def _gate_failure_reasons(
    graded: Any,
    delta_pct: float | None,
    keep_threshold_pct: float,
    *also: str,
) -> list[str]:
    """Everything the KEEP gate held against this measurement, in the order it was weighed."""
    reasons: list[str] = []
    if not graded.comparable:
        reasons.append(f"performance comparison unavailable: {graded.degrade_reason}")
    elif graded.graded_on_intvty and delta_pct is None:
        reasons.append(
            graded.veto_reason
            or f"interactivity p50 {graded.reference:.1f}->{graded.candidate:.1f} did not clear the interactivity KEEP gate"
        )
    elif delta_pct is None:
        reasons.append("no measurable throughput")
    elif delta_pct < keep_threshold_pct:
        reasons.append(f"throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%")
    elif graded.veto_reason:
        reasons.append(graded.veto_reason)
    reasons.extend(reason for reason in also if reason)
    return reasons
