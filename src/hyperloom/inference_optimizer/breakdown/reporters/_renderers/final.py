# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final / validated result renderer.

Read off ``outcome`` and ``close``. The throughput, its latency and the launch
that produced them all come from the one validation row that settled the
stack, so the block cannot pair a figure from one measurement with a latency
from another -- which is what the export's disk walk for the latency could do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hyperloom_common.coerce import optional_positive_float
from hyperloom.common.perf_metric import INTVTY_OBJECTIVES

from hyperloom_common.coerce import to_mapping

from ..base import (
    Decision,
    RenderedSection,
    close_of,
    events_of,
    fmt_pct,
    md_kv_list,
    outcome_of,
    register_renderer,
    session_of,
    task_config_of,
    validation_of,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import GEAK_DROPPED_STATUSES
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_FAILED_STATUSES
from ._agentx_metrics import graded_axes_facts, render_graded_axes
from ._invocation import render_invocation_block

_GEAK_DROPPED_FACT = (
    "GEAK candidate (self-reported {gain}) was DROPPED without revalidation (status={status}, {reason})."
)
_GEAK_DROPPED_WARNING = (
    "A measured GEAK e2e candidate was abandoned because its same-harness revalidation {failure} ({reason}). It was "
    "never judged on merit, so this session's gain may understate what the optimizer actually found — the "
    "candidate's artefacts are on disk but absent from current_best / action_path / the validated gain."
)
_GEAK_AWAITING_FACT = (
    "GEAK candidate (self-reported {gain}) is AWAITING a main-flow rebench — excluded from the headline gain and "
    "final stack until a measured rebench validates it."
)
_GEAK_AWAITING_WARNING = (
    "A GEAK(GEAK) e2e candidate self-reported a win but has NOT been confirmed by a same-harness main-flow rebench, "
    "so it is intentionally kept out of current_best / action_path / the validated gain. Its self-reported number is "
    "audit-only and must not be presented as the headline result."
)


def _gain_axis_note(final: dict[str, Any]) -> str:
    """`` (<axis>)`` when the gain was measured on an interactivity axis, else nothing."""
    axis = final.get("graded_on")
    return f" ({axis})" if axis in INTVTY_OBJECTIVES else ""


def _validation_gap_warnings(*, validated: bool, stack_changed: bool, gain_missing: bool) -> list[str]:
    """Warnings for a validation that fell behind the shipped stack, or that never ran."""
    warnings: list[str] = []
    if validated and stack_changed:
        warnings.append(
            "The last whole-stack validation predates the final adoptions, so "
            "the validated cumulative gain describes a shorter stack than the "
            "one that shipped."
        )
    if gain_missing:
        warnings.append(
            "No validated gain is recorded while baseline/final throughput are set — no validation run completed."
        )
    return warnings


def _last_geak_rebench(breakdown: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The last GEAK visit's rebench campaign and the claim it ruled on.

    A terminal status outlives the visit that stamped it, so a session with
    several GEAK visits must be read off the last one: an earlier visit's
    ``failed`` would otherwise outrank a later visit's live candidate.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The rebench block and the
            claim block, each ``{}`` when no visit engaged the GEAK route.
    """
    for event in reversed(events_of(breakdown, "kernel")):
        geak = to_mapping(to_mapping(event.get("ext")).get("geak"))
        if not geak:
            continue
        rebench = to_mapping(geak.get("rebench"))
        if rebench:
            return rebench, to_mapping(geak.get("claim"))
    return {}, {}


def _settled_failure(rebench: dict[str, Any], action_path: list[Any]) -> bool:
    """Whether a settled GEAK failure describes a candidate that never shipped.

    Args:
        rebench (dict[str, Any]): The last visit's rebench campaign.
        action_path (list[Any]): The final stack, whose entries are ``action``
            or ``action:variant``.

    Returns:
        bool: True when the rebench carries a terminal failure and the
            candidate is absent from the final stack. The absence is read off
            the stack rather than trusted to the status, because a 2b rebench
            that failed stamps ``failed`` and nothing retracts it when the 2a
            fallback then promotes the candidate for real.
    """
    if str(rebench.get("final_status") or "") not in REBENCH_FAILED_STATUSES:
        return False
    return not any(str(step).split(":", 1)[0] == "geak_e2e" for step in action_path)


def _geak_candidate_standing(
    candidate: dict[str, Any], rebench: dict[str, Any], claim: dict[str, Any], action_path: list[Any]
) -> tuple[list[str], list[str]]:
    """The ``(facts, warnings)`` saying why a GEAK e2e candidate is absent from the final stack; empty when none is."""
    status = str(candidate.get("status") or "")
    self_gain = candidate.get("self_reported_gain_pct")
    self_gain_str = fmt_pct(self_gain, plus=True) if isinstance(self_gain, (int, float)) else "unknown"
    if status in GEAK_DROPPED_STATUSES:
        reason = str(candidate.get("revalidation_error") or "").strip() or "reason not recorded"
        return (
            [_GEAK_DROPPED_FACT.format(gain=self_gain_str, status=status, reason=reason)],
            [_GEAK_DROPPED_WARNING.format(failure="could not land", reason=reason)],
        )
    if status == "awaiting_rebench":
        return [_GEAK_AWAITING_FACT.format(gain=self_gain_str)], [_GEAK_AWAITING_WARNING]
    # Last: a settled ``failed`` on the rebench outlives the candidate slot it was recorded from, so a LIVE candidate
    # in a later macro-cycle must win over a terminal status left behind by an earlier one.
    if not _settled_failure(rebench, action_path):
        return [], []
    geak_gain = claim.get("self_reported_gain_pct")
    geak_gain_str = fmt_pct(geak_gain, plus=True) if isinstance(geak_gain, (int, float)) else "unknown"
    reason = str(rebench.get("final_error") or "").strip() or "reason not recorded"
    return (
        [_GEAK_DROPPED_FACT.format(gain=geak_gain_str, status=f"rebench_{rebench.get('final_status')}", reason=reason)],
        [_GEAK_DROPPED_WARNING.format(failure="failed", reason=reason)],
    )


@dataclass(frozen=True)
class _FinalFigures:
    """The figures the final section reads off ``outcome`` and the validation row."""

    final: dict[str, Any]
    validation: dict[str, Any]
    final_tput: Any
    base_tput: Any
    gain_v: Any
    val_stack_len: Any
    val_ts: Any
    stack_changed: bool
    extra_args: Any
    action_path: Any


def _read_figures(breakdown: dict[str, Any]) -> _FinalFigures:
    """Read the final and baseline throughput, validated gain, validation stamp and final stack."""
    outcome = outcome_of(breakdown)
    f = to_mapping(outcome.get("final"))
    b = to_mapping(outcome.get("baseline"))
    validation = validation_of(breakdown)
    return _FinalFigures(
        final=f,
        validation=validation,
        final_tput=f.get("output_throughput"),
        base_tput=b.get("output_throughput"),
        gain_v=f.get("gain_pct"),
        val_stack_len=validation.get("validated_at_stack_len"),
        val_ts=validation.get("validated_ts"),
        stack_changed=bool(validation.get("stack_changed_after_validation")),
        extra_args=f.get("extra_server_args") or "",
        action_path=f.get("action_path") or [],
    )


def _render_markdown(fig: _FinalFigures, candidate: dict[str, Any], fw: Any, session: dict[str, Any]) -> str:
    """Render the final key-value block, the validation round's graded axes and the invocation block."""
    from .... import framework_registry

    md_kv = md_kv_list(
        [
            ("final_output_throughput", fig.final_tput),
            ("throughput_unit", framework_registry.throughput_unit(fw)),
            ("validated_gain_pct", fig.gain_v),
            ("geak_candidate", candidate or None),
            ("validated_at_stack_len", fig.val_stack_len),
            ("validated_ts", fig.val_ts),
            ("measurement_basis", fig.validation.get("measurement_basis")),
            ("stack_changed_after_validation", fig.stack_changed),
            ("extra_server_args", fig.extra_args),
            ("action_path", fig.action_path),
            ("ttft_mean_ms", optional_positive_float(fig.final.get("ttft_mean_ms"))),
            ("e2el_mean_ms", fig.final.get("e2el_mean_ms")),
            ("ttft_e2el_source", fig.validation.get("ttft_e2el_source")),
            ("extra_envs", fig.final.get("extra_envs") or None),
        ]
    )

    md_parts = [md_kv]
    # The axes the validation round was graded on, beside the throughput figure above. On an AgentX session the two
    # answer different questions: the verdict was decided on these, not on ``final_output_throughput``.
    graded_md = render_graded_axes(fig.validation.get("perf"))
    if graded_md:
        md_parts.append("")
        md_parts.append("**AgentX graded axes** (validation round):")
        md_parts.append(graded_md)
    # Kept last: an empty invocation block then leaves only trailing newlines, which the strip below drops.
    md_parts.append("")
    md_parts.append(render_invocation_block(fig.final.get("invocation"), session.get("image")))
    return "\n".join(md_parts).strip()


@register_renderer("final")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the final / validated-result section.

    Surfaces final throughput, the delta and validated cumulative gain vs.
    baseline, the action path and final server args, plus data-quality
    warnings (stale validation, missing validated gain). Skipped when
    neither final throughput nor a validated gain is present.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered final-result section.
    """
    fig = _read_figures(breakdown)
    session = session_of(breakdown)
    # The candidate's standing at close: no kernel event can hold it, since it
    # is settled after every one of them has closed.
    candidate = to_mapping(close_of(breakdown).get("geak_candidate"))
    rebench, claim = _last_geak_rebench(breakdown)

    facts: list[str] = []
    decisions: list[Decision] = []

    from .... import framework_registry

    fw = task_config_of(breakdown).get("framework_name")
    _unit = framework_registry.primary_metric_unit(fw)
    if fig.final_tput:
        facts.append(f"Final: {framework_registry.format_primary_metric(fw, fig.final_tput, precision=2)}.")
        if fig.base_tput:
            base_v = framework_registry.primary_metric_value(fw, fig.base_tput)
            final_v = framework_registry.primary_metric_value(fw, fig.final_tput)
            if base_v is not None and final_v is not None:
                # For latency-based metrics, an improvement shows as a negative delta.
                note = " (negative = faster)" if framework_registry.is_scriptable(fw) else ""
                facts.append(f"Delta vs baseline: {final_v - base_v:+.2f} {_unit}{note}.")
    if fig.gain_v is not None:
        facts.append(f"Validated cumulative gain: {fmt_pct(fig.gain_v, plus=True)}{_gain_axis_note(fig.final)}.")
        decisions.append(
            Decision(
                kind="kept" if fig.gain_v > 0 else "attempted",
                subject="final",
                metric_pct=float(fig.gain_v),
                rationale=f"validated at stack_len={fig.val_stack_len} ts={fig.val_ts}",
            )
        )
    facts.extend(graded_axes_facts(fig.validation.get("perf"), label="Final"))
    geak_facts, warnings = _geak_candidate_standing(candidate, rebench, claim, fig.action_path)
    facts.extend(geak_facts)
    if fig.action_path:
        facts.append("Final stack: " + " → ".join(f"`{p}`" for p in fig.action_path))
    if fig.extra_args:
        facts.append(f"Final extra_server_args: `{fig.extra_args}`.")
    warnings.extend(
        _validation_gap_warnings(
            validated=fig.val_stack_len is not None,
            stack_changed=fig.stack_changed,
            gain_missing=fig.gain_v is None and bool(fig.final_tput or fig.base_tput),
        )
    )

    return RenderedSection(
        section_id="final",
        title="Final Result",
        key_facts=facts,
        markdown_block=_render_markdown(fig, candidate, fw, session),
        decisions=decisions,
        warnings=warnings,
        skipped=not (fig.final_tput or fig.gain_v),
    )
