# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final / validated result renderer.

Read off ``outcome`` and ``close``. The throughput, its latency and the launch
that produced them all come from the one validation row that settled the
stack, so the block cannot pair a figure from one measurement with a latency
from another -- which is what the export's disk walk for the latency could do.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from hyperloom.common.perf_metric import INTVTY_OBJECTIVES

from .... import framework_registry
from ..base import (
    Decision,
    RenderedSection,
    as_dict,
    close_of,
    events_of,
    fmt_pct,
    md_kv_list,
    measured_ms,
    outcome_of,
    register_renderer,
    session_of,
    task_config_of,
    validation_of,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import GEAK_DROPPED_STATUSES
from ._agentx_metrics import graded_axes_facts, render_graded_axes
from ._invocation import render_invocation_block

# ``geak.rebench.final_status`` values with the same meaning, but settled: the
# rebench ran and could not produce a verdict. ``no_material`` / ``no_promote``
# are deliberately absent — those ARE verdicts, so the candidate was judged.
_GEAK_DROPPED_RESULT_STATUSES: frozenset[str] = frozenset({"failed", "fallback_failed"})


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
        geak = as_dict(as_dict(event.get("ext")).get("geak"))
        if not geak:
            continue
        rebench = as_dict(geak.get("rebench"))
        if rebench:
            return rebench, as_dict(geak.get("claim"))
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
    if str(rebench.get("final_status") or "") not in _GEAK_DROPPED_RESULT_STATUSES:
        return False
    return not any(str(step).split(":", 1)[0] == "geak_e2e" for step in action_path)


class _Note(NamedTuple):
    """A key fact and the warning that must be read beside it."""

    fact: str
    warning: str


def _self_reported_gain(record: dict[str, Any]) -> str:
    """The gain a GEAK record self-reported, as a signed percentage or ``unknown``."""
    gain = record.get("self_reported_gain_pct")
    return fmt_pct(gain, plus=True) if isinstance(gain, (int, float)) else "unknown"


def _dropped_note(gain: str, status: str, error: Any, revalidation: str) -> _Note:
    """The note on a GEAK candidate dropped before its same-harness revalidation could judge it."""
    reason = str(error or "").strip() or "reason not recorded"
    return _Note(
        fact=f"GEAK candidate (self-reported {gain}) was DROPPED without revalidation (status={status}, {reason}).",
        warning=(
            "A measured GEAK e2e candidate was abandoned because its same-harness "
            f"revalidation {revalidation} ({reason}). It was never judged on merit, "
            "so this session's gain may understate what the optimizer actually "
            "found — the candidate's artefacts are on disk but absent from "
            "current_best / action_path / the validated gain."
        ),
    )


def _geak_note(
    candidate: dict[str, Any], rebench: dict[str, Any], claim: dict[str, Any], action_path: list[Any]
) -> _Note | None:
    """The note on a GEAK candidate kept out of the final stack, or ``None`` when there is none."""
    gain = _self_reported_gain(candidate)
    status = str(candidate.get("status") or "")
    if status in GEAK_DROPPED_STATUSES:
        return _dropped_note(gain, status, candidate.get("revalidation_error"), "could not land")
    if status == "awaiting_rebench":
        return _Note(
            fact=(
                f"GEAK candidate (self-reported {gain}) is AWAITING a "
                "main-flow rebench — excluded from the headline gain and final stack "
                "until a measured rebench validates it."
            ),
            warning=(
                "A GEAK(GEAK) e2e candidate self-reported a win but has NOT been "
                "confirmed by a same-harness main-flow rebench, so it is intentionally "
                "kept out of current_best / action_path / the validated gain. Its "
                "self-reported number is audit-only and must not be presented as the "
                "headline result."
            ),
        )
    # Last: a settled ``failed`` on the rebench outlives the candidate slot it was recorded from, so a LIVE candidate
    # in a later macro-cycle must win over a terminal status left behind by an earlier one.
    if not _settled_failure(rebench, action_path):
        return None
    return _dropped_note(
        _self_reported_gain(claim), f"rebench_{rebench.get('final_status')}", rebench.get("final_error"), "failed"
    )


def _throughput_facts(fw: Any, final_tput: Any, base_tput: Any) -> list[str]:
    """The final figure on the framework's primary metric, and its delta against the baseline's."""
    facts: list[str] = []
    if final_tput:
        facts.append(f"Final: {framework_registry.format_primary_metric(fw, final_tput, precision=2)}.")
    if base_tput and final_tput:
        base_v = framework_registry.primary_metric_value(fw, base_tput)
        final_v = framework_registry.primary_metric_value(fw, final_tput)
        if base_v is not None and final_v is not None:
            # For latency-based metrics, an improvement shows as a negative delta.
            note = " (negative = faster)" if framework_registry.is_scriptable(fw) else ""
            unit = framework_registry.primary_metric_unit(fw)
            facts.append(f"Delta vs baseline: {final_v - base_v:+.2f} {unit}{note}.")
    return facts


def _shipped_stack_facts(action_path: list[Any], extra_args: Any) -> list[str]:
    """The stack the session shipped, and the extra server args it launched with."""
    facts: list[str] = []
    if action_path:
        facts.append("Final stack: " + " → ".join(f"`{p}`" for p in action_path))
    if extra_args:
        facts.append(f"Final extra_server_args: `{extra_args}`.")
    return facts


def _final_markdown(
    final: dict[str, Any], validation: dict[str, Any], candidate: dict[str, Any], fw: Any, session: dict[str, Any]
) -> str:
    """The result as a key/value list, then the axes it was graded on and the launch that produced it."""
    md_kv = md_kv_list(
        [
            ("final_output_throughput", final.get("output_throughput")),
            ("throughput_unit", framework_registry.throughput_unit(fw) or None),
            ("validated_gain_pct", final.get("gain_pct")),
            ("revalidation_pending", bool(candidate.get("revalidation_pending")) or None),
            ("geak_candidate", candidate or None),
            ("validated_at_stack_len", validation.get("validated_at_stack_len")),
            ("validated_ts", validation.get("validated_ts")),
            ("measurement_basis", validation.get("measurement_basis")),
            ("stack_changed_after_validation", bool(validation.get("stack_changed_after_validation"))),
            ("extra_server_args", final.get("extra_server_args") or None),
            ("action_path", final.get("action_path") or None),
            ("ttft_mean_ms", measured_ms(final.get("ttft_mean_ms"))),
            ("e2el_mean_ms", final.get("e2el_mean_ms")),
            ("ttft_e2el_source", validation.get("ttft_e2el_source")),
            ("extra_envs", final.get("extra_envs") or None),
        ]
    )

    md_parts = [md_kv]
    # The axes the validation round was graded on, beside the throughput figure above. On an AgentX session the two
    # answer different questions: the verdict was decided on these, not on ``final_output_throughput``.
    graded_md = render_graded_axes(validation.get("perf"))
    if graded_md:
        md_parts.append("")
        md_parts.append("**AgentX graded axes** (validation round):")
        md_parts.append(graded_md)
    inv_md = render_invocation_block(final.get("invocation"), session.get("image"))
    if inv_md:
        md_parts.append("")
        md_parts.append(inv_md)
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
    outcome = outcome_of(breakdown)
    f = as_dict(outcome.get("final"))
    validation = validation_of(breakdown)
    final_tput = f.get("output_throughput")
    base_tput = as_dict(outcome.get("baseline")).get("output_throughput")
    gain_v = f.get("gain_pct")
    val_stack_len = validation.get("validated_at_stack_len")
    action_path = f.get("action_path") or []
    # The candidate's standing at close: no kernel event can hold it, since it
    # is settled after every one of them has closed.
    candidate = as_dict(close_of(breakdown).get("geak_candidate"))
    rebench, claim = _last_geak_rebench(breakdown)
    fw = task_config_of(breakdown).get("framework_name")

    facts = _throughput_facts(fw, final_tput, base_tput)
    decisions: list[Decision] = []
    if gain_v is not None:
        facts.append(f"Validated cumulative gain: {fmt_pct(gain_v, plus=True)}{_gain_axis_note(f)}.")
        decisions.append(
            Decision(
                kind="kept" if (gain_v or 0) > 0 else "attempted",
                subject="final",
                metric_pct=float(gain_v),
                rationale=f"validated at stack_len={val_stack_len} ts={validation.get('validated_ts')}",
            )
        )
    facts.extend(graded_axes_facts(validation.get("perf"), label="Final"))
    warnings: list[str] = []
    geak = _geak_note(candidate, rebench, claim, action_path)
    if geak is not None:
        facts.append(geak.fact)
        warnings.append(geak.warning)
    facts.extend(_shipped_stack_facts(action_path, f.get("extra_server_args")))
    warnings.extend(
        _validation_gap_warnings(
            validated=val_stack_len is not None,
            stack_changed=bool(validation.get("stack_changed_after_validation")),
            gain_missing=gain_v is None and bool(final_tput or base_tput),
        )
    )
    return RenderedSection(
        section_id="final",
        title="Final Result",
        key_facts=facts,
        markdown_block=_final_markdown(f, validation, candidate, fw, session_of(breakdown)),
        decisions=decisions,
        warnings=warnings,
        skipped=not (final_tput or gain_v),
    )
