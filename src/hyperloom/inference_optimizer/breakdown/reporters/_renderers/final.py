# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final / validated result renderer.

Read off ``outcome`` and ``close``. The throughput, its latency and the launch
that produced them all come from the one validation row that settled the
stack, so the block cannot pair a figure from one measurement with a latency
from another -- which is what the export's disk walk for the latency could do.
"""

from __future__ import annotations

from typing import Any

from ..base import (
    Decision,
    RenderedSection,
    as_dict,
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
from ._invocation import render_invocation_block

# ``geak_candidate.status`` values meaning the candidate was measured but its
# revalidation never landed, so the win was abandoned rather than judged.
_GEAK_DROPPED_STATUSES: frozenset[str] = frozenset({"rebench_cancelled", "rebench_unavailable"})

# ``geak.rebench.final_status`` values with the same meaning, but settled: the
# rebench ran and could not produce a verdict. ``no_material`` / ``no_promote``
# are deliberately absent — those ARE verdicts, so the candidate was judged.
_GEAK_DROPPED_RESULT_STATUSES: frozenset[str] = frozenset({"failed", "fallback_failed"})


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
    b = as_dict(outcome.get("baseline"))
    validation = validation_of(breakdown)
    session = session_of(breakdown)
    final_tput = f.get("throughput_tok_s_per_gpu")
    base_tput = b.get("throughput_tok_s_per_gpu")
    gain_v = f.get("gain_pct")
    val_stack_len = validation.get("validated_at_stack_len")
    val_ts = validation.get("validated_ts")
    stack_changed = bool(validation.get("stack_changed_after_validation"))
    extra_args = f.get("extra_server_args") or ""
    action_path = f.get("action_path") or []
    # The candidate's standing at close: no kernel event can hold it, since it
    # is settled after every one of them has closed.
    candidate = as_dict(close_of(breakdown).get("geak_candidate"))
    rebench, claim = _last_geak_rebench(breakdown)
    revalidation_pending = bool(candidate.get("revalidation_pending"))
    candidate_status = str(candidate.get("status") or "")
    pending_awaiting = candidate_status == "awaiting_rebench"
    # Gain is provisional when a cross-harness revalidation is pending with no confirmed validated number.
    is_provisional = revalidation_pending and not (isinstance(gain_v, (int, float)) and gain_v > 0)
    # Headline is unvalidated when a GEAK candidate is pending with no positive validated gain.
    headline_unvalidated = pending_awaiting and not (isinstance(gain_v, (int, float)) and gain_v > 0)

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    from .... import framework_registry

    fw = task_config_of(breakdown).get("framework_name")
    _unit = framework_registry.primary_metric_unit(fw)
    if final_tput:
        facts.append(f"Final: {framework_registry.format_primary_metric(fw, final_tput, precision=2)}.")
    if base_tput and final_tput:
        base_v = framework_registry.primary_metric_value(fw, base_tput)
        final_v = framework_registry.primary_metric_value(fw, final_tput)
        if base_v is not None and final_v is not None:
            # For latency-based metrics, an improvement shows as a negative delta.
            note = " (negative = faster)" if framework_registry.is_scriptable(fw) else ""
            facts.append(f"Delta vs baseline: {final_v - base_v:+.2f} {_unit}{note}.")
    if is_provisional:
        facts.append("Cumulative gain is PENDING same-harness revalidation; no validated number exists yet.")
        warnings.append(
            "The recorded gain basis is PROVISIONAL and cross-harness: measured by the "
            "delegated optimizer's harness against the orchestrator baseline, so "
            "no gain is reported here. A same-harness full-stack rebench is "
            "pending and will supply the validated number."
        )
    elif gain_v is not None and not headline_unvalidated:
        facts.append(f"Validated cumulative gain: {fmt_pct(gain_v, plus=True)}.")
        decisions.append(
            Decision(
                kind="kept" if (gain_v or 0) > 0 else "attempted",
                subject="final",
                metric_pct=float(gain_v),
                rationale=f"validated at stack_len={val_stack_len} ts={val_ts}",
            )
        )
    self_gain = candidate.get("self_reported_gain_pct")
    self_gain_str = fmt_pct(self_gain, plus=True) if isinstance(self_gain, (int, float)) else "unknown"
    if candidate_status in _GEAK_DROPPED_STATUSES:
        drop_reason = str(candidate.get("revalidation_error") or "").strip() or "reason not recorded"
        facts.append(
            f"GEAK candidate (self-reported {self_gain_str}) was DROPPED without "
            f"revalidation (status={candidate_status}, {drop_reason})."
        )
        warnings.append(
            "A measured GEAK e2e candidate was abandoned because its same-harness "
            f"revalidation could not land ({drop_reason}). It was never judged on "
            "merit, so this session's gain may understate what the optimizer "
            "actually found — the candidate's artefacts are on disk but absent "
            "from current_best / action_path / the validated gain."
        )
    elif pending_awaiting:
        facts.append(
            f"GEAK candidate (self-reported {self_gain_str}) is AWAITING a "
            "main-flow rebench — excluded from the headline gain and final stack "
            "until a measured rebench validates it."
        )
        warnings.append(
            "A GEAK(GEAK) e2e candidate self-reported a win but has NOT been "
            "confirmed by a same-harness main-flow rebench, so it is intentionally "
            "kept out of current_best / action_path / the validated gain. Its "
            "self-reported number is audit-only and must not be presented as the "
            "headline result."
        )
    # Last: a settled ``failed`` on the rebench outlives the candidate slot it
    # was recorded from, so a LIVE candidate in a later macro-cycle must win
    # over a terminal status left behind by an earlier one.
    elif _settled_failure(rebench, action_path):
        geak_gain = claim.get("self_reported_gain_pct")
        geak_gain_str = fmt_pct(geak_gain, plus=True) if isinstance(geak_gain, (int, float)) else "unknown"
        drop_reason = str(rebench.get("final_error") or "").strip() or "reason not recorded"
        facts.append(
            f"GEAK candidate (self-reported {geak_gain_str}) was DROPPED without "
            f"revalidation (status=rebench_{rebench.get('final_status')}, {drop_reason})."
        )
        warnings.append(
            "A measured GEAK e2e candidate was abandoned because its same-harness "
            f"revalidation failed ({drop_reason}). It was never judged on merit, "
            "so this session's gain may understate what the optimizer actually "
            "found — the candidate's artefacts are on disk but absent from "
            "current_best / action_path / the validated gain."
        )
    if action_path:
        facts.append("Final stack: " + " → ".join(f"`{p}`" for p in action_path))
    if extra_args:
        facts.append(f"Final extra_server_args: `{extra_args}`.")
    if stack_changed:
        warnings.append(
            "The last whole-stack validation predates the final adoptions, so "
            "the validated cumulative gain describes a shorter stack than the "
            "one that shipped."
        )
    if gain_v is None and not is_provisional and (final_tput or base_tput):
        warnings.append(
            "No validated gain is recorded while baseline/final throughput are set — no validation run completed."
        )

    md_kv = md_kv_list(
        [
            ("final_throughput_tok_s_per_gpu", final_tput),
            ("throughput_unit", _unit or None),
            ("validated_gain_pct", gain_v),
            ("revalidation_pending", revalidation_pending or None),
            ("geak_candidate", candidate or None),
            ("validated_at_stack_len", val_stack_len),
            ("validated_ts", val_ts),
            ("measurement_basis", validation.get("measurement_basis")),
            ("stack_changed_after_validation", stack_changed),
            ("extra_server_args", extra_args or None),
            ("action_path", action_path or None),
            ("ttft_mean_ms", f.get("ttft_mean_ms")),
            ("e2el_mean_ms", f.get("e2el_mean_ms")),
            ("ttft_e2el_source", validation.get("ttft_e2el_source")),
            ("extra_envs", f.get("extra_envs") or None),
        ]
    )

    md_parts = [md_kv]
    inv_md = render_invocation_block(f.get("invocation"), session.get("image"))
    if inv_md:
        md_parts.append("")
        md_parts.append(inv_md)

    return RenderedSection(
        section_id="final",
        title="Final Result",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        decisions=decisions,
        warnings=warnings,
        skipped=not (final_tput or gain_v),
    )
