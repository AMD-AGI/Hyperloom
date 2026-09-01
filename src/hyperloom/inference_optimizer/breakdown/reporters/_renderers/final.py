# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final / validated result renderer."""

from __future__ import annotations

from typing import Any

from ..base import (
    Decision,
    RenderedSection,
    fmt_pct,
    md_kv_list,
    register_renderer,
)
from ._invocation import render_invocation_block

# ``geak_pending.status`` values meaning the candidate was measured but its
# revalidation never landed, so the win was abandoned rather than judged.
_GEAK_DROPPED_PENDING_STATUSES: frozenset[str] = frozenset({"rebench_cancelled", "rebench_unavailable"})

# ``geak_result.revalidation_status`` values with the same meaning, but settled:
# the rebench ran and could not produce a verdict. ``no_material`` / ``no_promote``
# are deliberately absent — those ARE verdicts, so the candidate was judged.
_GEAK_DROPPED_RESULT_STATUSES: frozenset[str] = frozenset({"failed", "fallback_failed"})


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
    f = breakdown.get("final") or {}
    b = breakdown.get("baseline") or {}
    session = breakdown.get("session") or {}
    final_tput = f.get("throughput_tok_s_per_gpu")
    base_tput = b.get("throughput_tok_s_per_gpu")
    gain_v = f.get("cumulative_gain_pct_validated")
    val_stack_len = f.get("validated_at_stack_len")
    val_ts = f.get("validated_ts")
    stack_changed = bool(f.get("stack_changed_after_validation"))
    extra_args = f.get("extra_server_args") or ""
    action_path = f.get("action_path") or []
    revalidation_pending = bool(f.get("revalidation_pending"))
    # Self-reported GEAK candidate excluded from the headline; surfaced as an audit-only note.
    geak_pending = f.get("geak_pending") if isinstance(f.get("geak_pending"), dict) else {}
    geak = breakdown.get("geak") if isinstance(breakdown.get("geak"), dict) else {}
    pending_awaiting = geak_pending.get("status") == "awaiting_rebench"
    # Gain is provisional when a cross-harness revalidation is pending with no confirmed validated number.
    is_provisional = revalidation_pending and not (isinstance(gain_v, (int, float)) and gain_v > 0)
    # Headline is unvalidated when a GEAK candidate is pending with no positive validated gain.
    headline_unvalidated = pending_awaiting and not (isinstance(gain_v, (int, float)) and gain_v > 0)

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    from .... import framework_registry

    fw = (breakdown.get("workload") or {}).get("framework_name")
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
    geak_pending_status = str(geak_pending.get("status") or "")
    geak_revalidation_status = str(geak.get("revalidation_status") or "")
    # ``action_path`` entries are ``action`` or ``action:variant``.
    geak_in_final_stack = any(str(step).split(":", 1)[0] == "geak_e2e" for step in action_path)
    if geak_pending and geak_pending_status in _GEAK_DROPPED_PENDING_STATUSES:
        self_gain = geak_pending.get("self_reported_gain_pct")
        self_gain_str = fmt_pct(self_gain, plus=True) if isinstance(self_gain, (int, float)) else "unknown"
        drop_reason = str(geak_pending.get("revalidation_error") or "").strip() or "reason not recorded"
        facts.append(
            f"GEAK candidate (self-reported {self_gain_str}) was DROPPED without "
            f"revalidation (status={geak_pending_status}, {drop_reason})."
        )
        warnings.append(
            "A measured GEAK e2e candidate was abandoned because its same-harness "
            f"revalidation could not land ({drop_reason}). It was never judged on "
            "merit, so this session's gain may understate what the optimizer "
            "actually found — the candidate's artefacts are on disk but absent "
            "from current_best / action_path / the validated gain."
        )
    elif geak_pending and geak_pending_status == "awaiting_rebench":
        self_gain = geak_pending.get("self_reported_gain_pct")
        self_gain_str = fmt_pct(self_gain, plus=True) if isinstance(self_gain, (int, float)) else "unknown"
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
    # Last: a settled ``failed`` on ``geak_result`` outlives the pending slot it
    # was recorded from, so a LIVE candidate in a later macro-cycle must win over
    # a terminal status left behind by an earlier one. ``render.py`` orders the
    # same three cases the same way.
    #
    # ``not geak_in_final_stack`` is the same guard one step further: a 2b
    # rebench that failed stamps ``failed``, and nothing clears it when the 2a
    # GEAK-harness fallback then promotes the candidate for real. The claim
    # here is that the candidate is ABSENT from the final stack, so read that
    # off the stack rather than trusting a status no writer retracts.
    elif geak_revalidation_status in _GEAK_DROPPED_RESULT_STATUSES and not geak_in_final_stack:
        self_gain = geak.get("gain_pct")
        self_gain_str = fmt_pct(self_gain, plus=True) if isinstance(self_gain, (int, float)) else "unknown"
        drop_reason = str(geak.get("revalidation_error") or "").strip() or "reason not recorded"
        facts.append(
            f"GEAK candidate (self-reported {self_gain_str}) was DROPPED without "
            f"revalidation (status=rebench_{geak_revalidation_status}, {drop_reason})."
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
            "stack_changed_after_validation = true — optimization_stack grew "
            "after the last successful validate_stack run, so the reported "
            "cumulative_gain_pct_validated may be stale relative to the "
            "current best."
        )
    if gain_v is None and not is_provisional and (final_tput or base_tput):
        warnings.append(
            "cumulative_gain_pct_validated is null while baseline/final "
            "throughput are set — validate_stack never ran or the snapshot "
            "predates V2."
        )

    md_kv = md_kv_list(
        [
            ("final_throughput_tok_s_per_gpu", final_tput),
            ("throughput_unit", f.get("throughput_unit") or None),
            ("cumulative_gain_pct_validated", gain_v),
            ("revalidation_pending", revalidation_pending or None),
            ("geak_pending", geak_pending or None),
            ("validated_at_stack_len", val_stack_len),
            ("validated_ts", val_ts),
            ("stack_changed_after_validation", stack_changed),
            ("extra_server_args", extra_args or None),
            ("action_path", action_path or None),
            ("ttft_mean_ms", f.get("ttft_mean_ms")),
            ("e2el_mean_ms", f.get("e2el_mean_ms")),
            ("ttft_e2el_source", f.get("ttft_e2el_source") or None),
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
