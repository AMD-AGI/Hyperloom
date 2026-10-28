# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline measurement renderer.

Read off the ``baseline`` events on the V6 timeline, whose actions the
executor recorded as each measurement ran. The flat projection this replaces
was assembled at export time, which is why it had to reconstruct the latency
by walking ``runs/baseline/`` on disk and could pair a throughput from one
measurement with a TTFT from another.
"""

from __future__ import annotations

from typing import Any

from hyperloom.common.coerce import optional_positive_float, to_mapping
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    anchoring_baseline_action,
    quality_ref_actions,
)

from ..base import (
    Decision,
    RenderedSection,
    md_kv_list,
    md_table,
    register_renderer,
    session_of,
    task_config_of,
    timeline_of,
)
from ._agentx_metrics import graded_axes_facts, render_graded_axes
from ._invocation import render_invocation_block


def _attempts(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Every quality-reference baseline action, oldest first, including the failed ones."""
    return [row.action for row in quality_ref_actions(timeline_of(breakdown))]


def _anchor(breakdown: dict[str, Any]) -> dict[str, Any]:
    """The action ``outcome.baseline`` is published from, or ``{}`` when none produced a figure."""
    anchored = anchoring_baseline_action(timeline_of(breakdown))
    return anchored.action if anchored else {}


@register_renderer("baseline")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the baseline-measurement section.

    Surfaces baseline throughput, accuracy and latency, the attempts
    history, and the launch invocation block, emitting data-quality
    warnings (e.g. missing throughput or TTFT). Skipped when neither
    throughput nor attempts were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered baseline section.
    """
    attempts = _attempts(breakdown)
    anchor = _anchor(breakdown)
    measurement = to_mapping(anchor.get("measurement"))
    session = session_of(breakdown)
    tput = measurement.get("output_throughput")
    acc = measurement.get("accuracy")
    ttft = optional_positive_float(measurement.get("ttft_mean_ms"))
    e2el = measurement.get("e2el_mean_ms")
    ttft_source = str(measurement.get("ttft_e2el_source") or "")
    # The streak this measurement was dispatched under, which is the count of
    # baselines that had failed before it landed. Read off the anchor rather
    # than recounted here: the session's own counter is advanced by the
    # write-back, so the action states what it was dispatched into.
    fail_streak = int(to_mapping(anchor.get("request")).get("failure_streak_before") or 0)

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    from .... import framework_registry

    fw = task_config_of(breakdown).get("framework_name")
    if tput:
        facts.append(f"Baseline: {framework_registry.format_primary_metric(fw, tput, precision=2)}.")
        decisions.append(
            Decision(
                kind="attempted",
                subject="baseline",
                metric_pct=None,
                rationale=f"baseline_tput={float(tput):.2f}",
            )
        )
    else:
        warnings.append("No baseline_tput recorded — every subsequent gain is uncomputable.")
        decisions.append(
            Decision(
                kind="not_attempted",
                subject="baseline",
                rationale="no throughput captured",
            )
        )
    if acc:
        facts.append(f"Baseline accuracy: {float(acc):.4g}.")
    if ttft is not None:
        facts.append(f"Baseline TTFT mean: {ttft:.1f} ms.")
    elif measurement.get("ttft_mean_ms") is None:
        warnings.append(
            "ttft_mean_ms is null — the measured round reported no latency, so "
            "the baseline's throughput cannot be read against a per-request cost."
        )
    if e2el is not None:
        facts.append(f"Baseline e2el mean: {float(e2el):.1f} ms.")
    # The axes an AgentX session is actually ranked on. ``output_throughput`` above is the synthetic-workload
    # figure, so on an agentic round it is the one number a reader must not read the verdict off.
    perf = measurement.get("perf")
    facts.extend(graded_axes_facts(perf, label="Baseline"))
    if fail_streak:
        warnings.append(f"baseline_failure_streak={fail_streak} — baseline retried after failure(s).")
    if attempts:
        facts.append(f"Baseline attempts recorded: {len(attempts)}.")

    md_parts: list[str] = []
    md_parts.append(
        md_kv_list(
            [
                ("output_throughput", tput),
                ("throughput_unit", measurement.get("throughput_unit") or None),
                ("accuracy", acc),
                ("submission_valid", measurement.get("submission_valid")),
                # Only ever non-empty on a round upstream refused, which is the round whose axes a reader is most
                # likely to be puzzling over. ``md_kv_list`` drops the empty list, so an accepted round says nothing.
                ("submission_invalid_reasons", measurement.get("submission_invalid_reasons") or None),
                ("ttft_mean_ms", ttft),
                ("e2el_mean_ms", e2el),
                ("ttft_e2el_source", ttft_source),
                ("config_path", to_mapping(anchor.get("request")).get("config_path") or None),
                ("benchmark_report_path", measurement.get("benchmark_report_path") or None),
                ("failure_streak", fail_streak or None),
            ]
        )
    )
    graded_md = render_graded_axes(perf)
    if graded_md:
        md_parts.append("")
        md_parts.append("**AgentX graded axes**:")
        md_parts.append(graded_md)
    if attempts:
        rows = [
            [
                a.get("end_time") or a.get("start_time"),
                a.get("status"),
                a.get("decision"),
                to_mapping(a.get("measurement")).get("output_throughput"),
                to_mapping(a.get("failure")).get("error_class"),
            ]
            for a in attempts[:10]
        ]
        md_parts.append("")
        md_parts.append("**Baseline attempts** (first 10):")
        md_parts.append(
            md_table(
                ["ts", "status", "decision", "key_metric", "error_class"],
                rows,
            )
        )

    # Kept last: an empty invocation block then leaves only trailing newlines, which the strip below drops.
    md_parts.append("")
    md_parts.append(render_invocation_block(anchor.get("invocation"), session.get("image")))

    return RenderedSection(
        section_id="baseline",
        title="Baseline",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        decisions=decisions,
        warnings=warnings,
        skipped=not (tput or attempts),
    )
