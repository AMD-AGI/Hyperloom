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

from dataclasses import dataclass
from typing import Any

from hyperloom_common.coerce import optional_positive_float, to_mapping
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


@dataclass(frozen=True)
class _BaselineFigures:
    """The figures the baseline section reads off its anchoring measurement."""

    measurement: dict[str, Any]
    tput: Any
    acc: Any
    ttft: float | None
    e2el: Any
    ttft_source: str
    fail_streak: int
    perf: Any


def _read_figures(anchor: dict[str, Any]) -> _BaselineFigures:
    """Read the throughput, accuracy, latency and failure-streak figures off the anchoring action."""
    measurement = to_mapping(anchor.get("measurement"))
    # The streak this measurement was dispatched under, which is the count of
    # baselines that had failed before it landed. Read off the anchor rather
    # than recounted here: the session's own counter is advanced by the
    # write-back, so the action states what it was dispatched into.
    fail_streak = int(to_mapping(anchor.get("request")).get("failure_streak_before") or 0)
    return _BaselineFigures(
        measurement=measurement,
        tput=measurement.get("output_throughput"),
        acc=measurement.get("accuracy"),
        ttft=optional_positive_float(measurement.get("ttft_mean_ms")),
        e2el=measurement.get("e2el_mean_ms"),
        ttft_source=str(measurement.get("ttft_e2el_source") or ""),
        fail_streak=fail_streak,
        perf=measurement.get("perf"),
    )


def _render_markdown(
    anchor: dict[str, Any], fig: _BaselineFigures, attempts: list[dict[str, Any]], session: dict[str, Any]
) -> str:
    """Render the baseline key-value block, graded axes, attempts table and invocation block."""
    md_parts: list[str] = []
    md_parts.append(
        md_kv_list(
            [
                ("output_throughput", fig.tput),
                ("throughput_unit", fig.measurement.get("throughput_unit") or None),
                ("accuracy", fig.acc),
                ("submission_valid", fig.measurement.get("submission_valid")),
                # Only ever non-empty on a round upstream refused, which is the round whose axes a reader is most
                # likely to be puzzling over. ``md_kv_list`` drops the empty list, so an accepted round says nothing.
                ("submission_invalid_reasons", fig.measurement.get("submission_invalid_reasons") or None),
                ("ttft_mean_ms", fig.ttft),
                ("e2el_mean_ms", fig.e2el),
                ("ttft_e2el_source", fig.ttft_source),
                ("config_path", to_mapping(anchor.get("request")).get("config_path") or None),
                ("benchmark_report_path", fig.measurement.get("benchmark_report_path") or None),
                ("failure_streak", fig.fail_streak or None),
            ]
        )
    )
    graded_md = render_graded_axes(fig.perf)
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
    return "\n".join(md_parts).strip()


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
    fig = _read_figures(anchor)
    session = session_of(breakdown)

    facts: list[str] = []
    warnings: list[str] = []
    decisions: list[Decision] = []

    from .... import framework_registry

    fw = task_config_of(breakdown).get("framework_name")
    if fig.tput:
        facts.append(f"Baseline: {framework_registry.format_primary_metric(fw, fig.tput, precision=2)}.")
        decisions.append(
            Decision(
                kind="attempted",
                subject="baseline",
                metric_pct=None,
                rationale=f"baseline_tput={float(fig.tput):.2f}",
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
    if fig.acc:
        facts.append(f"Baseline accuracy: {float(fig.acc):.4g}.")
    if fig.ttft is not None:
        facts.append(f"Baseline TTFT mean: {fig.ttft:.1f} ms.")
    elif fig.measurement.get("ttft_mean_ms") is None:
        warnings.append(
            "ttft_mean_ms is null — the measured round reported no latency, so "
            "the baseline's throughput cannot be read against a per-request cost."
        )
    if fig.e2el is not None:
        facts.append(f"Baseline e2el mean: {float(fig.e2el):.1f} ms.")
    # The axes an AgentX session is actually ranked on. ``output_throughput`` above is the synthetic-workload
    # figure, so on an agentic round it is the one number a reader must not read the verdict off.
    facts.extend(graded_axes_facts(fig.perf, label="Baseline"))
    if fig.fail_streak:
        warnings.append(f"baseline_failure_streak={fig.fail_streak} — baseline retried after failure(s).")
    if attempts:
        facts.append(f"Baseline attempts recorded: {len(attempts)}.")

    return RenderedSection(
        section_id="baseline",
        title="Baseline",
        key_facts=facts,
        markdown_block=_render_markdown(anchor, fig, attempts, session),
        decisions=decisions,
        warnings=warnings,
        skipped=not (fig.tput or attempts),
    )
