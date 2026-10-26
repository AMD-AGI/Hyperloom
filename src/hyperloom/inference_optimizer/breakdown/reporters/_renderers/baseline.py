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

from ..base import (
    Decision,
    RenderedSection,
    as_dict,
    dict_rows,
    events_of,
    md_kv_list,
    md_table,
    measured_ms,
    register_renderer,
    session_of,
    task_config_of,
)
from ._agentx_metrics import graded_axes_facts, render_graded_axes
from ._invocation import render_invocation_block

#: Action statuses whose figure the session went on to use. ``degraded`` is a
#: baseline standing on its cold warmup round because the budget would not hold
#: the hot pass: knowingly depressed, but it is the number every later gain was
#: read against. Kept in step with the same set in ``collectors/v6.py``, which
#: publishes ``outcome.baseline`` off the same actions.
_ANCHORING_STATUSES = frozenset({"succeeded", "degraded"})


def _anchoring_actions(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Every action that was dispatched as a session baseline, oldest first.

    Three different dispatches reach the baseline executor and each lands an
    action on a ``baseline`` event: the genuine baseline, ``replay_warm_recipe``
    and the kernel lane's throughput-only probes. Only the first anchors the
    session, so the selection reads the action's own
    ``establishes_quality_ref`` rather than re-deciding from the task kind.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        list[dict[str, Any]]: The baseline actions, including the ones that
            failed -- those are the attempts history.
    """
    actions: list[tuple[str, dict[str, Any]]] = []
    for event in events_of(breakdown, "baseline"):
        for action in dict_rows(as_dict(event.get("ext")).get("actions")):
            if not as_dict(action.get("request")).get("establishes_quality_ref"):
                continue
            actions.append((str(action.get("start_time") or action.get("end_time") or ""), action))
    actions.sort(key=lambda row: row[0])
    return [action for _stamp, action in actions]


def _anchor(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """The measurement the session's gains were read against.

    Args:
        actions (list[dict[str, Any]]): The baseline actions, oldest first.

    Returns:
        dict[str, Any]: The latest action that produced a usable figure, or
            ``{}`` when none did. A baseline re-measured after an enablement
            fix legitimately re-anchors, so the latest wins; ordered on the
            actions' own end stamps because a re-measure in a later phase is a
            separate event.
    """
    usable = [action for action in actions if str(action.get("status") or "").strip().lower() in _ANCHORING_STATUSES]
    if not usable:
        return {}
    usable.sort(key=lambda action: str(action.get("end_time") or action.get("start_time") or ""))
    return usable[-1]


def _failure_streak(anchor: dict[str, Any]) -> int:
    """How many baselines had failed before the anchor was dispatched, as its own request recorded."""
    return int(as_dict(anchor.get("request")).get("failure_streak_before") or 0)


def _attempts_table(attempts: list[dict[str, Any]]) -> str:
    """The first ten baseline attempts, one row each."""
    rows = [
        [
            a.get("end_time") or a.get("start_time"),
            a.get("status"),
            a.get("decision"),
            as_dict(a.get("measurement")).get("output_throughput"),
            as_dict(a.get("failure")).get("error_class"),
        ]
        for a in attempts[:10]
    ]
    return md_table(
        ["ts", "status", "decision", "key_metric", "error_class"],
        rows,
    )


def _baseline_markdown(anchor: dict[str, Any], attempts: list[dict[str, Any]], session: dict[str, Any]) -> str:
    """The anchor's figures as a key/value list, then its graded axes, the attempts history and its launch."""
    measurement = as_dict(anchor.get("measurement"))
    md_parts: list[str] = []
    md_parts.append(
        md_kv_list(
            [
                ("output_throughput", measurement.get("output_throughput")),
                ("throughput_unit", measurement.get("throughput_unit") or None),
                ("accuracy", measurement.get("accuracy")),
                ("submission_valid", measurement.get("submission_valid")),
                # Only ever non-empty on a round upstream refused, which is the round whose axes a reader is most
                # likely to be puzzling over. ``md_kv_list`` drops the empty list, so an accepted round says nothing.
                ("submission_invalid_reasons", measurement.get("submission_invalid_reasons") or None),
                ("ttft_mean_ms", measured_ms(measurement.get("ttft_mean_ms"))),
                ("e2el_mean_ms", measurement.get("e2el_mean_ms")),
                ("ttft_e2el_source", str(measurement.get("ttft_e2el_source") or "") or None),
                ("config_path", as_dict(anchor.get("request")).get("config_path") or None),
                ("benchmark_report_path", measurement.get("benchmark_report_path") or None),
                ("failure_streak", _failure_streak(anchor) or None),
            ]
        )
    )
    graded_md = render_graded_axes(measurement.get("perf"))
    if graded_md:
        md_parts.append("")
        md_parts.append("**AgentX graded axes**:")
        md_parts.append(graded_md)
    if attempts:
        md_parts.append("")
        md_parts.append("**Baseline attempts** (first 10):")
        md_parts.append(_attempts_table(attempts))

    inv_md = render_invocation_block(anchor.get("invocation"), session.get("image"))
    if inv_md:
        md_parts.append("")
        md_parts.append(inv_md)
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
    attempts = _anchoring_actions(breakdown)
    anchor = _anchor(attempts)
    measurement = as_dict(anchor.get("measurement"))
    tput = measurement.get("output_throughput")
    acc = measurement.get("accuracy")
    ttft = measured_ms(measurement.get("ttft_mean_ms"))
    e2el = measurement.get("e2el_mean_ms")
    # The streak this measurement was dispatched under, which is the count of
    # baselines that had failed before it landed. Read off the anchor rather
    # than recounted here: the session's own counter is advanced by the
    # write-back, so the action states what it was dispatched into.
    fail_streak = _failure_streak(anchor)

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
    facts.extend(graded_axes_facts(measurement.get("perf"), label="Baseline"))
    if fail_streak:
        warnings.append(f"baseline_failure_streak={fail_streak} — baseline retried after failure(s).")
    if attempts:
        facts.append(f"Baseline attempts recorded: {len(attempts)}.")
    return RenderedSection(
        section_id="baseline",
        title="Baseline",
        key_facts=facts,
        markdown_block=_baseline_markdown(anchor, attempts, session_of(breakdown)),
        decisions=decisions,
        warnings=warnings,
        skipped=not (tput or attempts),
    )
