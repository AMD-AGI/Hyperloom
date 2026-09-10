#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a Hyperloom session's per-LLM-call ledger as a structured HTML report.

``dump_llm_call_report`` already walks the same ledgers and prints the full
phase -> task tree. That report is exhaustive, and exhaustive is the problem: it
answers "what happened" in a few thousand rows and leaves "where did the money
go, and what did it buy" to the reader. This renderer takes the identical input
and arranges it as a small number of questions, each with its own section:

* what the session *bought* -- baseline to final, and which source the validated
  gain is attributed to (``session_breakdown.json``);
* what each phase *cost* to buy it, and so the cost per +1% throughput;
* inside an expensive phase, which task path carries the spend;
* where in a conversation the cost is incurred, and which models did the work.

This is the Hyperloom-wide counterpart of :mod:`render_geak_html_report`, which
covers a standalone GEAK run. Where that report sees one kernel agent, this one
sees every phase of a session -- PRELUDE, SWEEP, FRAMEWORK_AGENT, KERNEL_AGENT
and whatever else the ledger carries -- because a session's bill is not spent in
the kernel phase alone. Both pages share their formatting (:mod:`_report_html`)
so a number means the same thing on either.

Three rules hold throughout, the same three the other reports follow:

* nothing is modelled or extrapolated -- every number is computed from rows in
  the ledgers, and the coverage banner says what the ledgers do not contain;
* a quantity that was not recorded renders as "not recorded", never as zero;
* spend is attributed to a phase only where the ledger says so. Gain is
  attributed only where ``session_breakdown.json`` says so, through the declared
  map in :data:`ATTRIBUTION_PHASES` -- and any phase outside it is shown as
  unattributed rather than credited with someone else's win.

Usage::

    python3 -m hyperloom.inference_optimizer.tools.render_hyperloom_html_report \\
        --session-dir <SESSION> -o <SESSION>/reports/hyperloom_report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.tools._report_html import (
    CSS,
    bar,
    esc,
    fmt_hms,
    fmt_int,
    fmt_pct,
    fmt_usd,
    num,
    sparkline,
)
from hyperloom.inference_optimizer.tools.dump_llm_call_report import Node, build_tree, call_identity, load_ledgers
from hyperloom.inference_optimizer.tools import _report_agents as agents_mod
from hyperloom.inference_optimizer.tools._report_agents import UNLABELLED
from hyperloom.inference_optimizer.tools.render_geak_html_report import CALLS_FILENAME as GEAK_LEDGER
from hyperloom.inference_optimizer.tools.render_geak_html_report import derive_role as derive_geak_role
from hyperloom.inference_optimizer.tools.render_geak_html_report import load_calls as load_geak_ledger

BREAKDOWN_FILENAME = "session_breakdown.json"
DEFAULT_OUTPUT = "hyperloom_report.html"

#: How ``outcome.validation.attribution.by_source`` names map onto ledger phases.
#: The breakdown attributes gain to a *source*; the ledger attributes spend to a
#: *phase*. Joining the two needs this map, and the map is declared here rather
#: than guessed per run so that a phase it does not cover is reported as
#: unattributed instead of being quietly credited with another phase's gain.
ATTRIBUTION_PHASES: dict[str, tuple[str, ...]] = {
    "framework_agent": ("FRAMEWORK_AGENT",),
    "kernel": ("KERNEL_AGENT", "GEAK", "FORGE"),
    "warm_replay": ("WARM_REPLAY",),
}

#: Decisions in ``phase_timeline`` that mean the change was kept.
KEEP_DECISIONS = {"KEEP", "PROMOTE", "promoted"}

#: The ledger component whose rows the GEAK harvester writes into ``ext/``. When
#: GEAK's own ledger is grafted in, these are dropped so the spend is not counted
#: twice -- the harvested rows and the grafted ones describe the same API calls.
GEAK_COMPONENT = "geak"

#: The phase a grafted GEAK run is nested under. GEAK is invoked as this phase.
GEAK_PHASE = "KERNEL_AGENT"


def load_breakdown(path: Path) -> dict[str, Any] | None:
    """Read ``session_breakdown.json``, or return ``None`` when it is unusable.

    Args:
        path: Path to the breakdown file.

    Returns:
        The parsed object, or ``None`` if it is missing or not a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def phase_records(root: Node) -> list[dict[str, Any]]:
    """One record per phase, heaviest spend first.

    Args:
        root: The tree root from :func:`build_tree`.

    Returns:
        A list of ``{phase, totals, node}`` dicts.
    """
    records = [{"phase": kid.name, "totals": kid.total, "node": kid} for kid in root.children.values()]
    records.sort(key=lambda r: (-r["totals"].usd_total, -r["totals"].calls, r["phase"]))
    return records


def call_rows(turns: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The flat list of billable rows, counted the way :func:`build_tree` counts them.

    A turn that expanded into detail rows is represented by those rows only; a
    turn with none is represented by itself. That is the tree's own rule, so a
    figure derived here always agrees with the tree's totals rather than
    double-counting the turns that have a per-call breakdown.

    Args:
        turns: Turn-level rows.
        details: Per-API-call rows.

    Returns:
        The rows that carry spend.
    """
    detailed = {call_identity(r) for r in details}
    rows = list(details)
    rows += [r for r in turns if call_identity(r) not in detailed]
    return rows


def growth_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median ISL and mean cost by position within a conversation.

    A long conversation re-sends its history on every turn, so the call index is
    the axis along which context -- and therefore cost -- grows. Buckets with
    fewer than three calls are dropped: a median of one call is not a median,
    and a curve that tails off into single samples invites over-reading.

    Args:
        rows: Billable rows.

    Returns:
        ``[{index, calls, isl_median, usd_mean}]`` ordered by index.
    """
    by_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        index = int(num(row.get("api_call_index")) or num(row.get("turn")))
        if index <= 0:
            continue
        by_index[index].append((num(row.get("isl")), num(row.get("cost_usd"))))

    out: list[dict[str, Any]] = []
    for index in sorted(by_index):
        bucket = by_index[index]
        if len(bucket) < 3:
            continue
        isls = sorted(item[0] for item in bucket)
        mid = len(isls) // 2
        median = isls[mid] if len(isls) % 2 else (isls[mid - 1] + isls[mid]) / 2
        out.append(
            {
                "index": index,
                "calls": len(bucket),
                "isl_median": median,
                "usd_mean": sum(item[1] for item in bucket) / len(bucket),
            }
        )
    return out


def model_mix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spend and volume per model, dearest first.

    Args:
        rows: Billable rows.

    Returns:
        ``[{model, calls, usd, isl, osl}]``.
    """
    acc: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "usd": 0.0, "isl": 0.0, "osl": 0.0})
    for row in rows:
        entry = acc[str(row.get("model") or "unrecorded")]
        entry["calls"] += 1
        entry["usd"] += num(row.get("cost_usd"))
        entry["isl"] += num(row.get("isl"))
        entry["osl"] += num(row.get("osl"))
    out = [{"model": name, **vals} for name, vals in acc.items()]
    out.sort(key=lambda r: (-r["usd"], -r["calls"], r["model"]))
    return out


def outcome_ladder(breakdown: dict[str, Any] | None) -> dict[str, Any]:
    """The measured throughput ladder and the gain attribution, as recorded.

    Nothing is recomputed here. The breakdown is the harness's own record of
    what it measured; this only reshapes it for display, and a field the harness
    did not write stays absent so the page can say it was not measured.

    Args:
        breakdown: Parsed ``session_breakdown.json``, or ``None``.

    Returns:
        A dict with ``present`` and, when present, the ladder and attribution.
    """
    if not breakdown:
        return {"present": False, "steps": [], "by_source": {}, "notes": []}

    outcome = breakdown.get("outcome") or {}
    final = outcome.get("final") or {}
    validation = outcome.get("validation") or {}
    attribution = validation.get("attribution") or {}

    steps = [
        {
            "phase": str(row.get("phase") or ""),
            "action": str(row.get("action") or ""),
            "decision": str(row.get("decision") or ""),
            "metric": row.get("key_metric"),
            "metric_kind": str(row.get("key_metric_kind") or ""),
            "ts": str(row.get("ts") or ""),
        }
        for row in breakdown.get("phase_timeline") or []
        if isinstance(row, dict)
    ]

    return {
        "present": True,
        "baseline": outcome.get("baseline") or {},
        "final": final,
        "gain_pct": final.get("gain_pct"),
        "status": str(outcome.get("status") or ""),
        "stop_reason": str(outcome.get("stop_reason") or ""),
        "stage_reached": str(outcome.get("stage_reached") or ""),
        "action_path": [str(a) for a in (final.get("action_path") or [])],
        "steps": steps,
        "by_source": attribution.get("by_source") or {},
        "attribution_available": bool(attribution.get("available")),
        "unattributed_pct": validation.get("unattributed_gain_pct"),
        "gap_pct": validation.get("reconciliation_gap_pct"),
        "notes": [str(n) for n in (validation.get("notes") or [])],
    }


def join_gain(phases: list[dict[str, Any]], ladder: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach each phase's attributed gain, or mark it unattributed.

    Args:
        phases: Records from :func:`phase_records`.
        ladder: The result of :func:`outcome_ladder`.

    Returns:
        The same records with ``source``, ``gain_pct`` and ``timeline_keeps``.
    """
    by_phase: dict[str, dict[str, Any]] = {}
    for source, payload in (ladder.get("by_source") or {}).items():
        if not isinstance(payload, dict):
            continue
        for phase_name in ATTRIBUTION_PHASES.get(source, ()):
            by_phase[phase_name] = {
                "source": source,
                "gain_pct": payload.get("total_gain_pct"),
                "keeps": payload.get("keep_count"),
            }

    keeps: dict[str, int] = defaultdict(int)
    for step in ladder.get("steps") or []:
        if step["decision"] in KEEP_DECISIONS and step["phase"]:
            keeps[step["phase"]] += 1

    joined = []
    for record in phases:
        hit = by_phase.get(record["phase"])
        joined.append(
            {
                **record,
                "source": (hit or {}).get("source"),
                "gain_pct": (hit or {}).get("gain_pct"),
                "keeps": (hit or {}).get("keeps"),
                "timeline_keeps": keeps.get(record["phase"], 0),
            }
        )
    return joined


def session_identity(session_dir: Path, breakdown: dict[str, Any] | None) -> dict[str, str]:
    """Name the session and its model, preferring what the run recorded.

    ``metadata.session.session_dir`` records the *writer's* mount, which is not
    necessarily the path the report is generated from, so the directory on disk
    is a fallback rather than the authority.

    Args:
        session_dir: The session root as read.
        breakdown: Parsed breakdown, or ``None``.

    Returns:
        ``{session_id, model, revision, elapsed_min}``, "" for anything absent.
    """
    meta = ((breakdown or {}).get("metadata") or {}).get("session") or {}
    model = ""
    for source in ((breakdown or {}).get("session_meta") or {}, (breakdown or {}).get("model_info") or {}):
        if not isinstance(source, dict):
            continue
        for key in ("model_name", "model", "model_type"):
            if source.get(key):
                model = str(source[key])
                break
        if model:
            break
    return {
        "session_id": str(meta.get("session_id") or session_dir.name),
        "model": model or session_dir.parent.name,
        "revision": str(meta.get("code_revision") or ""),
        "elapsed_min": str(meta.get("elapsed_minutes") or ""),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #



def agent_identity(row: dict[str, Any]) -> tuple[str, str]:
    """The conversation a row belongs to, and the readable name for it.

    Hyperloom's ledger has no agent column. What it has is ``task_path``, which
    for a specialist reads ``specialist/<instance id>/turn-N`` -- so the instance
    is the conversation and ``turn-N`` is a position inside it -- and for
    everything else is either a bare name (``critic``) or absent. ``role`` is the
    readable name, but it is written by some producers and not others: on the
    runs seen so far it is present on orchestration and critic rows and empty on
    every specialist row. So identity comes from the task path, which is
    populated, and the name from ``role`` when the producer recorded one.

    Args:
        row: A ledger row.

    Returns:
        ``(agent id, role)``. The role is :data:`UNLABELLED` when nothing
        recorded one -- never inferred from the id.
    """
    path = str(row.get("task_path") or "").strip("/")
    component = str(row.get("component") or "").strip()
    segments = [seg for seg in path.split("/") if seg]
    if len(segments) >= 2:
        agent = "/".join(segments[:2])
    elif segments:
        agent = segments[0]
    else:
        agent = component or "unattributed"
    role = str(row.get("role") or "").strip()
    if not role:
        role = component if component and component != agent else UNLABELLED
    return agent, role


def normalise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape ledger rows into the record the shared deep dive works on.

    Ordering within a conversation is ``(turn, api_call_index, ts)`` and the
    position is then renumbered from zero, so the position curve measures where
    a call sat in its own conversation rather than in the session.

    Args:
        rows: Billable rows from :func:`call_rows`.

    Returns:
        Normalised rows, ordered by agent then position.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[agent_identity(row)[0]].append(row)

    out: list[dict[str, Any]] = []
    for agent, calls in grouped.items():
        calls.sort(key=lambda r: (num(r.get("turn")), num(r.get("api_call_index")), str(r.get("ts") or "")))
        for index, row in enumerate(calls):
            isl = num(row.get("isl")) or (
                num(row.get("input_tokens"))
                + num(row.get("cache_read_input_tokens"))
                + num(row.get("cache_creation_input_tokens"))
            )
            thinking = num(row.get("reasoning_output_tokens"))
            # Hyperloom's ledger puts thinking beside output, not inside it.
            osl = num(row.get("osl")) or (num(row.get("output_tokens")) + thinking)
            tools = [
                str(t.get("tool"))
                for t in (row.get("tool_calls") or [])
                if isinstance(t, dict) and t.get("tool")
            ]
            out.append(
                {
                    "phase": str(row.get("phase") or "unattributed"),
                    "agent": agent,
                    "role": agent_identity(row)[1],
                    "call_index": index,
                    "ts": row.get("ts"),
                    "isl": isl,
                    "osl": osl,
                    "thinking": thinking,
                    "usd": num(row.get("cost_usd")),
                    "dt_s": num(row.get("latency_ms")) / 1000.0,
                    "tools": tools,
                    "model": row.get("model"),
                    "component": row.get("component"),
                    # Hyperloom records no prompt text; the drill-down column stays blank
                    # rather than being filled with something the ledger did not say.
                    "prompt": "",
                }
            )
    return out


def hyperloom_role_of(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Name an agent from what its rows recorded, in that order of preference.

    A recorded ``role`` wins wherever the producer wrote one. Grafted GEAK rows
    carry prompt text instead, so they fall back to the role sentence every GEAK
    prompt opens with -- the same derivation GEAK's own report uses, so an agent
    is named identically on both pages. Nothing else is inferred.

    Args:
        calls: One agent's calls, in order.

    Returns:
        The identity mapping :func:`agents_mod.agent_records` merges in.
    """
    for call in calls:
        role = str(call.get("role") or "").strip()
        if role and role != UNLABELLED:
            return {"role": role, "specialty": None, "round": None, "derived": False}
    prompt = str(calls[0].get("prompt") or "") if calls else ""
    if prompt:
        return derive_geak_role(prompt)
    return {"role": UNLABELLED, "specialty": None, "round": None, "derived": False}


def geak_eval_dir(session_dir: Path) -> Path | None:
    """Where the session's GEAK cycle wrote its own artifacts, as the run recorded it.

    Read from the handoff records rather than guessed from the directory layout,
    because a resumed session can carry more than one cycle and only the record
    says which one ran.

    Args:
        session_dir: Session root.

    Returns:
        The eval directory, or ``None`` when no record names one.
    """
    for name in ("geak/result.json", "geak/handoff.json", "state.json"):
        try:
            data = json.loads((session_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        candidate = data.get("eval_dir") or (data.get("geak") or {}).get("eval_dir")
        if isinstance(candidate, str) and candidate:
            path = Path(candidate)
            if path.is_dir():
                return path
    return None


def graft_geak(session_dir: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace the harvested GEAK summary rows with GEAK's own per-call ledger.

    GEAK is invoked as the ``KERNEL_AGENT`` phase and has historically been the
    majority of a session's bill, but the session ledger sees it from outside:
    the harvester writes a handful of turn rows into an ``ext`` shard, so the
    phase that cost the most is the phase with the least to show. GEAK writes a
    full per-call ledger of its own, and when the run left one behind this
    grafts it in so ``KERNEL_AGENT`` drills down to GEAK's own agents.

    The harvested rows are dropped in the same step. They describe the same API
    calls as the grafted ones, so keeping both would count GEAK's spend twice.

    Args:
        session_dir: Session root.
        rows: Normalised session rows.

    Returns:
        ``(rows, status)`` -- the status says what was grafted, or why not, and
        is rendered in the coverage banner rather than left implicit.
    """
    eval_dir = geak_eval_dir(session_dir)
    status: dict[str, Any] = {
        "eval_dir": str(eval_dir) if eval_dir else "",
        "grafted": False,
        "calls": 0,
        "usd": 0.0,
        "dropped_harvested": 0,
        "reason": "",
    }
    harvested = [r for r in rows if str(r.get("component") or "") == GEAK_COMPONENT]
    if eval_dir is None:
        status["reason"] = "no GEAK eval directory is named in this session's records"
        return rows, status

    ledger = eval_dir / "reports" / GEAK_LEDGER
    geak_rows = load_geak_ledger(ledger)
    if not geak_rows:
        status["reason"] = (
            f"GEAK left no <code>{esc(GEAK_LEDGER)}</code> under <code>{esc(eval_dir / 'reports')}</code>. "
            "A GEAK build without the trace mirror writes its call ledger only into the Claude config "
            "home, which does not survive the container; run <code>dump_geak_call_report</code> against "
            "a surviving home to rebuild it."
        )
        return rows, status

    kept = [r for r in rows if str(r.get("component") or "") != GEAK_COMPONENT]
    grafted = [
        {**row, "phase": f"{GEAK_PHASE} / {row.get('phase') or 'unattributed'}", "component": GEAK_COMPONENT}
        for row in geak_rows
    ]
    status.update(
        {
            "grafted": True,
            "calls": len(grafted),
            "usd": sum(num(r.get("usd")) for r in grafted),
            "dropped_harvested": len(harvested),
            "dropped_usd": sum(num(r.get("usd")) for r in harvested),
        }
    )
    return kept + grafted, status


def _share(part: float, whole: float) -> str:
    """A share of a total, rendered as a bar plus its percentage."""
    fraction = (part / whole) if whole else 0.0
    return f'{bar(fraction)} <span class="mut">{fraction * 100:.1f}%</span>'


def _headline_cards(ident: dict[str, str], totals: Any, ladder: dict[str, Any]) -> str:
    """The four numbers a reader wants before they scroll."""
    gain = ladder.get("gain_pct") if ladder.get("present") else None
    gain_html = fmt_pct(gain) if gain is not None else '<span class="none">not recorded</span>'
    tone = "good" if num(gain) > 0 else ("bad" if num(gain) < 0 else "")
    return (
        '<div class="cards">'
        f'<div class="card"><h3>Validated gain</h3><p class="big {tone}">{gain_html}</p>'
        f'<p class="mut">{esc(ladder.get("stop_reason") or "outcome not recorded")}</p></div>'
        f'<div class="card"><h3>LLM spend</h3><p class="big">{fmt_usd(totals.usd_total)}</p>'
        f'<p class="mut">{fmt_int(totals.calls)} API calls over {fmt_int(totals.turns)} turns</p></div>'
        f'<div class="card"><h3>Tokens in / out</h3><p class="big">{fmt_int(totals.isl)}</p>'
        f'<p class="mut">{fmt_int(totals.osl)} out -- a {num(totals.isl) / (num(totals.osl) or 1):,.0f}:1 ratio</p></div>'
        f'<div class="card"><h3>Agent time</h3><p class="big">{fmt_hms(totals.ms_total / 1000.0)}</p>'
        f'<p class="mut">summed over calls, not wall clock</p></div>'
        "</div>"
    )


def _coverage_section(
    cov: dict[str, Any],
    breakdown_seen: bool,
    agents: list[dict[str, Any]],
    geak: dict[str, Any],
) -> str:
    """The banner that says what these numbers are and are not. Always rendered."""
    caveats: list[str] = []
    unlabelled = agents_mod.unlabelled_count(agents)
    if unlabelled:
        caveats.append(
            f"{fmt_int(unlabelled)} of {fmt_int(len(agents))} agents carry no <code>role</code>: that "
            "field is written by some producers and not others, so those agents are identified by their "
            "task path and shown as <i>unlabelled</i> rather than given an inferred name."
        )
    if geak.get("grafted"):
        caveats.append(
            f"GEAK's own per-call ledger is grafted in beneath <code>{esc(GEAK_PHASE)}</code>: "
            f"{fmt_int(geak['calls'])} calls worth {fmt_usd(geak['usd'])}, replacing the "
            f"{fmt_int(geak['dropped_harvested'])} harvested summary rows "
            f"({fmt_usd(geak.get('dropped_usd', 0.0))}) that describe the same calls. The headline "
            "cards above are the session ledger's own totals and still carry the harvested figure, so "
            "the two differ by whatever the harvest missed."
        )
    elif geak.get("reason"):
        caveats.append(f"No GEAK detail beneath <code>{esc(GEAK_PHASE)}</code>: {geak['reason']}")
    if cov.get("turns_without_detail"):
        caveats.append(
            f"{fmt_int(cov['turns_without_detail'])} of {fmt_int(cov['turns_total'])} turns wrote no per-call "
            f"detail rows and {'is' if cov['turns_without_detail'] == 1 else 'are'} counted once from the turn "
            "row, so their internal API calls are not separable here."
        )
    if cov.get("calls_unpriced"):
        caveats.append(
            f"{fmt_int(cov['calls_unpriced'])} calls could not be priced and are <b>left out of the USD "
            "total</b> rather than counted as free -- the totals below are therefore a floor."
        )
    if cov.get("calls_untimed"):
        caveats.append(f"{fmt_int(cov['calls_untimed'])} calls carry no timing, so agent time is a floor too.")
    if cov.get("calls_timing_apportioned"):
        caveats.append(
            f"{fmt_int(cov['calls_timing_apportioned'])} calls had their thinking/output time split "
            "apportioned from the token ratio rather than measured off stream events."
        )
    if cov.get("detail_rows_orphaned"):
        caveats.append(
            f"{fmt_int(cov['detail_rows_orphaned'])} detail rows match no turn row -- an out-of-process "
            "child writing through an <code>ext</code> shard, or a producer that recorded neither a "
            "call id nor a task path. They are counted once, and their turn-level context is unknown."
        )
    if not breakdown_seen:
        caveats.append(
            f"No <code>{BREAKDOWN_FILENAME}</code> was readable, so this page can say what the session "
            "<b>cost</b> but not what it <b>bought</b>."
        )
    sources = ", ".join(f"{esc(k)} ({fmt_int(v)})" for k, v in (cov.get("cost_sources") or {}).items())
    body = "".join(f"<li>{c}</li>" for c in caveats) or "<li>No gaps recorded: every call is priced and timed.</li>"
    return (
        '<h2 id="coverage">What these numbers cover</h2>'
        '<p class="lede">Every figure on this page is computed from the rows in this session\'s ledgers. '
        "Nothing is modelled, and nothing is carried over from another session.</p>"
        f'<ul class="caveats">{body}</ul>' + (f'<p class="mut">Cost sources: {sources}.</p>' if sources else "")
    )


def _outcome_section(ladder: dict[str, Any]) -> str:
    """What the session bought: the measured ladder, then who is credited with it."""
    if not ladder.get("present"):
        return (
            '<h2 id="bought">What the session bought</h2>'
            f'<p class="lede">No readable <code>{BREAKDOWN_FILENAME}</code>, so the outcome is '
            "<b>not recorded here</b>. That is not the same as no gain, and it is not reported as zero.</p>"
        )

    base = num((ladder.get("baseline") or {}).get("throughput_tok_s_per_gpu"))
    final = num((ladder.get("final") or {}).get("throughput_tok_s_per_gpu"))
    step_rows = []
    for step in ladder["steps"]:
        tone = "good" if step["decision"] in KEEP_DECISIONS else "mut"
        metric = f"{num(step['metric']):,.2f}" if step["metric"] is not None else "--"
        stamp = step["ts"][11:19] if len(step["ts"]) > 18 else step["ts"]
        step_rows.append(
            "<tr><td>"
            + esc(step["phase"] or "--")
            + "</td><td>"
            + esc(step["action"])
            + "</td>"
            + f'<td class="{tone}">'
            + esc(step["decision"])
            + "</td>"
            + f'<td class="mono">{metric}</td>'
            + '<td class="mut">'
            + esc(step["metric_kind"] or "")
            + "</td>"
            + '<td class="mono mut">'
            + esc(stamp)
            + "</td></tr>"
        )
    rows = "".join(step_rows)

    no_phase = '<span class="none">no phase mapped</span>'
    credit = "".join(
        f"<tr><td>{esc(source)}</td>"
        f"<td>{', '.join(ATTRIBUTION_PHASES.get(source, ())) or no_phase}</td>"
        f"<td>{fmt_pct(payload.get('total_gain_pct'))}</td>"
        f"<td>{fmt_int(payload.get('keep_count'))}</td></tr>"
        for source, payload in sorted(
            ((k, v) for k, v in ladder["by_source"].items() if isinstance(v, dict)),
            key=lambda kv: -num(kv[1].get("total_gain_pct")),
        )
    )

    notes = "".join(f"<li>{esc(n)}</li>" for n in ladder.get("notes") or [])
    return (
        '<h2 id="bought">What the session bought</h2>'
        f'<p class="lede">Baseline <b>{base:,.2f}</b> to final <b>{final:,.2f}</b> tok/s/GPU '
        f"({fmt_pct(ladder.get('gain_pct'))}), status <code>{esc(ladder.get('status'))}</code>, stopped because "
        f"<code>{esc(ladder.get('stop_reason'))}</code> at stage <code>{esc(ladder.get('stage_reached'))}</code>."
        + (
            f" The kept path was <code>{esc(' -> '.join(ladder['action_path']))}</code>."
            if ladder.get("action_path")
            else ""
        )
        + "</p>"
        "<h3>Who is credited with the gain</h3>"
        '<p class="mut">The harness attributes gain to a <i>source</i>; the ledger attributes spend to a '
        "<i>phase</i>. The middle column is the declared map between them -- a source with no phase mapped "
        "cannot be joined to spend, and is shown rather than dropped.</p>"
        '<table class="grid"><thead><tr><th>Source</th><th>Ledger phase</th><th>Attributed gain</th>'
        f"<th>Keeps</th></tr></thead><tbody>{credit or '<tr><td colspan=4>none recorded</td></tr>'}</tbody></table>"
        f'<p class="mut">Unattributed {fmt_pct(ladder.get("unattributed_pct"))}, reconciliation gap '
        f"{fmt_pct(ladder.get('gap_pct'))}.</p>"
        + (f'<ul class="caveats">{notes}</ul>' if notes else "")
        + "<h3>Every decision the session recorded</h3>"
        '<table class="grid"><thead><tr><th>Phase</th><th>Action</th><th>Decision</th><th>Key metric</th>'
        f"<th>Kind</th><th>UTC</th></tr></thead><tbody>{rows or '<tr><td colspan=6>none</td></tr>'}"
        "</tbody></table>"
    )


def _phase_table(joined: list[dict[str, Any]], total_usd: float) -> str:
    """The money question: what each phase cost, and what it bought."""
    rows = []
    for record in joined:
        totals = record["totals"]
        gain = record.get("gain_pct")
        if gain is None:
            bought = '<span class="none">not attributed</span>'
            per_pct = "--"
        else:
            bought = f'<span class="{"good" if num(gain) > 0 else "mut"}">{fmt_pct(gain)}</span>'
            per_pct = fmt_usd(totals.usd_total / num(gain)) if num(gain) > 0 else '<span class="none">n/a</span>'
        rows.append(
            f"<tr><td><b>{esc(record['phase'])}</b>"
            f'<div class="mut">{fmt_int(record["timeline_keeps"])} kept decisions</div></td>'
            f"<td>{fmt_usd(totals.usd_total)}</td><td>{_share(totals.usd_total, total_usd)}</td>"
            f"<td>{fmt_int(totals.calls)}</td><td>{fmt_int(totals.turns)}</td>"
            f"<td>{fmt_int(totals.isl)}</td><td>{fmt_int(totals.osl)}</td>"
            f"<td>{fmt_hms(totals.ms_total / 1000.0)}</td>"
            f"<td>{bought}</td><td>{per_pct}</td></tr>"
        )
    return (
        '<h2 id="phases">What each phase cost, and what it bought</h2>'
        "<p class=\"lede\">Spend is the ledger's own attribution. Gain is the breakdown's, joined through the "
        "declared source map -- so a phase can legitimately show real spend against no attributed gain, and "
        "that is the single most useful row on the page.</p>"
        '<table class="grid"><thead><tr><th>Phase</th><th>USD</th><th>Share</th><th>Calls</th><th>Turns</th>'
        "<th>ISL</th><th>OSL</th><th>Agent time</th><th>Bought</th><th>Per +1%</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _tree_rows(node: Node, depth: int, parent_usd: float, out: list[str], limit: int) -> None:
    """Render one subtree as indented table rows, heaviest child first."""
    if depth > limit:
        return
    kids = sorted(node.children.values(), key=lambda n: (-n.total.usd_total, n.name))
    for kid in kids:
        totals = kid.total
        out.append(
            f'<tr><td style="padding-left:{depth * 18}px">{esc(kid.name)}</td>'
            f"<td>{fmt_usd(totals.usd_total)}</td><td>{_share(totals.usd_total, parent_usd)}</td>"
            f"<td>{fmt_int(totals.calls)}</td><td>{fmt_int(totals.isl)}</td>"
            f"<td>{fmt_int(totals.osl)}</td><td>{fmt_hms(totals.ms_total / 1000.0)}</td></tr>"
        )
        _tree_rows(kid, depth + 1, totals.usd_total or parent_usd, out, limit)


def _task_tree(record: dict[str, Any], depth_limit: int) -> str:
    """The task-path tree beneath one phase, heaviest branch first."""
    totals = record["totals"]
    rows: list[str] = []
    _tree_rows(record["node"], 0, totals.usd_total, rows, depth_limit)
    if not rows:
        return ""
    return (
        "<h3>The task path beneath this phase</h3>"
        '<p class="lede">The same tree <code>dump_llm_call_report</code> prints. Share is against the '
        "parent, not the session.</p>"
        '<div class="scroll"><table><thead><tr><th>Task path</th><th>USD</th><th>Share of parent</th>'
        "<th>Calls</th><th>ISL</th><th>OSL</th><th>Agent time</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _deepdive_section(
    phases: list[dict[str, Any]],
    total_usd: float,
    trees: dict[str, str],
) -> str:
    """Inside each phase: the shared anatomy and roster, plus this page's task tree.

    The anatomy and roster are rendered by the module both reports share, so a
    column means the same thing here as on a GEAK run's own page. The task tree
    is Hyperloom's alone and is appended beneath them.

    Args:
        phases: Phase records from the shared module, most expensive first.
        total_usd: The bill shares are taken against.
        trees: Rendered task-path tree per phase name, "" where there is none.

    Returns:
        The section HTML.
    """
    return agents_mod.deepdive_html(
        phases,
        total_usd,
        lede=(
            "Expand a phase to see where its money went: how cost moves as a conversation grows, how "
            "few agents carry the total, what tools the work actually consisted of, and the full agent "
            "roster. Any agent row opens into its own API calls. A phase named "
            "<code>KERNEL_AGENT / ...</code> is a GEAK phase grafted in from GEAK's own ledger."
        ),
        extra_of=lambda phase: trees.get(phase["phase"], ""),
    )


def _growth_section(curve: list[dict[str, Any]]) -> str:
    """Where in a conversation the cost is incurred."""
    if not curve:
        return (
            '<h2 id="growth">Where in a conversation the cost lands</h2>'
            '<p class="lede">No call index recorded on these rows, so the position of a call within its '
            "conversation is <b>not recorded</b> for this session.</p>"
        )
    spark = sparkline([point["isl_median"] for point in curve])
    rows = "".join(
        f"<tr><td>{point['index']}</td><td>{fmt_int(point['calls'])}</td>"
        f"<td>{fmt_int(point['isl_median'])}</td><td>{fmt_usd(point['usd_mean'])}</td></tr>"
        for point in curve
        if point["index"] in {1, 5, 10, 20, 40, 60, 80, 100, 150, 200} or point is curve[-1]
    )
    first, last = curve[0]["isl_median"], curve[-1]["isl_median"]
    ratio = (last / first) if first else 0.0
    return (
        '<h2 id="growth">Where in a conversation the cost lands</h2>'
        f'<p class="lede">Median input length grows {ratio:,.2f}x between call {curve[0]["index"]} and call '
        f"{curve[-1]['index']} of a conversation. Every turn re-sends what came before, so turn count and "
        "context size multiply -- which is why a long agent is dearer than its token count suggests.</p>"
        f"<p>{spark}</p>"
        '<table class="grid"><thead><tr><th>Call index</th><th>Calls at this index</th><th>Median ISL</th>'
        f"<th>Mean USD</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _models_section(mix: list[dict[str, Any]], total_usd: float) -> str:
    """Which models did the work, and what each cost."""
    rows = "".join(
        f"<tr><td><code>{esc(entry['model'])}</code></td><td>{fmt_usd(entry['usd'])}</td>"
        f"<td>{_share(entry['usd'], total_usd)}</td><td>{fmt_int(entry['calls'])}</td>"
        f"<td>{fmt_int(entry['isl'])}</td><td>{fmt_int(entry['osl'])}</td></tr>"
        for entry in mix
    )
    return (
        '<h2 id="models">Which models did the work</h2>'
        '<p class="lede">A phase served by an expensive model for cheap work is the easiest saving on this '
        "page, and the only one visible without changing what the session does.</p>"
        '<table class="grid"><thead><tr><th>Model</th><th>USD</th><th>Share</th><th>Calls</th><th>ISL</th>'
        f"<th>OSL</th></tr></thead><tbody>{rows or '<tr><td colspan=6>none recorded</td></tr>'}</tbody></table>"
    )


def render(session_dir: Path, title: str | None = None, depth_limit: int = 3) -> str:
    """Build the whole self-contained document.

    Args:
        session_dir: Session root directory.
        title: Override for the document heading.
        depth_limit: How deep to walk each phase's task tree.

    Returns:
        The HTML document.
    """
    turns, details = load_ledgers(session_dir)
    root, cov = build_tree(turns, details)
    breakdown = load_breakdown(session_dir / BREAKDOWN_FILENAME)
    ladder = outcome_ladder(breakdown)
    joined = join_gain(phase_records(root), ladder)
    rows = call_rows(turns, details)
    ident = session_identity(session_dir, breakdown)

    # The deep dive runs on the normalised rows, with GEAK's own ledger grafted in
    # place of the handful of summary rows the harvester wrote for it.
    normalised, geak_status = graft_geak(session_dir, normalise(rows))
    agents = agents_mod.agent_records(normalised, hyperloom_role_of)
    deep_phases = agents_mod.phase_records(normalised, agents)
    deep_usd = sum(p["usd"] for p in deep_phases)
    trees = {record["phase"]: _task_tree(record, depth_limit) for record in joined}

    heading = title or f"Hyperloom session - where the time and the money went ({ident['model']})"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    read = ["llm_calls.jsonl", "llm_calls_detail.jsonl"] + ([BREAKDOWN_FILENAME] if breakdown else [])
    sources = ", ".join(f"<code>{esc(name)}</code>" for name in read)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(heading)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{esc(heading)}</h1>
<p class="sub">Session <code>{esc(ident["session_id"])}</code>
{f"| revision <code>{esc(ident['revision'])}</code>" if ident["revision"] else ""}
{f"| {esc(ident['elapsed_min'])} min elapsed" if ident["elapsed_min"] else ""}
| generated {generated} from {sources}</p>
<nav><a href="#bought">What it bought</a><a href="#phases">Spend by phase</a>
<a href="#deep">Inside each phase</a><a href="#delegate">Delegation signals</a>
<a href="#growth">Conversation growth</a>
<a href="#models">Models</a><a href="#coverage">Coverage</a></nav>
{_headline_cards(ident, root.total, ladder)}
{_coverage_section(cov, breakdown is not None, agents, geak_status)}
{_outcome_section(ladder)}
{_phase_table(joined, root.total.usd_total)}
{_deepdive_section(deep_phases, deep_usd, trees)}
{agents_mod.delegation_html(agents_mod.delegation_signals(agents))}
{_growth_section(growth_curve(rows))}
{_models_section(model_mix(rows), root.total.usd_total)}
<h2>How to reproduce this</h2>
<p class="lede">Everything above is computed from the files this session wrote. Regenerate with:</p>
<pre class="mono">python3 -m hyperloom.inference_optimizer.tools.render_hyperloom_html_report \\
    --session-dir {esc(session_dir)} -o {esc(session_dir / "reports" / DEFAULT_OUTPUT)}</pre>
</div>
<script>const AGENTS={agents_mod.agent_payload(agents)};{agents_mod.drill_js("llm_calls_detail.jsonl")}</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` when the session has no readable ledger.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-dir", "-s", type=Path, required=True, help="Session root directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="HTML file to write")
    parser.add_argument("--title", default=None, help="Override the document heading")
    parser.add_argument("--max-depth", type=int, default=3, help="How deep to walk each phase's task tree")
    args = parser.parse_args(argv)

    session_dir = args.session_dir
    turns, details = load_ledgers(session_dir)
    if not turns and not details:
        print(f"error: no llm_calls.jsonl or detail rows under {session_dir}", file=sys.stderr)
        return 2

    output = args.output or (session_dir / "reports" / DEFAULT_OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_text = render(session_dir, title=args.title, depth_limit=args.max_depth)
    output.write_text(html_text, encoding="utf-8")
    print(f"wrote {output} ({len(html_text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
