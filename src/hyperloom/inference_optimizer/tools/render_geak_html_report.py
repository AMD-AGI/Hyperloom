#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a GEAK run's per-LLM-call ledger as a structured, self-contained HTML report.

The tree report (``dump_geak_call_report``) is exhaustive but flat: 5,000 call rows
in one document answer "what happened" and not "where did the money go". This
renderer takes the same ``geak_calls.jsonl`` and arranges it as a small number of
questions, each with its own section:

* what did the run *buy*, and what did each phase *cost* to buy it (joined against
  ``geak_outcome.json`` when the run wrote one);
* inside an expensive phase, which agents carry the spend;
* inside an agent, where in the conversation the cost is incurred;
* which tasks look cheap enough to delegate to a smaller model.

Two rules hold throughout, the same two the outcome report follows:

* nothing is modelled or extrapolated -- every number is computed from the rows in
  the ledger, and the coverage banner says what the ledger does not contain;
* a quantity that was not recorded is rendered as "not recorded", never as zero,
  because "we did not measure it" and "it was nothing" are different claims.

Role labels are the one derived field. Claude Code records ``{phase, label}`` per
agent in the workflow record, but ``geak_calls.jsonl`` carries only the phase, so a
role is inferred from the agent's first prompt and is marked as derived wherever it
is shown. An agent whose prompt does not identify a role is shown as unlabelled, not
guessed at.

Usage::

    python3 -m hyperloom.inference_optimizer.tools.render_geak_html_report \\
        --reports-dir <RUN>/reports -o <RUN>/reports/geak_report.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.tools._report_html import CSS  # noqa: F401
from hyperloom.inference_optimizer.tools._report_html import bar as _bar
from hyperloom.inference_optimizer.tools._report_html import esc as _esc
from hyperloom.inference_optimizer.tools._report_html import fmt_hms as _hms
from hyperloom.inference_optimizer.tools._report_html import fmt_int as _int
from hyperloom.inference_optimizer.tools._report_html import fmt_pct as _pct
from hyperloom.inference_optimizer.tools._report_html import fmt_usd as _usd
from hyperloom.inference_optimizer.tools._report_html import num as _num
from hyperloom.inference_optimizer.tools._report_agents import (
    DECILES,
    MAX_CALLS_EMBEDDED,
    MIN_CALLS_FOR_CURVE,
    UNLABELLED,
    agent_key,
    agent_payload as _agent_payload,
    concentration,
    cost_curve,
    delegation_html as _delegation_section,
    delegation_signals,
    drill_js,
    phase_records,
)
from hyperloom.inference_optimizer.tools import _report_agents as _agents

CALLS_FILENAME = "geak_calls.jsonl"
OUTCOME_FILENAME = "geak_outcome.json"
DEFAULT_OUTPUT = "geak_report.html"

# Matches the role sentence GEAK's role prompts open with ("You are the tech_lead",
# "You are Engineer r1_d2 (specialty=memory)"). geak_calls.jsonl truncates prompts,
# so this finds a role often but not always -- hence UNLABELLED rather than a guess.
ROLE_RE = re.compile(r"You are (?:the )?([A-Za-z][\w \-/]{0,48}?)(?:[.,\n(]|\s+for\s|\s+committing\s)")
SPECIALTY_RE = re.compile(r"specialty=(\w+)")
ROUND_RE = re.compile(r"\br(\d+)_d(\d+)\b")

#: The drill-down script, pointed at this report's own ledger file.
JS = drill_js(CALLS_FILENAME)


def load_calls(path: Path) -> list[dict[str, Any]]:
    """Read the ledger, skipping lines that are not JSON objects.

    A missing ledger is empty, not fatal. Claude Code writes the ledger into a
    config home the run does not own, so a run whose home was a container overlay
    can finish, measure a real result, and still have no ledger left to read. That
    run's outcome is still worth a page; the page just has to say the cost half is
    gone rather than imply the phases were free.
    """
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_outcome(path: Path) -> dict[str, Any] | None:
    """Read ``geak_outcome.json`` if the run wrote one; ``None`` is 'not measured'."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def derive_role(prompt: str) -> dict[str, Any]:
    """Infer a readable role from an agent's first prompt. Never guesses."""
    text = prompt or ""
    match = ROLE_RE.search(text)
    role = match.group(1).strip() if match else UNLABELLED
    specialty = SPECIALTY_RE.search(text)
    rounds = ROUND_RE.search(role) or ROUND_RE.search(text[:400])
    return {
        "role": role,
        "specialty": specialty.group(1) if specialty else None,
        "round": int(rounds.group(1)) if rounds else None,
        "derived": True,
    }

def _role_of(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Name an agent from the first prompt of its conversation.

    GEAK's ledger carries no label, so the role is recovered from the role
    sentence every GEAK prompt opens with. See :func:`derive_role`.
    """
    return derive_role(str(calls[0].get("prompt", "")) if calls else "")


def agent_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the call rows into one record per (phase, agent), ordered by spend."""
    return _agents.agent_records(rows, _role_of)


def _anatomy(phase: dict[str, Any]) -> str:
    """Where inside a phase the cost is actually incurred."""
    return _agents.anatomy_html(phase)


def _roster(phase: dict[str, Any]) -> str:
    """Every agent in the phase, ranked by spend, each drillable down to its calls."""
    return _agents.roster_html(phase)


def _deepdive_section(joined: list[dict[str, Any]], total_usd: float) -> str:
    """One collapsible block per phase, each headed with the kernel it went after."""
    return _agents.deepdive_html(joined, total_usd, code_of=_kernel_tasks)


def coverage(rows: list[dict[str, Any]], agents: list[dict[str, Any]]) -> dict[str, Any]:
    """What the ledger does and does not contain. Read this before quoting a number."""
    unpriced = sum(1 for r in rows if not _num(r.get("usd")))
    untimed = sum(1 for r in rows if not _num(r.get("dt_s")))
    unlabelled = sum(1 for a in agents if a["role"] == UNLABELLED)
    models = sorted({str(r.get("model")) for r in rows if r.get("model")})
    return {
        "calls": len(rows),
        "agents": len(agents),
        "phases": len({str(r.get("phase", "")) for r in rows}),
        "models": models,
        "unpriced_calls": unpriced,
        "untimed_calls": untimed,
        "unlabelled_agents": unlabelled,
        "usd": sum(_num(r.get("usd")) for r in rows),
        "isl": sum(_num(r.get("isl")) for r in rows),
        "osl": sum(_num(r.get("osl")) for r in rows),
        "thinking": sum(_num(r.get("thinking")) for r in rows),
        "tool_calls": sum(len(r.get("tools") or []) for r in rows),
        "wall_seconds": sum(_num(r.get("dt_s")) for r in rows),
    }


def join_outcome(phases: list[dict[str, Any]], outcome: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Put each phase's measured throughput change beside what that phase cost.

    A phase that measured nothing gets ``outcome: None`` -- rendered as "not
    measured", never as 0%, because the two mean opposite things.

    Delta stages carry the phase name the ledger uses. Kernel stages are all
    recorded under the phase "HeadKernel" and distinguished by ``task``
    ("h0_gemm_..."), while the ledger names those phases "P8 HeadKernel h0", so the
    head token is what joins them.
    """
    deltas: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, Any]] = {}
    kernels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stage in (outcome or {}).get("stages") or []:
        if not isinstance(stage, dict):
            continue
        if stage.get("kind") == "reference":
            references[str(stage.get("phase", ""))] = stage
        elif stage.get("kind") == "delta":
            deltas[str(stage.get("phase", ""))] = stage
        elif stage.get("kind") == "kernel":
            task = str(stage.get("task", ""))
            head = task.split("_", 1)[0] if "_" in task else task
            if head:
                kernels[head].append(stage)

    joined: list[dict[str, Any]] = []
    for phase in phases:
        name = phase["phase"]
        found: dict[str, Any] | None = None
        if name in deltas:
            stage = deltas[name]
            found = {
                "kind": "delta",
                "gain_pct": stage.get("delta_pct"),
                "before_tok_s": stage.get("before_tok_s"),
                "after_tok_s": stage.get("after_tok_s"),
                "gate": stage.get("gate") or stage.get("correctness_gate"),
                "source": stage.get("source"),
            }
        elif name in references:
            stage = references[name]
            found = {
                "kind": "reference",
                "gain_pct": None,
                "after_tok_s": stage.get("after_tok_s"),
                "measurement_mode": stage.get("measurement_mode"),
                "source": stage.get("source"),
            }
        else:
            head = name.rsplit(" ", 1)[-1] if " " in name else ""
            if head in kernels:
                stages = kernels[head]
                measured = [s for s in stages if s.get("present")]
                found = {
                    "kind": "kernel",
                    # The end-to-end A/B of a kernel the run wrote is the real
                    # result when there is one; the opbench ceiling is only the
                    # upper bound that applies when nothing was validated.
                    "gain_pct": max(
                        (_num(s["integration"].get("e2e_delta_pct")) for s in stages if s.get("integration")),
                        default=(
                            max((_num(s.get("amdahl_ceiling_e2e_pct")) for s in measured), default=None)
                            if measured
                            else None
                        ),
                    ),
                    "tasks": [
                        {
                            "task": s.get("task"),
                            "present": bool(s.get("present")),
                            "isolated_speedup": s.get("isolated_speedup"),
                            "amdahl_ceiling_e2e_pct": s.get("amdahl_ceiling_e2e_pct"),
                            "pct_gpu_time": s.get("pct_gpu_time"),
                            "winner_backend": s.get("winner_backend"),
                            "integration": s.get("integration"),
                        }
                        for s in stages
                    ],
                }
        joined.append({**phase, "outcome": found})
    return joined


def _coverage_section(cov: dict[str, Any], outcome: dict[str, Any] | None) -> str:
    """The banner that says what these numbers are and are not. Always rendered."""
    caveats: list[str] = []
    if not cov["calls"]:
        # Zero calls and zero dollars mean the ledger is gone, not that the run
        # was free. Say which, at the top, before any number is read.
        caveats.append(
            f"No {CALLS_FILENAME} was found, so this run's LLM ledger is not available and every cost, "
            f"token and wall-clock figure below is absent rather than zero. Claude Code writes that "
            f"ledger into its own config home; if that home was a container overlay it died with the "
            f"container. What the run measured survived, and is reported in full."
        )
    if cov["unpriced_calls"]:
        caveats.append(f"{cov['unpriced_calls']:,} calls carry no cost and contribute $0 to every total below.")
    if cov["untimed_calls"]:
        share = cov["untimed_calls"] / max(1, cov["calls"]) * 100
        caveats.append(
            f"{cov['untimed_calls']:,} calls ({share:.1f}%) have no recorded duration, so wall-clock "
            f"sums are lower bounds, not the elapsed time of the run."
        )
    if cov["unlabelled_agents"]:
        caveats.append(
            f"{cov['unlabelled_agents']} of {cov['agents']} agents could not be given a role: the ledger "
            f"truncates prompts, and the authoritative label lives in the wf_*.json workflow record. "
            f"They are shown as '{UNLABELLED}' rather than guessed at."
        )
    if not cov["thinking"] and cov["calls"]:
        caveats.append(
            "No thinking tokens were recorded for this run, so the thinking columns are zero because "
            "there was nothing to record -- not because thinking was measured at zero."
        )
    if outcome is None:
        caveats.append(
            f"No {OUTCOME_FILENAME} was found beside the ledger, so this report can say what each phase "
            f"cost but not what it bought. Sections that need it are omitted rather than estimated."
        )
    elif (outcome.get("summary") or {}).get("compounded_is_estimate"):
        caveats.append(
            "The run's compounded end-to-end speedup is flagged as an estimate: phases handed off at "
            "different measured throughputs, so multiplying the per-phase gains does not give a "
            "measured figure. The observed first-to-last number is the measured one."
        )
    if cov["calls"]:
        caveats.append(
            "Roles are derived from each agent's first prompt, not read from the workflow record. "
            "They are a reading aid; the agent id is the identifier."
        )
    items = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    return (
        '<div class="note"><b>Coverage - read this before quoting a number</b>'
        + (
            "<div>Outcome only: what this run measured, without its LLM ledger.</div>"
            if not cov["calls"]
            else f"<div>{cov['calls']:,} API calls across {cov['agents']} agents and {cov['phases']} phases, "
            f"model{'s' if len(cov['models']) != 1 else ''} "
            f"{_esc(', '.join(cov['models']) or 'not recorded')}.</div>"
        )
        + f"<ul>{items}</ul></div>"
    )


def _headline_cards(cov: dict[str, Any], outcome: dict[str, Any] | None) -> str:
    summary = (outcome or {}).get("summary") or {}
    observed = summary.get("observed_delta_pct_first_to_last")
    per_pct = None
    if observed and _num(observed) > 0:
        per_pct = cov["usd"] / _num(observed)
    base = summary.get("reference_baseline_tok_s")
    final = summary.get("last_measured_after_tok_s")
    cards = []
    if observed is not None:
        cards.append(("Throughput gained", _pct(observed), "measured, first stage to last"))
    if base and final:
        cards.append(
            (
                "Throughput",
                f"{_int(base)} -&gt; {_int(final)}",
                "tok/s, baseline to last measured stage",
            )
        )
    if not cov["calls"]:
        # No ledger: a $0.00 card next to a real throughput number reads as
        # "this run was free", which is the one thing it definitely was not.
        cards.append(("Spend", "no ledger", "the LLM ledger for this run was not preserved"))
        body = "".join(
            f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{v}</div>'
            f'<div class="n">{_esc(n)}</div></div>'
            for k, v, n in cards
        )
        return f'<div class="cards">{body}</div>'
    cards += [
        ("Total spend", _usd(cov["usd"]), f"{cov['calls']:,} API calls"),
        ("Input tokens", _int(cov["isl"]), f"{cov['isl'] / max(1, cov['isl'] + cov['osl']) * 100:.1f}% of all tokens"),
        ("Output tokens", _int(cov["osl"]), "what the model actually wrote"),
        ("Recorded wall time", _hms(cov["wall_seconds"]), "sum of per-call durations"),
        ("Tool calls", _int(cov["tool_calls"]), "shell, file and monitor actions"),
    ]
    if per_pct is not None:
        cards.append(("Cost per +1%", _usd(per_pct), "whole-run average"))
    body = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{v}</div><div class="n">{_esc(n)}</div></div>'
        for k, v, n in cards
    )
    return f'<div class="cards">{body}</div>'


def _outcome_section(
    joined: list[dict[str, Any]], total_usd: float, outcome: dict[str, Any] | None, no_ledger: bool = False
) -> str:
    """The join the whole report exists for: spend beside measured result."""
    if outcome is None:
        return (
            '<h2 id="bought">What each phase bought</h2>'
            f'<p class="lede">Not available: no <code>{OUTCOME_FILENAME}</code> beside the ledger. '
            "Spend is still reported below; the result each phase measured is not, because nothing "
            "in the ledger records it and inferring it would be a guess.</p>"
        )
    rows: list[tuple[str, float, float, str, str]] = []
    for phase in joined:
        out = phase["outcome"]
        share = phase["usd"] / total_usd if total_usd else 0.0
        if out is None:
            bought = '<span class="none">no throughput measurement for this phase</span>'
            efficiency = '<span class="none">n/a</span>'
        elif out["kind"] == "reference":
            bought = (
                f'<span class="mut">established the baseline every later gain is measured against: '
                f"{_int(out['after_tok_s'])} tok/s"
                + (f" ({_esc(out['measurement_mode'])})" if out.get("measurement_mode") else "")
                + "</span>"
            )
            efficiency = '<span class="none">n/a</span>'
        elif out["kind"] == "delta":
            gain = _num(out["gain_pct"])
            tone = "good" if gain > 0 else "bad" if gain < 0 else "mut"
            gate = out.get("gate")
            bought = (
                f'<span class="{tone}">{_pct(out["gain_pct"])}</span> '
                f'<span class="mut">({_int(out["before_tok_s"])} -&gt; {_int(out["after_tok_s"])} tok/s'
                + (f", gate {_esc(gate)}" if gate else "")
                + ")</span>"
            )
            efficiency = _usd(phase["usd"] / gain) if gain > 0 else '<span class="none">n/a</span>'
        else:
            tasks = out.get("tasks") or []
            measured = [t for t in tasks if t["present"]]
            ab = [t for t in tasks if t.get("integration")]
            if ab:
                best_task = max(ab, key=lambda t: _num(t["integration"].get("e2e_delta_pct")))
                best = best_task["integration"]
                delta = _num(best.get("e2e_delta_pct"))
                tone = "good" if delta > 0 else "bad" if delta < 0 else "mut"
                gate = best.get("gate")
                acc = (
                    "accuracy held" if _num(best.get("gsm8k_cand")) >= _num(best.get("gsm8k_ref")) else "accuracy fell"
                )
                bought = (
                    f'<span class="{tone}">{delta:+.2f}%</span> <span class="mut">end-to-end, '
                    f"measured A/B of the kernel this run wrote for "
                    f"<code>{_esc(best_task.get('task'))}</code> "
                    f"({_esc(best.get('candidate'))}, {_num(best.get('isolated_speedup')):.4f}x isolated, "
                    f"{acc}"
                    + (f", gate {_esc(gate)}" if gate else "")
                    + f"; ceiling was {_num(best.get('amdahl_ceiling_pct')):.2f}%)</span>"
                )
                efficiency = _usd(phase["usd"] / delta) if delta > 0 else '<span class="none">n/a</span>'
                rows.append((phase["phase"], phase["usd"], share, bought, efficiency))
                continue
            ceiling = max((_num(t["amdahl_ceiling_e2e_pct"]) for t in measured), default=0.0)
            detail = ", ".join(
                f"{_esc(t['task'])}: "
                + (
                    f"{_num(t['isolated_speedup']):.4f}x isolated, {_num(t['pct_gpu_time']):.2f}% GPU time"
                    if t["present"]
                    else "never benchmarked"
                )
                for t in tasks
            )
            tone = "good" if ceiling > 0 else "bad"
            bought = (
                f'<span class="{tone}">{ceiling:+.2f}%</span> <span class="mut">end-to-end ceiling - {detail}</span>'
            )
            efficiency = '<span class="none">n/a</span>' if ceiling <= 0 else _usd(phase["usd"] / ceiling)
        rows.append((phase["phase"], phase["usd"], share, bought, efficiency))
    summary = outcome.get("summary") or {}
    estimate_note = ""
    if summary.get("compounded_is_estimate"):
        seams = summary.get("handoff_seams") or []
        seam_text = "; ".join(
            f"{_esc(s.get('from_phase'))} handed off at {_int(s.get('handoff_from_tok_s'))} tok/s but "
            f"{_esc(s.get('to_phase'))} measured its own baseline at {_int(s.get('handoff_to_tok_s'))} "
            f"({_num(s.get('gap_pct')):+.2f}%)"
            for s in seams
            if isinstance(s, dict)
        )
        estimate_note = (
            f'<p class="lede"><b>Why the per-phase gains do not multiply out to the headline.</b> '
            f"{seam_text}. The measured end-to-end figure is "
            f"{_pct(summary.get('observed_delta_pct_first_to_last'))} "
            f"({_num(summary.get('observed_speedup_first_to_last')):.4f}x); compounding the phase gains "
            f"would give {_num(summary.get('compounded_speedup')):.4f}x, which is an estimate.</p>"
        )
    if no_ledger:
        # Every cost cell would read $0.00 and every share 0.0%, which is a claim
        # about spend this report cannot make. Drop the columns instead.
        body = "".join(
            f'<tr><td><b>{_esc(name)}</b></td><td style="text-align:left">{bought}</td></tr>'
            for name, _usd_, _share_, bought, _eff_ in rows
        )
        return (
            '<h2 id="bought">What each phase bought</h2>'
            '<p class="lede">The result each phase measured, from the artifacts the run wrote at each '
            "phase boundary. What it cost to buy is not shown: this run's LLM ledger was not preserved, "
            "so there is no spend to report and a zero would be a false one.</p>"
            '<div class="scroll"><table><thead><tr><th>Phase</th>'
            f"<th>Measured result</th></tr></thead><tbody>{body}</tbody></table></div>{estimate_note}"
        )
    body = "".join(
        f"<tr><td><b>{_esc(name)}</b></td>"
        f"<td>{_usd(usd)}{_bar(share)}</td>"
        f"<td>{share * 100:.1f}%</td>"
        f'<td style="text-align:left">{bought}</td>'
        f"<td>{efficiency}</td></tr>"
        for name, usd, share, bought, efficiency in rows
    )
    return (
        '<h2 id="bought">What each phase bought, and what it cost to buy it</h2>'
        '<p class="lede">Spend comes from the call ledger; the result comes from the artifacts the run '
        "wrote at each phase boundary. A phase with no measurement is marked as such and is never "
        "shown as zero.</p>"
        '<div class="scroll"><table><thead><tr><th>Phase</th><th>Cost</th><th>Share</th>'
        "<th>Measured result</th><th>Cost per +1%</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>{estimate_note}"
    )


def performance_ladder(joined: list[dict[str, Any]], outcome: dict[str, Any] | None) -> dict[str, Any]:
    """Order the measured stages the way the run walked them, and rank the gains.

    Every figure here is a throughput the run measured and wrote down. A stage's
    share is its own tok/s gain over the summed tok/s gains of the stages that
    measured one -- arithmetic on recorded numbers, not an attribution model, and
    it deliberately does not equal the end-to-end figure, because the handoff
    seams between stages lose some of it. Kernel stages carry an Amdahl ceiling
    instead of a throughput, and a stage that measured nothing stays empty.
    """
    if outcome is None:
        return {}
    steps: list[dict[str, Any]] = []
    for stage in outcome.get("stages") or []:
        kind = stage.get("kind")
        if kind == "reference":
            steps.append(
                {
                    "kind": "reference",
                    "phase": str(stage.get("phase", "")),
                    "after": _num(stage.get("after_tok_s")),
                    "mode": stage.get("measurement_mode"),
                }
            )
        elif kind == "delta":
            before, after = _num(stage.get("before_tok_s")), _num(stage.get("after_tok_s"))
            steps.append(
                {
                    "kind": "delta",
                    "phase": str(stage.get("phase", "")),
                    "before": before,
                    "after": after,
                    "abs_gain": after - before,
                    "pct": _num(stage.get("delta_pct")),
                    "gate": stage.get("gate"),
                    "mode": stage.get("measurement_mode"),
                }
            )
    total_gain = sum(s["abs_gain"] for s in steps if s["kind"] == "delta" and s["abs_gain"] > 0)
    for step in steps:
        if step["kind"] == "delta":
            step["share"] = step["abs_gain"] / total_gain if total_gain else None
    costed = {p["phase"]: p["usd"] for p in joined}
    for step in steps:
        step["usd"] = costed.get(step["phase"])
    kernel_only = [p for p in joined if p["outcome"] and p["outcome"]["kind"] == "kernel"]
    # Kernel tasks the ledger has no phase for. A run can benchmark an operator
    # without a "P<n> HeadKernel <head>" phase existing to join it to, and those
    # tasks would otherwise vanish from the page entirely.
    attributed = {t["task"] for p in kernel_only for t in (p["outcome"].get("tasks") or []) if t.get("task")}
    orphans = [
        stage
        for stage in outcome.get("stages") or []
        if isinstance(stage, dict) and stage.get("kind") == "kernel" and stage.get("task") not in attributed
    ]
    silent = [p["phase"] for p in joined if p["outcome"] is None]
    return {
        "steps": steps,
        "total_gain": total_gain,
        "kernels": kernel_only,
        "orphan_kernels": orphans,
        "silent": silent,
        "summary": outcome.get("summary") or {},
    }


def _performance_section(ladder: dict[str, Any], no_ledger: bool = False) -> str:
    """Which phase made the run faster, and by how much."""
    if not ladder:
        return (
            '<h2 id="perf">What each phase contributed to throughput</h2>'
            f'<p class="lede">Not available: no <code>{OUTCOME_FILENAME}</code> beside the ledger. '
            "Every phase's contribution is unmeasured, which is not the same as zero, so none is "
            "shown.</p>"
        )
    summary = ladder["summary"]
    rows = []
    for step in ladder["steps"]:
        if step["usd"] is not None:
            cost = _usd(step["usd"])
        else:
            cost = '<span class="none">' + ("no ledger" if no_ledger else "not billed") + "</span>"
        if step["kind"] == "reference":
            rows.append(
                f'<tr class="ref"><td><b>{_esc(step["phase"])}</b></td>'
                f'<td colspan="2"><span class="mut">baseline</span></td>'
                f"<td><b>{_int(step['after'])}</b> tok/s</td>"
                f'<td colspan="2"><span class="mut">the number every later gain is measured '
                f"against{' (' + _esc(step['mode']) + ')' if step.get('mode') else ''}</span></td>"
                f"<td>{cost}</td></tr>"
            )
            continue
        tone = "good" if step["pct"] > 0 else "bad" if step["pct"] < 0 else "mut"
        share = step.get("share")
        share_cell = f"{share * 100:.1f}%{_bar(share)}" if share is not None else '<span class="none">n/a</span>'
        gate = f' <span class="mut">gate {_esc(step["gate"])}</span>' if step.get("gate") else ""
        rows.append(
            f"<tr><td><b>{_esc(step['phase'])}</b>{gate}</td>"
            f"<td>{_int(step['before'])}</td><td>{_int(step['after'])}</td>"
            f'<td class="{tone}"><b>{step["pct"]:+.2f}%</b></td>'
            f"<td>+{_int(step['abs_gain'])} tok/s</td>"
            f"<td>{share_cell}</td><td>{cost}</td></tr>"
        )
    notes = []
    validated = [
        (p, best)
        for p in ladder["kernels"]
        for best in [
            max(
                (t for t in (p["outcome"].get("tasks") or []) if t.get("integration")),
                key=lambda t: _num(t["integration"].get("e2e_delta_pct")),
                default=None,
            )
        ]
        if best
    ]
    unvalidated = [
        p for p in ladder["kernels"] if not any(t.get("integration") for t in (p["outcome"].get("tasks") or []))
    ]
    for phase, best_task in validated:
        integ = best_task["integration"]
        delta = _num(integ.get("e2e_delta_pct"))
        tone = "good" if delta > 0 else "bad"
        reason = integ.get("reason")
        head = (
            "<b>"
            + _esc(phase["phase"])
            + " shipped a kernel and measured "
            + '<span class="'
            + tone
            + '">'
            + f"{delta:+.2f}%"
            + "</span> end to end.</b> "
        )
        body = (
            "The kernel is <code>"
            + _esc(best_task.get("task"))
            + "</code>. The run wrote its own candidate ("
            + _esc(integ.get("candidate"))
            + "), served it "
            "against the reference and measured " + _int(integ.get("e2e_throughput_tok_s")) + " tok/s "
            "-- " + f"{_num(integ.get('isolated_speedup')):.4f}" + "x on the kernel in isolation, "
            "against a " + f"{_num(integ.get('amdahl_ceiling_pct')):.2f}" + "% ceiling. Accuracy was "
            "checked rather than assumed: gsm8k "
            + f"{_num(integ.get('gsm8k_ref')):.2f}"
            + " -&gt; "
            + f"{_num(integ.get('gsm8k_cand')):.2f}"
            + "."
        )
        tail = (" The harness recorded this verdict as <em>" + _esc(reason) + "</em>") if reason else ""
        notes.append(head + body + tail)
    if unvalidated:
        detail = "; ".join(
            f"{_esc(p['phase'])} ("
            + ", ".join(f"<code>{_esc(t['task'])}</code>" for t in (p["outcome"].get("tasks") or []) if t.get("task"))
            + ") "
            + f"{max((_num(t['amdahl_ceiling_e2e_pct']) for t in (p['outcome'].get('tasks') or []) if t['present']), default=0.0):+.2f}%"
            for p in unvalidated
        )
        notes.append(
            "<b>The other kernel phases contributed no measured end-to-end throughput.</b> No "
            "candidate they wrote reached an end-to-end A/B, and among the library backends the "
            "incumbent stayed fastest, so the Amdahl ceiling on any gain is "
            f"{detail}. That is a measured result, not a missing one: the work ran, was "
            "benchmarked, and did not beat what was already there."
        )
    if ladder.get("orphan_kernels"):
        detail = "; ".join(
            f"<code>{_esc(s.get('task'))}</code> "
            + (
                f"{_num(s.get('isolated_speedup')):.4f}x isolated, "
                f"{_num(s.get('pct_gpu_time')):.2f}% GPU time, "
                f"ceiling {_num(s.get('amdahl_ceiling_e2e_pct')):+.2f}%"
                if s.get("present")
                else "never benchmarked"
            )
            for s in ladder["orphan_kernels"]
        )
        notes.append(
            "<b>Kernels the run benchmarked without a phase of their own:</b> "
            + detail
            + ". No phase in the LLM ledger is named after them, so nothing here is attributed "
            "to them either way -- they are named because they are part of this run's kernel work."
        )
    if ladder["silent"]:
        notes.append(
            "<b>Phases with no throughput measurement of their own:</b> "
            + ", ".join(f"<code>{_esc(p)}</code>" for p in ladder["silent"])
            + ". They profile, strategise or hand off rather than ending on a benchmark, so "
            "nothing here is attributed to them either way."
        )
    if summary.get("compounded_is_estimate"):
        notes.append(
            "<b>The shares do not add up to the end-to-end figure, and should not.</b> "
            f"Measured end to end: {_pct(summary.get('observed_delta_pct_first_to_last'))} "
            f"({_num(summary.get('observed_speedup_first_to_last')):.4f}x, "
            f"{_int(summary.get('reference_baseline_tok_s'))} -&gt; "
            f"{_int(summary.get('last_measured_after_tok_s'))} tok/s). Multiplying the per-phase "
            f"gains would give {_num(summary.get('compounded_speedup')):.4f}x. The gap is the "
            "handoff seams: a phase does not always start from where the previous one finished."
        )
    note_html = "".join(f'<p class="lede">{n}</p>' for n in notes)
    return (
        '<h2 id="perf">What each phase contributed to throughput</h2>'
        '<p class="lede">Read in run order. Every tok/s here was measured by the run at a phase '
        "boundary and written to its own artifacts; share is a phase's tok/s gain over the summed "
        "gains of the phases that measured one.</p>"
        '<div class="scroll"><table><thead><tr><th>Phase</th><th>From</th><th>To</th>'
        "<th>Gain</th><th>Absolute</th><th>Share of measured gain</th><th>Cost</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{note_html}"
    )


def _kernel_tasks(phase: dict[str, Any]) -> str:
    """The kernel task names behind a HeadKernel phase, or "" for any other phase.

    "P8 HeadKernel h0" names the head, not the operator it went after. The task
    ("h0_gemm_a8w8_blockscale_task") is the only place the kernel is named, so
    every heading that carries a kernel phase carries the task with it.
    """
    out = phase.get("outcome") or {}
    if out.get("kind") != "kernel":
        return ""
    return ", ".join(str(t["task"]) for t in (out.get("tasks") or []) if t.get("task"))


def _phase_table(joined: list[dict[str, Any]], total_usd: float, total_isl: float) -> str:
    peak = max((p["usd"] for p in joined), default=1.0) or 1.0
    rows = "".join(
        f"<tr><td><b>{_esc(p['phase'])}</b>"
        + (f'<div class="mut"><code>{_esc(_kernel_tasks(p))}</code></div>' if _kernel_tasks(p) else "")
        + f"</td><td>{p['agents']}</td><td>{p['calls']:,}</td>"
        f"<td>{_usd(p['usd'])}{_bar(p['usd'] / peak)}</td>"
        f"<td>{p['usd'] / total_usd * 100 if total_usd else 0:.1f}%</td>"
        f"<td>{_int(p['isl'])}</td><td>{p['isl'] / total_isl * 100 if total_isl else 0:.1f}%</td>"
        f"<td>{_int(p['osl'])}</td><td>{_hms(p['seconds'])}</td>"
        f"<td>{p['tool_calls']:,}</td><td>{_usd(p['usd'] / max(1, p['calls']))}</td></tr>"
        for p in joined
    )
    return (
        '<h2 id="phases">Spend by phase</h2>'
        '<p class="lede">Ranked by cost. Input tokens carry almost all of it: every call re-sends the '
        "conversation, so a phase's bill tracks how long its agents talked, not how much they wrote.</p>"
        '<div class="scroll"><table><thead><tr><th>Phase</th><th>Agents</th><th>Calls</th><th>Cost</th>'
        "<th>Share</th><th>Input</th><th>Input share</th><th>Output</th><th>Recorded time</th>"
        f"<th>Tools</th><th>$/call</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def render(calls_path: Path, outcome_path: Path, title: str | None = None) -> str:
    """Build the whole self-contained document."""
    rows = load_calls(calls_path)
    outcome = load_outcome(outcome_path)
    agents = agent_records(rows)
    phases = phase_records(rows, agents)
    joined = join_outcome(phases, outcome)
    cov = coverage(rows, agents)
    signals = delegation_signals(agents)
    ladder = performance_ladder(joined, outcome)

    run_id = (outcome or {}).get("run_id") or calls_path.parent.parent.name
    heading = title or ("GEAK run - what it measured" if not rows else "GEAK run - where the time and the money went")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # Name only the files actually read. Crediting a ledger that was not there is
    # the kind of small untruth that makes a reader distrust the real numbers.
    read = [calls_path.name] if rows else []
    read += [outcome_path.name] if outcome else []
    sources = " and ".join(f"<code>{_esc(name)}</code>" for name in read) or "no readable input"

    # Without a ledger the three spend sections have no rows to show. An empty
    # table under a confident heading reads as "nothing happened here"; leaving
    # them out, and the nav entries with them, says the opposite and is true.
    no_ledger = not cov["calls"]
    nav_items = [("#perf", "Throughput"), ("#bought", "Cost vs result" if not no_ledger else "What it bought")]
    if not no_ledger:
        nav_items += [("#phases", "Spend by phase"), ("#deep", "Inside each phase")]
        nav_items += [("#delegate", "Delegation signals")]
    nav = "".join(f'<a href="{href}">{_esc(label)}</a>' for href, label in nav_items)
    ledger_sections = (
        ""
        if no_ledger
        else _phase_table(joined, cov["usd"], cov["isl"])
        + _deepdive_section(joined, cov["usd"])
        + _delegation_section(signals)
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(heading)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(heading)}</h1>
<p class="sub">Run <code>{_esc(run_id)}</code> | generated {generated} from
{sources}</p>
<nav>{nav}</nav>
{_headline_cards(cov, outcome)}
{_coverage_section(cov, outcome)}
{_performance_section(ladder, no_ledger)}
{_outcome_section(joined, cov["usd"], outcome, no_ledger)}
{ledger_sections}
<h2>How to reproduce this</h2>
<p class="lede">Everything above is computed from {"the file" if len(read) == 1 else "two files"}
this run wrote. Nothing is modelled or carried over from another run. Regenerate with:</p>
<pre class="mono">python3 -m hyperloom.inference_optimizer.tools.render_geak_html_report \\
    --reports-dir {_esc(calls_path.parent)} -o {_esc(calls_path.parent / DEFAULT_OUTPUT)}</pre>
</div>
<script>const AGENTS={_agent_payload(agents)};{JS}</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, required=True, help="Run's reports/ directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="HTML file to write")
    parser.add_argument("--title", default=None, help="Override the document heading")
    args = parser.parse_args(argv)

    calls_path = args.reports_dir / CALLS_FILENAME
    outcome_path = args.reports_dir / OUTCOME_FILENAME
    if not calls_path.is_file() and not outcome_path.is_file():
        print(f"error: neither {CALLS_FILENAME} nor {OUTCOME_FILENAME} in {args.reports_dir}", file=sys.stderr)
        return 2
    if not calls_path.is_file():
        print(f"warning: no {CALLS_FILENAME}; rendering the outcome half only", file=sys.stderr)
    output = args.output or args.reports_dir / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(calls_path, outcome_path, args.title), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
