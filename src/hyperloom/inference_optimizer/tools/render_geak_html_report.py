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
import html
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALLS_FILENAME = "geak_calls.jsonl"
OUTCOME_FILENAME = "geak_outcome.json"
DEFAULT_OUTPUT = "geak_report.html"

# Matches the role sentence GEAK's role prompts open with ("You are the tech_lead",
# "You are Engineer r1_d2 (specialty=memory)"). geak_calls.jsonl truncates prompts,
# so this finds a role often but not always -- hence UNLABELLED rather than a guess.
ROLE_RE = re.compile(r"You are (?:the )?([A-Za-z][\w \-/]{0,48}?)(?:[.,\n(]|\s+for\s|\s+committing\s)")
SPECIALTY_RE = re.compile(r"specialty=(\w+)")
ROUND_RE = re.compile(r"\br(\d+)_d(\d+)\b")
UNLABELLED = "unlabelled"

# A conversation is split into this many equal position buckets to show how cost
# moves as context accumulates. Ten is enough to see the trend and few enough to read.
DECILES = 10
# Agents shorter than this have too few calls for a position curve to mean anything.
MIN_CALLS_FOR_CURVE = 10
# Per-agent call rows embedded for the drill-down. The ledger stays the full record;
# this only bounds the size of the HTML.
MAX_CALLS_EMBEDDED = 400


def load_calls(path: Path) -> list[dict[str, Any]]:
    """Read the ledger, skipping lines that are not JSON objects."""
    rows: list[dict[str, Any]] = []
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


def _num(value: Any) -> float:
    """Coerce to float, treating anything non-numeric as 0.0 for summation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    return float(value)


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


def agent_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the call rows into one record per (phase, agent), ordered by spend."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("phase", "")), str(row.get("agent", "")))].append(row)

    agents: list[dict[str, Any]] = []
    for (phase, agent), calls in grouped.items():
        calls.sort(key=lambda r: (_num(r.get("call_index")), str(r.get("ts", ""))))
        isls = [_num(c.get("isl")) for c in calls]
        tools: dict[str, int] = defaultdict(int)
        for call in calls:
            for tool in call.get("tools") or []:
                tools[tool if isinstance(tool, str) else json.dumps(tool)[:40]] += 1
        usd = sum(_num(c.get("usd")) for c in calls)
        seconds = sum(_num(c.get("dt_s")) for c in calls)
        agents.append(
            {
                "phase": phase,
                "agent": agent,
                "calls": len(calls),
                "usd": usd,
                "seconds": seconds,
                "isl": sum(isls),
                "osl": sum(_num(c.get("osl")) for c in calls),
                "thinking": sum(_num(c.get("thinking")) for c in calls),
                "tool_calls": sum(tools.values()),
                "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
                "first_isl": isls[0] if isls else 0.0,
                "last_isl": isls[-1] if isls else 0.0,
                "median_isl": statistics.median(isls) if isls else 0.0,
                "usd_per_call": usd / len(calls) if calls else 0.0,
                "started": str(calls[0].get("ts", "")) if calls else "",
                "models": sorted({str(c.get("model")) for c in calls if c.get("model")}),
                **derive_role(str(calls[0].get("prompt", "")) if calls else ""),
                "rows": calls[:MAX_CALLS_EMBEDDED],
                "rows_truncated": max(0, len(calls) - MAX_CALLS_EMBEDDED),
            }
        )
    agents.sort(key=lambda a: -a["usd"])
    return agents


def phase_records(rows: list[dict[str, Any]], agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per phase, with the agent roster that produced its spend."""
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in agents:
        by_phase[agent["phase"]].append(agent)

    calls_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        calls_by_phase[str(row.get("phase", ""))].append(row)

    phases: list[dict[str, Any]] = []
    for phase, roster in by_phase.items():
        calls = calls_by_phase[phase]
        stamps = sorted(str(c.get("ts", "")) for c in calls if c.get("ts"))
        tools: dict[str, int] = defaultdict(int)
        for agent in roster:
            for tool, count in agent["tools"].items():
                tools[tool] += count
        phases.append(
            {
                "phase": phase,
                "agents": len(roster),
                "calls": len(calls),
                "usd": sum(a["usd"] for a in roster),
                "seconds": sum(a["seconds"] for a in roster),
                "isl": sum(a["isl"] for a in roster),
                "osl": sum(a["osl"] for a in roster),
                "tool_calls": sum(a["tool_calls"] for a in roster),
                "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
                "first_ts": stamps[0] if stamps else "",
                "last_ts": stamps[-1] if stamps else "",
                "roster": roster,
                "cost_curve": cost_curve(roster),
                "concentration": concentration(roster),
            }
        )
    phases.sort(key=lambda p: -p["usd"])
    return phases


def cost_curve(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spend and median ISL by position within a conversation.

    Every agent is stretched onto the same 0-9 scale, so this answers "does a call
    get more expensive the longer its conversation runs?" without being dominated by
    whichever agent happened to run longest. Agents too short to bucket are excluded,
    and the count of those is reported so the exclusion is visible.
    """
    spend: dict[int, float] = defaultdict(float)
    islers: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    used = 0
    for agent in roster:
        calls = agent["rows"]
        total = len(calls)
        if total < MIN_CALLS_FOR_CURVE:
            continue
        used += 1
        for index, call in enumerate(calls):
            bucket = min(DECILES - 1, int(DECILES * index / total))
            spend[bucket] += _num(call.get("usd"))
            islers[bucket].append(_num(call.get("isl")))
            counts[bucket] += 1
    if not used:
        return []
    return [
        {
            "decile": bucket,
            "usd": spend[bucket],
            "calls": counts[bucket],
            "median_isl": statistics.median(islers[bucket]) if islers[bucket] else 0.0,
        }
        for bucket in range(DECILES)
    ]


def concentration(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cumulative share of phase spend as agents are added most-expensive first."""
    total = sum(a["usd"] for a in roster)
    if total <= 0:
        return []
    running = 0.0
    points: list[dict[str, Any]] = []
    for rank, agent in enumerate(sorted(roster, key=lambda a: -a["usd"]), start=1):
        running += agent["usd"]
        points.append({"rank": rank, "cum_pct": running / total * 100.0})
    return points


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


def delegation_signals(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-agent signals bearing on 'could a smaller model have done this?'.

    These are observations, not a recommendation. A low output-per-call agent spent
    its budget reading rather than writing; a single-tool agent ran a mechanical loop.
    Both are reasons to *look* at an agent, and neither is evidence on its own that a
    cheaper model would have reached the same result.
    """
    signals: list[dict[str, Any]] = []
    for agent in agents:
        calls = max(1, agent["calls"])
        signals.append(
            {
                "phase": agent["phase"],
                "agent": agent["agent"],
                "role": agent["role"],
                "usd": agent["usd"],
                "calls": agent["calls"],
                "osl_per_call": agent["osl"] / calls,
                "isl_per_call": agent["isl"] / calls,
                "output_share": (agent["osl"] / agent["isl"]) if agent["isl"] else None,
                "tool_variety": len(agent["tools"]),
                "top_tool": next(iter(agent["tools"]), None),
                "thinking": agent["thinking"],
            }
        )
    signals.sort(key=lambda s: -s["usd"])
    return signals


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


def _esc(value: Any) -> str:
    """HTML-escape, and fold every non-ASCII character to a numeric entity.

    The page is written UTF-8 and declares it, but it gets opened from shared
    storage, out of archives and through viewers that ignore the declaration --
    and a mis-decoded em dash renders as mojibake with no clue as to why. Pure
    ASCII bytes cannot be mis-decoded, so the document is kept pure ASCII and
    anything outside it travels as an entity the browser resolves itself.
    """
    text = html.escape(str(value), quote=True)
    if text.isascii():
        return text
    return "".join(ch if ord(ch) < 128 else f"&#{ord(ch)};" for ch in text)


def _usd(value: Any) -> str:
    number = _num(value)
    return f"${number:,.2f}" if number >= 0.01 or number == 0 else f"${number:,.4f}"


def _int(value: Any) -> str:
    return f"{int(_num(value)):,}"


def _hms(seconds: Any) -> str:
    total = int(_num(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _pct(value: Any, digits: int = 2) -> str:
    """Format a percentage, or say plainly that it was never measured."""
    if value is None:
        return '<span class="none">not measured</span>'
    return f"{_num(value):+.{digits}f}%"


def _bar(fraction: float, tone: str = "spend") -> str:
    width = max(0.0, min(100.0, fraction * 100.0))
    return f'<span class="bar {tone}"><i style="width:{width:.2f}%"></i></span>'


def _sparkline(values: list[float], width: int = 220, height: int = 44) -> str:
    """A dependency-free inline SVG line. Empty input renders nothing, not a flat line."""
    if not values:
        return '<span class="none">no data</span>'
    top = max(values)
    bottom = min(values)
    span = (top - bottom) or 1.0
    step = width / max(1, len(values) - 1)
    points = " ".join(
        f"{index * step:.1f},{height - (value - bottom) / span * (height - 6) - 3:.1f}"
        for index, value in enumerate(values)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" role="img"><polyline points="{points}"/></svg>'
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0da;--card:#fff;
--accent:#2d6cdf;--good:#1a7f4b;--bad:#b3261e;--warn:#8a6d00;--spend:#2d6cdf;--time:#8a5cd0;}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#eceae6;--mut:#9d9a94;--line:#2e2e34;
--card:#1d1d22;--accent:#7aa5f5;--good:#5cc98d;--bad:#f0857c;--warn:#e0be4e;--spend:#7aa5f5;--time:#b394ea;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:23px;margin:0 0 4px}h2{font-size:18px;margin:38px 0 6px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}
.sub{color:var(--mut);margin:0 0 18px}
.lede{color:var(--mut);margin:0 0 14px;max-width:80ch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:20px;font-weight:600;margin-top:3px}
.card .n{color:var(--mut);font-size:11.5px;margin-top:2px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:7px;padding:11px 14px;margin:14px 0}
.note b{display:block;margin-bottom:3px}
.note ul{margin:6px 0 0;padding-left:20px}.note li{margin:2px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:hover{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.scroll{overflow-x:auto}
.bar{display:inline-block;width:74px;height:7px;background:var(--line);border-radius:4px;
overflow:hidden;vertical-align:middle;margin-left:7px}
.bar i{display:block;height:100%;background:var(--spend)}
.bar.time i{background:var(--time)}
tr.ref td{background:var(--card)}
.good{color:var(--good)}.bad{color:var(--bad)}.none{color:var(--mut);font-style:italic}
.mut{color:var(--mut)}
.spark{display:block;margin-top:4px}
.spark polyline{fill:none;stroke:var(--accent);stroke-width:1.8;vector-effect:non-scaling-stroke}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
details{background:var(--card);border:1px solid var(--line);border-radius:9px;margin:10px 0;padding:0}
summary{cursor:pointer;padding:11px 14px;font-weight:600;list-style:none;display:flex;
justify-content:space-between;gap:14px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary::before{content:"\\25b8";color:var(--mut);margin-right:8px;transition:transform .15s}
details[open]>summary::before{transform:rotate(90deg);display:inline-block}
summary .r{color:var(--mut);font-weight:500;font-size:12.5px}
.body{padding:0 14px 14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.tag{display:inline-block;background:color-mix(in srgb,var(--accent) 13%,transparent);
color:var(--accent);border-radius:4px;padding:0 6px;font-size:11px;margin-left:5px}
.tag.q{background:color-mix(in srgb,var(--mut) 16%,transparent);color:var(--mut)}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:9px 0;margin-bottom:10px;z-index:5;font-size:13px}
nav a{color:var(--accent);text-decoration:none;margin-right:16px}
nav a:hover{text-decoration:underline}
.calls{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:7px;margin-top:8px}
.calls table{margin:0}.calls th{position:sticky;top:0;background:var(--card)}
button.drill{background:none;border:1px solid var(--line);border-radius:5px;color:var(--accent);
cursor:pointer;font-size:11.5px;padding:2px 8px}
button.drill:hover{border-color:var(--accent)}
.pw{max-width:56ch;overflow:hidden;text-overflow:ellipsis;color:var(--mut);font-size:11.5px}
"""

JS = """
function fmtUsd(v){return v==null?'-':'$'+Number(v).toFixed(4);}
function fmtInt(v){return v==null?'-':Number(v).toLocaleString();}
function drill(key,btn){
  var host=document.getElementById('calls-'+key);
  if(host.dataset.built){host.hidden=!host.hidden;btn.textContent=host.hidden?'calls':'hide';return;}
  var a=AGENTS[key];
  var h='<table><thead><tr><th>#</th><th>time</th><th>ISL</th><th>OSL</th><th>think</th>'
       +'<th>secs</th><th>USD</th><th>tools</th><th style="text-align:left">first line of prompt</th>'
       +'</tr></thead><tbody>';
  a.rows.forEach(function(r){
    var p=(r.prompt||'').split('\\n').find(function(x){return x.trim();})||'';
    h+='<tr><td>'+r.i+'</td><td class="mono">'+(r.ts||'').slice(11,19)+'</td><td>'+fmtInt(r.isl)
      +'</td><td>'+fmtInt(r.osl)+'</td><td>'+fmtInt(r.think)+'</td><td>'+(r.dt==null?'-':r.dt.toFixed(1))
      +'</td><td>'+fmtUsd(r.usd)+'</td><td>'+(r.tools||[]).join(', ')
      +'</td><td class="pw" title="'+p.replace(/"/g,'&quot;').slice(0,400)+'">'+p.slice(0,110)+'</td></tr>';
  });
  h+='</tbody></table>';
  if(a.more){h+='<p class="mut" style="padding:8px 10px;margin:0">'+a.more
    +' further calls are in geak_calls.jsonl; this table is capped so the page stays loadable.</p>';}
  host.innerHTML=h;host.dataset.built='1';host.hidden=false;btn.textContent='hide';
}
"""


def _coverage_section(cov: dict[str, Any], outcome: dict[str, Any] | None) -> str:
    """The banner that says what these numbers are and are not. Always rendered."""
    caveats: list[str] = []
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
    if not cov["thinking"]:
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
    caveats.append(
        "Roles are derived from each agent's first prompt, not read from the workflow record. "
        "They are a reading aid; the agent id is the identifier."
    )
    items = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    return (
        '<div class="note"><b>Coverage - read this before quoting a number</b>'
        f"<div>{cov['calls']:,} API calls across {cov['agents']} agents and {cov['phases']} phases, "
        f"model{'s' if len(cov['models']) != 1 else ''} {_esc(', '.join(cov['models']) or 'not recorded')}."
        f"</div><ul>{items}</ul></div>"
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


def _outcome_section(joined: list[dict[str, Any]], total_usd: float, outcome: dict[str, Any] | None) -> str:
    """The join the whole report exists for: spend beside measured result."""
    if outcome is None:
        return (
            '<h2 id="bought">What each phase bought</h2>'
            f'<p class="lede">Not available: no <code>{OUTCOME_FILENAME}</code> beside the ledger. '
            "Spend is still reported below; the result each phase measured is not, because nothing "
            "in the ledger records it and inferring it would be a guess.</p>"
        )
    rows = []
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
            ab = [t["integration"] for t in tasks if t.get("integration")]
            if ab:
                best = max(ab, key=lambda i: _num(i.get("e2e_delta_pct")))
                delta = _num(best.get("e2e_delta_pct"))
                tone = "good" if delta > 0 else "bad" if delta < 0 else "mut"
                gate = best.get("gate")
                acc = (
                    "accuracy held" if _num(best.get("gsm8k_cand")) >= _num(best.get("gsm8k_ref")) else "accuracy fell"
                )
                bought = (
                    f'<span class="{tone}">{delta:+.2f}%</span> <span class="mut">end-to-end, '
                    f"measured A/B of the kernel this run wrote "
                    f"({_esc(best.get('candidate'))}, {_num(best.get('isolated_speedup')):.4f}x isolated, "
                    f"{acc}"
                    + (f", gate {_esc(gate)}" if gate else "")
                    + f"; ceiling was {_num(best.get('amdahl_ceiling_pct')):.2f}%)</span>"
                )
                efficiency = _usd(phase["usd"] / delta) if delta > 0 else '<span class="none">n/a</span>'
                rows.append(
                    f"<tr><td><b>{_esc(phase['phase'])}</b></td>"
                    f"<td>{_usd(phase['usd'])}{_bar(share)}</td>"
                    f"<td>{share * 100:.1f}%</td>"
                    f'<td style="text-align:left">{bought}</td>'
                    f"<td>{efficiency}</td></tr>"
                )
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
        rows.append(
            f"<tr><td><b>{_esc(phase['phase'])}</b></td>"
            f"<td>{_usd(phase['usd'])}{_bar(share)}</td>"
            f"<td>{share * 100:.1f}%</td>"
            f'<td style="text-align:left">{bought}</td>'
            f"<td>{efficiency}</td></tr>"
        )
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
    return (
        '<h2 id="bought">What each phase bought, and what it cost to buy it</h2>'
        '<p class="lede">Spend comes from the call ledger; the result comes from the artifacts the run '
        "wrote at each phase boundary. A phase with no measurement is marked as such and is never "
        "shown as zero.</p>"
        '<div class="scroll"><table><thead><tr><th>Phase</th><th>Cost</th><th>Share</th>'
        "<th>Measured result</th><th>Cost per +1%</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{estimate_note}"
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
    silent = [p["phase"] for p in joined if p["outcome"] is None]
    return {
        "steps": steps,
        "total_gain": total_gain,
        "kernels": kernel_only,
        "silent": silent,
        "summary": outcome.get("summary") or {},
    }


def _performance_section(ladder: dict[str, Any]) -> str:
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
        cost = _usd(step["usd"]) if step["usd"] is not None else '<span class="none">not billed</span>'
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
        (p, i)
        for p in ladder["kernels"]
        for i in [
            max(
                (t["integration"] for t in (p["outcome"].get("tasks") or []) if t.get("integration")),
                key=lambda x: _num(x.get("e2e_delta_pct")),
                default=None,
            )
        ]
        if i
    ]
    unvalidated = [
        p for p in ladder["kernels"] if not any(t.get("integration") for t in (p["outcome"].get("tasks") or []))
    ]
    for phase, integ in validated:
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
            "The run wrote its own candidate (" + _esc(integ.get("candidate")) + "), served it "
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
            f"{_esc(p['phase'])} {max((_num(t['amdahl_ceiling_e2e_pct']) for t in (p['outcome'].get('tasks') or []) if t['present']), default=0.0):+.2f}%"
            for p in unvalidated
        )
        notes.append(
            "<b>The other kernel phases contributed no measured end-to-end throughput.</b> No "
            "candidate they wrote reached an end-to-end A/B, and among the library backends the "
            "incumbent stayed fastest, so the Amdahl ceiling on any gain is "
            f"{detail}. That is a measured result, not a missing one: the work ran, was "
            "benchmarked, and did not beat what was already there."
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


def _anatomy(phase: dict[str, Any]) -> str:
    """Where inside a phase the cost is actually incurred."""
    curve = phase["cost_curve"]
    conc = phase["concentration"]
    blocks: list[str] = []

    if curve:
        spend = [c["usd"] for c in curve]
        isls = [c["median_isl"] for c in curve]
        total = sum(spend) or 1.0
        rows = "".join(
            f"<tr><td>{c['decile'] * 10}&ndash;{c['decile'] * 10 + 10}%</td>"
            f"<td>{c['calls']:,}</td><td>{_int(c['median_isl'])}</td>"
            f"<td>{_usd(c['usd'])}{_bar(c['usd'] / max(spend))}</td>"
            f"<td>{c['usd'] / total * 100:.1f}%</td></tr>"
            for c in curve
        )
        growth = (isls[-1] / isls[0]) if isls and isls[0] else None
        lede = (
            f"Each call re-sends the conversation so far, so the input grows as an agent works. "
            f"Median input goes from {_int(isls[0])} tokens in the first tenth of a conversation to "
            f"{_int(isls[-1])} in the last"
            + (f", a {growth:.2f}x growth" if growth else "")
            + f"; the last tenth costs {spend[-1] / total * 100:.1f}% of the phase against "
            f"{spend[0] / total * 100:.1f}% for the first."
        )
        blocks.append(
            "<div><h3>Cost by position in the conversation</h3>"
            f'<p class="lede">{_esc(lede)}</p>'
            "<table><thead><tr><th>Position</th><th>Calls</th><th>Median ISL</th><th>Spend</th>"
            f"<th>Share</th></tr></thead><tbody>{rows}</tbody></table>"
            f'<p class="mut">Median input tokens across the conversation:</p>{_sparkline(isls)}</div>'
        )
    else:
        blocks.append(
            "<div><h3>Cost by position in the conversation</h3>"
            f'<p class="none">Not shown: no agent in this phase ran {MIN_CALLS_FOR_CURVE} or more calls, '
            "which is too few for a position curve to mean anything.</p></div>"
        )

    if conc:

        def agents_to(pct: float) -> int:
            for point in conc:
                if point["cum_pct"] >= pct:
                    return point["rank"]
            return len(conc)

        half, most = agents_to(50.0), agents_to(80.0)
        rows = "".join(
            f"<tr><td>top {p['rank']}</td><td>{p['cum_pct']:.1f}%{_bar(p['cum_pct'] / 100)}</td></tr>"
            for p in conc
            if p["rank"] in {1, 3, 5, 10, 20, max(1, len(conc) // 2), len(conc)}
        )
        blocks.append(
            "<div><h3>How concentrated the spend is</h3>"
            f'<p class="lede">{half} of this phase\'s {len(conc)} agents account for half its cost, and '
            f"{most} account for 80%. Ranked most expensive first.</p>"
            f"<table><thead><tr><th>Agents</th><th>Cumulative share</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    if phase["tools"]:
        total_tools = sum(phase["tools"].values()) or 1
        rows = "".join(
            f"<tr><td><code>{_esc(name)}</code></td><td>{count:,}</td>"
            f"<td>{count / total_tools * 100:.1f}%{_bar(count / max(phase['tools'].values()))}</td></tr>"
            for name, count in list(phase["tools"].items())[:10]
        )
        blocks.append(
            "<div><h3>What the agents actually did</h3>"
            f'<p class="lede">{total_tools:,} tool calls across {phase["calls"]:,} API calls '
            f"({total_tools / max(1, phase['calls']):.2f} per call). A phase dominated by one tool is "
            "running a mechanical loop; a varied mix is exploratory work.</p>"
            f"<table><thead><tr><th>Tool</th><th>Calls</th><th>Share</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    return f'<div class="grid2">{"".join(blocks)}</div>'


def _roster(phase: dict[str, Any]) -> str:
    """Every agent in the phase, ranked by spend, each drillable down to its calls."""
    total = phase["usd"] or 1.0
    peak = max((a["usd"] for a in phase["roster"]), default=1.0) or 1.0
    rows: list[str] = []
    for agent in phase["roster"]:
        key = agent_key(agent)
        tag = (
            f'<span class="tag">{_esc(agent["role"])}</span>'
            if agent["role"] != UNLABELLED
            else f'<span class="tag q">{UNLABELLED}</span>'
        )
        if agent["specialty"]:
            tag += f'<span class="tag q">{_esc(agent["specialty"])}</span>'
        growth = f"{agent['last_isl'] / agent['first_isl']:.2f}x" if agent["first_isl"] else "-"
        rows.append(
            f"<tr><td><code>{_esc(agent['agent'][:12])}</code>{tag}</td>"
            f"<td>{agent['calls']:,}</td>"
            f"<td>{_usd(agent['usd'])}{_bar(agent['usd'] / peak)}</td>"
            f"<td>{agent['usd'] / total * 100:.1f}%</td>"
            f"<td>{_usd(agent['usd_per_call'])}</td>"
            f"<td>{_hms(agent['seconds'])}</td>"
            f"<td>{_int(agent['median_isl'])}</td>"
            f"<td>{growth}</td>"
            f"<td>{_int(agent['osl'])}</td>"
            f"<td>{agent['tool_calls']:,}</td>"
            f'<td><button class="drill" onclick="drill(\'{key}\',this)">calls</button></td></tr>'
            f'<tr><td colspan="11" style="padding:0">'
            f'<div class="calls" id="calls-{key}" hidden></div></td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Agent</th><th>Calls</th><th>Cost</th><th>Share</th>'
        "<th>$/call</th><th>Recorded time</th><th>Median ISL</th><th>ISL growth</th><th>Output</th>"
        f"<th>Tools</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def agent_key(agent: dict[str, Any]) -> str:
    """A DOM-safe id for an agent's drill-down panel."""
    raw = f"{agent['phase']}-{agent['agent']}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def _deepdive_section(joined: list[dict[str, Any]], total_usd: float) -> str:
    """One collapsible block per phase, expensive first, the top two open by default."""
    blocks: list[str] = []
    for rank, phase in enumerate(joined):
        share = phase["usd"] / total_usd * 100 if total_usd else 0.0
        window = ""
        if phase["first_ts"] and phase["last_ts"]:
            window = f"{phase['first_ts'][11:19]}&ndash;{phase['last_ts'][11:19]} UTC"
        blocks.append(
            f"<details{' open' if rank < 2 else ''}>"
            f"<summary><span>{_esc(phase['phase'])}</span>"
            f'<span class="r">{_usd(phase["usd"])} | {share:.1f}% of spend | '
            f"{phase['agents']} agents | {phase['calls']:,} calls | "
            f"{_int(phase['isl'])} in / {_int(phase['osl'])} out"
            + (f" | {window}" if window else "")
            + "</span></summary>"
            f'<div class="body">{_anatomy(phase)}<h3>Agents, most expensive first</h3>{_roster(phase)}</div>'
            "</details>"
        )
    return (
        '<h2 id="deep">Inside each phase</h2>'
        '<p class="lede">Expand a phase to see where its money went: how cost moves as a conversation '
        "grows, how few agents carry the total, what tools the work actually consisted of, and the "
        "full agent roster. Any agent row opens into its own API calls.</p>" + "".join(blocks)
    )


def _delegation_section(signals: list[dict[str, Any]], limit: int = 20) -> str:
    """Signals bearing on 'could a cheaper model have done this?' -- signals, not a verdict."""
    rows = "".join(
        f"<tr><td><code>{_esc(s['agent'][:12])}</code>"
        + (f'<span class="tag">{_esc(s["role"])}</span>' if s["role"] != UNLABELLED else "")
        + f"</td><td>{_esc(s['phase'])}</td><td>{_usd(s['usd'])}</td><td>{s['calls']:,}</td>"
        f"<td>{_int(s['isl_per_call'])}</td><td>{_int(s['osl_per_call'])}</td>"
        + (
            f"<td>{s['output_share'] * 100:.3f}%</td>"
            if s["output_share"] is not None
            else '<td><span class="none">n/a</span></td>'
        )
        + f"<td>{s['tool_variety']}</td><td><code>{_esc(s['top_tool'] or '-')}</code></td></tr>"
        for s in signals[:limit]
    )
    return (
        '<h2 id="delegate">Signals for delegating work to a cheaper model</h2>'
        '<p class="lede">These are observations, not a recommendation. An agent that reads far more '
        "than it writes spent its budget on context rather than reasoning, and one that uses a single "
        "tool repeatedly is running a mechanical loop. Both are reasons to <em>look</em> at an agent; "
        "neither is evidence that a smaller model would have reached the same result. The only way to "
        "know that is to run the arm and compare the measured throughput.</p>"
        '<div class="scroll"><table><thead><tr><th>Agent</th><th>Phase</th><th>Cost</th><th>Calls</th>'
        "<th>Input/call</th><th>Output/call</th><th>Output share</th><th>Tool variety</th>"
        f"<th>Main tool</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f'<p class="mut">Showing the {min(limit, len(signals))} most expensive agents of {len(signals)}.</p>'
    )


def _agent_payload(agents: list[dict[str, Any]]) -> str:
    """The per-call rows the drill-down renders, keyed by DOM id."""
    payload = {
        agent_key(a): {
            "more": a["rows_truncated"],
            "rows": [
                {
                    "i": int(_num(r.get("call_index"))),
                    "ts": r.get("ts"),
                    "isl": _num(r.get("isl")),
                    "osl": _num(r.get("osl")),
                    "think": _num(r.get("thinking")),
                    "dt": _num(r.get("dt_s")),
                    "usd": _num(r.get("usd")),
                    "tools": [t for t in (r.get("tools") or []) if isinstance(t, str)],
                    "prompt": (r.get("prompt") or "")[:400],
                }
                for r in a["rows"]
            ],
        }
        for a in agents
    }
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def _phase_table(joined: list[dict[str, Any]], total_usd: float, total_isl: float) -> str:
    peak = max((p["usd"] for p in joined), default=1.0) or 1.0
    rows = "".join(
        f"<tr><td><b>{_esc(p['phase'])}</b></td><td>{p['agents']}</td><td>{p['calls']:,}</td>"
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
    heading = title or "GEAK run - where the time and the money went"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(heading)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(heading)}</h1>
<p class="sub">Run <code>{_esc(run_id)}</code> | generated {generated} from
<code>{_esc(calls_path.name)}</code>
{"and <code>" + _esc(outcome_path.name) + "</code>" if outcome else ""}</p>
<nav><a href="#perf">Throughput</a><a href="#bought">Cost vs result</a><a href="#phases">Spend by phase</a>
<a href="#deep">Inside each phase</a><a href="#delegate">Delegation signals</a></nav>
{_headline_cards(cov, outcome)}
{_coverage_section(cov, outcome)}
{_performance_section(ladder)}
{_outcome_section(joined, cov["usd"], outcome)}
{_phase_table(joined, cov["usd"], cov["isl"])}
{_deepdive_section(joined, cov["usd"])}
{_delegation_section(signals)}
<h2>How to reproduce this</h2>
<p class="lede">Everything above is computed from two files this run wrote. Nothing is modelled or
carried over from another run. Regenerate with:</p>
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
    if not calls_path.is_file():
        print(f"error: no {CALLS_FILENAME} in {args.reports_dir}", file=sys.stderr)
        return 2
    output = args.output or args.reports_dir / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(calls_path, args.reports_dir / OUTCOME_FILENAME, args.title), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
