#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-phase deep dive shared by the GEAK and Hyperloom HTML reports.

Both reports answer the same question one level down from "which phase cost the
most": *inside* an expensive phase, where did the money actually go? The answer
is the same four panels either way -- how cost moves as a conversation grows,
how few agents carry the total, what tools the work consisted of, and the agent
roster with every agent drillable to its own API calls -- so the machinery lives
here once and a number means the same thing on either page.

The two ledgers are not the same shape, so callers normalise their rows into the
small record this module works on before calling anything here:

======================  =========================================================
key                     meaning
======================  =========================================================
``phase``               the phase the call belongs to
``agent``               a stable id for the conversation the call is part of
``call_index``          position of the call within that conversation, from 0
``ts``                  ISO timestamp
``isl`` / ``osl``       input (with cache) and output tokens
``thinking``            thinking tokens; a subset of ``osl`` or beside it, per ledger
``usd``                 derived cost
``dt_s``                seconds attributed to the call, 0 when not recorded
``tools``               list of tool names invoked by the call
``model``               model id
``prompt``              first 400-odd characters of what the model was handed
======================  =========================================================

Identity is the one thing the two ledgers genuinely disagree on, so it is
injected rather than assumed. GEAK's ledger carries prompt text and no label, so
it recovers a role with a regex; Hyperloom's carries a ``role`` field that is
populated for some components and empty for others, so it falls back to the
task path. Callers pass a ``role_of`` resolver and this module never guesses:
an agent whose role could not be established is :data:`UNLABELLED`, and the
coverage banner is expected to say how many those were.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from typing import Any, Callable

from hyperloom.inference_optimizer.tools._report_html import bar as _bar
from hyperloom.inference_optimizer.tools._report_html import esc as _esc
from hyperloom.inference_optimizer.tools._report_html import fmt_hms as _hms
from hyperloom.inference_optimizer.tools._report_html import fmt_int as _int
from hyperloom.inference_optimizer.tools._report_html import fmt_usd as _usd
from hyperloom.inference_optimizer.tools._report_html import num as _num
from hyperloom.inference_optimizer.tools._report_html import sparkline as _sparkline

#: An agent whose role could not be established from the ledger. Never a guess.
UNLABELLED = "unlabelled"

# A conversation is split into this many equal position buckets to show how cost
# moves as context accumulates. Ten is enough to see the trend and few enough to read.
DECILES = 10
# Agents shorter than this have too few calls for a position curve to mean anything.
MIN_CALLS_FOR_CURVE = 10
# Per-agent call rows embedded for the drill-down. The ledger stays the full record;
# this only bounds the size of the HTML.
MAX_CALLS_EMBEDDED = 400

#: A ``role_of`` resolver: given an agent's calls, name it. Must not guess.
RoleResolver = Callable[[list[dict[str, Any]]], dict[str, Any]]


def agent_records(rows: list[dict[str, Any]], role_of: RoleResolver) -> list[dict[str, Any]]:
    """Fold the call rows into one record per (phase, agent), ordered by spend.

    Args:
        rows: Normalised call rows.
        role_of: Resolver handed the agent's calls in order; returns at least a
            ``role`` key, optionally ``specialty``, ``round`` and ``derived``.

    Returns:
        One record per agent, most expensive first.
    """
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
        identity = {"role": UNLABELLED, "specialty": None, "round": None, "derived": True}
        identity.update(role_of(calls) or {})
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
                **identity,
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


def unlabelled_count(agents: list[dict[str, Any]]) -> int:
    """How many agents could not be given a role. For the coverage banner."""
    return sum(1 for a in agents if a["role"] == UNLABELLED)


def anatomy_html(phase: dict[str, Any]) -> str:
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
            '<div class="scroll"><table><thead><tr><th>Position</th><th>Calls</th><th>Median ISL</th>'
            "<th>Spend</th><th>Share</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
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
            '<div class="scroll"><table><thead><tr><th>Agents</th><th>Cumulative share</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div></div>"
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
            '<div class="scroll"><table><thead><tr><th>Tool</th><th>Calls</th><th>Share</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div></div>"
        )
    return f'<div class="grid2">{"".join(blocks)}</div>'


def agent_key(agent: dict[str, Any]) -> str:
    """A DOM-safe id for an agent's drill-down panel."""
    raw = f"{agent['phase']}-{agent['agent']}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def roster_html(phase: dict[str, Any]) -> str:
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


def deepdive_html(
    joined: list[dict[str, Any]],
    total_usd: float,
    code_of: Callable[[dict[str, Any]], str] | None = None,
    lede: str | None = None,
    extra_of: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """One collapsible block per phase, expensive first, the top two open by default.

    Args:
        joined: Phase records, most expensive first.
        total_usd: The bill the shares are taken against.
        code_of: Optional extra label for a phase heading -- GEAK uses it to name
            the kernel a HeadKernel phase went after, which the phase name omits.
        lede: Optional replacement for the introductory paragraph.
        extra_of: Optional extra panel appended inside a phase block, after the
            roster -- the Hyperloom report uses it for its task-path tree.
    """
    blocks: list[str] = []
    for rank, phase in enumerate(joined):
        share = phase["usd"] / total_usd * 100 if total_usd else 0.0
        window = ""
        if phase["first_ts"] and phase["last_ts"]:
            window = f"{phase['first_ts'][11:19]}&ndash;{phase['last_ts'][11:19]} UTC"
        code = code_of(phase) if code_of else ""
        blocks.append(
            f"<details{' open' if rank < 2 else ''}>"
            f"<summary><span>{_esc(phase['phase'])}"
            + (f" <code>{_esc(code)}</code>" if code else "")
            + "</span>"
            f'<span class="r">{_usd(phase["usd"])} | {share:.1f}% of spend | '
            f"{phase['agents']} agents | {phase['calls']:,} calls | "
            f"{_int(phase['isl'])} in / {_int(phase['osl'])} out"
            + (f" | {window}" if window else "")
            + "</span></summary>"
            f'<div class="body">{anatomy_html(phase)}<h3>Agents, most expensive first</h3>'
            f"{roster_html(phase)}{extra_of(phase) if extra_of else ''}</div>"
            "</details>"
        )
    intro = lede or (
        "Expand a phase to see where its money went: how cost moves as a conversation grows, how "
        "few agents carry the total, what tools the work actually consisted of, and the full agent "
        "roster. Any agent row opens into its own API calls."
    )
    return f'<h2 id="deep">Inside each phase</h2><p class="lede">{intro}</p>' + "".join(blocks)


def delegation_html(signals: list[dict[str, Any]], limit: int = 20) -> str:
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


def agent_payload(agents: list[dict[str, Any]]) -> str:
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


def drill_js(ledger_name: str) -> str:
    """The drill-down script. ``ledger_name`` is the file a capped table points at."""
    return """
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
    +' further calls are in LEDGER_NAME; this table is capped so the page stays loadable.</p>';}
  host.innerHTML=h;host.dataset.built='1';host.hidden=false;btn.textContent='hide';
}
""".replace("LEDGER_NAME", ledger_name)
