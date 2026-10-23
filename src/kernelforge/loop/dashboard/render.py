# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a dashboard snapshot as one self-contained HTML page: no network, every agent text escaped.

Its one script reloads the page every minute, except while a message is being typed into its form, and keeps open
the rounds the reader opened. Without scripts a meta refresh does the reloading.
"""

from __future__ import annotations

import math
import time
from html import escape

REFRESH_SEC = 60

# Decision -> (status role, glyph, label). Status colours never carry meaning alone: the glyph and label go with them.
_GOOD = ("good", "✓", "kept")
_NEUTRAL = ("neutral", "–", "not faster")
_CRITICAL = ("critical", "✗", "wrong or broken")
_WARNING = ("warning", "⚠", "infrastructure")
_DECISIONS = {
    "KEEP": _GOOD,
    "REVERT_PERF": _NEUTRAL,
    "NO_CHANGES": ("neutral", "∅", "no change"),
    "REVERT_VALIDATION": _CRITICAL,
    "REVERT_VALIDATION_TIMEOUT": _CRITICAL,
    "REVERT_VALIDATION_ERROR": _CRITICAL,
    "BUILD_FAILED": _CRITICAL,
    "CRASH": _CRITICAL,
    "REVERT_INTEGRITY": _CRITICAL,
}

_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--neutral:#898781}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--series-1:#3987e5}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5}
body{margin:0}
.viz-root{background:var(--page);color:var(--text-primary);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
font-size:14px;padding:20px;min-height:100vh;box-sizing:border-box}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--text-secondary);margin-bottom:16px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:14px}
.tile .label{color:var(--text-secondary);font-size:13px}.tile .value{font-size:26px;font-weight:600;margin-top:4px}
.hero .value{font-size:38px;white-space:nowrap}.tile .note{color:var(--muted);font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top;overflow-wrap:anywhere}
th{color:var(--text-secondary);font-weight:600}td.num,th.num{text-align:right}
.status{display:inline-flex;gap:4px;align-items:center}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%}
.good{background:var(--good)}.warning{background:var(--warning)}.critical{background:var(--critical)}
.neutral{background:var(--neutral)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--text-secondary);font-size:12px;margin:6px 0}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font-size:12px;color:var(--text-secondary)}
details summary{cursor:pointer;color:var(--text-secondary);font-size:12px;margin-top:8px}
svg text{fill:var(--muted);font-size:11px}
details.round{border-top:1px solid var(--grid);padding:6px 0}
details.round>summary{font-size:14px;color:var(--text-primary);margin:0;list-style-position:inside}
details.round>summary .meta{color:var(--text-secondary)}
.round-body{margin:8px 0 4px 18px}.round-body h3{font-size:12px;color:var(--text-secondary);margin:10px 0 4px;
text-transform:uppercase;letter-spacing:.04em}.round-body pre{color:var(--text-primary);font-size:13px;
font-family:inherit;line-height:1.45}
form.say textarea{width:100%;box-sizing:border-box;font:inherit;padding:8px;border:1px solid var(--axis);
border-radius:6px;background:var(--page);color:var(--text-primary);resize:vertical}
form.say .buttons{display:flex;gap:8px;margin:8px 0 4px}
form.say button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--axis);cursor:pointer;
background:var(--surface-1);color:var(--text-primary)}form.say button.primary{background:var(--series-1);
border-color:var(--series-1);color:#fff}
"""

# Reload every REFRESH_SEC unless a message is half typed, and reopen the rounds that were open before the reload.
_SCRIPT = """
(function(){var k='forge-open';try{JSON.parse(sessionStorage.getItem(k)||'[]').forEach(function(id){
var d=document.getElementById(id);if(d)d.open=true;});}catch(e){}
setInterval(function(){var t=document.querySelector('form.say textarea');
if(t&&(t.value.trim()||document.activeElement===t))return;
var open=[].map.call(document.querySelectorAll('details[id][open]'),function(d){return d.id;});
try{sessionStorage.setItem(k,JSON.stringify(open));}catch(e){}location.reload();},%d);})();
""" % (REFRESH_SEC * 1000)


def _fmt(value, spec: str = ".4f", default: str = "—") -> str:
    if value is None:
        return default
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return escape(str(value))


def _duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}h {rest // 60:02d}m"


def _decision_style(decision: str) -> tuple[str, str, str]:
    return _DECISIONS.get(decision, _WARNING)


def _status_chip(decision: str) -> str:
    role, glyph, label = _decision_style(decision)
    return f'<span class="status"><span class="dot {role}"></span>{glyph} {escape(decision or label)}</span>'


def _tile(label: str, value: str, note: str = "", *, hero: bool = False) -> str:
    extra = f'<div class="note">{note}</div>' if note else ""
    css = "card tile hero" if hero else "card tile"
    return f'<div class="{css}"><div class="label">{escape(label)}</div><div class="value">{value}</div>{extra}</div>'


def _noise_tile(snapshot: dict) -> str:
    floor = snapshot.get("noise_floor") or {}
    if not floor:
        return ""
    value = floor.get("value")
    shown = f"{float(value):.2%}" if value is not None else "—"
    note = f"A-vs-A, {len(floor.get('scores') or [])} pairs at iteration {floor.get('iteration')}"
    if floor.get("failure"):
        note = f"last attempt failed: {floor['failure']}"
    return _tile("Noise floor", shown, escape(note))


def _keep_bar(bar) -> str:
    if not isinstance(bar, dict):
        return ""
    if bar.get("mode") == "paired":
        retried = f", {bar['retries']} arm run(s) repeated" if bar.get("retries") else ""
        return (
            f"paired ×{bar.get('pairs')}: lower {_fmt(bar.get('lower'))} vs {1 + float(bar.get('floor') or 0):.4f}"
            f"{retried}"
        )
    return f"needs {_fmt(bar.get('required'))}"


def _header(snapshot: dict) -> str:
    live, campaign = snapshot.get("live") or {}, snapshot.get("campaign") or {}
    status = live.get("status") or campaign.get("session_status") or "unknown"
    reason = live.get("termination_reason") or campaign.get("termination_reason") or ""
    rendered = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.get("rendered_at") or 0))
    parts = [f"status <strong>{escape(status)}</strong>"]
    if reason:
        parts.append(f"ended: {escape(reason)}")
    if live.get("anchor_label"):
        parts.append(f"scores are speedups over {escape(live['anchor_label'])}")
    parts.append(f"rendered {rendered}, refreshes every {REFRESH_SEC} s (not while you type a message)")
    title = escape(campaign.get("campaign_id") or snapshot.get("campaign_root") or "forge-loop")
    return f"<h1>forge-loop · {title}</h1><div class='sub'>{' · '.join(parts)}</div>"


def _best_tile(campaign: dict) -> str:
    best = campaign.get("best") or {}
    if best.get("mean_case_speedup") is not None:
        return _tile(
            "Best speedup",
            f"{_fmt(best['mean_case_speedup'])}×",
            f"iteration {escape(str(best.get('iteration') or '—'))}",
            hero=True,
        )
    start = campaign.get("search_start_mean_case_speedup")
    return _tile("Best speedup", f"{_fmt(start)}×" if start is not None else "—", "the starting kernel", hero=True)


def _time_left(snapshot: dict) -> float | None:
    """From the deadline when there is one, so a page re-rendered later does not show a stale countdown."""
    live = snapshot.get("live") or {}
    if live.get("status") not in (None, "running"):
        return 0.0
    if live.get("deadline_unix"):
        return float(live["deadline_unix"]) - float(snapshot.get("rendered_at") or time.time())
    return live.get("time_remaining_sec")


def _time_tile(snapshot: dict, deadline_text: str) -> str:
    live = snapshot.get("live") or {}
    if live.get("status") not in (None, "running"):
        reason = live.get("termination_reason") or (snapshot.get("campaign") or {}).get("termination_reason") or ""
        return _tile("Time left", "ended", escape(reason or str(live.get("status"))))
    return _tile("Time left", _duration(_time_left(snapshot)), escape(deadline_text))


def _tiles(snapshot: dict) -> str:
    live, campaign = snapshot.get("live") or {}, snapshot.get("campaign") or {}
    counters = campaign.get("cumulative") or {}
    usage = snapshot.get("usage") or {}
    deadline = live.get("deadline_unix")
    deadline_text = time.strftime("until %Y-%m-%d %H:%M", time.localtime(deadline)) if deadline else ""
    tokens = sum(int(usage.get(key) or 0) for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"))
    cost = usage.get("total_cost_usd")
    cost_text = f"${float(cost):,.2f}" if cost is not None and usage.get("cost_available", True) else "—"
    return (
        "<div class='tiles'>"
        + "".join(
            [
                _best_tile(campaign),
                _tile(
                    "Iterations",
                    escape(str(counters.get("iterations") or 0)),
                    f"{counters.get('kept') or 0} kept · {counters.get('reverted') or 0} reverted",
                ),
                _time_tile(snapshot, deadline_text),
                _tile("LLM spend", cost_text, f"{tokens:,} tokens · {usage.get('calls') or 0} calls"),
                _noise_tile(snapshot),
            ]
        )
        + "</div>"
    )


def _score_chart(rows: list[dict], start: float | None = None) -> str:
    points = [row for row in rows if row.get("iter") is not None]
    if start is not None and points:
        # Where the search began, so a single round still draws the step it made.
        points = [{"iter": 0, "decision": "start", "best_after": start}, *points]
    scored = [row for row in points if row.get("score") is not None or row.get("best_after") is not None]
    if not scored:
        return "<div class='card'><h2>Score by iteration</h2><div class='sub'>No measured iteration yet.</div></div>"
    width, height, left, right, top, bottom = 760, 240, 48, 16, 12, 28
    iters = [int(row["iter"]) for row in scored]
    values = [float(v) for row in scored for v in (row.get("score"), row.get("best_after")) if v is not None] + [1.0]
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 or 0.05
    low, high = low - pad, high + pad
    first, last = min(iters), max(iters)

    def x(it: int) -> float:
        return left + (it - first) / max(1, last - first) * (width - left - right)

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (height - top - bottom)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Score by iteration">']
    ticks = _ticks(low, high)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) + (
        1 if round(step / 10 ** math.floor(math.log10(step)), 6) == 2.5 else 0
    )
    for tick in ticks:
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)"/>'
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.{decimals}f}</text>'
        )
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(1.0):.1f}" y2="{y(1.0):.1f}" stroke="var(--axis)"/>')
    path, previous = [], None
    for row in scored:
        best = row.get("best_after")
        if best is None:
            continue
        px, py = x(int(row["iter"])), y(float(best))
        path.append(f"M{px:.1f},{py:.1f}" if previous is None else f"H{px:.1f}V{py:.1f}")
        previous = py
    if path:
        parts.append(
            f'<path d="{" ".join(path)}" fill="none" stroke="var(--series-1)" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    for row in scored:
        it, score = int(row["iter"]), row.get("score")
        if score is not None:
            tip = f"iteration {it}: {_fmt(score)}× ({row.get('decision', '')})"
            parts.append(
                f'<circle cx="{x(it):.1f}" cy="{y(float(score)):.1f}" r="3" fill="var(--neutral)" opacity=".7">'
                f"<title>{escape(tip)}</title></circle>"
            )
        if row.get("decision") == "KEEP" and row.get("best_after") is not None:
            parts.append(
                f'<circle cx="{x(it):.1f}" cy="{y(float(row["best_after"])):.1f}" r="5" fill="var(--series-1)" '
                f'stroke="var(--surface-1)" stroke-width="2"><title>KEEP at iteration {it}: '
                f"{_fmt(row['best_after'])}×</title></circle>"
            )
    parts.append(f'<text x="{left}" y="{height - 6}">{"start" if first == 0 else f"iteration {first}"}</text>')
    parts.append(f'<text x="{width - right}" y="{height - 6}" text-anchor="end">iteration {last}</text></svg>')
    legend = (
        "<div class='legend'><span><svg width='18' height='8'><line x1='0' x2='18' y1='4' y2='4' "
        "stroke='var(--series-1)' stroke-width='2'/></svg> best so far (dot = KEEP)</span>"
        "<span><span class='dot neutral'></span> candidate score</span>"
        "<span><svg width='18' height='8'><line x1='0' x2='18' y1='4' y2='4' stroke='var(--axis)'/></svg> "
        "anchor (1.0×)</span></div>"
    )
    return f"<div class='card'><h2>Score by iteration</h2>{legend}{''.join(parts)}{_iteration_table(rows)}</div>"


def _ticks(low: float, high: float, target: int = 4) -> list[float]:
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) spanning ``low``..``high``."""
    raw = (high - low) / target
    if raw <= 0:
        return [low]
    base = 10 ** math.floor(math.log10(raw))
    step = next(multiple * base for multiple in (1, 2, 2.5, 5, 10) if multiple * base >= raw)
    ticks, value = [], math.ceil(low / step) * step
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _iteration_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><td class='num'>{escape(str(row.get('iter')))}</td><td>{_status_chip(row.get('decision', ''))}</td>"
        f"<td class='num'>{_fmt(row.get('score'))}</td><td class='num'>{_fmt(row.get('best_after'))}</td>"
        f"<td>{escape(_keep_bar(row.get('keep_bar')))}</td>"
        f"<td>{escape(str(row.get('direction_id') or ''))}</td><td>{escape(row.get('plan', ''))}</td></tr>"
        for row in reversed(rows)
    )
    return (
        "<details><summary>Iterations as a table</summary><table><tr><th class='num'>iter</th><th>decision</th>"
        "<th class='num'>score</th><th class='num'>best after</th><th>KEEP bar</th><th>direction</th><th>plan</th></tr>"
        f"{body}</table></details>"
    )


def _decision_strip(rows: list[dict]) -> str:
    if not rows:
        return ""
    cell, gap = 14, 2
    width = max(1, len(rows)) * (cell + gap)
    cells = []
    for index, row in enumerate(rows):
        role, glyph, label = _decision_style(row.get("decision", ""))
        tip = f"iteration {row.get('iter')}: {row.get('decision', '')} ({label})"
        cells.append(
            f'<rect x="{index * (cell + gap)}" y="0" width="{cell}" height="{cell}" rx="3" fill="var(--{role})">'
            f"<title>{escape(tip)}</title></rect>"
        )
    legend = "".join(
        f"<span class='status'><span class='dot {role}'></span>{glyph} {label}</span>"
        for role, glyph, label in (_GOOD, _NEUTRAL, _CRITICAL, _WARNING)
    )
    return (
        f"<div class='card'><h2>Decisions</h2><div class='legend'>{legend}</div>"
        f'<svg viewBox="0 0 {width} {cell}" width="{min(width, 760)}" height="{cell}" role="img" '
        f'aria-label="Decision per iteration">{"".join(cells)}</svg></div>'
    )


def _case_table(snapshot: dict) -> str:
    cases = snapshot.get("cases") or []
    if not cases:
        return ""
    threshold = (snapshot.get("live") or {}).get("serving_speedup")
    rows = []
    served = 0
    for case in cases:
        anchor, best = case.get("anchor_ms"), case.get("best_ms")
        speedup = (float(anchor) / float(best)) if anchor and best else None
        cell = "—"
        if threshold is not None and speedup is not None:
            ok = speedup >= float(threshold)
            served += ok
            cell = _status_mark(ok)
        rows.append(
            f"<tr><td>{escape(str(case.get('case_id')))}</td><td class='num'>{_micro(anchor)}</td>"
            f"<td class='num'>{_micro(best)}</td><td class='num'>{_fmt(speedup, '.3f')}</td><td>{cell}</td></tr>"
        )
    head = "per-case speedup of the best kernel over the anchor (point estimate from the loop's own timing)"
    if threshold is not None:
        head += f"; {served}/{len(cases)} at or above {float(threshold):g}×"
    return (
        f"<div class='card'><h2>Cases</h2><div class='sub'>{escape(head)}</div><table><tr><th>case</th>"
        "<th class='num'>anchor µs</th><th class='num'>best µs</th><th class='num'>speedup</th>"
        f"<th>at threshold</th></tr>{''.join(rows)}</table></div>"
    )


def _micro(milliseconds) -> str:
    return _fmt(None if milliseconds is None else float(milliseconds) * 1000.0, ".2f")


def _status_mark(ok: bool) -> str:
    if ok:
        return '<span class="status"><span class="dot good"></span>✓ yes</span>'
    return '<span class="status"><span class="dot neutral"></span>– no</span>'


def _usage_table(snapshot: dict) -> str:
    by_role = (snapshot.get("usage") or {}).get("by_role") or {}
    if not by_role:
        return ""
    rows = "".join(
        f"<tr><td>{escape(role)}</td><td class='num'>{int(c.get('input_tokens') or 0):,}</td>"
        f"<td class='num'>{int(c.get('output_tokens') or 0):,}</td>"
        f"<td class='num'>{_fmt(c.get('total_cost_usd'), '.2f')}</td></tr>"
        for role, c in by_role.items()
    )
    return (
        "<div class='card'><h2>LLM spend by role</h2><table><tr><th>role</th><th class='num'>input tokens</th>"
        f"<th class='num'>output tokens</th><th class='num'>USD</th></tr>{rows}</table></div>"
    )


def _text_card(title: str, text: str) -> str:
    if not text:
        return ""
    return f"<div class='card'><h2>{escape(title)}</h2><pre>{escape(text)}</pre></div>"


def _events(snapshot: dict) -> str:
    events = snapshot.get("events") or []
    if not events:
        return ""
    rows = "".join(
        f"<tr><td>{escape(str(e.get('ts', '')))}</td><td class='num'>{escape(str(e.get('iter', '')))}</td>"
        f"<td>{escape(str(e.get('type', '')))}</td><td>{escape(str(e.get('detail', '')))}</td></tr>"
        for e in reversed(events)
    )
    return (
        "<div class='card'><h2>Recent events</h2><table><tr><th>time</th><th class='num'>iter</th><th>event</th>"
        f"<th>detail</th></tr>{rows}</table></div>"
    )


_DIRECTION_STATUS = {
    "OPEN": ("neutral", "○", "open"),
    "ACTIVE": ("series-1", "◐", "being tried"),
    "KEPT": ("good", "✓", "produced a KEEP"),
    "EXHAUSTED": ("neutral", "–", "mined out"),
    "INFEASIBLE": ("critical", "✗", "ruled out"),
}


def _directions_panel(snapshot: dict) -> str:
    directions = snapshot.get("directions") or {}
    ledger = directions.get("ledger") or []
    if not directions:
        return ""
    rows = []
    for entry in ledger:
        role, glyph, label = _DIRECTION_STATUS.get(entry.get("status"), ("neutral", "?", "unknown"))
        expected = entry.get("expected_gain_pct")
        rows.append(
            f"<tr><td>{escape(str(entry.get('id')))}</td><td>{escape(str(entry.get('title')))}</td>"
            f"<td>{escape(str(entry.get('category') or ''))}</td>"
            f"<td><span class='status'><span class='dot' style='background:var(--{role})'></span>{glyph} "
            f"{escape(label)}</span></td>"
            f"<td class='num' title='{escape(str(entry.get('expected_gain_reason') or ''))}'>"
            f"{_fmt(expected, '.1f') + '%' if expected is not None else 'unranked'}</td>"
            f"<td class='num'>{entry.get('attempts', 0)}</td>"
            f"<td class='num'>{entry.get('keeps', 0)}</td><td class='num'>{_fmt(entry.get('gain_pct'), '.2f')}%</td>"
            f"<td>{escape(str(entry.get('source') or ''))}</td></tr>"
        )
    policy, standing = directions.get("policy") or {}, directions.get("standing") or {}
    window_hours = float(policy.get("window_hours") or 0)
    rule = (
        f"Early stop: the last {window_hours:g} h of attempts gain under {float(policy.get('min_gain') or 0) * 100:g}%, "
        f"no direction is still expected to gain {float(policy.get('priority_gain') or 0) * 100:g}% or more, and an "
        "audit finds nothing new."
    )
    if standing.get("window_hours") is None:
        now = f"Now: under {window_hours:g} h of attempts so far."
    else:
        now = f"Now: {_fmt((standing.get('window_gain') or 0) * 100, '.2f')}% over {standing['window_hours']:.1f} h."
    if standing.get("criteria_met"):
        verdict = "Every criterion holds: " + (
            "an audit for missed directions runs next." if standing.get("audit_due") else "the audit found nothing new."
        )
    else:
        verdict = "Not yet: " + "; ".join(standing.get("unmet") or []) + "."
    focus = standing.get("focus") or []
    focus_text = f" Progress is flat, so plans must pursue: {', '.join(focus)}." if focus else ""
    replans, audits = directions.get("replans") or [], directions.get("audits") or []
    last_replan = (
        f"Last replan at iteration {replans[-1].get('iteration')}: ranked {len(replans[-1].get('ranked') or [])}, "
        f"retired {len(replans[-1].get('retired') or [])}, added {len(replans[-1].get('added') or [])}."
        if replans
        else "No replan yet."
    )
    last_audit = (
        f"Last audit at iteration {audits[-1].get('iteration')} added {len(audits[-1].get('added') or [])}."
        if audits
        else "No audit yet."
    )
    head = f"{rule} {now} {verdict}{focus_text} {last_replan} {last_audit}"
    return (
        f"<div class='card'><h2>Directions ({len(ledger)})</h2><div class='sub'>{escape(head)}</div>"
        "<table><tr><th>id</th><th>direction</th><th>category</th><th>status</th><th class='num'>expected</th>"
        "<th class='num'>tries</th><th class='num'>keeps</th><th class='num'>gain</th><th>from</th></tr>"
        f"{''.join(rows)}</table></div>"
    )


_ABLATION_STATUS = {
    "required_for_correctness": ("neutral", "■", "needed for correctness"),
    "not_separable": ("neutral", "⋈", "tangled with a later change"),
    "driver_changed": ("warning", "⚠", "driver differs"),
    "failed": ("critical", "✗", "bench failed"),
    "skipped_budget": ("neutral", "…", "not reached"),
}


def _pct(fraction, spec: str = "+.2f") -> str:
    return "—" if fraction is None else f"{_fmt(fraction * 100, spec)}%"


def _ablation_verdict(arm: dict) -> tuple[str, str, str]:
    if arm.get("status") != "measured":
        return _ABLATION_STATUS.get(arm.get("status"), ("neutral", "?", "unknown"))
    if arm.get("within_noise"):
        return ("neutral", "–", "within noise")
    if (arm.get("cost") or {}).get("mean", 0) > 0:
        return ("good", "✓", "pays")
    return ("warning", "⚠", "faster without it")


def _ablation_panel(snapshot: dict) -> str:
    ablation = snapshot.get("ablation") or {}
    arms = ablation.get("arms") or {}
    if not arms:
        return ""
    rows = []
    for key, arm in arms.items():
        role, glyph, label = _ablation_verdict(arm)
        cost = arm.get("cost") or {}
        name = "all off (the start)" if key == "all_off" else f"{key[:10]} {arm.get('title') or ''}"
        bounds = f"{_pct(cost['lower'])} … {_pct(cost['upper'])}" if cost else "—"
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{escape(str(arm.get('direction_id') or ''))}</td>"
            f"<td class='num'>{_pct(arm.get('gain_when_kept'))}</td><td class='num'>{_pct(cost.get('mean'))}</td>"
            f"<td class='num'>{escape(bounds)}</td>"
            f"<td><span class='status'><span class='dot' style='background:var(--{role})'></span>{glyph} "
            f"{escape(label)}</span></td></tr>"
        )
    head = f"best {str(ablation.get('best_commit', ''))[:12]}; noise floor {_pct(ablation.get('noise_floor'), '.2f')}"
    if "singles_combined" in ablation:
        head += (
            f"; single costs combined {_pct(ablation['singles_combined'])} against all-off "
            f"{_pct(ablation.get('all_off_cost'))}"
        )
    if ablation.get("completed") is False:
        head += "; budget ran out, rerun forge-ablate for the rest"
    return (
        f"<div class='card'><h2>Ablation</h2><div class='sub'>{escape(head)}. Cost = how much slower the best kernel "
        "runs without the change.</div><table><tr><th>change</th><th>direction</th><th class='num'>gain when kept</th>"
        f"<th class='num'>cost without it</th><th class='num'>95% bounds</th><th>verdict</th></tr>{''.join(rows)}"
        "</table></div>"
    )


def _inbox_panel(snapshot: dict) -> str:
    inbox = snapshot.get("inbox") or {}
    pending, active = inbox.get("pending") or [], inbox.get("active") or []
    items = [f"<li>waiting for the next iteration: {escape(text)}</li>" for text in pending]
    items += [f"<li>{escape(text)}</li>" for text in active]
    standing = f"<ul>{''.join(items)}</ul>" if items else "<div class='sub'>No message stands.</div>"
    return (
        "<div class='card'><h2>Operator messages</h2><div class='sub'>Every agent reads them from the next iteration "
        "on, ahead of its own plans, until they are cleared. Sending works when the page is served by "
        "<code>kernelforge forge-dashboard</code> or a relay; opened as a file, use <code>forge-say</code>.</div>"
        "<form class='say' method='post' action='say'><textarea name='message' rows='3' "
        "placeholder='e.g. Focus on the M&lt;=64 cases; leave the large ones alone.'></textarea>"
        "<div class='buttons'><button class='primary' type='submit'>Send</button>"
        "<button type='submit' name='clear' value='1'>Clear all messages</button></div></form>"
        f"{standing}</div>"
    )


def _plan_title(plan: str) -> str:
    for line in plan.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.lower().startswith("direction:"):
            return line[:140]
    return ""


def _round_entry(entry: dict, *, open_: bool) -> str:
    iteration, score, before = entry.get("iter"), entry.get("score"), entry.get("best_before")
    change = ""
    if score is not None and before:
        change = f" ({float(score) / float(before) - 1:+.2%} vs best {_fmt(before)}×)"
    meta = [f"iteration {escape(str(iteration))}"]
    if score is not None:
        meta.append(f"{_fmt(score)}×{change}")
    if entry.get("direction_id"):
        meta.append(f"direction {escape(str(entry['direction_id']))}")
    title = _plan_title(entry.get("plan") or "")
    verdict = [part for part in (_keep_bar(entry.get("keep_bar")),) if part]
    if entry.get("session_end_reason"):
        verdict.append(f"session ended: {entry['session_end_reason']}")
    if entry.get("turns"):
        verdict.append(f"{entry['turns']} turns")
    sections = [
        ("Plan", entry.get("plan")),
        ("What the implementer reports", entry.get("account")),
        ("Why it failed", entry.get("validation")),
        ("Verdict", " · ".join(verdict)),
    ]
    body = "".join(f"<h3>{name}</h3><pre>{escape(str(text))}</pre>" for name, text in sections if text)
    return (
        f"<details class='round' id='round-{escape(str(iteration))}'{' open' if open_ else ''}><summary>"
        f"{_status_chip(entry.get('decision', ''))} <span class='meta'>· {' · '.join(meta)}</span>"
        f"{' — ' + escape(title) if title else ''}</summary><div class='round-body'>{body}</div></details>"
    )


def _rounds_panel(snapshot: dict) -> str:
    rounds = snapshot.get("rounds") or []
    if not rounds:
        return ""
    entries = "".join(_round_entry(entry, open_=index == 0) for index, entry in enumerate(rounds))
    return (
        "<div class='card'><h2>Rounds</h2><div class='sub'>newest first: what each round planned, what its "
        f"implementer reports doing, and the verdict</div>{entries}</div>"
    )


def render_dashboard(snapshot: dict) -> str:
    """The whole page for one snapshot."""
    rows = snapshot.get("iterations") or []
    body = [
        _header(snapshot),
        _tiles(snapshot),
        _score_chart(rows, (snapshot.get("campaign") or {}).get("search_start_mean_case_speedup")),
        _decision_strip(rows),
        _inbox_panel(snapshot),
        _rounds_panel(snapshot),
        _directions_panel(snapshot),
        _ablation_panel(snapshot),
        _case_table(snapshot),
        _text_card("Latest supervisor ruling", snapshot.get("supervisor") or ""),
        _usage_table(snapshot),
        _events(snapshot),
    ]
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<noscript><meta http-equiv='refresh' content='{REFRESH_SEC}'></noscript>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>forge-loop dashboard</title><style>{_CSS}</style></head>"
        f"<body><div class='viz-root'>{''.join(part for part in body if part)}</div>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
