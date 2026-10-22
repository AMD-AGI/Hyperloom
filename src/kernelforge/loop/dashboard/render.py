# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a dashboard snapshot as one self-contained HTML page: no script, no network, every agent text escaped."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
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
.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600}td.num,th.num{text-align:right}
.status{display:inline-flex;gap:4px;align-items:center}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%}
.good{background:var(--good)}.warning{background:var(--warning)}.critical{background:var(--critical)}
.neutral{background:var(--neutral)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--text-secondary);font-size:12px;margin:6px 0}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font-size:12px;color:var(--text-secondary)}
details summary{cursor:pointer;color:var(--text-secondary);font-size:12px;margin-top:8px}
svg text{fill:var(--muted);font-size:11px}
"""


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
    parts.append(f"rendered {rendered}, refreshes every {REFRESH_SEC} s")
    title = escape(campaign.get("campaign_id") or snapshot.get("campaign_root") or "forge-loop")
    return f"<h1>forge-loop · {title}</h1><div class='sub'>{' · '.join(parts)}</div>"


def _tiles(snapshot: dict) -> str:
    live, campaign = snapshot.get("live") or {}, snapshot.get("campaign") or {}
    best = campaign.get("best") or {}
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
                _tile(
                    "Best speedup",
                    f"{_fmt(best.get('mean_case_speedup'))}×",
                    f"iteration {escape(str(best.get('iteration') or '—'))}",
                    hero=True,
                ),
                _tile(
                    "Iterations",
                    escape(str(counters.get("iterations") or 0)),
                    f"{counters.get('kept') or 0} kept · {counters.get('reverted') or 0} reverted",
                ),
                _tile("Time left", _duration(live.get("time_remaining_sec")), escape(deadline_text)),
                _tile("LLM spend", cost_text, f"{tokens:,} tokens · {usage.get('calls') or 0} calls"),
            ]
        )
        + "</div>"
    )


def _score_chart(rows: list[dict]) -> str:
    points = [row for row in rows if row.get("iter") is not None]
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
    parts.append(f'<text x="{left}" y="{height - 6}">iteration {first}</text>')
    parts.append(f'<text x="{width - right}" y="{height - 6}" text-anchor="end">iteration {last}</text></svg>')
    legend = (
        "<div class='legend'><span><svg width='18' height='8'><line x1='0' x2='18' y1='4' y2='4' "
        "stroke='var(--series-1)' stroke-width='2'/></svg> best so far (dot = KEEP)</span>"
        "<span><span class='dot neutral'></span> candidate score</span>"
        "<span><svg width='18' height='8'><line x1='0' x2='18' y1='4' y2='4' stroke='var(--axis)'/></svg> "
        "anchor (1.0×)</span></div>"
    )
    return f"<div class='card'><h2>Score by iteration</h2>{legend}{''.join(parts)}{_iteration_table(points)}</div>"


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
        f"<td>{escape(str(row.get('direction_id') or ''))}</td><td>{escape(row.get('plan', ''))}</td></tr>"
        for row in reversed(rows)
    )
    return (
        "<details><summary>Iterations as a table</summary><table><tr><th class='num'>iter</th><th>decision</th>"
        "<th class='num'>score</th><th class='num'>best after</th><th>direction</th><th>plan</th></tr>"
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
            f"<tr><td>{escape(str(case.get('case_id')))}</td><td class='num'>{_fmt(anchor, '.4f')}</td>"
            f"<td class='num'>{_fmt(best, '.4f')}</td><td class='num'>{_fmt(speedup, '.3f')}</td><td>{cell}</td></tr>"
        )
    head = "per-case speedup of the best kernel over the anchor (point estimate from the loop's own timing)"
    if threshold is not None:
        head += f"; {served}/{len(cases)} at or above {float(threshold):g}×"
    return (
        f"<div class='card'><h2>Cases</h2><div class='sub'>{escape(head)}</div><table><tr><th>case</th>"
        "<th class='num'>anchor ms</th><th class='num'>best ms</th><th class='num'>speedup</th>"
        f"<th>at threshold</th></tr>{''.join(rows)}</table></div>"
    )


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


# Panels in page order. Later features register theirs in EXTRA_PANELS, keyed off their own snapshot entries.
EXTRA_PANELS: list[Callable[[dict], str]] = []


def render_dashboard(snapshot: dict) -> str:
    """The whole page for one snapshot."""
    rows = snapshot.get("iterations") or []
    body = [
        _header(snapshot),
        _tiles(snapshot),
        _score_chart(rows),
        _decision_strip(rows),
        _case_table(snapshot),
        *(panel(snapshot) for panel in EXTRA_PANELS),
        _text_card("Latest supervisor ruling", snapshot.get("supervisor") or ""),
        _usage_table(snapshot),
        _events(snapshot),
    ]
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='{REFRESH_SEC}'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>forge-loop dashboard</title><style>{_CSS}</style></head>"
        f"<body><div class='viz-root'>{''.join(part for part in body if part)}</div></body></html>"
    )
