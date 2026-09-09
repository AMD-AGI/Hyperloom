#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Formatting and page chrome shared by the HTML run reports.

Two reports render the same kind of document -- a GEAK run
(:mod:`render_geak_html_report`) and a Hyperloom session
(:mod:`render_hyperloom_html_report`) -- and a reader who opens both should not
have to work out whether "$1,234.56" means the same thing on each page. The
formatters and the stylesheet therefore live here once rather than being copied,
so the two pages cannot drift into disagreeing about how a number is written.

Nothing here decides what to say; it only decides how a value is spelled. The
one rule that carries meaning is in :func:`fmt_pct`: ``None`` renders as "not
measured" rather than as zero, because "we did not measure it" and "it was
nothing" are different claims and a report that conflates them is lying.
"""

from __future__ import annotations

import html
from typing import Any


def num(value: Any) -> float:
    """Coerce a ledger value to a float, treating anything unusable as 0.0.

    A bool is not a token count and NaN is not a cost, so both are rejected
    rather than summed: ``True`` would otherwise add 1 to a total and a single
    NaN would poison every total it touches.

    Args:
        value: A value off a JSON row.

    Returns:
        The value as a float, or ``0.0`` when it is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    return float(value)


def esc(value: Any) -> str:
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


def fmt_usd(value: Any) -> str:
    number = num(value)
    return f"${number:,.2f}" if number >= 0.01 or number == 0 else f"${number:,.4f}"


def fmt_int(value: Any) -> str:
    return f"{int(num(value)):,}"


def fmt_hms(seconds: Any) -> str:
    total = int(num(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    """Format a percentage, or say plainly that it was never measured."""
    if value is None:
        return '<span class="none">not measured</span>'
    return f"{num(value):+.{digits}f}%"


def bar(fraction: float, tone: str = "spend") -> str:
    width = max(0.0, min(100.0, fraction * 100.0))
    return f'<span class="bar {tone}"><i style="width:{width:.2f}%"></i></span>'


def sparkline(values: list[float], width: int = 220, height: int = 44) -> str:
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
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
/* Grid items default to min-width:auto, so a nowrap table refuses to shrink below its
   min-content width and bleeds over the next column. Let the item shrink, and let the
   table scroll inside its own column instead of over its neighbour. */
.grid2>*{min-width:0}
.grid2 .scroll{max-width:100%}
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
