# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared data structures + registry for ``session_breakdown`` section renderers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

__all__ = [
    "Decision",
    "RenderedSection",
    "RendererFn",
    "REGISTRY",
    "as_dict",
    "register_renderer",
    "render_section",
]


def as_dict(value: Any) -> dict[str, Any]:
    """Narrow a breakdown section to a mapping, since no producer is schema-checked."""
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Decision:
    """One structured verdict surfaced by a renderer."""

    kind: str
    subject: str
    metric_pct: float | None = None
    rationale: str = ""


@dataclass(frozen=True)
class RenderedSection:
    """A single section's render output."""

    section_id: str
    title: str
    key_facts: list[str] = field(default_factory=list)
    markdown_block: str = ""
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


RendererFn = Callable[[dict[str, Any]], RenderedSection]


# Renderers self-register at import time; walked in insertion order.
REGISTRY: list[tuple[str, RendererFn]] = []


def register_renderer(section_id: str) -> Callable[[RendererFn], RendererFn]:
    """Decorator: register ``fn`` under ``section_id`` (re-registration replaces the prior entry)."""

    def _wrap(fn: RendererFn) -> RendererFn:
        """Register ``fn`` under ``section_id`` and return it unchanged."""
        for i, (sid, _) in enumerate(REGISTRY):
            if sid == section_id:
                REGISTRY[i] = (section_id, fn)
                return fn
        REGISTRY.append((section_id, fn))
        return fn

    return _wrap


def render_section(
    section_id: str,
    fn: RendererFn,
    breakdown: dict[str, Any],
) -> RenderedSection:
    """Run one renderer so a failing section costs itself, not the report."""
    try:
        return fn(breakdown)
    except Exception as exc:  # noqa: BLE001 — one bad section must not lose the report
        log.exception("report section %s failed to render", section_id)
        return RenderedSection(
            section_id=section_id,
            title=section_id.replace("_", " ").title(),
            warnings=[f"section could not be rendered: {type(exc).__name__}: {exc}"],
        )


# Small markdown helpers.
def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    """Render a GitHub-flavored markdown table; empty rows yield ``\"\"``."""
    rows = list(rows)
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
    return "\n".join(out)


def md_kv_list(items: list[tuple[str, Any]]) -> str:
    """Render ``[(k, v), ...]`` as a bullet list, skipping ``None`` / empty-string values."""
    out = []
    for k, v in items:
        if v in (None, "", []):
            continue
        out.append(f"- **{k}**: {_md_cell(v)}")
    return "\n".join(out)


def _md_cell(v: Any) -> str:
    """Format a single value for display inside a markdown table cell."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.3g}" if abs(v) < 1 or abs(v) >= 1e4 else f"{v:.2f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_md_cell(x) for x in v) if v else "—"
    s = str(v)
    return s.replace("|", "\\|").replace("\n", " ")


def fmt_pct(v: Any, *, plus: bool = False) -> str:
    """Format a numeric value as a percentage string."""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if (plus and x > 0) else ""
    return f"{sign}{x:.2f}%"
