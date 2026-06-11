# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared data structures + registry for ``session_breakdown`` section renderers.

A renderer is a deterministic function turning one breakdown section into
a :class:`RenderedSection`. Numbers stay deterministic (the LLM has
historically hallucinated them); the LLM only narrates ``key_facts``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

__all__ = [
    "Decision",
    "RenderedSection",
    "RendererFn",
    "REGISTRY",
    "register_renderer",
    "renderer_names",
]


@dataclass(frozen=True)
class Decision:
    """One structured verdict surfaced by a renderer.

    ``kind`` vocabulary: ``kept`` / ``reverted`` / ``rejected`` /
    ``partial`` / ``attempted`` / ``not_attempted``.
    """

    kind: str
    subject: str
    metric_pct: float | None = None
    rationale: str = ""


@dataclass(frozen=True)
class RenderedSection:
    """A single section's render output.

    ``markdown_block`` is preserved verbatim; the LLM only sees
    ``key_facts`` / ``decisions`` / ``warnings``. ``skipped`` tells the
    compose layer to emit a one-line "not run" note instead of the block.
    """

    section_id: str
    title: str
    key_facts: list[str] = field(default_factory=list)
    markdown_block: str = ""
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


RendererFn = Callable[[dict[str, Any]], RenderedSection]


# Renderers self-register at import time; compose.py walks this in
# insertion order so section ordering is stable.
REGISTRY: list[tuple[str, RendererFn]] = []


def register_renderer(section_id: str) -> Callable[[RendererFn], RendererFn]:
    """Decorator: register ``fn`` under ``section_id`` (re-registration replaces the prior entry).

    Args:
        section_id: Identifier under which the decorated renderer is stored.

    Returns:
        A decorator that registers the renderer function and returns it
        unchanged.
    """

    def _wrap(fn: RendererFn) -> RendererFn:
        """Register ``fn`` under ``section_id`` and return it unchanged.

        Args:
            fn (RendererFn): The renderer function being decorated.

        Returns:
            RendererFn: The same function, after registration.
        """
        for i, (sid, _) in enumerate(REGISTRY):
            if sid == section_id:
                REGISTRY[i] = (section_id, fn)
                return fn
        REGISTRY.append((section_id, fn))
        return fn

    return _wrap


def renderer_names() -> list[str]:
    """List the section ids of all registered renderers, in order.

    Returns:
        list[str]: Section identifiers in registry insertion order.
    """
    return [sid for sid, _ in REGISTRY]


# Small markdown helpers — kept here so individual renderers stay terse.
def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    """Render a GitHub-flavored markdown table; empty rows yield ``""``.

    Args:
        headers (list[str]): Column header labels.
        rows (Iterable[list[Any]]): Row values; each inner list is rendered as
            one table row via :func:`_md_cell`.

    Returns:
        str: The markdown table text, or an empty string when there are no rows.
    """
    rows = list(rows)
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
    return "\n".join(out)


def md_kv_list(items: list[tuple[str, Any]]) -> str:
    """Render ``[(k, v), ...]`` as a bullet list, skipping ``None`` /
    empty-string values.

    Args:
        items (list[tuple[str, Any]]): Key/value pairs to render as bold-keyed
            bullet points; entries whose value is ``None``, ``""`` or ``[]``
            are omitted.

    Returns:
        str: The newline-joined markdown bullet list.
    """
    out = []
    for k, v in items:
        if v in (None, "", []):
            continue
        out.append(f"- **{k}**: {_md_cell(v)}")
    return "\n".join(out)


def _md_cell(v: Any) -> str:
    """Format a single value for display inside a markdown table cell.

    Handles ``None`` (em dash), booleans (check/cross marks), floats
    (compact numeric formatting, NaN as em dash), sequences (comma-joined)
    and escapes pipe/newline characters in strings.

    Args:
        v (Any): The value to format.

    Returns:
        str: A markdown-safe cell string.
    """
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
    """Format a numeric value as a percentage string.

    Args:
        v (Any): The value to format; non-numeric or ``None`` yields an em dash.
        plus (bool): If ``True``, prefix a ``+`` for strictly positive values.

    Returns:
        str: A string like ``"+12.34%"`` / ``"12.34%"``, or ``"—"`` when the
            value is missing or non-numeric.
    """
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if (plus and x > 0) else ""
    return f"{sign}{x:.2f}%"


def fmt_int(v: Any) -> str:
    """Format a value as a thousands-separated integer string.

    Args:
        v (Any): The value to format.

    Returns:
        str: The integer with thousands separators (e.g. ``"1,234"``), the
            raw ``str(v)`` when it is non-numeric, or ``"—"`` when ``None``.
    """
    if v is None:
        return "—"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)
