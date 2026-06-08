# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared data structures + registry for ``session_breakdown`` section
renderers.

A *renderer* is a deterministic function that takes one section of the
``session_breakdown.json`` dict and produces a :class:`RenderedSection`.
The compose layer collects every renderer's output, optionally hands the
``key_facts`` / ``decisions`` to an LLM for narrative connecting, and
stitches the result back together with the deterministic markdown blocks
unchanged.

Why this layering exists:

* The numbers (throughputs, gains, kernel counts, paths) MUST be
  deterministic — the LLM has historically hallucinated them
  (``715.2%`` gain attributed to GEAK on a session that never ran
  GEAK, ``MI355X`` written into a report for an MI300X run, etc.).
* The LLM is good at prose connecting / interpretation, so it is only
  allowed to talk *about* the facts in ``key_facts``; the system
  prompt enforces "do not invent numbers".
* Every renderer is independently testable against a synthetic
  ``session_breakdown.json`` fixture, and the compose layer is
  testable with the LLM swapped out.
"""

from __future__ import annotations

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

    ``kind`` is a small, well-known vocabulary so downstream consumers
    (UI badges, dashboards, the LLM prompt) can render verdicts without
    parsing free-form strings.

    Vocabulary:

    * ``"kept"``          — capability / action produced a promoted entry.
    * ``"reverted"``      — applied + then rolled back (e.g. integrate REVERT).
    * ``"rejected"``      — rejected without applying (e.g. patch fail).
    * ``"partial"``       — passed compile / correctness but no speedup.
    * ``"attempted"``     — ran but no decision recorded yet.
    * ``"not_attempted"`` — capability never invoked this session.
    """

    kind: str
    subject: str
    metric_pct: float | None = None
    rationale: str = ""


@dataclass(frozen=True)
class RenderedSection:
    """A single section's render output.

    The compose layer guarantees ``markdown_block`` is preserved
    verbatim; the LLM only sees ``key_facts`` + ``decisions`` +
    ``warnings`` and is forbidden from rewriting numbers.

    ``skipped`` is the renderer's signal to "do not narrate this
    section at all" — used for sections whose underlying capability
    was not exercised (e.g. ``sweep`` when the session never ran a
    sweep). The compose layer renders a one-line "not run this
    session" note instead of a full section block.
    """

    section_id: str
    title: str
    key_facts: list[str] = field(default_factory=list)
    markdown_block: str = ""
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


RendererFn = Callable[[dict[str, Any]], RenderedSection]


# Module-level registry. Renderers register themselves at import time
# via the @register_renderer decorator; compose.py walks this list in
# insertion order so the final report's section ordering is stable.
REGISTRY: list[tuple[str, RendererFn]] = []


def register_renderer(section_id: str) -> Callable[[RendererFn], RendererFn]:
    """Decorator: append ``fn`` to :data:`REGISTRY` under ``section_id``.

    Re-registration replaces the prior entry (so reload-in-place during
    development stays sane), which is why we walk + replace instead of
    append-only.

    Args:
        section_id (str): The stable identifier for the section this renderer
            produces; used as the registry key and report ordering anchor.

    Returns:
        Callable[[RendererFn], RendererFn]: A decorator that registers the
            wrapped renderer function and returns it unchanged.
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


# ---------------------------------------------------------------------------
# Small markdown helpers — kept here so individual renderers stay terse.
# ---------------------------------------------------------------------------
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
        if v != v:  # NaN
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
