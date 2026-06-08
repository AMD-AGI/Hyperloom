# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared data structures + registry for ``session_breakdown`` section renderers.

A renderer is a deterministic function turning one breakdown section into
a :class:`RenderedSection`. Numbers stay deterministic (the LLM has
historically hallucinated them); the LLM only narrates ``key_facts``.
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
    """Decorator: register ``fn`` under ``section_id`` (re-registration replaces the prior entry)."""

    def _wrap(fn: RendererFn) -> RendererFn:
        for i, (sid, _) in enumerate(REGISTRY):
            if sid == section_id:
                REGISTRY[i] = (section_id, fn)
                return fn
        REGISTRY.append((section_id, fn))
        return fn

    return _wrap


def renderer_names() -> list[str]:
    return [sid for sid, _ in REGISTRY]


# Small markdown helpers — kept here so individual renderers stay terse.
def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    """Render a GitHub-flavored markdown table; empty rows yield ``""``."""
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
    empty-string values."""
    out = []
    for k, v in items:
        if v in (None, "", []):
            continue
        out.append(f"- **{k}**: {_md_cell(v)}")
    return "\n".join(out)


def _md_cell(v: Any) -> str:
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
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if (plus and x > 0) else ""
    return f"{sign}{x:.2f}%"


def fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)
