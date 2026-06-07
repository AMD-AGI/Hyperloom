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
    """

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


# ---------------------------------------------------------------------------
# Small markdown helpers — kept here so individual renderers stay terse.
# ---------------------------------------------------------------------------
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
