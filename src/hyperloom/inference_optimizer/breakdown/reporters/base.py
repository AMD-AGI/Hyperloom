# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared data structures + registry for ``session_breakdown`` section renderers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from hyperloom.common.coerce import dict_rows, to_float, to_mapping


log = logging.getLogger(__name__)

__all__ = [
    "Decision",
    "RenderedSection",
    "RendererFn",
    "REGISTRY",
    "close_of",
    "critic_iterations_of",
    "events_of",
    "outcome_of",
    "register_renderer",
    "render_section",
    "session_of",
    "stop_reason_of",
    "task_config_of",
    "timeline_of",
    "validation_of",
]


def timeline_of(breakdown: Any) -> list[dict[str, Any]]:
    """The ordered event timeline.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``timeline``, oldest event first, or ``[]`` when absent.
    """
    return dict_rows(to_mapping(breakdown).get("timeline"))


def events_of(breakdown: Any, event_type: str) -> list[dict[str, Any]]:
    """Every timeline event of one type, in timeline order.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.
        event_type: The ``type`` to select on.

    Returns:
        The matching events, oldest first.
    """
    return [event for event in timeline_of(breakdown) if str(event.get("type") or "") == event_type]


def session_of(breakdown: Any) -> dict[str, Any]:
    """The session identity and lifecycle block.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``metadata.session``, or ``{}`` when absent.
    """
    return to_mapping(to_mapping(breakdown).get("metadata")).get("session") or {}


def task_config_of(breakdown: Any) -> dict[str, Any]:
    """The launch-time workload and model configuration block.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``metadata.task_config``, or ``{}`` when absent.
    """
    return to_mapping(to_mapping(breakdown).get("metadata")).get("task_config") or {}


def _target_label(target: dict[str, Any]) -> str:
    """One target as ``kind>=value`` (numeric), ``kind=value`` (e.g. a directory) or bare ``kind``."""
    kind = str(target.get("kind") or "")
    raw = target.get("value")
    value = to_float(raw)
    if not kind:
        return ""
    if value is not None:
        return f"{kind}>={value:g}"
    return f"{kind}={raw}" if raw not in (None, "") else kind


def stop_target_text(objective: Any) -> str:
    """The session's stop targets, one label per target joined by ``", "``.

    Args:
        objective: ``metadata.task_config.objective``; several targets arrive as
            the first one plus an ``objectives`` list holding all of them.

    Returns:
        The label, or ``""`` when no target was recorded.
    """
    target = to_mapping(objective)
    targets = (
        [to_mapping(t) for t in target.get("objectives") or []] if isinstance(target.get("objectives"), list) else []
    )
    return ", ".join(label for t in targets or [target] if (label := _target_label(t)))


def outcome_of(breakdown: Any) -> dict[str, Any]:
    """The session result block.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``outcome``, or ``{}`` when absent.
    """
    return to_mapping(to_mapping(breakdown).get("outcome"))


def validation_of(breakdown: Any) -> dict[str, Any]:
    """The stack ledger's reconciliation of its parts against the whole.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``outcome.validation``, or ``{}`` when absent.
    """
    return to_mapping(outcome_of(breakdown).get("validation"))


def close_of(breakdown: Any) -> dict[str, Any]:
    """What the session settled at close.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``close``, or ``{}`` when absent.
    """
    return to_mapping(to_mapping(breakdown).get("close"))


def critic_iterations_of(breakdown: Any) -> list[dict[str, Any]]:
    """The critic agent's own review passes, in the order it ran them.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        ``critic.iterations``, or ``[]`` when the critic never ran.
    """
    return dict_rows(to_mapping(to_mapping(breakdown).get("critic")).get("iterations"))


def stop_reason_of(breakdown: Any) -> str:
    """Why the run ended.

    The reason is an outcome of the session rather than part of its identity,
    so it lives on ``outcome`` and not alongside the session ids.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        The stop reason, or ``""`` when the run recorded none.
    """
    return str(to_mapping(to_mapping(breakdown).get("outcome")).get("stop_reason") or "")


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
    except Exception as exc:  # isolation boundary: one report section renderer
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
