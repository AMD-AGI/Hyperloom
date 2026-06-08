# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session identification + lifecycle renderer.

Surfaces just enough of the ``session`` section so the report header
is self-contained: session_id, claw join keys, host, code revision,
stop reason, elapsed time. Workload (model / framework / GPU / TP /
ISL / OSL) lives in its own section.
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, md_kv_list, register_renderer


@register_renderer("session")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the session-identification + lifecycle section.

    Surfaces session/claw ids, host, container image, code revision, stop
    reason, elapsed time and tick count so the report header is
    self-contained. Skipped when neither session id nor host is present.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered session section.
    """
    s = breakdown.get("session") or {}
    facts: list[str] = []
    warnings: list[str] = []

    sid = str(s.get("session_id") or "")
    claw = s.get("claw_session_id")
    sandbox = s.get("sandbox_user_id")
    stop_reason = str(s.get("stop_reason") or "")
    elapsed = s.get("elapsed_minutes")
    host = str(s.get("host") or "")
    image = s.get("image")
    code = str(s.get("code_revision") or "")
    tick = s.get("tick_count")

    if sid:
        facts.append(f"Hyperloom session_id=`{sid}`")
    if claw:
        facts.append(f"Joined to Primus-Claw session `{claw}`")
    else:
        facts.append(
            "No claw_session_id recorded — this is a standalone "
            "Hyperloom run or a pre-V2 session."
        )
    if sandbox:
        facts.append(f"Sandbox user `{sandbox}`")
    if stop_reason:
        facts.append(f"Run ended with stop_reason=`{stop_reason}`.")
    if elapsed is not None:
        facts.append(f"Wall-clock elapsed: {float(elapsed):.1f} minutes.")
    if tick and int(tick) > 0:
        facts.append(f"Coordinator advanced through {int(tick)} ticks.")
    elif tick == 0:
        warnings.append(
            "tick_count = 0 — Coordinator either resumed without a fresh "
            "scheduler loop or this session predates per-tick counters."
        )
    if host:
        facts.append(f"Ran on host `{host}`.")
    if isinstance(image, str) and image.strip():
        facts.append(f"Container image `{image}`.")
    if code:
        facts.append(f"Hyperloom code revision `{code}`.")

    md = md_kv_list([
        ("session_id",       sid),
        ("claw_session_id",  claw),
        ("sandbox_user_id",  sandbox),
        ("stop_reason",      stop_reason or None),
        ("elapsed_minutes",  elapsed),
        ("tick_count",       tick),
        ("host",             host or None),
        # Always include the image row so reviewers can see at a glance
        # whether the run had its container image recorded; "(not
        # configured)" makes the gap obvious instead of silently
        # omitting the field.
        ("image",            image if (isinstance(image, str) and image.strip()) else "(not configured)"),
        ("code_revision",    code or None),
        ("created_at_utc",   s.get("created_at_utc")),
        ("ended_at_utc",     s.get("ended_at_utc")),
        ("max_minutes",      s.get("max_minutes")),
        ("session_dir",      s.get("session_dir")),
    ])

    decisions = [Decision(
        kind="attempted" if stop_reason else "not_attempted",
        subject=f"session:{sid or 'unknown'}",
        rationale=f"stop_reason={stop_reason or 'unset'}",
    )] if sid else []

    return RenderedSection(
        section_id="session",
        title="Session",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=warnings,
        skipped=not (sid or host),
    )
