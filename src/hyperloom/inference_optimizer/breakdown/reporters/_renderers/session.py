# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session identification + lifecycle renderer (id, claw keys, host, revision, stop reason, elapsed)."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, md_kv_list, register_renderer, session_of, stop_reason_of


def _elapsed_facts(session: dict[str, Any]) -> list[str]:
    """The run leg's and the whole session's wall-clock time, each under its own name."""
    leg = session.get("elapsed_minutes")
    total = session.get("total_elapsed_minutes")
    facts: list[str] = []
    if leg is not None:
        facts.append(f"Wall-clock elapsed{' (this leg)' if total is not None else ''}: {float(leg):.1f} minutes.")
    if total is not None:
        facts.append(f"Total elapsed across all legs: {float(total):.1f} minutes.")
    return facts


def _identity_facts(sid: str, claw: Any, sandbox: Any, stop_reason: str) -> list[str]:
    """The session's id, the claw session and sandbox user it ran under, and why it stopped."""
    facts: list[str] = []
    if sid:
        facts.append(f"Hyperloom session_id=`{sid}`")
    if claw:
        facts.append(f"Joined to Primus-Claw session `{claw}`")
    else:
        facts.append("No claw_session_id recorded — this is a standalone Hyperloom run or a pre-V2 session.")
    if sandbox:
        facts.append(f"Sandbox user `{sandbox}`")
    if stop_reason:
        facts.append(f"Run ended with stop_reason=`{stop_reason}`.")
    return facts


def _image_configured(image: Any) -> bool:
    """Whether the session recorded a container image that is not blank."""
    return isinstance(image, str) and bool(image.strip())


def _placement_facts(host: str, image: Any, code: str) -> list[str]:
    """The host, container image and Hyperloom code revision the session ran on."""
    facts: list[str] = []
    if host:
        facts.append(f"Ran on host `{host}`.")
    if _image_configured(image):
        facts.append(f"Container image `{image}`.")
    if code:
        facts.append(f"Hyperloom code revision `{code}`.")
    return facts


def _session_decisions(sid: str, stop_reason: str) -> list[Decision]:
    """Whether the identified session was attempted, read off its stop reason."""
    if not sid:
        return []
    return [
        Decision(
            kind="attempted" if stop_reason else "not_attempted",
            subject=f"session:{sid}",
            rationale=f"stop_reason={stop_reason or 'unset'}",
        )
    ]


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
    s = session_of(breakdown)
    sid = str(s.get("session_id") or "")
    claw = s.get("claw_session_id")
    sandbox = s.get("sandbox_user_id")
    stop_reason = stop_reason_of(breakdown)
    host = str(s.get("host") or "")
    image = s.get("image")
    code = str(s.get("code_revision") or "")
    tick = s.get("tick_count")

    facts = _identity_facts(sid, claw, sandbox, stop_reason)
    facts.extend(_elapsed_facts(s))
    warnings: list[str] = []
    if tick and int(tick) > 0:
        facts.append(f"Coordinator advanced through {int(tick)} ticks.")
    elif tick == 0:
        warnings.append(
            "tick_count = 0 — Coordinator either resumed without a fresh "
            "scheduler loop or this session predates per-tick counters."
        )
    facts.extend(_placement_facts(host, image, code))

    md = md_kv_list(
        [
            ("session_id", sid),
            ("claw_session_id", claw),
            ("sandbox_user_id", sandbox),
            ("stop_reason", stop_reason or None),
            ("elapsed_minutes", s.get("elapsed_minutes")),
            ("total_elapsed_minutes", s.get("total_elapsed_minutes")),
            ("tick_count", tick),
            ("host", host or None),
            # Always emit the image row so a missing image is visible.
            ("image", image if _image_configured(image) else "(not configured)"),
            ("code_revision", code or None),
            ("created_at_utc", s.get("created_at_utc")),
            ("ended_at_utc", s.get("ended_at_utc")),
            ("max_minutes", s.get("max_minutes")),
            ("session_dir", s.get("session_dir")),
        ]
    )
    return RenderedSection(
        section_id="session",
        title="Session",
        key_facts=facts,
        markdown_block=md,
        decisions=_session_decisions(sid, stop_reason),
        warnings=warnings,
        skipped=not (sid or host),
    )
