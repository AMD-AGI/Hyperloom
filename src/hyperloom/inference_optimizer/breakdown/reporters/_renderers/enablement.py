# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement renderer — admission status, round lifecycle, build attempts."""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, as_dict, md_kv_list, md_table, register_renderer


def _rounds_table(rounds: list[Any]) -> str:
    """Render the round ledger as a compact table (newest-first, capped at 10)."""
    rows: list[list[Any]] = []
    for r in rounds[:10]:
        if not isinstance(r, dict):
            continue
        rows.append(
            [
                r.get("round_id") or "—",
                r.get("outcome") or "open",
                r.get("attempts") or 0,
                r.get("opened_at") or "—",
            ]
        )
    if not rows:
        return ""
    return md_table(["round_id", "outcome", "attempts", "opened_at"], rows)


def _builds_table(builds: list[Any]) -> str:
    """Render the build-attempt ledger as a table (capped at 10)."""
    rows: list[list[Any]] = []
    for b in builds[:10]:
        if not isinstance(b, dict):
            continue
        rows.append(
            [
                b.get("attempt_root") or "—",
                b.get("status") or "—",
                b.get("framework") or "—",
                b.get("component") or "—",
            ]
        )
    if not rows:
        return ""
    return md_table(["attempt_root", "status", "framework", "component"], rows)


@register_renderer("enablement")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the enablement admission and round-lifecycle section."""
    e = as_dict(breakdown.get("enablement"))
    if not e:
        return RenderedSection(section_id="enablement", title="Enablement", skipped=True)

    facts: list[str] = []
    warnings: list[str] = []

    engaged = e.get("engaged")
    mode = str(e.get("mode") or "")
    succeeded = e.get("succeeded")
    attempts = e.get("attempts")
    failure_kind = e.get("failure_kind")

    if engaged:
        facts.append(f"Enablement engaged (mode={mode or 'unset'}).")
    elif mode:
        facts.append(f"Enablement mode={mode!r} — not yet engaged this session.")
    if succeeded:
        facts.append("Enablement succeeded.")
    elif engaged and succeeded is False:
        facts.append("Enablement did not produce a KEEP this session.")
    if failure_kind:
        facts.append(f"Last classified failure: {failure_kind}.")

    kv = md_kv_list(
        [
            ("mode", mode or None),
            ("engaged", engaged),
            ("origin", e.get("origin") or None),
            ("trigger_kind", e.get("trigger_kind") or None),
            ("attempts", attempts),
            ("succeeded", succeeded),
            ("failure_kind", failure_kind or None),
            ("round_count", e.get("round_count") or None),
        ]
    )

    parts: list[str] = [kv] if kv else []

    round_outcomes = e.get("round_outcomes")
    if isinstance(round_outcomes, dict) and round_outcomes:
        outcome_rows = [[k, v] for k, v in sorted(round_outcomes.items())]
        parts.append("\n**Round outcomes**\n\n" + md_table(["outcome", "count"], outcome_rows))

    rounds_md = _rounds_table(e.get("rounds") or [])
    if rounds_md:
        parts.append("\n**Rounds**\n\n" + rounds_md)

    builds_md = _builds_table(e.get("build_attempts") or [])
    if builds_md:
        parts.append("\n**Build attempts**\n\n" + builds_md)

    return RenderedSection(
        section_id="enablement",
        title="Enablement",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        warnings=warnings,
        skipped=False,
    )
