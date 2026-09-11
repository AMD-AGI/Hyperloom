# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement renderer — admission status, round ledger, targeted-build attempts."""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, as_dict, md_kv_list, md_table, register_renderer

_ROUND_COLUMNS = ("round_id", "state", "outcome", "holder_task_id")
_BUILD_COLUMNS = ("component", "ref", "gpu_arch", "ok", "failure_class")


def _table(entries: Any, columns: tuple[str, ...]) -> str:
    """Render a list of mappings as a table of ``columns``."""
    return md_table(list(columns), [[e.get(c) for c in columns] for e in entries or []])


def _outcomes_table(outcomes: Any) -> str:
    """Render the settled-round outcome counts."""
    if not outcomes:
        return ""
    return md_table(["outcome", "count"], [[k, v] for k, v in sorted(outcomes.items())])


@register_renderer("enablement")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the enablement admission and round-lifecycle section."""
    e = as_dict(breakdown.get("enablement"))
    if not e:
        return RenderedSection(section_id="enablement", title="Enablement", skipped=True)

    mode = e.get("mode") or "unset"
    facts: list[str] = []
    if e.get("engaged"):
        facts.append(f"Enablement engaged (mode={mode}, attempts={e.get('attempts') or 0}).")
    else:
        facts.append(f"Enablement did not engage (mode={mode}).")
    if e.get("succeeded"):
        facts.append("Enablement produced a KEEP.")
    if e.get("failure_kind"):
        facts.append(f"Last classified failure: {e['failure_kind']}.")

    parts = [
        md_kv_list(
            [
                ("mode", e.get("mode")),
                ("engaged", e.get("engaged")),
                ("origin", e.get("origin")),
                ("trigger_kind", e.get("trigger_kind")),
                ("attempts", e.get("attempts")),
                ("succeeded", e.get("succeeded")),
                ("failure_kind", e.get("failure_kind")),
                ("round_count", e.get("round_count")),
            ]
        )
    ]
    for title, block in (
        ("Round outcomes", _outcomes_table(e.get("round_outcomes"))),
        ("Rounds", _table(e.get("rounds"), _ROUND_COLUMNS)),
        ("Build attempts", _table(e.get("build_attempts"), _BUILD_COLUMNS)),
    ):
        if block:
            parts.append(f"\n**{title}**\n\n{block}")

    return RenderedSection(
        section_id="enablement",
        title="Enablement",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
    )
