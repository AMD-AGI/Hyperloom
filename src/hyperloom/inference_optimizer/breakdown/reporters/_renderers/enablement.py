# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement renderer — admission, authoring rounds, and targeted builds.

Read off the ``enablement`` events on the V6 timeline. The lane's facts are
written by six modules on different ticks, so the event is the only place they
meet.
"""

from __future__ import annotations

from typing import Any


from hyperloom_common.coerce import to_mapping

from ..base import RenderedSection, events_of, md_kv_list, md_table, register_renderer

_ATTEMPT_COLUMNS = ("attempt", "status", "failure_kind", "reason")
_BUILD_COLUMNS = ("component", "ref", "gpu_arch", "ok", "failure_class")


def _table(rows: Any, columns: tuple[str, ...]) -> str:
    """Render a list of mappings as a table of ``columns``."""
    return md_table(list(columns), [[r.get(c) for c in columns] for r in rows or []])


@register_renderer("enablement")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the enablement section from the lane's timeline event."""
    events = events_of(breakdown, "enablement")
    if not events:
        return RenderedSection(section_id="enablement", title="Enablement", skipped=True)

    event = events[0]
    ext = to_mapping(event.get("ext"))
    attempts = to_mapping(ext.get("attempts"))
    builds = to_mapping(ext.get("builds"))
    revalidations = to_mapping(ext.get("revalidations"))
    result = to_mapping(ext.get("result"))
    status = str(event.get("status") or "")

    facts = [
        f"Enablement ran {attempts.get('count') or 0} authoring round(s) "
        f"(mode={ext.get('mode') or 'unset'}, origin={ext.get('origin') or 'unset'})."
    ]
    if attempts.get("landed"):
        facts.append(f"{attempts['landed']} round(s) landed a fix.")
    if attempts.get("validation_pending"):
        facts.append(f"{attempts['validation_pending']} kept round(s) awaiting revalidation.")
    if attempts.get("advanced"):
        facts.append(f"{attempts['advanced']} round(s) moved the boot forward without landing one.")
    if builds.get("failed"):
        facts.append(f"{builds['failed']} targeted build(s) failed.")
    if status:
        facts.append(f"Lane outcome: {status}.")

    parts = [
        md_kv_list(
            [
                ("mode", ext.get("mode")),
                ("origin", ext.get("origin")),
                ("status", status),
                ("outcome", result.get("outcome")),
                ("rounds", attempts.get("count")),
                ("settled", attempts.get("settled")),
                ("landed", attempts.get("landed")),
                ("validation_pending", attempts.get("validation_pending") or None),
                ("advanced", attempts.get("advanced")),
                ("builds", builds.get("count")),
                ("builds_failed", builds.get("failed")),
                ("revalidations", revalidations.get("count")),
                ("revalidations_promoted", revalidations.get("promoted")),
                ("human_review", to_mapping(ext.get("human_review")).get("count")),
            ]
        )
    ]
    for title, block in (
        ("Rounds", _table(attempts.get("rows"), _ATTEMPT_COLUMNS)),
        ("Build attempts", _table(builds.get("rows"), _BUILD_COLUMNS)),
    ):
        if block:
            parts.append(f"\n**{title}**\n\n{block}")

    return RenderedSection(
        section_id="enablement",
        title="Enablement",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
    )
