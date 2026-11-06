# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Output-protocol rule that a bridge patch ships behind a default-off switch, which integration enforces."""

from __future__ import annotations

from ..measurement.runtime_findings import RESTORED_PATH_RULES, lists_disabled_path
from ..state.gaps import STATIC_RECON_GAP_PREFIX


def bridge_switch_rule(*, gap_canonical_id: str, runtime_findings: str) -> list[str]:
    """The rule, for a dispatch on a static-recon gap or whose section 4c shows a switched-off path; else nothing."""
    if not (gap_canonical_id.startswith(STATIC_RECON_GAP_PREFIX) or lists_disabled_path(runtime_findings)):
        return []
    return [
        "- A **bridge** — a patch that relaxes a gate switching a path off (a static-recon gap, or a",
        f"  ``{'`` / ``'.join(sorted(RESTORED_PATH_RULES))}`` ``resolves_finding``) — MUST put the relaxed",
        "  branch behind its own environment switch that defaults OFF and declare it in ``framework_switches``",
        "  (``switch``, ``target``, ``evidence``); integration refuses an un-switched bridge before benching it.",
        "  A switched bridge that misses the KEEP gate stays in the tree, dormant, as a lever explore turns on",
        "  next to later changes; an un-switched one would be reverted and lost.",
    ]
