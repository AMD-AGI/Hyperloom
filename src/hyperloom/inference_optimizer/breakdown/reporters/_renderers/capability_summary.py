# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capability summary renderer — status/attempts/keeps table + per-capability decisions."""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer

_CAPABILITY_ORDER = (
    "explore",
    "geak",
    "specialist",
    "forge",
)


@register_renderer("capability_summary")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the capability-summary section.

    Produces a status/attempts/keeps table for each capability in a
    stable order, one structured :class:`Decision` per non-``not_attempted``
    capability, and a one-line fact per row. Skipped when no capabilities
    were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered capability-summary section.
    """
    cap = breakdown.get("capability_summary") or {}
    rows: list[list[Any]] = []
    facts: list[str] = []
    decisions: list[Decision] = []
    warnings: list[str] = []

    # Known capabilities first, then unknown extras.
    keys = list(_CAPABILITY_ORDER) + [k for k in cap if k not in _CAPABILITY_ORDER]
    for name in keys:
        v = cap.get(name)
        if v is None:
            continue
        status = str(v.get("status") or "not_attempted")
        attempts = int(v.get("attempts") or 0)
        keeps = int(v.get("keeps") or 0)
        extras: list[str] = []
        if "best_gain_pct" in v and v["best_gain_pct"] is not None:
            extras.append(f"best_gain={fmt_pct(v['best_gain_pct'])}")
        if "best_throughput" in v and v["best_throughput"] is not None:
            extras.append(f"best_tput={v['best_throughput']:.2f}")
        if "tested" in v and v["tested"] is not None:
            extras.append(f"tested={v['tested']}")
        if "last_validated_gain_pct" in v and v["last_validated_gain_pct"] is not None:
            extras.append(f"validated_gain={fmt_pct(v['last_validated_gain_pct'])}")
        if "grid_size" in v and v["grid_size"] is not None:
            extras.append(f"grid={v['grid_size']}")
        # Kernel-lane outcomes that ``keeps`` deliberately excludes. Without
        # them a reader cannot tell a lane that failed from one whose wins are
        # still waiting on integrate.
        if v.get("micro_only_keeps"):
            extras.append(f"micro_only={v['micro_only_keeps']}")
        if v.get("pending_integrate"):
            extras.append(f"pending_integrate={v['pending_integrate']}")
        if v.get("reverts"):
            extras.append(f"reverts={v['reverts']}")
        if v.get("e2e_gain_pct") is not None:
            extras.append(f"e2e_gain={fmt_pct(v['e2e_gain_pct'])}")
        # explore extras.
        if "keep_unstable_count" in v and v["keep_unstable_count"]:
            extras.append(f"keep_unstable={v['keep_unstable_count']}")
        if "winners_history" in v and v["winners_history"]:
            extras.append(f"history={v['winners_history']}")
        extras_str = " · ".join(extras) if extras else ""

        rows.append([name, status, attempts, keeps, extras_str])
        facts.append(
            f"`{name}` status={status}, attempts={attempts}, keeps={keeps}" + (f" ({extras_str})" if extras_str else "")
        )
        if status != "not_attempted":
            decisions.append(
                Decision(
                    kind=status,
                    subject=name,
                    metric_pct=None,
                    rationale=(
                        f"{keeps}/{attempts} attempts promoted"
                        if attempts
                        else "promoted via optimization_stack fallback"
                    ),
                )
            )

    if not rows:
        warnings.append(
            "capability_summary section is empty — collectors found no "
            "<action>_attempts and no optimization_stack entries."
        )

    md = md_table(["capability", "status", "attempts", "keeps", "extra"], rows)
    return RenderedSection(
        section_id="capability_summary",
        title="Capability Summary",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=warnings,
        skipped=not rows,
    )
