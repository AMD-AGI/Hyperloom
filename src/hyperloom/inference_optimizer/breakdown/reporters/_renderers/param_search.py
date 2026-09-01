# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parameter / backend explore-search state renderer."""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


@register_renderer("param_search")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the parameter / backend explore-search section.

    Summarizes the backends and params explore search (tested vs.
    accepted counts), discovered framework flags, backend-winner history
    and attempted synergy combos. Skipped when no search data exists.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered parameter/backend search section.
    """
    ps = breakdown.get("param_search") or {}
    explore = ps.get("explore") or {}
    backends = ps.get("backends") or {}
    params = ps.get("params") or {}
    flags = ps.get("discovered_flags") or {}
    synergy = ps.get("synergy_attempted") or []

    explore_accepted = len((explore.get("accepted") or []) if isinstance(explore, dict) else [])
    explore_tested = len((explore.get("tested") or {}) if isinstance(explore, dict) else {})
    backends_accepted = len((backends.get("accepted") or []) if isinstance(backends, dict) else [])
    backends_tested = len((backends.get("tested") or {}) if isinstance(backends, dict) else {})
    params_accepted = len((params.get("accepted") or []) if isinstance(params, dict) else [])
    params_tested = len((params.get("tested") or {}) if isinstance(params, dict) else {})
    has_explore = (
        explore_accepted
        or explore_tested
        or (explore.get("cursor") if isinstance(explore, dict) else None) is not None
        or (explore.get("last_round") if isinstance(explore, dict) else None) is not None
    )

    facts: list[str] = []
    facts.append(f"backends search: tested={backends_tested}, accepted={backends_accepted}")
    facts.append(f"params search: tested={params_tested}, accepted={params_accepted}")
    if has_explore:
        facts.append(f"explore ledger: tested={explore_tested}, accepted={explore_accepted}")
    if synergy:
        facts.append(f"synergy_attempted combos: {len(synergy)}.")
    if not flags:
        facts.append("discovered_flags: empty (framework AST never parsed).")
    else:
        for k, v in flags.items():
            if isinstance(v, dict):
                nb = len(v.get("backend_flags") or [])
                np = len(v.get("param_flags") or [])
                facts.append(f"discovered_flags[{k}]: backend={nb}, param={np}, source={v.get('source_path') or '?'}")

    md_parts: list[str] = []
    md_parts.append("**Backends search:**")
    md_parts.append(
        md_table(
            ["accepted", "tested", "cursor", "last_round"],
            [
                [
                    backends_accepted,
                    backends_tested,
                    (backends.get("cursor") if isinstance(backends, dict) else None),
                    (backends.get("last_round") if isinstance(backends, dict) else None),
                ]
            ],
        )
    )
    md_parts.append("")
    md_parts.append("**Params search:**")
    md_parts.append(
        md_table(
            ["accepted", "tested", "cursor", "last_round"],
            [
                [
                    params_accepted,
                    params_tested,
                    (params.get("cursor") if isinstance(params, dict) else None),
                    (params.get("last_round") if isinstance(params, dict) else None),
                ]
            ],
        )
    )
    if has_explore:
        md_parts.append("")
        md_parts.append("**Explore Search:**")
        md_parts.append(
            md_table(
                ["accepted", "tested", "cursor", "last_round"],
                [
                    [
                        explore_accepted,
                        explore_tested,
                        (explore.get("cursor") if isinstance(explore, dict) else None),
                        (explore.get("last_round") if isinstance(explore, dict) else None),
                    ]
                ],
            )
        )

    if synergy:
        md_parts.append("")
        md_parts.append("**Synergy combos attempted**: " + ", ".join(f"`{c}`" for c in synergy[:20]))
        if len(synergy) > 20:
            md_parts[-1] += f" (+{len(synergy) - 20} more)"

    no_search_data = (
        not synergy
        and not flags
        and explore_accepted == 0
        and explore_tested == 0
        and backends_accepted == 0
        and backends_tested == 0
        and params_accepted == 0
        and params_tested == 0
    )

    return RenderedSection(
        section_id="param_search",
        title="Parameter / Backend Search",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        warnings=[],
        skipped=no_search_data,
    )
