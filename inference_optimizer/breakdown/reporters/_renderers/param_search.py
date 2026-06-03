"""Parameter / backend DFS state renderer."""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


@register_renderer("param_search")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the parameter / backend DFS-search section.

    Summarizes the backends and params depth-first search (tested vs.
    accepted counts), discovered framework flags, backend-winner history
    and attempted synergy combos. Skipped when no search data exists.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered parameter/backend search section.
    """
    ps = breakdown.get("param_search") or {}
    backends = ps.get("backends") or {}
    params = ps.get("params") or {}
    flags = ps.get("discovered_flags") or {}
    winners = ps.get("backend_winners_history") or []
    synergy = ps.get("synergy_attempted") or []

    backends_accepted = len((backends.get("accepted") or []) if isinstance(backends, dict) else [])
    backends_tested = len((backends.get("tested") or {}) if isinstance(backends, dict) else {})
    params_accepted = len((params.get("accepted") or []) if isinstance(params, dict) else [])
    params_tested = len((params.get("tested") or {}) if isinstance(params, dict) else {})

    facts: list[str] = []
    facts.append(
        f"backends DFS: tested={backends_tested}, accepted={backends_accepted}"
    )
    facts.append(
        f"params DFS: tested={params_tested}, accepted={params_accepted}"
    )
    if winners:
        facts.append(f"backend_winners_history: {len(winners)} round(s).")
    if synergy:
        facts.append(f"synergy_attempted combos: {len(synergy)}.")
    if not flags:
        facts.append("discovered_flags: empty (framework AST never parsed).")
    else:
        for k, v in flags.items():
            if isinstance(v, dict):
                nb = len(v.get("backend_flags") or [])
                np = len(v.get("param_flags") or [])
                facts.append(
                    f"discovered_flags[{k}]: backend={nb}, param={np}, "
                    f"source={v.get('source_path') or '?'}"
                )

    md_parts: list[str] = []
    md_parts.append("**Backends DFS:**")
    md_parts.append(md_table(
        ["accepted", "tested", "cursor", "last_round"],
        [[backends_accepted, backends_tested,
          (backends.get("cursor") if isinstance(backends, dict) else None),
          (backends.get("last_round") if isinstance(backends, dict) else None)]],
    ))
    md_parts.append("")
    md_parts.append("**Params DFS:**")
    md_parts.append(md_table(
        ["accepted", "tested", "cursor", "last_round"],
        [[params_accepted, params_tested,
          (params.get("cursor") if isinstance(params, dict) else None),
          (params.get("last_round") if isinstance(params, dict) else None)]],
    ))

    if winners:
        md_parts.append("")
        md_parts.append(f"**Backend winners history** (last 5 of {len(winners)}):")
        rows = []
        for w in winners[-5:]:
            rows.append([
                w.get("round_id"), w.get("action"),
                w.get("base_tput"),
                ((w.get("best") or {}).get("name") if isinstance(w.get("best"), dict) else None),
                ((w.get("best") or {}).get("gain_pct") if isinstance(w.get("best"), dict) else None),
                w.get("ts"),
            ])
        md_parts.append(md_table(
            ["round", "action", "base_tput", "best_name", "best_gain_pct", "ts"],
            rows,
        ))

    if synergy:
        md_parts.append("")
        md_parts.append("**Synergy combos attempted**: "
                        + ", ".join(f"`{c}`" for c in synergy[:20]))
        if len(synergy) > 20:
            md_parts[-1] += f" (+{len(synergy)-20} more)"

    no_search_data = (
        not winners and not synergy and not flags
        and backends_accepted == 0 and backends_tested == 0
        and params_accepted == 0 and params_tested == 0
    )

    return RenderedSection(
        section_id="param_search",
        title="Parameter / Backend Search",
        key_facts=facts,
        markdown_block="\n".join(md_parts).strip(),
        warnings=[],
        skipped=no_search_data,
    )
