"""Capability summary renderer.

Renders the ``capability_summary`` section into:

* a markdown table listing each capability's status / attempts / keeps,
* one ``Decision`` per non-``not_attempted`` capability so the LLM has
  structured verdicts to reference,
* ``key_facts`` that paraphrase each row in one line each.

This renderer is the canonical example used as the design reference
for the other 13 renderers — keep it terse and deterministic.
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, fmt_pct, md_table, register_renderer

_CAPABILITY_ORDER = (
    # ``explore`` is the primary row for sessions
    #. backends / params / validate_stack
    # remain as compatibility aliases so legacy resume reports stay
    # readable.
    "explore",
    "backends", "params", "sweep", "geak", "oob", "validate_stack",
)


@register_renderer("capability_summary")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    cap = breakdown.get("capability_summary") or {}
    rows: list[list[Any]] = []
    facts: list[str] = []
    decisions: list[Decision] = []
    warnings: list[str] = []

    # Stable ordering: known capabilities first, then any extras the
    # exporter started writing without a renderer update.
    keys = list(_CAPABILITY_ORDER) + [k for k in cap if k not in _CAPABILITY_ORDER]
    for name in keys:
        v = cap.get(name)
        if v is None:
            continue
        status   = str(v.get("status") or "not_attempted")
        attempts = int(v.get("attempts") or 0)
        keeps    = int(v.get("keeps") or 0)
        # Disambiguate the validate_stack "not_attempted but I have a
        # validated_gain" case: it means the validate_stack action did
        # not re-run this session, but state carries a validated gain
        # from an earlier round. Show that as ``stale_validated`` so
        # the row is no longer self-contradicting.
        if (
            name == "validate_stack"
            and status == "not_attempted"
            and v.get("last_validated_gain_pct") is not None
        ):
            status = "stale_validated"
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
        # v0.8 M3 explore extras.
        if "keep_unstable_count" in v and v["keep_unstable_count"]:
            extras.append(f"keep_unstable={v['keep_unstable_count']}")
        if "winners_history" in v and v["winners_history"]:
            extras.append(f"history={v['winners_history']}")
        extras_str = " · ".join(extras) if extras else ""

        rows.append([name, status, attempts, keeps, extras_str])
        facts.append(
            f"`{name}` status={status}, attempts={attempts}, keeps={keeps}"
            + (f" ({extras_str})" if extras_str else "")
        )
        if status != "not_attempted":
            decisions.append(Decision(
                kind=status,
                subject=name,
                metric_pct=None,
                rationale=(
                    f"{keeps}/{attempts} attempts promoted"
                    if attempts else "promoted via optimization_stack fallback"
                ),
            ))

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
