"""Sweep matrix renderer.

Sweeps are concurrency / ISL-OSL grids run on the final stack so users
can see end-of-session frontier coverage. The renderer surfaces:

* grid size, accepted vs failed count,
* the best (highest output_throughput) point,
* all variants in a small table (truncated to 50 rows).
"""

from __future__ import annotations

from typing import Any

from ..base import Decision, RenderedSection, md_table, register_renderer

_MAX_ROWS = 50


@register_renderer("sweep")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the sweep section: grid coverage, best point and variants.

    Surfaces the concurrency / ISL-OSL grid run on the final stack,
    including success/failure counts, the best-throughput point, and a
    truncated variant table. Skipped when no sweep ran this session.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, or a skipped placeholder when
            there are no sweep variants.
    """
    sw = breakdown.get("sweep") or {}
    raw_variants = sw.get("all_variants") or []
    variants: list[dict[str, Any]] = [
        v if isinstance(v, dict) else {"raw": str(v)} for v in raw_variants
    ]
    if not variants:
        return RenderedSection(
            section_id="sweep",
            title="Sweep",
            key_facts=["No sweep run this session."],
            markdown_block="",
            decisions=[Decision(
                kind="not_attempted",
                subject="sweep",
                rationale="no sweep variants captured",
            )],
            warnings=[],
            skipped=True,
        )

    grid_size = sw.get("grid_size") or len(variants)
    successes = [v for v in variants if v.get("status") == "success"]
    failures = [v for v in variants if v.get("status") != "success"]

    def _ot(v: dict[str, Any]) -> float:
        """Extract a variant's output throughput as a float sort key.

        Args:
            v (dict[str, Any]): A sweep variant record.

        Returns:
            float: The ``output_throughput`` value, or ``0.0`` when missing or
                non-numeric.
        """
        try:
            return float(v.get("output_throughput") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    best = max(successes, key=_ot, default=None)

    facts: list[str] = []
    facts.append(
        f"Sweep grid={grid_size}, success={len(successes)}, failed={len(failures)}."
    )
    if best:
        facts.append(
            f"Best point: conc={best.get('conc')} isl={best.get('isl')} "
            f"osl={best.get('osl')} → tput={best.get('output_throughput')} "
            f"ttft={best.get('ttft')}ms"
        )

    decisions: list[Decision] = [Decision(
        kind="kept" if best else "attempted",
        subject="sweep",
        rationale=f"{len(successes)}/{grid_size} variants succeeded",
    )]

    head = variants[:_MAX_ROWS]
    rows = []
    for v in head:
        rows.append([
            v.get("conc"), v.get("isl"), v.get("osl"),
            v.get("status"),
            v.get("output_throughput"),
            v.get("ttft"),
            v.get("e2el"),
            (v.get("error") or "")[:60],
        ])
    md = md_table(
        ["conc", "isl", "osl", "status", "tput", "ttft", "e2el", "error"],
        rows,
    )
    if len(variants) > _MAX_ROWS:
        md = f"_Showing first {_MAX_ROWS} of {len(variants)} variants._\n\n" + md

    return RenderedSection(
        section_id="sweep",
        title="Sweep",
        key_facts=facts,
        markdown_block=md,
        decisions=decisions,
        warnings=[],
        skipped=False,
    )
