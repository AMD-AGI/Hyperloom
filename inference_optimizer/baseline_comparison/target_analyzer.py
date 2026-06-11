# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Top-level orchestration for the external baseline comparison step.

The :func:`analyze` entry point is the **only** function the action
executor calls; everything else here is helper code kept module-local
so the executor file stays tiny.

Flow:

1. Map the local model path to a canonical display name. Miss →
   return a ``BaselineSummary`` with ``status="skipped"``.
2. Read the LLM-authored ``competitor_target.json`` (produced by the
   research scout, every datapoint source-backed). No sourced rows →
   ``status="no_match"``.
3. Project each per-concurrency target row into a ``BaselinePoint``;
   pick the best per-GPU throughput row as ``best`` and record every
   concurrency (deduplicated, sorted by conc) for the report.
4. Materialise the summary to disk:

   * ``target_analysis/target_baseline.json`` — machine-readable
   * ``target_analysis/target_analysis_report.md`` — short human note

All paths derived via :mod:`session_paths` — no hand-rolled string
concatenation under the session dir.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inferencex_client import DEFAULT_BASE_URL
from .name_mapping import to_inferencex_name
from .types import BaselinePoint, BaselineQuery, BaselineSummary


log = logging.getLogger(__name__)

LLM_AUTHORED_SOURCE = "llm_authored"


def _iso_utc_now() -> str:
    """Return the current UTC time as a second-precision ISO string.

    Returns:
        str: Timestamp formatted as ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dedup_by_conc(points: list[BaselinePoint]) -> list[BaselinePoint]:
    """Keep the highest ``tput_per_gpu`` per (conc, decode_tp) combo.

    Upstream sometimes contains multiple rows for the same (conc, tp)
    (different dates or sweep methods). The report only needs the best
    one per slot — keeping all of them just clutters the markdown.

    Args:
        points (list[BaselinePoint]): Candidate points, possibly with
            duplicate ``(conc, decode_tp)`` combos.

    Returns:
        list[BaselinePoint]: Best point per ``(conc, decode_tp)``,
            sorted by those two keys.
    """
    best: dict[tuple[int, int], BaselinePoint] = {}
    for p in points:
        key = (p.conc, p.decode_tp)
        cur = best.get(key)
        if cur is None or p.tput_per_gpu > cur.tput_per_gpu:
            best[key] = p
    return sorted(best.values(), key=lambda p: (p.conc, p.decode_tp))


def _format_report_md(summary: BaselineSummary) -> str:
    """Render a 10-15 line human-readable markdown summary.

    Intentionally avoids printing a gap percentage — the agreed
    contract is "facts only, no derived KPI" so this section never
    accidentally becomes an optimisation target (see S2 in the design
    chat).

    Args:
        summary (BaselineSummary): The summary to render.

    Returns:
        str: The markdown report text (newline-terminated).
    """
    q = summary.query
    lines: list[str] = []
    lines.append(f"# Target analysis — external baseline ({summary.status})")
    lines.append("")
    lines.append(f"- Source: {summary.source or DEFAULT_BASE_URL}")
    lines.append(f"- Fetched at: {summary.fetched_at}")
    lines.append(
        "- Query: "
        f"model=`{q.model or '(unset)'}`  "
        f"gpu=`{q.gpu or '(unset)'}`  "
        f"framework=`{q.framework or '(any)'}`  "
        f"precision=`{q.precision or '(any)'}`  "
        f"ISL/OSL=`{q.isl or '(any)'}/{q.osl or '(any)'}`"
    )
    lines.append(f"- Rows matched: {summary.row_count}")
    if summary.warning:
        lines.append(f"- Warning: {summary.warning}")
    lines.append("")

    if summary.status != "ok" or summary.best is None:
        lines.append(
            "> No reference data point is available — the orchestrator was "
            "**not** affected by this step (target_analysis only feeds the "
            "final report)."
        )
        return "\n".join(lines) + "\n"

    b = summary.best
    lines.append("## Reference best (per-GPU throughput)")
    lines.append("")
    lines.append(f"- Throughput/GPU: **{b.tput_per_gpu:.1f}** tok/s/GPU")
    lines.append(f"  - at concurrency {b.conc}, decode TP {b.decode_tp}")
    if b.output_tput_per_gpu:
        lines.append(f"- Output Throughput/GPU: {b.output_tput_per_gpu:.1f} tok/s/GPU")
    if b.mean_ttft_ms:
        lines.append(f"- Mean TTFT: {b.mean_ttft_ms:.1f} ms")
    if b.mean_tpot_ms:
        lines.append(f"- Mean TPOT: {b.mean_tpot_ms:.3f} ms")
    if b.mean_e2el_ms:
        lines.append(f"- Mean E2E latency: {b.mean_e2el_ms:.1f} ms")
    if b.date:
        lines.append(f"- Reference run date: {b.date}")
    lines.append("")

    if summary.all_concurrencies:
        lines.append("## All matched concurrencies")
        lines.append("")
        lines.append("| conc | decode_tp | tput/GPU | mean_tpot (ms) |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for p in summary.all_concurrencies:
            lines.append(
                f"| {p.conc} | {p.decode_tp} "
                f"| {p.tput_per_gpu:.1f} | {p.mean_tpot_ms:.3f} |"
            )
        lines.append("")

    lines.append(
        "> Advisory only. This data does **not** feed Objective, scoring, or "
        "any agent prompt — it is shown here for post-mortem comparison."
    )
    return "\n".join(lines) + "\n"


def _persist(
    summary: BaselineSummary,
    *,
    session_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON + MD into ``<session_dir>/target_analysis/``.

    Uses :mod:`session_paths` for path computation. Returns the
    ``(json_path, md_path)`` tuple so the executor can surface them
    on the bus event.

    Args:
        summary (BaselineSummary): The analysis artefact to serialise.
        session_dir (Path): Session root under which the
            ``target_analysis/`` output directory is created.

    Returns:
        tuple[Path, Path]: The ``(json_path, md_path)`` of the written
            JSON and Markdown report files.
    """
    from ..session_paths import (
        target_analysis_dir,
        target_analysis_report_md,
        target_baseline_json,
    )
    out_dir = target_analysis_dir(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_baseline_json(session_dir)
    md_path = target_analysis_report_md(session_dir)
    json_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_format_report_md(summary), encoding="utf-8")
    return json_path, md_path


def _target_row_to_point(row: dict[str, Any]) -> BaselinePoint | None:
    """Project one LLM-authored ``per_conc`` row into a ``BaselinePoint``.

    Returns ``None`` when ``tput_per_gpu`` is missing or non-positive.
    ``tpot_ms`` is the per-output-token latency; ``interactivity`` (when
    present) is informational only and not folded into the point shape.
    """
    def _fnum(key: str) -> float:
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            return 0.0

    tput = _fnum("tput_per_gpu")
    if tput <= 0:
        return None
    return BaselinePoint(
        tput_per_gpu=tput,
        output_tput_per_gpu=0.0,
        conc=int(row.get("conc") or 0),
        decode_tp=0,
        mean_ttft_ms=0.0,
        mean_tpot_ms=_fnum("tpot_ms"),
        mean_e2el_ms=0.0,
        date="",
    )


def analyze(
    *,
    session_dir: Path,
    model_path: str,
    compare_against_gpu: str,
    framework: str = "",
    precision: str = "",
    isl: int = 0,
    osl: int = 0,
) -> BaselineSummary:
    """Build the target-analysis summary from LLM-authored competitor data.

    The reference numbers come from ``competitor_target.json`` (produced
    by the research scout, every datapoint source-backed) rather than a
    live HTTP pull. Persists the same ``BaselineSummary`` disk contract so
    the report renderer is unchanged.

    Never raises. ``BaselineSummary.status`` is one of:

    * ``ok``       — competitor target had a usable per-conc row
    * ``skipped``  — model name mapping miss OR ``compare_against_gpu``
                     was empty
    * ``no_match`` — no ``competitor_target.json`` / no sourced rows

    ``reason`` mirrors ``status`` with finer granularity:
    ``ok`` / ``model_mapping_miss`` / ``no_target_gpu_configured`` /
    ``no_competitor_target``.
    """
    from ..orchestrator import research_hints

    canonical_model = to_inferencex_name(model_path) or ""
    query = BaselineQuery(
        model=canonical_model,
        gpu=compare_against_gpu.strip(),
        framework=framework.strip(),
        precision=precision.strip(),
        isl=int(isl or 0),
        osl=int(osl or 0),
    )
    now = _iso_utc_now()

    if not canonical_model:
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status="skipped",
            reason="model_mapping_miss",
            warning=(
                f"model name mapping miss for {model_path!r}; "
                "no display name found"
            ),
            source=LLM_AUTHORED_SOURCE,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    if not query.gpu:
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status="skipped",
            reason="no_target_gpu_configured",
            warning="compare_against_gpu is empty",
            source=LLM_AUTHORED_SOURCE,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    target = research_hints.load_competitor_target(Path(session_dir))
    rows = list((target or {}).get("per_conc") or [])
    points = [p for p in (_target_row_to_point(r) for r in rows) if p is not None]

    if not points:
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status="no_match",
            reason="no_competitor_target",
            warning=(
                "no sourced competitor_target.json available "
                "(research scout disabled or produced no targets)"
            ),
            source=LLM_AUTHORED_SOURCE,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    all_points = _dedup_by_conc(points)
    best = max(points, key=lambda p: p.tput_per_gpu)
    target_sources = sorted({
        str(r.get("source")).strip() for r in rows if str(r.get("source") or "").strip()
    })
    summary = BaselineSummary(
        query=query,
        fetched_at=now,
        row_count=len(points),
        best=best,
        all_concurrencies=all_points,
        status="ok",
        reason="ok",
        warning=(
            "sources: " + "; ".join(target_sources) if target_sources else ""
        ),
        source=LLM_AUTHORED_SOURCE,
    )
    _persist(summary, session_dir=session_dir)
    return summary


__all__ = [
    "analyze",
]
