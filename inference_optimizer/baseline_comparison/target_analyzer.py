"""Top-level orchestration for the external baseline comparison step.

Layered above :mod:`inferencex_client` and :mod:`name_mapping`. The
:func:`analyze` entry point is the **only** function the action
executor calls; everything else here is helper code kept module-local
so the executor file stays tiny.

Flow:

1. Map the local model path to an InferenceX display name. Miss →
   return a ``BaselineSummary`` with ``status="skipped"`` (no HTTP).
2. Fetch all rows for that model from InferenceX. Failure →
   ``status="fetch_error"``.
3. Filter by ``(gpu, framework, precision, isl, osl)``. Zero rows →
   ``status="no_match"``.
4. Pick the best per-GPU throughput row as ``best``; record every
   concurrency we saw (deduplicated, sorted by conc) for the report.
5. Materialise the summary to disk:

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

from .inferencex_client import DEFAULT_BASE_URL, fetch_rows
from .name_mapping import to_inferencex_name
from .types import BaselinePoint, BaselineQuery, BaselineSummary


log = logging.getLogger(__name__)


def _iso_utc_now() -> str:
    """Return the current UTC time as a second-precision ISO string.

    Returns:
        str: Timestamp formatted as ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_str(value: Any) -> str:
    """Coerce a value to a stripped string.

    Args:
        value (Any): Value to stringify; ``None`` becomes ``""``.

    Returns:
        str: The stripped string form of ``value``.
    """
    return str(value or "").strip()


def _strip_lower(value: Any) -> str:
    """Coerce a value to a stripped, case-folded string.

    Args:
        value (Any): Value to normalise.

    Returns:
        str: The stripped, case-folded string form of ``value``.
    """
    return _strip_str(value).casefold()


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    gpu: str,
    framework: str,
    precision: str,
    isl: int,
    osl: int,
) -> list[dict[str, Any]]:
    """Apply the per-row filter that the Go ``filterRows`` performs.

    All string comparisons are case-insensitive; integer comparisons
    are strict. Pass ``""`` / ``0`` to skip a dimension (matches the
    handler's query-param semantics).

    Args:
        rows (list[dict[str, Any]]): Upstream rows to filter.
        gpu (str): Hardware name to match (``""`` skips).
        framework (str): Framework to match (``""`` skips).
        precision (str): Precision to match (``""`` skips).
        isl (int): Input sequence length to match (``0`` skips).
        osl (int): Output sequence length to match (``0`` skips).

    Returns:
        list[dict[str, Any]]: The rows that pass every active filter.
    """
    out: list[dict[str, Any]] = []
    gpu_n = gpu.casefold()
    fw_n = framework.casefold()
    prec_n = precision.casefold()
    for r in rows:
        if not isinstance(r, dict):
            continue
        if gpu_n and _strip_lower(r.get("hardware")) != gpu_n:
            continue
        if fw_n and _strip_lower(r.get("framework")) != fw_n:
            continue
        if prec_n and _strip_lower(r.get("precision")) != prec_n:
            continue
        if isl and int(r.get("isl") or 0) != int(isl):
            continue
        if osl and int(r.get("osl") or 0) != int(osl):
            continue
        out.append(r)
    return out


def _row_to_point(row: dict[str, Any]) -> BaselinePoint | None:
    """Project one upstream row into a ``BaselinePoint``.

    Latency metrics are converted from seconds to milliseconds.

    Args:
        row (dict[str, Any]): One upstream baseline row.

    Returns:
        BaselinePoint | None: The projected point, or ``None`` if the
            row's metrics lack a positive ``tput_per_gpu``.
    """
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    try:
        tput = float(metrics.get("tput_per_gpu"))
    except (TypeError, ValueError):
        return None
    if tput <= 0:
        return None

    def _fnum(key: str) -> float:
        """Read a metric as a float, defaulting to ``0.0``.

        Args:
            key (str): Metric key to read from the enclosing row's
                ``metrics`` block.

        Returns:
            float: The metric value, or ``0.0`` when missing/invalid.
        """
        try:
            return float(metrics.get(key))
        except (TypeError, ValueError):
            return 0.0

    return BaselinePoint(
        tput_per_gpu=tput,
        output_tput_per_gpu=_fnum("output_tput_per_gpu"),
        conc=int(row.get("conc") or 0),
        decode_tp=int(row.get("decode_tp") or 0),
        mean_ttft_ms=_fnum("mean_ttft") * 1000.0,
        mean_tpot_ms=_fnum("mean_tpot") * 1000.0,
        mean_e2el_ms=_fnum("mean_e2el") * 1000.0,
        date=_strip_str(row.get("date")),
    )


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
    """Run the full target-analysis pipeline and persist the artefacts.

    Never raises. The returned ``BaselineSummary.status`` is one of:

    * ``ok``         — upstream had matching rows, ``best`` populated
    * ``skipped``    — model name mapping miss OR ``compare_against_gpu``
                      was empty; no HTTP issued
    * ``fetch_error`` — upstream HTTP / decode failed
    * ``no_match``   — upstream had data but nothing matched the filter

    Callers that need to branch on *why* the run was skipped should look
    at ``BaselineSummary.reason`` instead of regex-matching the warning
    string. ``reason`` is one of:

    * ``ok``                       — populated ``best``
    * ``model_mapping_miss``       — local model has no InferenceX name
    * ``no_target_gpu_configured`` — empty ``compare_against_gpu`` (the
                                      caller deliberately passed nothing)
    * ``fetch_error``              — upstream call failed
    * ``no_match``                 — fetched rows but filter dropped all

    The caller (target_analysis ActionRunner) treats every status as
    a success — the optimizer loop never blocks on this step.

    Args:
        session_dir (Path): Session root for persisting the artefacts.
        model_path (str): Local model path / HF repo name to map to an
            InferenceX display name.
        compare_against_gpu (str): Target GPU label to filter rows by;
            an empty value short-circuits to a ``skipped`` summary.
        framework (str): Optional inference framework filter.
        precision (str): Optional precision filter (e.g. ``fp8``).
        isl (int): Optional input sequence length filter.
        osl (int): Optional output sequence length filter.

    Returns:
        BaselineSummary: The persisted summary, whose ``status`` and
            ``reason`` describe the outcome (never raises).
    """
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
                "no InferenceX display name found"
            ),
            source=DEFAULT_BASE_URL,
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
            source=DEFAULT_BASE_URL,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    rows, fetch_warn = fetch_rows(canonical_model)
    if rows is None:
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status="fetch_error",
            reason="fetch_error",
            warning=fetch_warn or "unknown fetch failure",
            source=DEFAULT_BASE_URL,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    matched = _filter_rows(
        rows,
        gpu=query.gpu,
        framework=query.framework,
        precision=query.precision,
        isl=query.isl,
        osl=query.osl,
    )
    points = [p for p in (_row_to_point(r) for r in matched) if p is not None]

    if not points:
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status="no_match",
            reason="no_match",
            warning=(
                f"fetched {len(rows)} rows for {canonical_model!r} but none "
                f"matched filter (gpu={query.gpu!r}, framework={query.framework!r}, "
                f"precision={query.precision!r}, isl={query.isl}, osl={query.osl})"
            ),
            source=DEFAULT_BASE_URL,
        )
        _persist(summary, session_dir=session_dir)
        return summary

    all_points = _dedup_by_conc(points)
    best = max(points, key=lambda p: p.tput_per_gpu)
    summary = BaselineSummary(
        query=query,
        fetched_at=now,
        row_count=len(matched),
        best=best,
        all_concurrencies=all_points,
        status="ok",
        reason="ok",
        warning="",
        source=DEFAULT_BASE_URL,
    )
    _persist(summary, session_dir=session_dir)
    return summary


__all__ = [
    "analyze",
]
