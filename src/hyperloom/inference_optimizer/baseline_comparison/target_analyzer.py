# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level orchestration for the external baseline comparison step."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.common.timeutil import now_iso

from .inferencex_client import (
    DEFAULT_BASE_URL,
    base_url,
    fetch_agentic_interactivity,
    fetch_rows,
    find_reference_rows,
    normalize_benchmark_id,
)
from .types import BaselinePoint, BaselineQuery, BaselineReason, BaselineSummary, BenchmarkMode


# --- InferenceX model name mapping -------------------------------------------

KNOWN_INFERENCEX_MODELS: tuple[str, ...] = (
    "DeepSeek-R1-0528",
    "GLM-5",
    "GLM-5.2",
    "gpt-oss-120b",
    "Llama-3.3-70B-Instruct-FP8",
    "Qwen-3.5-397B-A17B",
    "Kimi-K2.5",
    "MiniMax-M2.5",
    "MiniMax-M3",
)

_VENDOR_PREFIX_RE = re.compile(
    r"^(MiniMaxAI[-_]|deepseek-ai[-_]|deepseek[-_]|meta-llama[-_]|"
    r"Qwen[-_]|moonshotai[-_]|openai[-_]|google[-_]|microsoft[-_]|"
    r"zhipuai[-_]|THUDM[-_])",
    re.IGNORECASE,
)


def to_inferencex_name(model_path_or_name: str) -> str | None:
    """Translate a local path / HF repo string into an InferenceX display name."""
    if not model_path_or_name:
        return None
    raw = str(model_path_or_name).strip()
    if not raw:
        return None

    candidate = Path(raw).name if ("/" in raw or "\\" in raw) else raw
    stripped = _VENDOR_PREFIX_RE.sub("", candidate, count=1)

    # Try the full candidate before the vendor-stripped form: several canonical InferenceX names start with a token
    # the prefix regex would strip (``DeepSeek-``, ``Qwen-``), so stripping first would break exact matches.
    for needle in (candidate.casefold(), stripped.casefold()):
        for known in KNOWN_INFERENCEX_MODELS:
            if known.casefold() == needle:
                return known
        for known in ("GLM-5.2", "MiniMax-M3"):
            if needle in {f"{known.casefold()}-{suffix}" for suffix in ("fp4", "fp8", "mxfp4", "nvfp4")}:
                return known

    return None


def _dedup_by_conc(points: list[BaselinePoint]) -> list[BaselinePoint]:
    """Keep the highest ``tput_per_gpu`` per (conc, decode_tp) combo.

    Args:
        points: Reference points; ties keep the first point in input order.
    """
    best: dict[tuple[int, int], BaselinePoint] = {}
    for p in points:
        key = (p.conc, p.decode_tp)
        cur = best.get(key)
        if cur is None or p.tput_per_gpu > cur.tput_per_gpu:
            best[key] = p
    return sorted(best.values(), key=lambda p: (p.conc, p.decode_tp))


def _format_report_md(summary: BaselineSummary) -> str:
    """Render a human-readable markdown summary of the external-baseline lookup."""
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
    if q.benchmark_mode == "agentx":
        lines.append("- Workload: agentic_traces (variable request lengths)")
    lines.append(f"- Rows matched: {summary.row_count}")
    if summary.warning:
        lines.append(f"- Warning: {summary.warning}")
    lines.append("")

    if summary.status != "ok" or summary.best is None:
        lines.append(
            "> No reference data point is available for prompt advisory or final-report comparison. "
            "Objective, scoring, and KEEP/REVERT gates do not depend on this reference."
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
            lines.append(f"| {p.conc} | {p.decode_tp} | {p.tput_per_gpu:.1f} | {p.mean_tpot_ms:.3f} |")
        lines.append("")

    lines.append(
        "> Advisory only. This InferenceX-measured reference never feeds the "
        "Objective, scoring, or any KEEP/REVERT gate; a matching row is "
        "surfaced to the gap advisory as direction only."
    )
    return "\n".join(lines) + "\n"


def persist_summary(
    summary: BaselineSummary,
    *,
    session_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON + MD into ``<session_dir>/target_analysis/``."""
    from ..session.session_paths import (
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


def _row_to_point(row: dict[str, Any]) -> BaselinePoint | None:
    """Project one raw InferenceX benchmark record into a ``BaselinePoint``."""
    if not isinstance(row, dict):
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    tput = to_float(metrics.get("tput_per_gpu"), default=0.0)
    if tput <= 0:
        return None
    return BaselinePoint(
        tput_per_gpu=tput,
        output_tput_per_gpu=to_float(metrics.get("output_tput_per_gpu"), default=0.0),
        conc=int(row.get("conc") or 0),
        decode_tp=int(row.get("decode_tp") or 0),
        mean_ttft_ms=to_float(metrics.get("mean_ttft"), default=0.0) * 1000.0,
        mean_tpot_ms=to_float(metrics.get("mean_tpot"), default=0.0) * 1000.0,
        mean_e2el_ms=to_float(metrics.get("mean_e2el"), default=0.0) * 1000.0,
        date=str(row.get("date") or ""),
    )


def _write_measured_competitor_target(
    session_dir: Path,
    query: BaselineQuery,
    points: list[BaselinePoint],
    source: str,
) -> bool:
    """Persist a measured ``competitor_target.json`` (``source`` = live API URL)."""
    per_conc: list[dict[str, Any]] = []
    for p in points:
        row = {"conc": p.conc, "tput_per_gpu": p.tput_per_gpu, "source": source}
        if query.benchmark_mode == "agentx":
            row.update(e2e_norm_intvty_p90=p.e2e_norm_intvty_p90, benchmark_id=p.benchmark_id, decode_tp=p.decode_tp)
        else:
            row.update(tpot_ms=p.mean_tpot_ms, interactivity=1000.0 / p.mean_tpot_ms if p.mean_tpot_ms > 0 else 0.0)
        per_conc.append(row)
    if not per_conc:
        return False
    try:
        from hyperloom.orchestrator.knowledge import research_hints

        target = {
            "gpu": query.gpu,
            "model": query.model,
            "framework": query.framework,
            "precision": query.precision,
            "per_conc": per_conc,
            "notes": f"InferenceX measured reference ({query.model} @ {query.gpu})",
        }
        if query.benchmark_mode == "agentx":
            target.update(benchmark_mode="agentx", throughput_basis="total_token_throughput_per_gpu")
        return research_hints.write_competitor_target(Path(session_dir), target)
    except Exception:  # noqa: BLE001 — advisory feed is best-effort
        return False


def clear_competitor_target(session_dir: Path) -> None:
    """Remove any existing ``competitor_target.json``. Best-effort, never raises."""
    try:
        from ..session import session_paths

        path = session_paths.competitor_target_json(Path(session_dir))
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


def analyze(
    *,
    session_dir: Path,
    model_path: str,
    compare_against_gpu: str,
    framework: str = "",
    precision: str = "",
    isl: int | None = 0,
    osl: int | None = 0,
    benchmark_mode: BenchmarkMode = "synthetic",
) -> BaselineSummary:
    """Build the target-analysis summary from live InferenceX measurements.

    Args:
        session_dir: Destination for the summary and measured competitor target.
        model_path: Local path or model name to map to InferenceX.
        compare_against_gpu: Target hardware; empty skips the lookup.
        framework: Query metadata only, not a row filter.
        precision: Optional filter; agentx maps MXFP4/NVFP4 to FP4.
        isl: Synthetic input length; ignored in agentx mode.
        osl: Synthetic output length; ignored in agentx mode.
        benchmark_mode: Synthetic fixed-shape or agentx agentic-trace matching.
    """
    canonical_model = to_inferencex_name(model_path) or ""
    agentx = benchmark_mode == "agentx"
    query_precision = precision.strip()
    if agentx and query_precision.casefold() in {"mxfp4", "nvfp4"}:
        query_precision = "fp4"
    query = BaselineQuery(
        model=canonical_model,
        gpu=compare_against_gpu.strip(),
        framework=framework.strip(),
        precision=query_precision,
        isl=None if agentx else int(isl or 0),
        osl=None if agentx else int(osl or 0),
        benchmark_mode=benchmark_mode,
    )
    now = now_iso(timespec="seconds", z_suffix=True)
    source = base_url()

    def _skip(status: str, reason: BaselineReason, warning: str) -> BaselineSummary:
        """Persist and return a no-data summary (skipped / no_match cases)."""
        summary = BaselineSummary(
            query=query,
            fetched_at=now,
            row_count=0,
            best=None,
            status=status,
            reason=reason,
            warning=warning,
            source=source,
        )
        persist_summary(summary, session_dir=session_dir)
        clear_competitor_target(session_dir)
        return summary

    if not canonical_model:
        return _skip(
            "skipped", "model_mapping_miss", f"model name mapping miss for {model_path!r}; no InferenceX name found"
        )

    if not query.gpu:
        return _skip("skipped", "no_target_gpu_configured", "compare_against_gpu is empty")

    rows = fetch_rows(canonical_model)
    if rows is None:
        return _skip("no_match", "fetch_error", f"InferenceX API fetch failed for model={canonical_model!r}")
    if not rows:
        return _skip(
            "no_match", "no_inferencex_data", f"InferenceX returned no benchmarks for model={canonical_model!r}"
        )

    matched = find_reference_rows(
        rows,
        hardware=query.gpu,
        isl=query.isl,
        osl=query.osl,
        precision=query.precision,
        benchmark_mode=query.benchmark_mode,
    )
    if not matched:
        hw = query.gpu.strip().casefold()
        has_gpu = any(isinstance(r, dict) and str(r.get("hardware") or "").strip().casefold() == hw for r in rows)
        if not has_gpu:
            return _skip(
                "no_match",
                "unsupported_target_gpu",
                f"InferenceX has no {query.gpu!r} data for model={canonical_model!r}",
            )
        # GPU present but no comparable row.
        if query.precision:
            shape_rows = find_reference_rows(
                rows,
                hardware=query.gpu,
                isl=query.isl,
                osl=query.osl,
                precision="",
                benchmark_mode=query.benchmark_mode,
            )
            if shape_rows:
                return _skip(
                    "no_match",
                    "precision_mismatch",
                    f"InferenceX has gpu={query.gpu} isl/osl={query.isl}/{query.osl} rows "
                    f"but none at precision={query.precision}",
                )
        return _skip(
            "no_match",
            "dimension_mismatch",
            f"no InferenceX row for gpu={query.gpu} isl/osl={query.isl}/{query.osl} "
            f"precision={query.precision or '(any)'}",
        )

    points: list[BaselinePoint] = []
    by_id: dict[str, BaselinePoint] = {}
    for row in matched:
        point = _row_to_point(row)
        if point is None:
            continue
        if agentx:
            try:
                point.benchmark_id = normalize_benchmark_id(row.get("id"))
            except ValueError:
                pass  # Keep the row as a throughput-only reference.
            if point.benchmark_id is not None:
                if point.benchmark_id in by_id:
                    continue
                by_id[point.benchmark_id] = point
        points.append(point)
    if not points:
        return _skip("no_match", "no_valid_rows", "matched InferenceX rows had no positive tput_per_gpu")

    warnings: list[str] = []
    if agentx:
        warnings.append(
            "cross-system reference, not a GPU-only comparison; precision bucket does not prove identical "
            "quantization, corpus or deployment configuration"
        )
        derived = fetch_agentic_interactivity(list(by_id)) if by_id else {}
        if derived is None:
            warnings.append("P90 fetch failed; total throughput remains available")
        else:
            for key, point in by_id.items():
                point.e2e_norm_intvty_p90 = derived.get(key)
        missing_p90 = sum(p.e2e_norm_intvty_p90 is None for p in points)
        if missing_p90:
            warnings.append(f"P90 unavailable for {missing_p90} of {len(points)} reference rows")

    all_points = _dedup_by_conc(points)
    best = max(points, key=lambda p: p.tput_per_gpu)
    dates = sorted({p.date for p in points if p.date})
    if dates:
        warnings.append("reference dates: " + ", ".join(dates))
    summary = BaselineSummary(
        query=query,
        fetched_at=now,
        row_count=len(points),
        best=best,
        all_concurrencies=all_points,
        status="ok",
        reason="ok",
        warning="; ".join(warnings),
        source=source,
    )
    persist_summary(summary, session_dir=session_dir)
    if not _write_measured_competitor_target(Path(session_dir), query, all_points, source):
        clear_competitor_target(session_dir)
    return summary


__all__ = [
    "analyze",
    "clear_competitor_target",
    "persist_summary",
    "to_inferencex_name",
]
