# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX-style concurrency sweep comparison chart.

Renders a throughput-vs-interactivity curve for baseline vs optimised arms
produced by :mod:`conc_sweep`.  The function is **best-effort**: any import
failure (missing ``matplotlib``) or data / IO error is logged and ``None``
is returned so callers can skip the chart without aborting the report.

Axes follow the metric the payload's ``summary`` was graded on, so the chart
shows the ranking the session was actually scored by -- read off the record
rather than re-derived, because the grading rule consults the environment of
the process that ran the sweep.

Total token throughput — the pair InferenceX ranks a submission by:
  x = intvty_p90                     (p90 interactivity, tok/s/user)
  y = total_token_throughput / tp    (tok/s per chip)

Output throughput — no aiperf export, so interactivity is approximated from
concurrency:
  x = output_throughput / conc       (tok/s per user)
  y = output_throughput / tp         (tok/s per GPU)

``benchmark_mode`` stays a separate question: it says whether the workload was
an agentic replay, which decides whether the session's ISL/OSL are real and so
whether the roofline can be drawn at all.

Rendered on a black background for a high-contrast dashboard look. Colours:
  baseline  — red        ``#FF4C4C``
  optimized — orange     ``#FF8C00``
  ceiling   — grey       ``#888888`` dashed (off by default)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from hyperloom.common.perf_metric import GRADED_OUTPUT, is_agentx_mode

log = logging.getLogger(__name__)


def _positive(value: Any) -> float | None:
    """Coerce to a strictly positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _agentx_xy(point: Mapping[str, Any], tp_eff: float) -> tuple[float, float] | None:
    """p90 interactivity against token throughput per chip."""
    intvty = _positive(point.get("intvty_p90"))
    total = _positive(point.get("total_token_throughput"))
    if intvty is None or total is None:
        return None
    return intvty, total / tp_eff


def _synthetic_xy(point: Mapping[str, Any], tp_eff: float) -> tuple[float, float] | None:
    """Output throughput per user against output throughput per GPU."""
    tput = _positive(point.get("output_throughput"))
    conc = _positive(point.get("conc"))
    if tput is None or conc is None:
        return None
    return tput / conc, tput / tp_eff


@dataclass(frozen=True)
class _Axes:
    """The pair a graded metric is plotted on."""

    point_xy: Callable[[Mapping[str, Any], float], tuple[float, float] | None]
    x_label: str
    y_label: str


def _graded_metric_of(data: Mapping[str, Any]) -> str:
    """The axis this sweep was graded on, as the sweep recorded it.

    Read rather than re-derived: the grading rule consults the environment,
    which describes the process that ran the sweep, not the one drawing the
    chart. A summary that names no metric was taken on
    ``conc_pair_comparison``'s default.
    """
    return str((data.get("summary") or {}).get("metric") or "").strip() or GRADED_OUTPUT


def _axes_for_metric(metric: str, tp_eff: float) -> _Axes:
    """Pick the axis pair a graded metric is read on."""
    if metric != GRADED_OUTPUT:
        return _Axes(
            point_xy=_agentx_xy,
            x_label="P90 Interactivity  (tok/s/user)",
            y_label=f"Token Throughput per Chip  (tok/s/chip, tp={int(tp_eff)})",
        )
    return _Axes(
        point_xy=_synthetic_xy,
        x_label="Interactivity  (output_throughput / concurrency,  tok/s/user)",
        y_label=f"Efficiency  (output_throughput / tp={int(tp_eff)},  tok/s/GPU)",
    )


def render_conc_sweep_curve(
    payload: dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    model_label: str = "",
    gpu_label: str = "",
    tp: int = 1,
    isl: int = 0,
    osl: int = 0,
    draw_ceiling: bool = False,
) -> Path | None:
    """Render a throughput-vs-interactivity PNG from a ``conc_sweep_summary.json``.

    Rendered on a dark (black) background for a high-contrast dashboard look.

    Args:
        payload: Either the already-parsed payload dict, or a file path to the
            ``conc_sweep_summary.json`` to load.
        out_path: Destination PNG path.
        model_label: Model name shown in the chart title.
        gpu_label: GPU label shown in the chart title.
        tp: Tensor-parallel size used to normalise y-axis to tok/s/GPU.
            When 0 the raw output_throughput is used (tp treated as 1).
        isl: Input sequence length (informational, shown in title).
        osl: Output sequence length (informational, shown in title).
        draw_ceiling: When ``True`` and ``roofline_ceiling`` data is present,
            draw a dashed theoretical peak line. Off by default.

    Returns:
        The resolved ``Path`` of the written PNG, or ``None`` on any failure
        (missing matplotlib, bad data, IO error).
    """
    try:
        return _render(
            payload=payload,
            out_path=Path(out_path),
            model_label=model_label,
            gpu_label=gpu_label,
            tp=tp,
            isl=isl,
            osl=osl,
            draw_ceiling=draw_ceiling,
        )
    except Exception:  # noqa: BLE001
        log.debug("conc_sweep_plot: render failed", exc_info=True)
        return None


def _load_payload(payload: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    path = Path(payload)
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_series(
    points: list[dict[str, Any]],
    tp_eff: float,
    axes: _Axes,
) -> tuple[list[float], list[float]]:
    """Extract one arm's (x, y) series on *axes*, sorted ascending by x.

    Args:
        points: Conc-sweep point dicts.
        tp_eff: Effective TP size for y-axis normalisation (must be >= 1).
        axes: The pair to read.

    Returns:
        ``(xs, ys)``. A point missing either axis is dropped rather than
        plotted at zero.
    """
    pairs: list[tuple[float, float]] = []
    for pt in points:
        xy = axes.point_xy(pt, tp_eff)
        if xy is not None:
            pairs.append(xy)
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _ceiling_series(
    ceiling_data: dict[str, Any],
    tp_eff: float,
) -> tuple[list[float], list[float]]:
    """Extract ceiling (x, y) from ``roofline_ceiling`` payload rows.

    Args:
        ceiling_data: The ``roofline_ceiling`` sub-dict from the payload.
        tp_eff: Effective TP size for y-axis normalisation.

    Returns:
        ``(xs, ys)`` for the theoretical peak line, sorted ascending by x.
        Returns ``([], [])`` when data is missing or malformed.
    """
    rows = ceiling_data.get("rows") or []
    pairs: list[tuple[float, float]] = []
    for row in rows:
        conc = row.get("conc")
        peak = row.get("t_peak_tok_s")
        if conc is None or peak is None:
            continue
        try:
            conc_f = float(conc)
            peak_f = float(peak)
        except (TypeError, ValueError):
            continue
        if conc_f <= 0 or peak_f <= 0:
            continue
        x = peak_f / conc_f  # interactivity at ceiling
        y = peak_f / tp_eff  # efficiency at ceiling
        pairs.append((x, y))
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _render(
    payload: dict[str, Any] | str | Path,
    out_path: Path,
    *,
    model_label: str,
    gpu_label: str,
    tp: int,
    isl: int,
    osl: int,
    draw_ceiling: bool,
) -> Path | None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    data = _load_payload(payload)
    tp_eff = float(max(tp, 1))
    # Two independent facts the payload records separately: the axis the sweep
    # was graded on, which an operator can pin against the workload's default,
    # and whether the workload was an agentic replay at all.
    metric = _graded_metric_of(data)
    agentic = is_agentx_mode(data.get("benchmark_mode"))
    axes = _axes_for_metric(metric, tp_eff)

    baseline_pts = (data.get("baseline") or {}).get("points") or []
    optimized_pts = (data.get("optimized") or {}).get("points") or []

    bx, by = _arm_series(baseline_pts, tp_eff, axes)
    ox, oy = _arm_series(optimized_pts, tp_eff, axes)

    # Need at least the baseline to draw something useful.
    if not bx and not ox:
        log.debug("conc_sweep_plot: no valid data points in either arm — skipping plot")
        return None

    # Dark theme palette.
    bg = "#000000"
    fg = "#E6E6E6"
    grid_c = "#3A3A3A"
    baseline_c = "#FF4C4C"
    optimized_c = "#FF8C00"
    ceiling_c = "#888888"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # The roofline is a decode-only output_throughput bound computed from the
    # session's ISL/OSL, and ``_ceiling_series`` returns it in that axis pair's
    # units. It needs both: a workload whose ISL/OSL are real, and a chart whose
    # y axis is the throughput the bound is on.
    if draw_ceiling and not agentic and metric == GRADED_OUTPUT:
        ceiling_data = data.get("roofline_ceiling") or {}
        cx, cy = _ceiling_series(ceiling_data, tp_eff)
        if cx:
            ax.plot(cx, cy, "--", color=ceiling_c, linewidth=1.2, label="Theoretical peak (roofline)")

    if bx:
        ax.plot(bx, by, "o-", color=baseline_c, linewidth=2, markersize=6, label="baseline")
        for x, y in zip(bx, by):
            ax.annotate(
                f"{y:.0f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=baseline_c,
            )

    if ox:
        ax.plot(ox, oy, "s-", color=optimized_c, linewidth=2, markersize=6, label="optimized")
        for x, y in zip(ox, oy):
            ax.annotate(
                f"{y:.0f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
                color=optimized_c,
            )

    ax.set_xlabel(axes.x_label, fontsize=10, color=fg)
    ax.set_ylabel(axes.y_label, fontsize=10, color=fg)

    title_parts = [model_label or "Model"]
    if gpu_label:
        title_parts.append(gpu_label)
    # An agentic replay takes its request shapes from the trace corpus, so the
    # session's ISL/OSL are inert placeholders and naming them would misreport
    # what was measured.
    if agentic:
        title_parts.append("Agentic")
    elif isl or osl:
        title_parts.append(f"ISL={isl} OSL={osl}")
    ax.set_title(
        "Concurrency Sweep — Throughput vs Interactivity\n" + " | ".join(title_parts),
        fontsize=11,
        color=fg,
    )

    ax.grid(True, alpha=0.3, color=grid_c)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
    legend = ax.legend(loc="upper right", fontsize=9, facecolor=bg, edgecolor=grid_c)
    for text in legend.get_texts():
        text.set_color(fg)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, format="png", facecolor=bg)
    plt.close(fig)
    log.info("conc_sweep_plot: wrote %s", out_path)
    return out_path


__all__ = ["render_conc_sweep_curve"]
