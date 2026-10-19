# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX-style concurrency sweep comparison chart."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

from hyperloom.common.perf_metric import GRADED_OUTPUT, is_agentx_mode

log = logging.getLogger(__name__)

# Dark theme palette.
_BG = "#000000"
_FG = "#E6E6E6"
_GRID_C = "#3A3A3A"
_BASELINE_C = "#FF4C4C"
_OPTIMIZED_C = "#FF8C00"
_CEILING_C = "#888888"


def _positive(value: Any) -> float | None:
    """Coerce to a strictly positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _agentx_xy(point: Mapping[str, Any], tp_eff: float) -> tuple[float, float] | None:
    """p90 interactivity against token throughput per chip."""
    intvty = _positive(point.get("e2e_norm_intvty_p90"))
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
    """The axis this sweep was graded on, as the sweep recorded it."""
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


@dataclass(frozen=True)
class ConcCurveOptions:
    """Labels and workload shape a sweep curve is drawn for.

    Attributes:
        model_label: Model name in the chart title.
        gpu_label: GPU name in the chart title.
        tp: Tensor-parallel size the per-chip axes divide by.
        isl: Input sequence length the roofline ceiling is computed for.
        osl: Output sequence length the roofline ceiling is computed for.
        draw_ceiling: Whether to draw the roofline ceiling.
    """

    model_label: str = ""
    gpu_label: str = ""
    tp: int = 1
    isl: int = 0
    osl: int = 0
    draw_ceiling: bool = False


def render_conc_sweep_curve(
    payload: dict[str, Any] | str | Path,
    out_path: str | Path,
    options: ConcCurveOptions = ConcCurveOptions(),
) -> Path | None:
    """Render a throughput-vs-interactivity PNG from a ``conc_sweep_summary.json``."""
    try:
        return _render(payload, Path(out_path), options)
    except Exception:
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
) -> tuple[list[float], list[float], list[float | None]]:
    """Extract one arm's (x, y, conc) series on *axes*, in concurrency order.

    A ladder is walked by concurrency, and interactivity need not fall as it rises, so sorting by x would join the
    points in an order nobody ran them in. A point with no recorded concurrency sorts last, by x.
    """
    rows: list[tuple[float | None, float, float]] = []
    for pt in points:
        xy = axes.point_xy(pt, tp_eff)
        if xy is not None:
            rows.append((_positive(pt.get("conc")), xy[0], xy[1]))
    rows.sort(key=lambda r: (r[0] is None, r[0] or 0.0, r[1]))
    return [r[1] for r in rows], [r[2] for r in rows], [r[0] for r in rows]


def _conc_text(conc: float | None) -> str:
    return "" if conc is None else f"c={conc:g}"


def _unpaired(concs: list[float | None], other: list[float | None]) -> list[bool]:
    """Which of *concs* the other arm has no plotted point for; nothing is unpaired against an empty arm."""
    if not other:
        return [False] * len(concs)
    return [c is not None and c not in other for c in concs]


@dataclass(frozen=True)
class _Arm:
    """One arm's series and how it is drawn."""

    name: str
    xs: list[float]
    ys: list[float]
    concs: list[float | None]
    color: str
    marker: str
    label_dy: int


def _draw_arm(ax: Any, arm: _Arm, other_concs: list[float | None]) -> bool:
    """Draw one arm, labelled by concurrency; True when some rung is hollow because the other arm lacks it."""
    xs, ys, concs, color, marker = arm.xs, arm.ys, arm.concs, arm.color, arm.marker
    if not xs:
        return False
    ax.plot(xs, ys, f"{marker}-", color=color, linewidth=2, markersize=6, label=arm.name)
    single = _unpaired(concs, other_concs)
    if any(single):
        # Hollow: the same concurrency has no point on the other arm, so there is nothing to compare it with.
        ax.plot(
            [x for x, s in zip(xs, single) if s],
            [y for y, s in zip(ys, single) if s],
            marker,
            markersize=6,
            markerfacecolor=_BG,
            markeredgecolor=color,
            markeredgewidth=1.5,
            linestyle="none",
        )
    for x, y, conc in zip(xs, ys, concs):
        ax.annotate(
            f"{_conc_text(conc)} {y:.0f}".strip(),
            (x, y),
            textcoords="offset points",
            xytext=(0, arm.label_dy),
            ha="center",
            fontsize=7,
            color=color,
        )
    return any(single)


def _ceiling_series(
    ceiling_data: dict[str, Any],
    tp_eff: float,
) -> tuple[list[float], list[float]]:
    """Extract ceiling (x, y) from ``roofline_ceiling`` payload rows."""
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


class _SweepSeries(NamedTuple):
    """Both arms of one sweep, projected onto the axes it was graded on."""

    baseline: _Arm
    optimized: _Arm
    axes: _Axes
    metric: str
    agentic: bool

    def shows_roofline(self) -> bool:
        """Whether a roofline ceiling describes what this sweep measured."""
        return not self.agentic and self.metric == GRADED_OUTPUT


def _sweep_series(data: Mapping[str, Any], tp_eff: float) -> _SweepSeries | None:
    """Both arms on the graded axes, ``None`` when neither arm has a point worth drawing."""
    # Two independent facts the payload records separately: the axis the sweep was graded on, which an operator can
    # pin against the workload's default, and whether the workload was an agentic replay at all.
    metric = _graded_metric_of(data)
    axes = _axes_for_metric(metric, tp_eff)
    bx, by, bc = _arm_series((data.get("baseline") or {}).get("points") or [], tp_eff, axes)
    ox, oy, oc = _arm_series((data.get("optimized") or {}).get("points") or [], tp_eff, axes)
    # Need at least the baseline to draw something useful.
    if not bx and not ox:
        return None
    return _SweepSeries(
        baseline=_Arm("baseline", bx, by, bc, _BASELINE_C, "o", 8),
        optimized=_Arm("optimized", ox, oy, oc, _OPTIMIZED_C, "s", -14),
        axes=axes,
        metric=metric,
        agentic=is_agentx_mode(data.get("benchmark_mode")),
    )


def _draw_roofline_ceiling(ax: Any, data: Mapping[str, Any], tp_eff: float) -> None:
    """Draw the decode-only output_throughput bound, in the graded axis pair's units."""
    cx, cy = _ceiling_series(data.get("roofline_ceiling") or {}, tp_eff)
    if cx:
        ax.plot(cx, cy, "--", color=_CEILING_C, linewidth=1.2, label="Theoretical peak (roofline)")


def _draw_arms(ax: Any, series: _SweepSeries) -> None:
    """Draw both arms, and the legend entry for rungs only one of them measured."""
    hollow = False
    for arm, other_concs in (
        (series.baseline, series.optimized.concs),
        (series.optimized, series.baseline.concs),
    ):
        hollow = _draw_arm(ax, arm, other_concs) or hollow
    if not hollow:
        return
    ax.plot(
        [],
        [],
        "o",
        markerfacecolor=_BG,
        markeredgecolor=_FG,
        markeredgewidth=1.5,
        linestyle="none",
        label="hollow: concurrency measured by one arm only",
    )


def _chart_title(options: ConcCurveOptions, agentic: bool) -> str:
    """The chart's title, naming the request shape only when the sweep fixed one."""
    parts = [options.model_label or "Model"]
    if options.gpu_label:
        parts.append(options.gpu_label)
    # An agentic replay takes its request shapes from the trace corpus, so the session's ISL/OSL are inert
    # placeholders and naming them would misreport what was measured.
    if agentic:
        parts.append("Agentic")
    elif options.isl or options.osl:
        parts.append(f"ISL={options.isl} OSL={options.osl}")
    return "Concurrency Sweep — Throughput vs Interactivity\n" + " | ".join(parts)


def _apply_dark_theme(fig: Any, ax: Any, axes: _Axes, title: str) -> None:
    """Label the figure and put its axes, grid and legend in the dark palette."""
    ax.set_xlabel(axes.x_label, fontsize=10, color=_FG)
    ax.set_ylabel(axes.y_label, fontsize=10, color=_FG)
    ax.set_title(title, fontsize=11, color=_FG)
    ax.grid(True, alpha=0.3, color=_GRID_C)
    ax.tick_params(colors=_FG)
    for spine in ax.spines.values():
        spine.set_color(_GRID_C)
    legend = ax.legend(loc="upper right", fontsize=9, facecolor=_BG, edgecolor=_GRID_C)
    for text in legend.get_texts():
        text.set_color(_FG)
    fig.tight_layout()


def _render(payload: dict[str, Any] | str | Path, out_path: Path, options: ConcCurveOptions) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_payload(payload)
    tp_eff = float(max(options.tp, 1))
    series = _sweep_series(data, tp_eff)
    if series is None:
        log.debug("conc_sweep_plot: no valid data points in either arm — skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    if options.draw_ceiling and series.shows_roofline():
        _draw_roofline_ceiling(ax, data, tp_eff)
    _draw_arms(ax, series)
    _apply_dark_theme(fig, ax, series.axes, _chart_title(options, series.agentic))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, format="png", facecolor=_BG)
    plt.close(fig)
    log.info("conc_sweep_plot: wrote %s", out_path)
    return out_path


__all__ = ["ConcCurveOptions", "render_conc_sweep_curve"]
