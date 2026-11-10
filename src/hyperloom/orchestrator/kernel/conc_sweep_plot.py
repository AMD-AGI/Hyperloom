# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX-style concurrency sweep comparison chart."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from hyperloom_common.coerce import optional_positive_float
from hyperloom.common.perf_metric import GRADED_OUTPUT, is_agentx_mode

log = logging.getLogger(__name__)


def _agentx_xy(point: Mapping[str, Any], tp_eff: float) -> tuple[float, float] | None:
    """p90 interactivity against token throughput per chip."""
    intvty = optional_positive_float(point.get("e2e_norm_intvty_p90"))
    total = optional_positive_float(point.get("total_token_throughput"))
    if intvty is None or total is None:
        return None
    return intvty, total / tp_eff


def _synthetic_xy(point: Mapping[str, Any], tp_eff: float) -> tuple[float, float] | None:
    """Output throughput per user against output throughput per GPU."""
    tput = optional_positive_float(point.get("output_throughput"))
    conc = optional_positive_float(point.get("conc"))
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
        isl: Input sequence length in the chart title.
        osl: Output sequence length in the chart title.
    """

    model_label: str = ""
    gpu_label: str = ""
    tp: int = 1
    isl: int = 0
    osl: int = 0


def render_conc_sweep_curve(
    payload: dict[str, Any],
    out_path: str | Path,
    options: ConcCurveOptions = ConcCurveOptions(),
) -> Path | None:
    """Render a throughput-vs-interactivity PNG from a ``conc_sweep_summary.json``."""
    try:
        return _render(payload, Path(out_path), options)
    except (ImportError, OSError):
        log.debug("conc_sweep_plot: render failed", exc_info=True)
        return None


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
            rows.append((optional_positive_float(pt.get("conc")), xy[0], xy[1]))
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


def _draw_arm(ax: Any, arm: _Arm, other_concs: list[float | None], bg: str) -> bool:
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
            markerfacecolor=bg,
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


def _chart_title(options: ConcCurveOptions, agentic: bool) -> str:
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


def _render(data: dict[str, Any], out_path: Path, options: ConcCurveOptions) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp_eff = float(max(options.tp, 1))
    # Two independent facts the payload records separately: the axis the sweep was graded on, which an operator can
    # pin against the workload's default, and whether the workload was an agentic replay at all.
    metric = _graded_metric_of(data)
    agentic = is_agentx_mode(data.get("benchmark_mode"))
    axes = _axes_for_metric(metric, tp_eff)

    bx, by, bc = _arm_series((data.get("baseline") or {}).get("points") or [], tp_eff, axes)
    ox, oy, oc = _arm_series((data.get("optimized") or {}).get("points") or [], tp_eff, axes)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    single_arm_note = False
    for arm, other_concs in (
        (_Arm("baseline", bx, by, bc, baseline_c, "o", 8), oc),
        (_Arm("optimized", ox, oy, oc, optimized_c, "s", -14), bc),
    ):
        single_arm_note = _draw_arm(ax, arm, other_concs, bg) or single_arm_note
    if single_arm_note:
        ax.plot(
            [],
            [],
            "o",
            markerfacecolor=bg,
            markeredgecolor=fg,
            markeredgewidth=1.5,
            linestyle="none",
            label="hollow: concurrency measured by one arm only",
        )

    ax.set_xlabel(axes.x_label, fontsize=10, color=fg)
    ax.set_ylabel(axes.y_label, fontsize=10, color=fg)

    ax.set_title(_chart_title(options, agentic), fontsize=11, color=fg)

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


__all__ = ["ConcCurveOptions", "render_conc_sweep_curve"]
