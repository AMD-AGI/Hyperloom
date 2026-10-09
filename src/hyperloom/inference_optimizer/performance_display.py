# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Display throughput without changing the measurements used for grading."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from hyperloom.common.agentx_mode import native_agentx_session

from . import framework_registry


def throughput_fields(
    value: float | None,
    framework: str | None,
    *,
    native_agentx: bool = False,
    gpu_count: Any = None,
) -> dict[str, Any]:
    """Project a measurement into display fields, using only its recorded topology.

    Native AgentX reports aggregate output throughput. TP and EP can overlap,
    so neither the process environment nor their product is a GPU count. An
    absent count leaves the per-GPU field null and explicitly labels the total.
    Legacy serving and scriptable measurements retain their existing units.
    """
    fields = {
        "throughput_tok_s_per_gpu": value,
        "throughput_unit": framework_registry.throughput_unit(framework),
    }
    if not native_agentx:
        return fields
    known_count = (
        isinstance(gpu_count, (int, float))
        and not isinstance(gpu_count, bool)
        and math.isfinite(gpu_count)
        and gpu_count > 0
        and float(gpu_count).is_integer()
    )
    fields.update(
        throughput_tok_s=value,
        throughput_tok_s_per_gpu=(value / gpu_count if value is not None else None) if known_count else None,
        throughput_unit="tok/s/GPU" if known_count else "tok/s (aggregate)",
    )
    return fields


def recorded_throughput(measurement: Mapping[str, Any]) -> float | None:
    """Read a display measurement, preferring the measured per-GPU figure."""
    value = measurement.get("throughput_tok_s_per_gpu")
    return measurement.get("throughput_tok_s") if value is None else value


def recorded_metric_unit(framework: str | None, measurement: Mapping[str, Any]) -> str:
    """The primary display unit, including native totals without known topology."""
    if measurement.get("throughput_unit") == "tok/s (aggregate)":
        return "tok/s (aggregate)"
    return framework_registry.primary_metric_unit(framework)


def format_recorded_metric(framework: str | None, measurement: Mapping[str, Any], *, precision: int = 1) -> str:
    """Format display fields without dividing a per-GPU value a second time."""
    value = recorded_throughput(measurement)
    if measurement.get("throughput_unit") == "tok/s (aggregate)":
        number = "n/a" if value is None else f"{value:.{precision}f}"
        return f"{number} tok/s (aggregate)"
    return framework_registry.format_primary_metric(framework, value, precision=precision)


def format_measurement_metric(framework: str | None, measurement: Mapping[str, Any], *, precision: int = 1) -> str:
    """Format an executor result; its native protocol marker identifies totals."""
    fields = throughput_fields(
        measurement.get("output_throughput"),
        framework,
        native_agentx=measurement.get("native_agentx_report") is True,
        gpu_count=measurement.get("agentx_gpu_count"),
    )
    return format_recorded_metric(framework, fields, precision=precision)


def format_session_metric(
    state: Any,
    value: float | None,
    *,
    measurement: Mapping[str, Any] | None = None,
    precision: int = 1,
) -> str:
    """Format a state or saved report's raw baseline/current-best throughput."""
    read = state.get if isinstance(state, Mapping) else lambda name, default=None: getattr(state, name, default)
    framework = read("framework", "")
    perf = measurement if measurement is not None else read("baseline_perf", None) or {}
    fields = throughput_fields(
        value,
        framework,
        native_agentx=read("native_agentx", False) is True or native_agentx_session(state),
        gpu_count=perf.get("agentx_gpu_count"),
    )
    return format_recorded_metric(framework, fields, precision=precision)
