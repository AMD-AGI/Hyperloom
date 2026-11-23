# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's wiring of the power recorder, and the reader of what it wrote.

A benchmark report's ``gpu_monitor`` block samples one card over the whole process, so it describes
neither the serving cards nor the load being measured; :class:`hyperloom_common.gpu_telemetry.PowerRecorder` does.
This module names that sampling for an inference_suite round, which runs the recorder itself
(:func:`round_gpu_telemetry`), and reads the round's energy back as measurement fields (:func:`round_power_fields`)
for the power-budget checks and the search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from hyperloom_common.amd_smi import metric_value
from hyperloom_common.env import EnvValueError, env_float
from hyperloom_common.gpu_telemetry import DEFAULT_INTERVAL_SEC, fast_interval
from hyperloom.common.gpu_power_settings import gpu_power_sampling_unavailable
from inference_suite.contracts import READABLE_REPORT_SCHEMAS, GpuTelemetry, PowerSummary
from inference_suite.contracts.layout import POWER

log = logging.getLogger(__name__)

__all__ = [
    "POWER_FIELDS",
    "round_gpu_telemetry",
    "round_power_fields",
    "round_power_summary",
]

#: The measurement fields a round's power fills, each from the :class:`PowerSummary` figure it names (``None`` for
#: ``power_valid``, which is the summary's own). Scalars only: ``hosted_executor`` publishes a round's finite numbers
#: and drops dicts and ``None``, so an invalid round carries ``power_valid`` 0 and no energy.
POWER_FIELDS = {
    "gpu_power_avg_w": "avg_power_w",
    "gpu_power_p75_w": "p75_power_w",
    "gpu_power_p90_w": "p90_power_w",
    "gpu_energy_j": "total_gpu_energy_j",
    "joules_per_input_token": "joules_per_input_token",
    "joules_per_output_token": "joules_per_output_token",
    "joules_per_total_token": "joules_per_total_token",
    "joules_per_successful_query": "joules_per_successful_query",
}

_INTERVAL_ENV = "HYPERLOOM_GPU_POWER_INTERVAL_S"

#: How far a recorder's start may precede the subprocess it is paired with and still be that round's.
_START_SLACK_SEC = 5.0


def _interval_sec() -> float:
    """The operator's sampling interval, as :func:`fast_interval` takes it.

    Raises:
        EnvValueError: the variable is not a number, or :func:`fast_interval` refuses it.
    """
    requested = env_float(_INTERVAL_ENV, DEFAULT_INTERVAL_SEC)
    try:
        return fast_interval(requested)
    except ValueError as exc:
        raise EnvValueError(f"{_INTERVAL_ENV}: {exc}") from exc


def round_gpu_telemetry() -> GpuTelemetry:
    """The GPU sampling an inference_suite round runs: on where ``amd-smi`` can sample, at the operator's interval."""
    try:
        interval_sec = _interval_sec()
    except EnvValueError:
        log.warning("gpu_power: the sampling interval is unusable; this round samples nothing", exc_info=True)
        return GpuTelemetry(enabled=False, interval_sec=DEFAULT_INTERVAL_SEC)
    return GpuTelemetry(enabled=not gpu_power_sampling_unavailable(), interval_sec=interval_sec)


def _read_power_json(workspace: Path, subprocess_started_unix: float | None) -> dict[str, Any] | None:
    """This round's ``power.json``, beside the report or one level up; ``None`` when no recorder ran for it."""
    for directory in (workspace, workspace.parent):
        try:
            payload = json.loads((directory / POWER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        started = metric_value(payload.get("started_unix"))
        if subprocess_started_unix is not None and (
            started is None or started < float(subprocess_started_unix) - _START_SLACK_SEC
        ):
            continue
        return payload
    return None


def round_power_summary(
    report: Mapping[str, Any] | None, workspace: Path | None, *, subprocess_started_unix: float | None = None
) -> PowerSummary | None:
    """The round's power: an inference_suite report's ``power`` section, else the ``power.json`` a recorder left in
    ``workspace``; ``None`` when no recorder ran for the round."""
    if report and report.get("schema") in READABLE_REPORT_SCHEMAS:
        section = report.get("power")
        return None if section is None else PowerSummary.from_document(section)
    if workspace is None:
        return None
    document = _read_power_json(Path(workspace), subprocess_started_unix)
    return None if document is None else PowerSummary.of(POWER, document)


def round_power_fields(summary: PowerSummary) -> dict[str, Any]:
    """The measurement fields of a round's power.

    ``gpu_power_by_gpu_w`` is ``None`` when any card's energy is, so a per-card budget never passes a card it cannot
    see; every figure is ``None`` unless ``power_valid`` is 1.
    """
    return {
        **{field: summary.metrics[name] for field, name in POWER_FIELDS.items()},
        "gpu_power_by_gpu_w": None if summary.per_gpu_avg_power_w is None else dict(summary.per_gpu_avg_power_w),
        "power_valid": summary.power_valid,
    }
