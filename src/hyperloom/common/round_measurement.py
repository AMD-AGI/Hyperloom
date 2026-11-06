# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's measurement of one inference_suite round, mapped from its report and launch.

This is the one place a round report is judged (D2): which round is a valid measurement, which is diagnostic only,
and which step a failed round stopped at. The measurement carries the keys a Magpie benchmark report normalises to,
plus every fixed metric key, so a reader does not need to know which runner measured the round.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom_common.coerce import to_float, to_int
from inference_suite.contracts import (
    FIXED_METRIC_KEYS,
    AgentxResult,
    Report,
    RunStatus,
    ScriptableResult,
    WorkloadKind,
)
from inference_suite.contracts.launch import Launch
from inference_suite.contracts.layout import REPORT_FILE

#: Where every latency figure of a round report comes from: the report itself, never a derivation.
LATENCY_SOURCE = REPORT_FILE

_LOAD_MEASURED = frozenset({RunStatus.COMPLETED, RunStatus.PARTIAL})
_MS_PER_SECOND = 1000.0
_COMPLETED_REQUESTS = "completed_requests"
_OUTPUT_THROUGHPUT = "output_throughput"
_REQUEST_THROUGHPUT = "request_throughput"
_DURATION_SECONDS = "duration_seconds"

#: Each AgentX throughput figure, by the measurement key it fills.
_AGENTX_THROUGHPUT = {
    _REQUEST_THROUGHPUT: _REQUEST_THROUGHPUT,
    _OUTPUT_THROUGHPUT: "output_tokens_per_second",
    "input_throughput": "input_tokens_per_second",
    "total_token_throughput": "total_tokens_per_second",
    _DURATION_SECONDS: _DURATION_SECONDS,
}

#: Each figure a scriptable script's result carries, by the measurement key it fills, with its coercion.
_SCRIPTABLE_FIELDS = (
    (_OUTPUT_THROUGHPUT, _OUTPUT_THROUGHPUT, to_float),
    (_REQUEST_THROUGHPUT, _REQUEST_THROUGHPUT, to_float),
    (_COMPLETED_REQUESTS, "completed", to_int),
    ("requested_requests", "num_prompts", to_int),
    (_DURATION_SECONDS, "duration", to_float),
    ("e2el_mean_ms", "mean_e2el_ms", to_float),
    ("latency_s", "latency_s", to_float),
)


def _agentx_metrics(result: AgentxResult) -> dict[str, Any]:
    metrics: dict[str, Any] = {key: result.throughput[raw] for key, raw in _AGENTX_THROUGHPUT.items()}
    metrics[_COMPLETED_REQUESTS] = result.counts["successful"]
    for key, seconds in result.latency_seconds.items():
        metrics[f"{key}_ms"] = None if seconds is None else seconds * _MS_PER_SECOND
    metrics.update({f"e2e_norm_intvty_{stat}": value for stat, value in result.e2e_norm_intvty.items()})
    metrics["agentx_counts"] = dict(result.counts)
    return metrics


def _scriptable_metrics(result: ScriptableResult) -> dict[str, Any]:
    document: Mapping[str, Any] = result.result
    metrics: dict[str, Any] = {key: coerce(document.get(raw)) for key, raw, coerce in _SCRIPTABLE_FIELDS}
    metrics["throughput_unit"] = document.get("throughput_unit")
    # The script writes its quality scores raw; the scriptable quality gate judges them.
    metrics["quality_gate"] = document.get("quality")
    return metrics


def _load_metrics(report: Report) -> dict[str, Any]:
    if report.fixed is not None:
        return dict(report.fixed.values)
    if report.agentx is not None:
        return _agentx_metrics(report.agentx)
    if report.scriptable is not None:
        return _scriptable_metrics(report.scriptable)
    return {}


def _round_facts(report: Report, launch: Launch | None) -> dict[str, Any]:
    return {
        "reported_success": report.run_status is RunStatus.COMPLETED,
        "run_status": report.run_status.value,
        "failed_step": None if report.failed_step is None else report.failed_step.value,
        "failure_detail": report.failure_detail,
        "framework": None if launch is None else launch.framework.name.value,
        "model": None if launch is None else launch.base.model.id,
        "tp": None if launch is None else launch.base.topology.tp,
        "ep": None if launch is None else launch.base.topology.ep,
        "conc": report.concurrency,
        "workload_kind": report.workload_kind.value,
        "workload_identity": report.workload_digest,
        "launch_digest": report.launch_digest,
        "throughput_unit": None,
        "quality_gate": None,
        "latency_s": None,
        **dict.fromkeys(FIXED_METRIC_KEYS),
        "raw_result_path": None,
        "ttft_e2el_source": LATENCY_SOURCE,
        "tpot_source": LATENCY_SOURCE,
        "nonfatal_warnings": [],
    }


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _load_valid(report: Report, measurement: Mapping[str, Any]) -> bool:
    """A round whose load ran and served: positive throughput, and a request served unless a script measured it."""
    if report.run_status not in _LOAD_MEASURED or not _positive(measurement[_OUTPUT_THROUGHPUT]):
        return False
    return report.scriptable is not None or _positive(measurement[_COMPLETED_REQUESTS])


def round_measurement(report: Report, launch: Launch | None) -> dict[str, Any]:
    """The measurement of the round ``report`` describes; ``launch`` is ``None`` for a round refused before one.

    A profiled round is diagnostic only: it is never a valid measurement. It, and a round that runs no load (boot
    only, or accuracy only), succeeds when it completes; a measuring round succeeds when it is valid.
    """
    measurement = {**_round_facts(report, launch), **_load_metrics(report)}
    diagnostic = report.profile is not None
    valid = _load_valid(report, measurement) and not diagnostic
    measurement["valid_measurement"] = valid
    measurement["diagnostic_only"] = diagnostic
    measures = report.workload_kind is not WorkloadKind.NONE and not diagnostic
    measurement["round_succeeded"] = valid if measures else report.run_status is RunStatus.COMPLETED
    return measurement


__all__ = ["LATENCY_SOURCE", "round_measurement"]
