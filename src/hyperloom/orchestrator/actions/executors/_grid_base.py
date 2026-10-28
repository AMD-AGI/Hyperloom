# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared value types and helpers for the ``explore`` executor's grid runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key

log = logging.getLogger(__name__)


# Per-variant KEEP threshold (gain-pct + accuracy gate); the grid noise floor.
DEFAULT_KEEP_THRESHOLD_PCT = 1.0

# Terminal outcomes that ended a grid variant without a graded comparison, so
# they are neither a KEEP nor a REVERT.
TS_FAILED = "FAILED"
TS_SKIPPED_DEDUP = "SKIPPED_DEDUP"


@dataclass
class GridVariant:
    """One row of the grid we're going to test."""

    name: str
    extra_server_args: str = ""
    extra_envs: dict[str, str] | None = None
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = field(default=None, kw_only=True)
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = field(default=None, kw_only=True)
    args_mode: str = field(default="append", kw_only=True)
    note: str = ""

    def __post_init__(self) -> None:
        """Drop unsafe envs and normalize the removal and args-mode controls."""
        self.extra_envs, dropped_envs = filter_untrusted_env_mapping(
            self.extra_envs,
            allow_predicate=is_allowed_variant_env_key,
        )
        if dropped_envs:
            log.warning("Variant %s: dropping unsafe extra_envs %s", self.name, ", ".join(sorted(dropped_envs)))
        self.remove_args = to_str_list(self.remove_args)
        self.unset_envs = to_str_list(self.unset_envs)
        mode = str(self.args_mode or "append").strip().lower()
        self.args_mode = mode if mode in {"append", "replace"} else "append"
        # Optional runtime override; injected into materialized YAML benchmark.envs by _build_variant_yaml so the
        # server subprocess resolves the attempt runtime.
        self.runtime_override: dict[str, str] = {}
        self.source_patches: list[dict[str, str]] = []
        self.overlay_pythonpath: str = ""


@dataclass
class VariantResult:
    """One bench run's parsed result."""

    name: str
    extra_server_args: str
    extra_envs: dict[str, str]
    status: str
    benchmark_backend: str | None = None
    benchmark_valid: bool | None = None
    diagnostic_only: bool | None = None
    workload_identity: str | None = None
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    gpu_power_avg_w: float | None = None
    gpu_power_by_gpu_w: dict[str, float] | None = None
    tpot_mean_ms: float | None = None
    input_throughput: float | None = None
    tpot_p90_ms: float | None = None
    intvty_p90: float | None = None
    intvty_p50: float | None = None
    request_error_rate: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Short failure-classification tag; empty for successes.
    error_class: str = ""
    note: str = ""
    # Wall-clock seconds the Magpie subprocess consumed.
    runtime_sec: float | None = None
    # True iff reaped by the overtime soft deadline.
    killed_overtime: bool = False
    # Rough output tok/s salvaged from server.log on the killed_overtime path; informational only, never feeds winner
    # selection.
    estimated_output_throughput: float | None = None
    server_log_path: str | None = None
    launch_evidence: dict[str, Any] = field(default_factory=dict)
    launch_evidence_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain JSON-friendly dict."""
        return {
            "benchmark_backend": self.benchmark_backend,
            "benchmark_valid": self.benchmark_valid,
            "diagnostic_only": self.diagnostic_only,
            "workload_identity": self.workload_identity,
            "name": self.name,
            "extra_server_args": self.extra_server_args,
            "extra_envs": self.extra_envs,
            "status": self.status,
            "output_throughput": self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds": self.duration_seconds,
            "ttft_mean_ms": self.ttft_mean_ms,
            "e2el_mean_ms": self.e2el_mean_ms,
            "gpu_power_avg_w": self.gpu_power_avg_w,
            "gpu_power_by_gpu_w": self.gpu_power_by_gpu_w,
            "tpot_mean_ms": self.tpot_mean_ms,
            "input_throughput": self.input_throughput,
            "tpot_p90_ms": self.tpot_p90_ms,
            "e2e_norm_intvty_p90": self.intvty_p90,
            "e2e_norm_intvty_p50": self.intvty_p50,
            "request_error_rate": self.request_error_rate,
            "workspace": self.workspace,
            "report_path": self.report_path,
            "raw_result_path": self.raw_result_path,
            "reported_success": self.reported_success,
            "returncode": self.returncode,
            "nonfatal_warnings": self.nonfatal_warnings,
            "error": self.error,
            "error_class": self.error_class,
            "note": self.note,
            "runtime_sec": self.runtime_sec,
            "killed_overtime": self.killed_overtime,
            "estimated_output_throughput": self.estimated_output_throughput,
            "server_log_path": self.server_log_path,
            "launch_evidence": self.launch_evidence,
            "launch_evidence_path": self.launch_evidence_path,
        }
