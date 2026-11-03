# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data shapes for the external baseline comparison layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BenchmarkMode = Literal["synthetic", "agentx"]
BaselineReason = Literal[
    "",
    "ok",
    "model_mapping_miss",
    "no_target_gpu_configured",
    "fetch_error",
    "no_inferencex_data",
    "unsupported_target_gpu",
    "precision_mismatch",
    "dimension_mismatch",
    "no_valid_rows",
]


@dataclass(frozen=True)
class TargetAnalysisRequest:
    """The session's workload as configured; analysis maps it onto an InferenceX :class:`BaselineQuery`.

    ``compare_against_gpu`` empty skips the lookup; ``framework`` is query metadata, not a row filter;
    ``isl`` / ``osl`` are ignored in agentx mode.
    """

    model_path: str
    compare_against_gpu: str
    framework: str = ""
    precision: str = ""
    isl: int | None = 0
    osl: int | None = 0
    benchmark_mode: BenchmarkMode = "synthetic"


@dataclass
class BaselineQuery:
    """Fully-resolved query against the InferenceX upstream."""

    model: str
    gpu: str
    framework: str = ""
    precision: str = ""
    isl: int | None = 0
    osl: int | None = 0
    benchmark_mode: BenchmarkMode = "synthetic"


@dataclass
class BaselinePoint:
    """One reference data point pulled out of the upstream rows."""

    tput_per_gpu: float
    output_tput_per_gpu: float
    conc: int
    decode_tp: int
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_e2el_ms: float
    date: str = ""
    benchmark_id: str | None = None
    e2e_norm_intvty_p90: float | None = None


@dataclass
class BaselineSummary:
    """The full target-analysis artefact persisted under the session dir."""

    query: BaselineQuery
    fetched_at: str
    row_count: int
    best: BaselinePoint | None
    all_concurrencies: list[BaselinePoint] = field(default_factory=list)
    status: str = "ok"
    warning: str = ""
    source: str = ""
    reason: BaselineReason = ""


__all__ = [
    "BenchmarkMode",
    "BaselineReason",
    "BaselineQuery",
    "BaselinePoint",
    "BaselineSummary",
    "TargetAnalysisRequest",
]
