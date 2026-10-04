# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data shapes for the external baseline comparison layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the query into a plain JSON-safe dict."""
        return {
            "model": self.model,
            "gpu": self.gpu,
            "framework": self.framework,
            "precision": self.precision,
            "isl": self.isl,
            "osl": self.osl,
            "benchmark_mode": self.benchmark_mode,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the data point into a plain JSON-safe dict."""
        return {
            "tput_per_gpu": self.tput_per_gpu,
            "output_tput_per_gpu": self.output_tput_per_gpu,
            "conc": self.conc,
            "decode_tp": self.decode_tp,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_tpot_ms": self.mean_tpot_ms,
            "mean_e2el_ms": self.mean_e2el_ms,
            "date": self.date,
            "benchmark_id": self.benchmark_id,
            "e2e_norm_intvty_p90": self.e2e_norm_intvty_p90,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary (and its nested points) to a JSON-safe dict."""
        return {
            "query": self.query.to_dict(),
            "fetched_at": self.fetched_at,
            "row_count": self.row_count,
            "best": self.best.to_dict() if self.best else None,
            "all_concurrencies": [p.to_dict() for p in self.all_concurrencies],
            "status": self.status,
            "reason": self.reason,
            "warning": self.warning,
            "source": self.source,
        }


__all__ = [
    "BenchmarkMode",
    "BaselineReason",
    "BaselineQuery",
    "BaselinePoint",
    "BaselineSummary",
]
