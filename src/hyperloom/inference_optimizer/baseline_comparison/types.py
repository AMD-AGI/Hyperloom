# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data shapes for the external baseline comparison layer.

These types live in their own module so the HTTP client and the
orchestration executor can import them without pulling in each other's
dependencies, tests can construct ``BaselineSummary`` directly, and the
on-disk JSON shape is pinned by ``BaselineSummary.to_dict``.

``BaselineSummary`` itself is report-only: only ``ReportExecutor`` reads the
on-disk JSON to render the "External baseline (advisory)" section in
``final.md``. The matched measured points are additionally derived into
``competitor_target.json`` (``research_hints.write_competitor_target``), which
feeds the gap advisory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaselineQuery:
    """Fully-resolved query against the InferenceX upstream.

    Merges the user-supplied ``--compare-against-gpu`` with process-derived
    fields (model display name, framework, precision, isl, osl). Persisted
    to disk so the report can show exactly what was queried.
    """

    model: str
    gpu: str
    framework: str = ""
    precision: str = ""
    isl: int = 0
    osl: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the query into a plain JSON-safe dict.

        Returns:
            dict[str, Any]: The query fields (model, gpu, framework,
                precision, isl, osl) as a flat dictionary.
        """
        return {
            "model": self.model,
            "gpu": self.gpu,
            "framework": self.framework,
            "precision": self.precision,
            "isl": self.isl,
            "osl": self.osl,
        }


@dataclass
class BaselinePoint:
    """One reference data point pulled out of the upstream rows.

    Kept minimal on purpose: only the fields the report needs.
    """

    tput_per_gpu: float
    output_tput_per_gpu: float
    conc: int
    decode_tp: int
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_e2el_ms: float
    date: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the data point into a plain JSON-safe dict.

        Returns:
            dict[str, Any]: The point's throughput, concurrency, latency,
                and date fields as a flat dictionary.
        """
        return {
            "tput_per_gpu": self.tput_per_gpu,
            "output_tput_per_gpu": self.output_tput_per_gpu,
            "conc": self.conc,
            "decode_tp": self.decode_tp,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_tpot_ms": self.mean_tpot_ms,
            "mean_e2el_ms": self.mean_e2el_ms,
            "date": self.date,
        }


@dataclass
class BaselineSummary:
    """The full target-analysis artefact persisted under the session dir.

    Shape (schematic, not literal JSON):

    .. code-block:: text

        {
          "query":         {model, gpu, framework, precision, isl, osl},
          "fetched_at":    "2026-05-12T07:00:34Z",
          "row_count":     22,
          "best":          {tput_per_gpu, conc, decode_tp, ...} | null,
          "all_concurrencies": [{conc, tput_per_gpu, decode_tp, ...}],
          "status":        "ok" | "no_match" | "skipped",
          "reason":        "ok" | "model_mapping_miss" |
                           "no_target_gpu_configured" | "unsupported_target_gpu" |
                           "dimension_mismatch" | "precision_mismatch" |
                           "no_inferencex_data" | "fetch_error" | "no_valid_rows",
          "warning":       "<human-readable note; on status=ok this carries the
                           matched rows' reference dates, empty if none>",
          "source":        "https://inferencex.semianalysis.com/api/v1"
        }

    The ``status`` field is the single source of truth: ``ok`` means
    ``best`` is populated and the report should render it; any other value
    means the report can show a one-line note and move on.

    The ``reason`` field is the structured machine-readable counterpart to
    ``warning``: callers should branch on it instead of regex-matching the
    human-readable warning string.
    """

    query: BaselineQuery
    fetched_at: str
    row_count: int
    best: BaselinePoint | None
    all_concurrencies: list[BaselinePoint] = field(default_factory=list)
    status: str = "ok"
    warning: str = ""
    source: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary (and its nested points) to a JSON-safe dict.

        Returns:
            dict[str, Any]: The full on-disk artefact shape, with the
                nested query and baseline points recursively serialised.
        """
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
    "BaselineQuery",
    "BaselinePoint",
    "BaselineSummary",
]
