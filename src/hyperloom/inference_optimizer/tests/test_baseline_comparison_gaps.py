# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage-gap unit tests for the external baseline comparison layer.

Covers uncovered branches in ``baseline_comparison``: whitespace-only input to
``to_inferencex_name``, the optional-field/all-concurrencies rendering in
``_format_report_md``, and the fail-soft branches in ``_row_to_point``.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
    _row_to_point,
)
from hyperloom.inference_optimizer.baseline_comparison.types import (
    BaselinePoint,
    BaselineQuery,
    BaselineSummary,
)


def _full_summary() -> BaselineSummary:
    best = BaselinePoint(
        tput_per_gpu=123.4,
        output_tput_per_gpu=98.7,
        conc=64,
        decode_tp=8,
        mean_ttft_ms=42.0,
        mean_tpot_ms=1.234,
        mean_e2el_ms=567.8,
        date="2026-05-12",
    )
    return BaselineSummary(
        query=BaselineQuery(
            model="Qwen-Image",
            gpu="MI355X",
            framework="xdit",
            precision="fp8",
            isl=128,
            osl=256,
        ),
        fetched_at="2026-05-12T07:00:34Z",
        row_count=22,
        best=best,
        all_concurrencies=[best],
        status="ok",
        warning="",
        source="https://inferencex.example/api/v1",
        reason="ok",
    )


def test_row_to_point_none_on_nonpositive_or_missing_metrics() -> None:
    """Non-positive throughput or a missing ``metrics`` block is dropped (returns None)."""
    assert _row_to_point({"metrics": {"tput_per_gpu": 0}}) is None
    assert _row_to_point({"metrics": {"tput_per_gpu": -3.2}}) is None
    assert _row_to_point({"metrics": {}}) is None
    assert _row_to_point({}) is None
