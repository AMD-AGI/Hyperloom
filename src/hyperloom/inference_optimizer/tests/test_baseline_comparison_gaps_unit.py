# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage-gap unit tests for the external baseline comparison layer.

Covers uncovered branches in ``baseline_comparison``: whitespace-only input to
``to_inferencex_name``, the optional-field/all-concurrencies rendering in
``_format_report_md``, and the fail-soft branches in ``_row_to_point``.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
    _format_report_md,
    _row_to_point,
    to_inferencex_name,
)
from hyperloom.inference_optimizer.baseline_comparison.types import (
    BaselinePoint,
    BaselineQuery,
    BaselineSummary,
)


def test_to_inferencex_name_none_on_empty() -> None:
    assert to_inferencex_name("") is None


def test_to_inferencex_name_none_on_whitespace_only() -> None:
    """A string that is empty after ``strip()`` returns None (never raises)."""
    assert to_inferencex_name("   \t  ") is None


def test_to_inferencex_name_none_on_unknown_model() -> None:
    assert to_inferencex_name("some/unknown-model-xyz") is None


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


def test_format_report_md_renders_all_optional_fields() -> None:
    """A best with every optional latency field + all_concurrencies exercises the conditional render branches."""
    md = _format_report_md(_full_summary())
    assert "Reference best" in md
    assert "Output Throughput/GPU" in md
    assert "Mean TTFT" in md
    assert "Mean TPOT" in md
    assert "Mean E2E latency" in md
    assert "Reference run date" in md
    assert "All matched concurrencies" in md


def test_row_to_point_valid_row() -> None:
    point = _row_to_point(
        {
            "conc": 32,
            "decode_tp": 4,
            "metrics": {
                "tput_per_gpu": 10.5,
                "output_tput_per_gpu": 5.25,
                "mean_ttft": 0.1,  # seconds
                "mean_tpot": 0.02,
                "mean_e2el": 3.0,
            },
            "date": "2026-05-12",
        }
    )
    assert point is not None
    assert point.tput_per_gpu == 10.5
    assert point.conc == 32
    assert point.decode_tp == 4
    # Latencies converted seconds → milliseconds.
    assert round(point.mean_tpot_ms, 1) == 20.0
    assert round(point.mean_ttft_ms, 1) == 100.0


def test_row_to_point_unparseable_numeric_is_zero() -> None:
    """A non-numeric latency fails soft to 0.0; a valid positive tput still builds the point."""
    point = _row_to_point({"conc": 8, "metrics": {"tput_per_gpu": 7.0, "mean_tpot": "not-a-number"}})
    assert point is not None
    assert point.mean_tpot_ms == 0.0


def test_row_to_point_none_on_nonpositive_or_missing_metrics() -> None:
    """Non-positive throughput or a missing ``metrics`` block is dropped (returns None)."""
    assert _row_to_point({"metrics": {"tput_per_gpu": 0}}) is None
    assert _row_to_point({"metrics": {"tput_per_gpu": -3.2}}) is None
    assert _row_to_point({"metrics": {}}) is None
    assert _row_to_point({}) is None
