# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage-gap unit tests for the external baseline comparison layer.

Closes the residual uncovered branches in ``baseline_comparison`` that the
main suite leaves out:

* ``name_mapping.to_inferencex_name`` — whitespace-only input -> ``None``.
* ``types.BaselineSummary.from_dict`` — full round-trip reconstruction.
* ``target_analyzer._format_report_md`` — the optional-latency-field and
  all-concurrencies rendering branches (only reached when ``best`` carries the
  optional metrics).
* ``target_analyzer._target_row_to_point`` — unparseable numeric field
  (fail-soft to ``0.0``) and non-positive throughput (row dropped -> ``None``).

These are report-only paths (no SharedState / scoring coupling), so they are
pure-function tests with no fixtures.
"""

from __future__ import annotations

from inference_optimizer.baseline_comparison.name_mapping import to_inferencex_name
from inference_optimizer.baseline_comparison.target_analyzer import (
    _format_report_md,
    _target_row_to_point,
)
from inference_optimizer.baseline_comparison.types import (
    BaselinePoint,
    BaselineQuery,
    BaselineSummary,
)


# ---------------------------------------------------------------------------
# name_mapping.to_inferencex_name
# ---------------------------------------------------------------------------
def test_to_inferencex_name_none_on_empty() -> None:
    assert to_inferencex_name("") is None


def test_to_inferencex_name_none_on_whitespace_only() -> None:
    """A string that is empty after ``strip()`` returns None (never raises)."""
    assert to_inferencex_name("   \t  ") is None


def test_to_inferencex_name_none_on_unknown_model() -> None:
    assert to_inferencex_name("some/unknown-model-xyz") is None


# ---------------------------------------------------------------------------
# types.BaselineSummary round-trip
# ---------------------------------------------------------------------------
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


def test_baseline_summary_from_dict_roundtrip() -> None:
    """``from_dict(to_dict(x))`` reconstructs an equal summary (nested points)."""
    original = _full_summary()
    restored = BaselineSummary.from_dict(original.to_dict())
    assert restored == original


def test_baseline_summary_from_dict_tolerates_partial() -> None:
    """A partial / older artefact loads with defaults and no ``best``."""
    restored = BaselineSummary.from_dict({"query": {"model": "m", "gpu": "g"}})
    assert restored.best is None
    assert restored.row_count == 0
    assert restored.status == "ok"
    assert restored.all_concurrencies == []


# ---------------------------------------------------------------------------
# target_analyzer._format_report_md optional-field branches
# ---------------------------------------------------------------------------
def test_format_report_md_renders_all_optional_fields() -> None:
    """A best with every optional latency field + all_concurrencies exercises
    the conditional render branches."""
    md = _format_report_md(_full_summary())
    assert "Reference best" in md
    assert "Output Throughput/GPU" in md
    assert "Mean TTFT" in md
    assert "Mean TPOT" in md
    assert "Mean E2E latency" in md
    assert "Reference run date" in md
    assert "All matched concurrencies" in md


# ---------------------------------------------------------------------------
# target_analyzer._target_row_to_point fail-soft branches
# ---------------------------------------------------------------------------
def test_target_row_to_point_valid_row() -> None:
    point = _target_row_to_point({"tput_per_gpu": "10.5", "conc": "32", "tpot_ms": "1.5"})
    assert point is not None
    assert point.tput_per_gpu == 10.5
    assert point.conc == 32


def test_target_row_to_point_unparseable_numeric_is_zero() -> None:
    """A non-numeric field fails soft to 0.0 (does not raise); with a valid
    positive tput the point is still built."""
    point = _target_row_to_point({"tput_per_gpu": "7.0", "tpot_ms": "not-a-number"})
    assert point is not None
    assert point.mean_tpot_ms == 0.0


def test_target_row_to_point_none_on_nonpositive_tput() -> None:
    """A row with non-positive throughput is dropped (returns None)."""
    assert _target_row_to_point({"tput_per_gpu": "0"}) is None
    assert _target_row_to_point({"tput_per_gpu": "-3.2"}) is None
    assert _target_row_to_point({}) is None
