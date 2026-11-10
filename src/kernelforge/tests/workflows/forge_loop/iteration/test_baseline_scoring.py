# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoring against the baseline: how a broken baseline is diagnosed, and how per-case coverage gates a score."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.workflows.forge_loop.iteration.runner import IterationConfig, IterationLoop
from kernelforge.tests.conftest import patch_iteration_name
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _make_loop


def _static_bench(result: dict):
    """A measurement stand-in that always returns ``result``."""

    async def bench(_run, **_kwargs):
        return dict(result)

    return bench


def test_baseline_crash_reports_the_driver_failure_not_the_output_format(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A crashed driver must be diagnosed by its own exit code and output."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    patch_iteration_name(
        monkeypatch,
        "measure_wallclock",
        _static_bench(
            {
                "success": False,
                "message": "BENCH CRASHED (exit 1)",
                "output": (
                    "Traceback (most recent call last):\n"
                    "FileNotFoundError: could not locate the invocation "
                    "specification JSON\n"
                ),
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "BENCH CRASHED (exit 1)" in printed
    assert "could not locate the invocation specification JSON" in printed
    assert "case_ms" not in printed


def test_baseline_timeout_reports_the_timeout(tmp_path, monkeypatch, capsys):
    """A timed-out driver reports no output tail, only its verdict."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    patch_iteration_name(
        monkeypatch,
        "measure_wallclock",
        _static_bench({"success": False, "message": "TIMEOUT after 300s"}),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "TIMEOUT after 300s" in printed
    assert "case_ms" not in printed


def test_baseline_without_case_lines_names_the_missing_contract(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A driver that ran but reported no per-case timings is a distinct fault."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    patch_iteration_name(
        monkeypatch,
        "measure_wallclock",
        _static_bench(
            {
                "success": True,
                "median_ms": 5.5635,
                "message": "BENCH: mean=5.5635 ms",
                "case_times": {},
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "case_ms" in printed
    assert "BENCH: mean=5.5635 ms" in printed
    assert "CRASHED" not in printed


def test_baseline_without_aggregate_line_is_not_silent(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A missing aggregate must be reported, not returned as a bare ``None``."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    patch_iteration_name(
        monkeypatch,
        "measure_wallclock",
        _static_bench(
            {
                "success": True,
                "median_ms": None,
                "message": "BENCH: cases only",
                "case_times": {"case": 1.0},
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "Baseline bench FAILED" in printed
    assert "median_ms" in printed


def test_baseline_case_coverage_drift_names_the_cases(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Coverage that moves between measurements must name what moved."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    async def bench(_run, **_kwargs):
        return {
            "success": False,
            "message": ("MEASUREMENT CASE COVERAGE MISMATCH: expected=['case_001', 'case_002'], got=['case_001']"),
        }

    patch_iteration_name(monkeypatch, "measure_wallclock", bench)

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "CASE COVERAGE MISMATCH" in printed
    assert "case_002" in printed


def test_case_metric_fails_closed_on_incomplete_candidate_coverage():
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            baseline_wall_ms=10.0,
        ),
        tracker=object(),
        config=object(),
    )
    loop._baseline_case_times = {"small": 2.0, "large": 8.0}
    bench = {
        "success": True,
        "median_ms": 7.0,
        "case_times": {"small": 1.0},
        "measurements": [
            {
                "success": True,
                "case_times": {"small": 1.0},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
    }

    loop._apply_mean_case_speedup_metric(bench)

    assert bench["success"] is False
    assert bench["mean_case_speedup"] is None
    assert bench["case_coverage_complete"] is False
    assert "large" in bench["message"]


def test_mean_case_speedup_metric_preserves_raw_mean_for_diagnostics():
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            baseline_wall_ms=5.0,
        ),
        tracker=object(),
        config=object(),
    )
    loop._baseline_case_times = {"small": 1.0, "large": 9.0}
    bench = {
        "success": True,
        "median_ms": 5.25,
        "case_times": {"small": 0.5, "large": 10.0},
        "measurements": [
            {
                "success": True,
                "case_times": {"small": 0.5, "large": 10.0},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
    }

    loop._apply_mean_case_speedup_metric(bench)

    assert bench["median_ms"] == 5.25
    assert bench["mean_case_speedup"] == pytest.approx(1.45)
