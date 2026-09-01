"""Coverage tests for the MCP server tool definitions and test/bench tools.

Hermetic: test/bench drivers are tiny Python scripts written to tmp_path and
run via the same interpreter. No GPU, no real kernels.
"""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    bench_wallclock,
    calculate_mean_case_speedup,
)

# Alias avoids pytest-asyncio (auto mode) collecting the imported coroutine as
# a test just because its name starts with "test_".
from kernelforge.mcp_server.tools.test import test_correctness as run_correctness


def _write_driver(tmp_path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def _run_and_flush(coro):
    """Run a coroutine, then pump the loop so a killed subprocess transport
    finishes closing before the loop is torn down (avoids a spurious
    'Event loop is closed' unraisable warning on the timeout path)."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
        loop.run_until_complete(asyncio.sleep(0.2))
        return result
    finally:
        loop.close()


def test_calculate_mean_case_speedup_weights_cases_equally():
    mean_case_speedup = calculate_mean_case_speedup(
        case_times={"small": 0.5, "large": 10.0},
        baseline_case_times={"small": 1.0, "large": 9.0},
    )

    # Mean speedup = (2.0 + 0.9) / 2 = 1.45, regardless of case duration.
    assert mean_case_speedup == pytest.approx(1.45)


def test_calculate_mean_case_speedup_requires_full_case_coverage():
    with pytest.raises(CaseCoverageError, match="large"):
        calculate_mean_case_speedup(
            case_times={"small": 0.5},
            baseline_case_times={"small": 1.0, "large": 9.0},
        )


def test_calculate_mean_case_speedup_rejects_unexpected_cases():
    with pytest.raises(CaseCoverageError, match="unexpected=.*extra"):
        calculate_mean_case_speedup(
            case_times={"small": 0.5, "extra": 1.0},
            baseline_case_times={"small": 1.0},
        )


def test_bench_rejects_duplicate_case_timings(tmp_path):
    driver = _write_driver(
        tmp_path,
        "duplicate_cases.py",
        "print('case_ms: repeated 1.0')\nprint('case_ms: repeated 2.0')\nprint('mean_ms: 1.5')\n",
    )

    result = asyncio.run(bench_wallclock(driver_script=driver))

    assert result["success"] is False
    assert result["message"] == "DUPLICATE CASE TIMINGS: repeated"


# ─── test_correctness ───


def test_correctness_pass_snr(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "print('SNR: 42.50 dB')\n")
    result = asyncio.run(run_correctness(drv, snr_threshold=30.0))
    assert result["passed"] is True
    assert result["outcome"] == "pass"
    assert result["snr_db"] == 42.5
    assert "output" not in result  # tail dropped on PASS


def test_correctness_fail_snr(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "print('SNR: 10.0 dB')\n")
    result = asyncio.run(run_correctness(drv, snr_threshold=30.0))
    assert result["passed"] is False
    assert result["outcome"] == "correctness_failure"
    assert "output" in result  # tail kept on FAIL


def test_correctness_allclose_and_maxdiff(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "print('allclose: True')\nprint('max_diff: 1.2e-05')\n")
    result = asyncio.run(run_correctness(drv))
    assert result["passed"] is True
    assert result["allclose"] is True
    assert result["max_diff"] == 1.2e-05


def test_correctness_no_metric(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "print('nothing useful')\n")
    result = asyncio.run(run_correctness(drv))
    assert result["passed"] is False
    assert result["outcome"] == "invalid_result"
    assert "NO CORRECTNESS METRIC" in result["message"]


def test_correctness_driver_crash(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "import sys; sys.exit(3)\n")
    result = asyncio.run(run_correctness(drv))
    assert result["passed"] is False
    assert result["outcome"] == "driver_error"
    assert "CRASHED" in result["message"]


def test_correctness_timeout(tmp_path):
    drv = _write_driver(tmp_path, "d.py", "import time; time.sleep(5)\n")
    result = _run_and_flush(run_correctness(drv, timeout_sec=1))
    assert result["passed"] is False
    assert result["outcome"] == "timeout"
    assert "TIMEOUT" in result["message"]


# ─── bench_wallclock ───


def test_bench_per_iter_median(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "for t in (1.0, 2.0, 3.0):\n    print(f'wall_ms: {t}')\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["success"] is True
    assert result["median_ms"] == 2.0
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 3.0
    assert result["n_samples"] == 3


def test_bench_case_times_and_callback(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "print('wall_ms: 2.0')\nprint('case_ms: caseA 5.5')\n")
    captured = {}
    result = asyncio.run(bench_wallclock(drv, on_result=captured.update))
    assert result["case_times"] == {"caseA": 5.5}
    assert captured["median_ms"] == 2.0


def test_bench_parses_case_bandwidth(tmp_path):
    """Parse byte counts and GB/s values without unit ambiguity."""
    drv = _write_driver(
        tmp_path,
        "bandwidth.py",
        "print('mean_ms: 1.0')\nprint('case_bw: caseA bytes=9007199254740993 algbw=12.5GB/s busbw=10.25GB/s')\n",
    )

    result = asyncio.run(bench_wallclock(drv))

    assert result["case_bandwidth"] == {
        "caseA": {
            "bytes": 9007199254740993,
            "algbw_gbs": 12.5,
            "busbw_gbs": 10.25,
        }
    }


def test_bench_aggregate_mean(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "print('mean_ms: 4.25')\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["success"] is True
    assert result["stat"] == "mean"
    assert result["median_ms"] == 4.25


def test_bench_aggregate_median(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "print('median_ms: 3.0')\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["stat"] == "median"


def test_bench_no_timing(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "print('no timings here')\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["success"] is False
    assert "NO TIMING DATA" in result["message"]


def test_bench_crash(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "import sys; sys.exit(1)\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["success"] is False
    assert "CRASHED" in result["message"]


def test_bench_timeout(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "import time; time.sleep(5)\n")
    result = _run_and_flush(bench_wallclock(drv, timeout_sec=1))
    assert result["success"] is False
    assert "TIMEOUT" in result["message"]


def test_bench_bad_case_ms_skipped(tmp_path):
    drv = _write_driver(tmp_path, "b.py", "print('wall_ms: 1.0')\nprint('case_ms: caseB notanumber')\n")
    result = asyncio.run(bench_wallclock(drv))
    assert result["case_times"] == {}


# ─── build / pmc / registers dispatch (mock the GPU-bound tool coroutines) ───


def _async_stub(return_value):
    async def _stub(*args, **kwargs):
        return return_value

    return _stub
