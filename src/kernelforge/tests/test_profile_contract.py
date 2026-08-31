# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the driver-owned kernel-only profiling contract."""

from __future__ import annotations

import asyncio

from kernelforge.loop.profile_contract import PROFILE_RUN_FLAG
from kernelforge.mcp_server.tools.bench import bench_wallclock


def test_profile_contract_exposes_only_profile_run_flag():
    assert PROFILE_RUN_FLAG == "--profile-run"


def test_bench_case_times_remain_available_for_scoring(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("print('mean_ms: 5.0')\nprint('case_ms: small 2.0')\nprint('case_ms: dominant 8.0')\n")

    result = asyncio.run(bench_wallclock(driver_script=str(driver)))

    assert result["success"]
    assert result["case_times"] == {"small": 2.0, "dominant": 8.0}
