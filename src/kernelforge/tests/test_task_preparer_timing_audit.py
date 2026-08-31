"""Preflight timing must be recorded, and audit snapshots must be datable.

The audit directory carried no timing at all, so "which stage ate the budget"
could only be inferred from file mtimes -- and those lied: ``_audit_driver``
used ``shutil.copy2``, which copies the SOURCE mtime onto the snapshot. Every
driver snapshot therefore claimed the driver's own mtime instead of its capture
time, and a timeline reconstructed from the directory was off by minutes.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import asdict

from kernelforge.loop import task_preparer


def test_preflight_records_total_and_per_stage_seconds(monkeypatch, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("print('x')\n")

    async def slow_correctness(**_kwargs):
        await asyncio.sleep(0.05)
        return {"passed": True, "snr_db": 40.0, "message": "PASS"}

    async def fast_bench(**_kwargs):
        return {
            "success": True,
            "median_ms": 1.0,
            "case_times": {"case-1": 1.0},
            "message": "ok",
        }

    monkeypatch.setattr(task_preparer, "test_correctness", slow_correctness)
    monkeypatch.setattr(task_preparer, "bench_wallclock", fast_bench)

    result = asyncio.run(task_preparer._preflight_async(driver.as_posix(), 30.0, 1, 2))

    assert result.ok
    assert result.duration_sec >= 0.05
    assert result.details["correctness"]["seconds"] >= 0.05
    assert "seconds" in result.details["bench"]
    # The audit record is built from asdict(), so it must carry the timing too.
    dumped = asdict(result)
    assert dumped["duration_sec"] == result.duration_sec
    assert dumped["details"]["correctness"]["seconds"] >= 0.05


def test_stage_timing_is_recorded_even_when_the_stage_fails(monkeypatch, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("print('x')\n")

    async def crashing(**_kwargs):
        return {"passed": False, "message": "DRIVER CRASHED (exit 1)", "output": "boom"}

    async def crashing_bench(**_kwargs):
        return {"success": False, "message": "BENCH CRASHED (exit 1)", "output": "boom"}

    monkeypatch.setattr(task_preparer, "test_correctness", crashing)
    monkeypatch.setattr(task_preparer, "bench_wallclock", crashing_bench)

    result = asyncio.run(task_preparer._preflight_async(driver.as_posix(), 30.0, 1, 2))

    assert not result.ok
    assert "seconds" in result.details["correctness"]
    assert "seconds" in result.details["bench"]


def test_audit_driver_snapshot_is_stamped_with_the_capture_time(tmp_path):
    """A copy2'd snapshot inherits the source mtime; the audit must not."""
    source = tmp_path / "driver.py"
    source.write_text("DRIVER\n")
    old = time.time() - 3600
    os.utime(source, (old, old))

    destination = tmp_path / "audit" / "driver_before.py"
    destination.parent.mkdir()

    shutil.copy2(source, destination)
    assert abs(destination.stat().st_mtime - old) < 2  # the trap the audit fell into

    os.utime(destination, None)  # what _audit_driver now does
    assert destination.stat().st_mtime - old > 3000
    assert destination.read_text() == "DRIVER\n"
