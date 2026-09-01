from __future__ import annotations

import asyncio

from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.loop.runner import IterationConfig


def test_full_suite_timeout_default_matches_cold_jit_budget():
    config = IterationConfig(kernel_file="kernel.py", driver_script="driver.py")
    assert config.validate_stage_timeout_sec == 1800


def test_validation_runs_driver_full_suite_once(monkeypatch):
    calls = []

    async def fake_correctness(**kwargs):
        calls.append(kwargs)
        return {
            "passed": True,
            "snr_db": 42.0,
            "message": "CORRECT",
            "output": "suite output",
        }

    monkeypatch.setattr(
        "kernelforge.loop.validation.test_correctness",
        fake_correctness,
    )

    report = asyncio.run(
        run_validation_pipeline(
            "driver.py",
            snr_threshold=31.0,
            timeout_per_stage=123,
        )
    )

    assert report.all_passed is True
    assert report.failed_stage is None
    assert report.results[0].stage_name == "Full suite"
    assert report.results[0].snr_db == 42.0
    assert calls == [
        {
            "driver_script": "driver.py",
            "driver_args": [],
            "snr_threshold": 31.0,
            "timeout_sec": 123,
        }
    ]


def test_validation_preserves_full_suite_failure(monkeypatch):
    async def fake_correctness(**_kwargs):
        return {
            "passed": False,
            "snr_db": 7.0,
            "message": "suite failed",
            "output": "case 4 mismatch",
        }

    monkeypatch.setattr(
        "kernelforge.loop.validation.test_correctness",
        fake_correctness,
    )

    report = asyncio.run(run_validation_pipeline("driver.py"))

    assert report.all_passed is False
    assert report.failed_stage == 1
    assert report.failed_output == "case 4 mismatch"
    assert "Full suite: FAIL" in report.summary()


def test_validation_distinguishes_timeout_from_correctness_failure(monkeypatch):
    async def fake_correctness(**_kwargs):
        return {
            "passed": False,
            "outcome": "timeout",
            "message": "TIMEOUT after 1800s",
            "output": "",
        }

    monkeypatch.setattr(
        "kernelforge.loop.validation.test_correctness",
        fake_correctness,
    )

    report = asyncio.run(run_validation_pipeline("driver.py"))

    assert report.all_passed is False
    assert report.failed_outcome == "timeout"
    assert "Full suite: TIMEOUT" in report.summary()
    assert "TIMEOUT" in str(report.results[0])
