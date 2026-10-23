from __future__ import annotations

import asyncio

import pytest

from kernelforge.evaluation.correctness.pipeline import ValidationResult, run_validation_pipeline


def test_full_suite_timeout_default_matches_cold_jit_budget():
    from kernelforge.loop.runner import IterationConfig

    config = IterationConfig(kernel_file="kernel.py", driver_script="driver.py")
    assert config.validate_stage_timeout_sec == 1800


@pytest.mark.parametrize(
    ("passed", "outcome", "expected"),
    [
        (True, "", "PASS"),
        (False, "timeout", "TIMEOUT"),
        (False, "driver_error", "ERROR"),
        (False, "invalid_result", "ERROR"),
        (False, "correctness_failure", "FAIL"),
    ],
)
def test_validation_result_status_label_contract(passed, outcome, expected):
    result = ValidationResult(1, "Full suite", passed, "details", outcome=outcome)
    assert result.status_label == expected


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
        "kernelforge.evaluation.correctness.pipeline.test_correctness",
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
            "env": None,
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
        "kernelforge.evaluation.correctness.pipeline.test_correctness",
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
        "kernelforge.evaluation.correctness.pipeline.test_correctness",
        fake_correctness,
    )

    report = asyncio.run(run_validation_pipeline("driver.py"))

    assert report.all_passed is False
    assert report.failed_outcome == "timeout"
    assert "Full suite: TIMEOUT" in report.summary()
    assert "TIMEOUT" in str(report.results[0])
