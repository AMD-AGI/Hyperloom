"""Tests that a failing preflight carries the driver's own output forward."""

from __future__ import annotations

import asyncio
import errno
import logging
from dataclasses import asdict

import pytest

from kernelforge.workflows.forge_loop.preparation import task_preparer_prompts
from kernelforge.workflows.forge_loop.preparation import task_preflight


_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File ".forge_driver_x.py", line 42, in <module>\n'
    "    out = aiter.paged_attention(q, k, v)\n"
    "TypeError: paged_attention() missing 1 required positional argument: 'scale'\n"
)


def _run_preflight(monkeypatch, tmp_path, *, correctness, bench):
    driver = tmp_path / "driver.py"
    driver.write_text("print('noop')\n")

    async def _fake_correctness(**kwargs):
        return correctness

    async def _fake_bench(_run, **kwargs):
        return bench

    monkeypatch.setattr(task_preflight, "test_correctness", _fake_correctness)
    monkeypatch.setattr(task_preflight, "bench_wallclock", _fake_bench)
    return asyncio.run(
        task_preflight._preflight_async(driver.as_posix(), task_preflight.DriverContract(snr_threshold=30.0), 1, 2)
    )


def test_crash_tail_reaches_result_and_audit(monkeypatch, tmp_path):
    result = _run_preflight(
        monkeypatch,
        tmp_path,
        correctness={
            "passed": False,
            "message": "DRIVER CRASHED (exit 1)",
            "output": _TRACEBACK,
        },
        bench={
            "success": False,
            "message": "BENCH CRASHED (exit 1)",
            "output": _TRACEBACK,
        },
    )

    assert not result.ok
    assert "TypeError" in result.diagnostics["correctness"]
    assert "TypeError" in result.diagnostics["bench"]
    # asdict() is what the audit record is written from.
    assert "TypeError" in asdict(result)["diagnostics"]["correctness"]
    # The one-line summary stays short for the log; detail_report() carries the tail.
    assert "TypeError" not in result.summary()
    assert "TypeError" in result.detail_report()


def test_tail_is_truncated(monkeypatch, tmp_path):
    result = _run_preflight(
        monkeypatch,
        tmp_path,
        correctness={
            "passed": False,
            "message": "DRIVER CRASHED (exit 1)",
            "output": "x" * (task_preflight.DIAG_TAIL_CHARS + 500),
        },
        bench={"success": True, "median_ms": 1.0, "message": "ok"},
    )

    assert len(result.diagnostics["correctness"]) == task_preflight.DIAG_TAIL_CHARS
    assert "bench" not in result.diagnostics


def test_passing_stage_records_nothing(monkeypatch, tmp_path):
    result = _run_preflight(
        monkeypatch,
        tmp_path,
        correctness={"passed": True, "snr_db": 48.0, "message": "PASS"},
        bench={
            "success": True,
            "median_ms": 1.0,
            "case_times": {"case-1": 1.0},
            "message": "ok",
        },
    )

    assert result.ok
    assert result.diagnostics == {}
    assert result.detail_report() == result.summary()


@pytest.mark.parametrize("stage", ["correctness", "bench"])
def test_a_stage_whose_driver_cannot_be_spawned_fails_with_its_stack(monkeypatch, tmp_path, caplog, stage):
    driver = tmp_path / "driver.py"
    driver.write_text("print('noop')\n")
    spawn = OSError(errno.EMFILE, "Too many open files")

    async def _correctness(**kwargs):
        if stage == "correctness":
            raise spawn
        return {"passed": True, "snr_db": 48.0, "message": "PASS"}

    async def _bench(_run, **kwargs):
        if stage == "bench":
            raise spawn
        return {"success": True, "median_ms": 1.0, "case_times": {"case-1": 1.0}, "message": "ok"}

    monkeypatch.setattr(task_preflight, "test_correctness", _correctness)
    monkeypatch.setattr(task_preflight, "bench_wallclock", _bench)
    with caplog.at_level(logging.WARNING, logger=task_preflight.log.name):
        result = asyncio.run(
            task_preflight._preflight_async(driver.as_posix(), task_preflight.DriverContract(snr_threshold=30.0), 1, 2)
        )

    assert (result.ok, result.correctness_ok, result.bench_ok) == (False, stage == "bench", stage == "correctness")
    assert result.reasons == [f"{stage} run raised OSError: [Errno 24] Too many open files"]
    assert list(result.diagnostics) == [stage]
    assert result.diagnostics[stage].startswith("Traceback (most recent call last):\n")
    assert result.diagnostics[stage].endswith("OSError: [Errno 24] Too many open files\n")
    (record,) = [r for r in caplog.records if r.getMessage().startswith(f"preflight {stage} run raised: ")]
    assert record.exc_info[1] is spawn


def test_retry_prompt_shows_the_traceback():
    failed = task_preflight.PreflightResult(
        ok=False,
        correctness_ok=False,
        bench_ok=False,
        reasons=["correctness mode produced no SNR/allclose metric (DRIVER CRASHED (exit 1))"],
        diagnostics={"correctness": _TRACEBACK},
    )
    prompt = task_preparer_prompts._build_prompt(
        task_preparer_prompts.PromptBrief(evidence="## Task metadata", driver_rel=".forge_driver_x.py"),
        "",
        task_preparer_prompts.RetryNote(failure="Deterministic preflight after your edit:\n" + failed.detail_report()),
    )

    assert "TypeError" in prompt
    assert "missing 1 required positional argument" in prompt
