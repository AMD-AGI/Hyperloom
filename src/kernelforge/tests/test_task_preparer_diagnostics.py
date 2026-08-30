"""Tests that a failing preflight carries the driver's own output forward.

``test_correctness`` / ``bench_wallclock`` already capture the child's
stdout+stderr on a non-zero exit, but ``_preflight_async`` used to keep only the
one-line verdict ("DRIVER CRASHED (exit 1)"). The repair agent therefore saw a
crash with no traceback and spent its attempt rediscovering it — the observed
failure mode behind "could not produce a conforming driver within the budget".
These tests pin the tail to the result, the audit dict, and the retry prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from kernelforge.loop import task_preparer


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

    async def _fake_bench(**kwargs):
        return bench

    monkeypatch.setattr(task_preparer, "test_correctness", _fake_correctness)
    monkeypatch.setattr(task_preparer, "bench_wallclock", _fake_bench)
    return asyncio.run(task_preparer._preflight_async(driver.as_posix(), 30.0, 1, 2))


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
            "output": "x" * (task_preparer.DIAG_TAIL_CHARS + 500),
        },
        bench={"success": True, "median_ms": 1.0, "message": "ok"},
    )

    assert len(result.diagnostics["correctness"]) == task_preparer.DIAG_TAIL_CHARS
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


def test_retry_prompt_shows_the_traceback():
    failed = task_preparer.PreflightResult(
        ok=False,
        correctness_ok=False,
        bench_ok=False,
        reasons=["correctness mode produced no SNR/allclose metric (DRIVER CRASHED (exit 1))"],
        diagnostics={"correctness": _TRACEBACK},
    )
    prompt = task_preparer._build_prompt(
        evidence="## Task metadata",
        driver_rel=".forge_driver_x.py",
        reference_note="",
        prior_failure="Deterministic preflight after your edit:\n" + failed.detail_report(),
    )

    assert "TypeError" in prompt
    assert "missing 1 required positional argument" in prompt
