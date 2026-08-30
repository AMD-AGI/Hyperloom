# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test tool — run the kernel's SNR correctness pre-filter.

Passing this is necessary but not sufficient: forge accepts a candidate -- a
kept iteration or an adopted warm start alike -- only after the task's own
``correctness_command`` also passes.
"""

from __future__ import annotations

import asyncio
import re
import sys

from ._subprocess import kill_process_group
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


async def test_correctness(
    driver_script: str,
    driver_args: list[str] | None = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    timeout_sec: int = 120,
) -> dict:
    """Run a kernel test driver and extract its SNR pre-filter verdict.

    The driver script MUST print at least one of:
      - "SNR: XX.XX dB"  (preferred)
      - "allclose: True/False"
      - "max_diff: X.XXe-XX"

    Args:
        driver_script: Path to Python test driver.
        driver_args: Additional arguments to pass to the driver.
        snr_threshold: Minimum SNR in dB to pass (default 30.0).
        timeout_sec: Maximum runtime before killing (default 120s).

    Returns:
        Dict with: passed, outcome, snr_db, max_diff, allclose, output.
        ``outcome`` is one of ``pass``, ``correctness_failure``, ``timeout``,
        ``driver_error``, or ``invalid_result``.
    """
    cmd = [sys.executable, driver_script] + (driver_args or [])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        await kill_process_group(proc)
        return {
            "passed": False,
            "outcome": "timeout",
            "message": f"TIMEOUT after {timeout_sec}s",
            "output": "",
        }
    except asyncio.CancelledError:
        await kill_process_group(proc)
        raise

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    full_output = stdout_text + "\n" + stderr_text

    if proc.returncode != 0:
        return {
            "passed": False,
            "outcome": "driver_error",
            "message": f"DRIVER CRASHED (exit {proc.returncode})",
            "output": full_output[-2000:],
        }

    # Parse SNR
    snr_match = re.search(r"SNR:\s*([-\d.]+)\s*dB", full_output)
    snr_db = float(snr_match.group(1)) if snr_match else None

    # Parse allclose
    allclose_match = re.search(r"allclose:\s*(True|False)", full_output, re.IGNORECASE)
    allclose = allclose_match.group(1).lower() == "true" if allclose_match else None

    # Parse max_diff
    diff_match = re.search(r"max_diff:\s*([\d.eE+-]+)", full_output)
    max_diff = float(diff_match.group(1)) if diff_match else None

    # Determine pass/fail
    if snr_db is not None:
        passed = snr_db >= snr_threshold
        verdict = f"SNR={snr_db:.2f} dB (threshold={snr_threshold})"
    elif allclose is not None:
        passed = allclose
        verdict = f"allclose={allclose}"
    else:
        passed = False
        verdict = "NO CORRECTNESS METRIC FOUND in output"
    outcome = (
        "pass"
        if passed
        else "invalid_result"
        if snr_db is None and allclose is None and max_diff is None
        else "correctness_failure"
    )

    result = {
        "passed": passed,
        "outcome": outcome,
        "snr_db": snr_db,
        "max_diff": max_diff,
        "allclose": allclose,
        "message": f"{'PASS' if passed else 'FAIL'}: {verdict}",
    }
    # On PASS, SNR/max_diff/allclose already carry the signal — the raw tail
    # is dead weight against the next turn's input budget. Keep on FAIL so
    # the agent can inspect warnings / numerical context.
    if not passed:
        result["output"] = full_output[-1500:]
    return result
