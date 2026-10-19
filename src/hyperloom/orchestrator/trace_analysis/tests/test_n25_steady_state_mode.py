# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""N25 — TraceLens splitter steady-state chunk selection contract."""

from __future__ import annotations

import subprocess
import sys


_TL_MODULE = "hyperloom.orchestrator.trace_analysis.tracelens_analysis"


def _run_help():
    """Invoke `python -m hyperloom.orchestrator.trace_analysis.tracelens_analysis --help` and return stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", _TL_MODULE, "--help"],
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def test_cli_flag_appears_in_help():
    """--steady-state-mode is wired into argparse and documented."""
    out = _run_help()
    assert "--steady-state-mode" in out
    # All three choices visible in usage/help.
    assert "mixed" in out
    assert "decode_only" in out
    assert "prefilldecode" in out


def test_cli_rejects_unknown_mode():
    """argparse choices=() must reject random strings."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            _TL_MODULE,
            "--trace-input",
            "/tmp/does-not-exist",
            "--workspace-path",
            "/tmp",
            "--steady-state-mode",
            "garbage_mode",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    err = proc.stderr or ""
    assert "garbage_mode" in err
    assert "choose from" in err or "invalid choice" in err
