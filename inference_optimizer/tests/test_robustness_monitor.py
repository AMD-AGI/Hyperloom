# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``optimizer_runs/robustness_monitor.sh.example`` session-dir
resolution.

The monitor resolves the session dir holding ``state.json`` from (1) an
explicit ``$INFERENCE_OPTIMIZER_SESSION_DIR`` or (2) the ``.session_dir`` field
of ``$LAUNCH_INFO_FILE``, and refuses to guess a timestamp/default path. When
the launch-info JSON is configured but not yet flushed (the monitor raced ahead
of the optimizer's launch-info write), it must poll for a bounded window rather
than exiting immediately.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR = REPO_ROOT / "optimizer_runs" / "robustness_monitor.sh.example"

# Vars the monitor consumes; strip from the inherited env so each test is
# hermetic and only sees what it sets explicitly.
_MONITOR_VARS = (
    "INFERENCE_OPTIMIZER_SESSION_DIR",
    "LAUNCH_INFO_FILE",
    "LAUNCH_INFO_WAIT_SEC",
    "PID_FILE",
    "REPO_ROOT",
    "MAX_HOURS",
    "MAGPIE_PYTHON",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for var in _MONITOR_VARS:
        env.pop(var, None)
    return env


def test_monitor_waits_for_delayed_launch_info(tmp_path):
    """LAUNCH_INFO_FILE set but written only after a short delay: the monitor
    must poll (bounded) and resolve it, then go terminal (exit 0) — not give up
    with exit 2 before the optimizer flushed launch-info."""
    sess = tmp_path / "sess"
    (sess / "reports").mkdir(parents=True)
    # Terminal marker so the first main-loop iteration exits 0 right after the
    # session dir is resolved (keeps the test from entering the resume loop).
    (sess / "reports" / "final.md").write_text("done\n", encoding="utf-8")

    launch = tmp_path / "launch.json"
    pidfile = tmp_path / "pid"
    pidfile.write_text("999999\n", encoding="utf-8")

    env = _clean_env()
    env.update(
        {
            "PID_FILE": str(pidfile),
            "REPO_ROOT": str(tmp_path),
            "LAUNCH_INFO_FILE": str(launch),
            "LAUNCH_INFO_WAIT_SEC": "15",
            "MAX_HOURS": "1",
        }
    )

    proc = subprocess.Popen(
        ["bash", str(MONITOR)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Monitor should be polling here, not already dead with exit 2.
    time.sleep(1.5)
    launch.write_text(json.dumps({"session_dir": str(sess)}), encoding="utf-8")
    out, err = proc.communicate(timeout=30)

    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={out}\nstderr={err}"


def test_monitor_fails_fast_when_no_session_source(tmp_path):
    """No explicit session dir AND no LAUNCH_INFO_FILE -> exit 2 immediately
    (no wasted polling), never silently falling back to /workspace/hyperloom."""
    pidfile = tmp_path / "pid"
    pidfile.write_text("1\n", encoding="utf-8")

    env = _clean_env()
    env.update(
        {
            "PID_FILE": str(pidfile),
            "REPO_ROOT": str(tmp_path),
            "MAX_HOURS": "1",
        }
    )

    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(MONITOR)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    elapsed = time.time() - t0

    assert proc.returncode == 2
    assert "session dir is unknown" in proc.stderr
    # Fail-fast: with no LAUNCH_INFO_FILE we must NOT enter the bounded wait
    # loop (default 60s). A generous bound tolerates CI process-spawn latency
    # while still proving we never polled a wait window.
    assert elapsed < 30, f"should fail fast (no wait loop), took {elapsed:.1f}s"


def test_monitor_times_out_when_launch_info_never_appears(tmp_path):
    """LAUNCH_INFO_FILE configured but the file never appears: the monitor
    polls for the bounded window then exits 2 (does not hang forever)."""
    launch = tmp_path / "never.json"
    pidfile = tmp_path / "pid"
    pidfile.write_text("1\n", encoding="utf-8")

    env = _clean_env()
    env.update(
        {
            "PID_FILE": str(pidfile),
            "REPO_ROOT": str(tmp_path),
            "LAUNCH_INFO_FILE": str(launch),
            "LAUNCH_INFO_WAIT_SEC": "3",
            "MAX_HOURS": "1",
        }
    )

    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(MONITOR)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    elapsed = time.time() - t0

    assert proc.returncode == 2
    # Waited at least most of the window before giving up.
    assert elapsed >= 2, f"should have polled the wait window, took {elapsed:.1f}s"
    assert "session dir is unknown" in proc.stderr


def test_monitor_tolerates_non_integer_wait_sec(tmp_path):
    """A malformed LAUNCH_INFO_WAIT_SEC must degrade to the default wait, not
    emit a bash arithmetic error and skip the bounded poll. Pre-fix, an invalid
    token ('60x') makes the ``$(( ... ))`` deadline computation error out under
    ``set -u``, so the monitor never polls the real window and bails (exit 2)
    with a confusing 'within 60xs' message before the delayed launch-info
    appears. Post-fix it warns, falls back to the default window, picks up the
    launch-info, and exits 0."""
    sess = tmp_path / "sess"
    (sess / "reports").mkdir(parents=True)
    (sess / "reports" / "final.md").write_text("done\n", encoding="utf-8")

    launch = tmp_path / "launch.json"
    pidfile = tmp_path / "pid"
    pidfile.write_text("999999\n", encoding="utf-8")

    env = _clean_env()
    env.update(
        {
            "PID_FILE": str(pidfile),
            "REPO_ROOT": str(tmp_path),
            "LAUNCH_INFO_FILE": str(launch),
            "LAUNCH_INFO_WAIT_SEC": "60x",  # invalid arithmetic token
            "MAX_HOURS": "1",
        }
    )

    proc = subprocess.Popen(
        ["bash", str(MONITOR)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.5)
    launch.write_text(json.dumps({"session_dir": str(sess)}), encoding="utf-8")
    out, err = proc.communicate(timeout=30)

    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={out}\nstderr={err}"
    assert "invalid LAUNCH_INFO_WAIT_SEC" in err


def test_monitor_handles_leading_zero_wait_sec(tmp_path):
    """A leading-zero LAUNCH_INFO_WAIT_SEC ('08') passes a naive ^[0-9]+$ check
    but bash arithmetic treats it as OCTAL -> '08: value too great for base'
    under ``set -u`` (the very crash the validation was meant to prevent). The
    deadline computation must force base-10 so '08' means 8 seconds. We point a
    delayed launch-info at a terminal session: post-fix the monitor polls, picks
    it up, and exits 0 with no octal arithmetic error on stderr."""
    sess = tmp_path / "sess"
    (sess / "reports").mkdir(parents=True)
    (sess / "reports" / "final.md").write_text("done\n", encoding="utf-8")

    launch = tmp_path / "launch.json"
    pidfile = tmp_path / "pid"
    pidfile.write_text("999999\n", encoding="utf-8")

    env = _clean_env()
    env.update(
        {
            "PID_FILE": str(pidfile),
            "REPO_ROOT": str(tmp_path),
            "LAUNCH_INFO_FILE": str(launch),
            "LAUNCH_INFO_WAIT_SEC": "08",  # valid digits, but octal-invalid in bash
            "MAX_HOURS": "1",
        }
    )

    proc = subprocess.Popen(
        ["bash", str(MONITOR)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.5)
    launch.write_text(json.dumps({"session_dir": str(sess)}), encoding="utf-8")
    out, err = proc.communicate(timeout=30)

    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={out}\nstderr={err}"
    assert "value too great for base" not in err
