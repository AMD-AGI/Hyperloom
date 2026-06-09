# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``optimizer_runs/robustness_monitor.sh.example`` session-dir resolution.

The monitor resolves the session dir from an explicit env var or the
``$LAUNCH_INFO_FILE`` ``.session_dir`` field (polling a bounded window if not
yet flushed), and refuses to guess a default path.
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

# Vars the monitor consumes; stripped from the inherited env so each test is hermetic.
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
    """A delayed LAUNCH_INFO_FILE: the monitor polls (bounded), resolves it, and exits 0 — not exit 2 before launch-info flushed."""
    sess = tmp_path / "sess"
    (sess / "reports").mkdir(parents=True)
    # Terminal marker so the first main-loop iteration exits 0 right after the session dir resolves.
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
    """No session dir and no LAUNCH_INFO_FILE -> exit 2 immediately, never falling back to a default path."""
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
    # Fail-fast: with no LAUNCH_INFO_FILE we must NOT enter the bounded wait loop.
    assert elapsed < 30, f"should fail fast (no wait loop), took {elapsed:.1f}s"


def test_monitor_times_out_when_launch_info_never_appears(tmp_path):
    """LAUNCH_INFO_FILE configured but the file never appears: the monitor polls the bounded window then exits 2."""
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
    """A malformed LAUNCH_INFO_WAIT_SEC ('60x') must degrade to the default wait, warn, poll, and exit 0 — not error out under ``set -u``."""
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
    """A leading-zero LAUNCH_INFO_WAIT_SEC ('08') must be parsed base-10 (not octal), so the monitor polls and exits 0 with no arithmetic error."""
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
