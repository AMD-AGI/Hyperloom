"""Tests that the bash skeleton scripts exist + DRY_RUN_MOCK works."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _scripts_runnable() -> bool:
    """Bash + the script must actually run.

    On Windows the resolver picks up ``wsl.exe`` which fails when no WSL
    distribution is installed; we probe with ``bash -c true`` to confirm
    the resolved interpreter actually executes shell commands.
    """
    if shutil.which("bash") is None:
        return False
    try:
        rc = subprocess.call(
            ["bash", "-c", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return rc == 0


# ---------------------------------------------------------------------------
def test_run_baseline_script_exists():
    assert (SCRIPTS_DIR / "run_baseline.sh").is_file()


def test_eval_accuracy_script_exists():
    assert (SCRIPTS_DIR / "eval_accuracy.sh").is_file()


def test_monitor_script_exists():
    assert (SCRIPTS_DIR / "monitor.sh").is_file()


@pytest.mark.skipif(not _scripts_runnable(), reason="bash unavailable")
def test_run_baseline_dry_run_writes_metrics(tmp_path: Path):
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        MODEL="fake/model", TP="2", PORT="9000",
        OUT_DIR=str(out_dir), DRY_RUN_MOCK="1",
    )
    rc = subprocess.call(
        ["bash", str(SCRIPTS_DIR / "run_baseline.sh")],
        env=env,
    )
    assert rc == 0
    metrics = (out_dir / "metrics.json")
    assert metrics.is_file()
    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert data["mocked"] is True
    assert data["tput_per_gpu"] == 5000.0


@pytest.mark.skipif(not _scripts_runnable(), reason="bash unavailable")
def test_eval_accuracy_dry_run_writes_summary(tmp_path: Path):
    res = tmp_path / "res"
    env = os.environ.copy()
    env.update(
        MODEL="fake/model", PORT="9000",
        RESULTS_DIR=str(res), DRY_RUN_MOCK="1",
        EVAL_TASK="gsm8k",
    )
    rc = subprocess.call(
        ["bash", str(SCRIPTS_DIR / "eval_accuracy.sh")],
        env=env,
    )
    assert rc == 0
    summary = res / "eval_summary_gsm8k.json"
    assert summary.is_file()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["score"] == 0.71


@pytest.mark.skipif(not _scripts_runnable(), reason="bash unavailable")
def test_run_baseline_real_path_returns_nonzero(tmp_path: Path):
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env.update(MODEL="x", TP="1", PORT="1", OUT_DIR=str(out_dir))
    env.pop("DRY_RUN_MOCK", None)
    rc = subprocess.call(
        ["bash", str(SCRIPTS_DIR / "run_baseline.sh")],
        env=env,
    )
    assert rc != 0
