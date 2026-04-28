"""Smoke tests for the skill-bundled shell scripts.

The scripts now live under
``.cursor/skills/inference-optimizer/scripts/`` (resolved via
:func:`paths.skill_scripts_dir`) and are the *real* sprint scripts that
launch sglang/vllm + run benchmarks + evaluate accuracy. We can only
verify two cheap things in a sandbox without GPU + WekaFS bundle:

1. The expected files exist and are executable.
2. Invoked without their required env vars they refuse to do anything
   (exit non-zero, do not leak files into the cwd).

The full DRY_RUN_MOCK fixture path was specific to the now-removed
``src/inference_optimizer/scripts/{run_baseline,eval_accuracy}.sh``
stubs; the new skill scripts assume real GPU + InferenceX bundle.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from inference_optimizer.paths import skill_scripts_dir


def _scripts_runnable() -> bool:
    if shutil.which("bash") is None:
        return False
    try:
        rc = subprocess.call(
            ["bash", "-c", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return rc == 0


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------
def test_skill_scripts_dir_exists():
    assert skill_scripts_dir().is_dir()


@pytest.mark.parametrize(
    "name",
    [
        "run_baseline.sh",
        "run_profile.sh",
        "run_sweep.sh",
        "eval_accuracy.sh",
        "common.sh",
        "executor.sh",
        "bootstrap.sh",
        "geak_ray_submit.py",
        "oob_ray_submit.py",
        "ray_submit.py",
        "patch_inductor.py",
        "trace_action.py",
    ],
)
def test_skill_script_exists(name):
    p = skill_scripts_dir() / name
    assert p.is_file(), f"skill script missing: {p}"


def test_run_baseline_executable_bit():
    """The shell scripts should be marked executable so subprocess.run
    can invoke them directly without a ``bash`` prefix."""
    p = skill_scripts_dir() / "run_baseline.sh"
    assert p.stat().st_mode & 0o111, f"{p} is not executable"


# ---------------------------------------------------------------------------
# Behaviour without env vars — must refuse, must not silently succeed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _scripts_runnable(), reason="bash unavailable")
def test_run_baseline_refuses_without_required_env(tmp_path: Path):
    """``MODEL`` / ``TP`` / ``CONC`` / ``ISL`` / ``OSL`` / ``INFERENCEX_PATH``
    are mandatory. Calling the script with none of these set must exit
    non-zero (set -u) before touching the GPU."""
    env = {"PATH": "/usr/bin:/bin"}  # minimal env, no fixtures
    rc = subprocess.call(
        ["bash", str(skill_scripts_dir() / "run_baseline.sh")],
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(tmp_path),
        timeout=10,
    )
    assert rc != 0


@pytest.mark.skipif(not _scripts_runnable(), reason="bash unavailable")
def test_eval_accuracy_refuses_without_required_env(tmp_path: Path):
    env = {"PATH": "/usr/bin:/bin"}
    rc = subprocess.call(
        ["bash", str(skill_scripts_dir() / "eval_accuracy.sh")],
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(tmp_path),
        timeout=10,
    )
    assert rc != 0
