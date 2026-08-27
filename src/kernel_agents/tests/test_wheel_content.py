"""Lock that the main wheel does not include test suites."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    wheel_dir = tmp_path_factory.mktemp("kfwheel")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheel_dir)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel build failed: {result.stderr[:400]}")
    wheels = list(wheel_dir.glob("kernel_agents-*.whl"))
    assert wheels, "wheel build succeeded but produced no kernel_agents-*.whl"
    return wheels[-1]


def test_wheel_contains_no_test_files(built_wheel):
    names = zipfile.ZipFile(built_wheel).namelist()
    leaked = [n for n in names if "/tests/" in n or n.split("/")[-1].startswith("test_")]
    assert not leaked, f"test files leaked into wheel ({len(leaked)}): {leaked[:5]}"


def test_wheel_contains_package_modules(built_wheel):
    names = zipfile.ZipFile(built_wheel).namelist()
    assert any(n.startswith("kernel_agents/") and n.endswith(".py") for n in names)
    assert any(n.startswith("kernel_agents/fusion/") and n.endswith(".py") for n in names)


def test_wheel_contains_staged_analysis_profiling_resources(built_wheel):
    names = set(zipfile.ZipFile(built_wheel).namelist())
    root = "kernel_agents/data/local_knowledge/common_methodology/profiling"

    assert f"{root}/rocpc_profile.py" in names
    assert f"{root}/rocprof_compute_workflow.md" in names
    assert f"{root}/reading_a_kernel_bottleneck.md" in names
    assert f"{root}/roofline_on_mi.md" in names
    assert f"{root}/benchmarking_methodology.md" in names
