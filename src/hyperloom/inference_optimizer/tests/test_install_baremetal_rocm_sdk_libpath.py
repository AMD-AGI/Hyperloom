# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install_baremetal.sh's TheRock ROCm SDK wheel lib-dir detection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_INSTALL_SH = _ASSETS / "install_baremetal.sh"


def _extract_function(name: str) -> str:
    return subprocess.run(
        ["sed", "-n", f"/^{name}()/,/^}}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _run_rocm_sdk_wheel_lib_dirs(pythonpath: str) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("rocm_sdk_wheel_lib_dirs")
    assert fn_src.strip(), "rocm_sdk_wheel_lib_dirs() not found in install_baremetal.sh"
    script = f"set -euo pipefail\n{fn_src}\nrocm_sdk_wheel_lib_dirs '{sys.executable}'\n"
    env = {"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True, env=env)


def test_rocm_sdk_wheel_lib_dirs_detects_therock_layout(tmp_path: Path) -> None:
    site = tmp_path / "site"
    core_lib = site / "_rocm_sdk_core" / "lib"
    core_lib.mkdir(parents=True)
    (site / "_rocm_sdk_core" / "__init__.py").write_text("")
    devel_host_math_lib = site / "_rocm_sdk_devel" / "lib" / "host-math" / "lib"
    devel_host_math_lib.mkdir(parents=True)
    (site / "_rocm_sdk_devel" / "__init__.py").write_text("")

    result = _run_rocm_sdk_wheel_lib_dirs(str(site))

    assert result.returncode == 0, result.stderr
    dirs = result.stdout.splitlines()
    assert str(core_lib) in dirs
    assert str(devel_host_math_lib) in dirs
    # devel/lib itself is a byproduct of the nested mkdir and must also be reported.
    assert str(site / "_rocm_sdk_devel" / "lib") in dirs


def test_rocm_sdk_wheel_lib_dirs_absent_on_standard_rocm_image(tmp_path: Path) -> None:
    # Empty PYTHONPATH dir: neither _rocm_sdk_core nor _rocm_sdk_devel importable,
    # matching a standard /opt/rocm image. Detection must report nothing.
    empty_site = tmp_path / "empty"
    empty_site.mkdir()

    result = _run_rocm_sdk_wheel_lib_dirs(str(empty_site))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
