# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sglang_kernel_layouts"
_INSTALL_SH = _ASSETS / "install_baremetal.sh"


def _kernel_dir(checkout: Path) -> subprocess.CompletedProcess[str]:
    fn_src = subprocess.run(
        ["sed", "-n", "/^sglang_kernel_rocm_build_dir()/,/^}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    script = f"set -euo pipefail\n{fn_src}\nsglang_kernel_rocm_build_dir '{checkout}'\n"
    return subprocess.run(
        ["bash", "-lc", script],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_suffix"),
    [
        ("legacy", "sgl-kernel"),
        ("v0517_aot", "python/sglang/kernels/aot"),
    ],
)
def test_sglang_kernel_rocm_build_dir_known_layouts(
    fixture_name: str, expected_suffix: str
) -> None:
    checkout = _FIXTURES / fixture_name
    result = _kernel_dir(checkout)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(expected_suffix)


def test_sglang_kernel_rocm_build_dir_missing_layout() -> None:
    checkout = _FIXTURES / "empty"
    result = _kernel_dir(checkout)
    assert result.returncode != 0
