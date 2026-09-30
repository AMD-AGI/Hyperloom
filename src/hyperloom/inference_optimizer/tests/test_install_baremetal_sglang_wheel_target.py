# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install_baremetal.sh's amd-sglang wheel target selection."""

from __future__ import annotations

import subprocess
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


def _run_extra_for_torch(tmp_path: Path, torch_hip: str | None) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("sglang_rocm_extra_for_torch")
    assert fn_src.strip(), "sglang_rocm_extra_for_torch() not found in install_baremetal.sh"
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    if torch_hip is not None:
        torch_pkg = site / "torch"
        torch_pkg.mkdir(exist_ok=True)
        (torch_pkg / "__init__.py").write_text("from . import version\n")
        (torch_pkg / "version.py").write_text(f'hip = "{torch_hip}"\n')
    script = f"set -uo pipefail\n{fn_src}\nsglang_rocm_extra_for_torch python3\n"
    env = {"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True, env=env)


def test_extra_for_torch_selects_rocm724_on_rocm72_stack(tmp_path: Path) -> None:
    result = _run_extra_for_torch(tmp_path, "7.2.4-abc123")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rocm724"


def test_extra_for_torch_selects_rocm700_on_rocm70_stack(tmp_path: Path) -> None:
    result = _run_extra_for_torch(tmp_path, "7.0.1-def456")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rocm700"


def test_extra_for_torch_is_empty_on_therock_rocm10_stack(tmp_path: Path) -> None:
    # The rocm10/gfx950 image reports HIP 7.15, which is not a ROCm release and
    # has no published amd-sglang wheel. Must not claim a rocm724 wheel target.
    result = _run_extra_for_torch(tmp_path, "7.15.26333")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_extra_for_torch_is_empty_without_torch(tmp_path: Path) -> None:
    result = _run_extra_for_torch(tmp_path, None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _run_version_for_extra(extra: str) -> str:
    fn_src = _extract_function("sglang_pypi_version_for_extra")
    assert fn_src.strip(), "sglang_pypi_version_for_extra() not found in install_baremetal.sh"
    script = f"set -euo pipefail\n{fn_src}\nsglang_pypi_version_for_extra '{extra}'\n"
    out = subprocess.run(["bash", "-lc", script], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def test_version_for_extra_maps_known_targets() -> None:
    assert _run_version_for_extra("rocm700") == "7.0.0"
    assert _run_version_for_extra("rocm724") == "7.2.4"


def test_default_sglang_rocm_extra_is_not_hardcoded() -> None:
    # Regression guard: the default must come from the detected stack, so the
    # config block must not pin a wheel target of its own.
    text = _INSTALL_SH.read_text()
    assert 'SGLANG_ROCM_EXTRA="${SGLANG_ROCM_EXTRA:-}"' in text
    assert 'SGLANG_ROCM_EXTRA="${SGLANG_ROCM_EXTRA:-rocm724}"' not in text
