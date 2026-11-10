# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.isolation (disk preflight, attempt venvs)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.agents.framework.isolation import (
    DiskPreflightError,
    create_venv,
    disk_preflight,
)


def _venv_python(venv: Path, code: str) -> str:
    return subprocess.run(
        [str(venv / "bin" / "python"), "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_venv_sees_the_packages_of_the_interpreter_that_created_it(tmp_path: Path) -> None:
    """Created from a venv, ``--system-site-packages`` alone would hide its packages, ROCm torch included."""
    venv = tmp_path / "venv"
    create_venv(venv)
    assert _venv_python(venv, "import yaml; print(yaml.__file__)") == __import__("yaml").__file__


def test_a_venv_sees_the_eggs_and_editable_installs_of_the_interpreter_that_created_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image's sglang-kernel is an egg and its aiter an editable install: both reach sys.path through a .pth."""
    inherited = tmp_path / "inherited-site"
    (inherited / "sgl_kernel-0.4.7.egg" / "sgl_kernel").mkdir(parents=True)
    (inherited / "sgl_kernel-0.4.7.egg" / "sgl_kernel" / "__init__.py").write_text("", encoding="utf-8")
    (inherited / "easy-install.pth").write_text("./sgl_kernel-0.4.7.egg\n", encoding="utf-8")
    monkeypatch.setattr("site.getsitepackages", lambda: [str(inherited)])
    venv = tmp_path / "venv"
    create_venv(venv)
    expected = str(inherited / "sgl_kernel-0.4.7.egg" / "sgl_kernel" / "__init__.py")
    assert _venv_python(venv, "import sgl_kernel; print(sgl_kernel.__file__)") == expected


def test_a_package_installed_into_the_venv_shadows_the_interpreters_copy(tmp_path: Path) -> None:
    """An editable install lands as a ``.pth`` too; it must still win over the inherited site directories."""
    venv = tmp_path / "venv"
    create_venv(venv)
    src = tmp_path / "src" / "yaml"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("ORIGIN = 'attempt'\n", encoding="utf-8")
    purelib = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (purelib / "__editable__.yaml-0.0.pth").write_text(f"{src.parent}\n", encoding="utf-8")
    assert _venv_python(venv, "import yaml; print(yaml.ORIGIN)") == "attempt"


def test_disk_preflight_passes_when_enough_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1GB floor is easily satisfied by a normal scratch mount."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "0.001")
    disk_preflight(tmp_path / "wd", n_candidates=2)


def test_disk_preflight_raises_when_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Floor higher than available free GB triggers DiskPreflightError."""
    usage = shutil.disk_usage(str(tmp_path))
    impossible_gb = (usage.free / (1024**3)) + 10_000
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", str(impossible_gb))
    with pytest.raises(DiskPreflightError) as exc:
        disk_preflight(tmp_path / "wd", n_candidates=1)
    assert "insufficient disk" in str(exc.value)
    assert "required" in str(exc.value)


def test_disk_preflight_n_candidates_scales_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """required = max(floor, n * per_candidate) — large n breaches a low floor."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "0.001")
    usage = shutil.disk_usage(str(tmp_path))
    free_gb = usage.free / (1024**3)
    n = int(free_gb / 1.5) + 50
    with pytest.raises(DiskPreflightError):
        disk_preflight(tmp_path / "wd", n_candidates=n, per_candidate_gb=1.5)


def test_disk_preflight_env_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FRAMEWORK_EXPLORER_DISK_MIN_GB sets the floor."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "999999")
    with pytest.raises(DiskPreflightError):
        disk_preflight(tmp_path / "wd", n_candidates=1)
