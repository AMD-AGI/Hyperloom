# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Both installers resolve ``python3`` to the image's own Python, ahead of the system bins they prepend."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2]
_INSTALLERS = [
    _SRC / "inference_optimizer" / "assets" / "install.sh",
    _SRC / "agents" / "kernel" / "scripts" / "install.sh",
]


_PARENT_PYTHON_GUARD = 'if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then\n'


def _venv_loop(installer: Path) -> str:
    text = installer.read_text(encoding="utf-8")
    start = text.index("for _venv_bin in ")
    guard = text.rfind(_PARENT_PYTHON_GUARD, 0, start)
    if guard != -1:
        return text[guard : text.index("\nfi\n", start) + len("\nfi\n")]
    return text[start : text.index("\ndone\n", start) + len("\ndone\n")]


def _python3_dir(installer: Path, tmp_path: Path, images: list[str], python: str | None = None) -> str:
    for image in images:
        bin_dir = tmp_path / image.lstrip("/")
        bin_dir.mkdir(parents=True, exist_ok=True)
        for name in ("python", "python3"):
            (bin_dir / name).write_text("#!/bin/sh\n", encoding="utf-8")
            (bin_dir / name).chmod(0o755)
    loop = _venv_loop(installer).replace(" /opt/", f" {tmp_path}/opt/").replace(" /venv/", f" {tmp_path}/venv/")
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHON"}}
    env["PATH"] = "/usr/bin:/bin"
    if python is not None:
        env["PYTHON"] = python
    script = f'{loop}p=$(command -v python3); printf %s "${{p%/*}}"'
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    return out.stdout


@pytest.mark.parametrize("installer", _INSTALLERS, ids=["inference_optimizer", "kernel_agent"])
def test_rocm10_image_python_is_put_first(installer: Path, tmp_path: Path) -> None:
    assert _python3_dir(installer, tmp_path, ["/opt/python/bin"]) == f"{tmp_path}/opt/python/bin"


@pytest.mark.parametrize("installer", _INSTALLERS, ids=["inference_optimizer", "kernel_agent"])
def test_opt_venv_still_wins_over_opt_python(installer: Path, tmp_path: Path) -> None:
    images = ["/opt/venv/bin", "/opt/python/bin"]
    assert _python3_dir(installer, tmp_path, images) == f"{tmp_path}/opt/venv/bin"


@pytest.mark.parametrize("installer", _INSTALLERS, ids=["inference_optimizer", "kernel_agent"])
def test_no_image_python_leaves_path_unchanged(installer: Path, tmp_path: Path) -> None:
    assert _python3_dir(installer, tmp_path, []) in {"/usr/bin", "/bin"}


def test_kernel_agent_keeps_the_parent_selected_python(tmp_path: Path) -> None:
    images = ["/custom/venv/bin", "/opt/python/bin"]
    python = f"{tmp_path}/custom/venv/bin/python3"
    assert _python3_dir(_INSTALLERS[1], tmp_path, images, python) == f"{tmp_path}/custom/venv/bin"


def test_kernel_agent_ignores_a_parent_python_that_is_not_executable(tmp_path: Path) -> None:
    python = f"{tmp_path}/missing/bin/python3"
    assert _python3_dir(_INSTALLERS[1], tmp_path, ["/opt/python/bin"], python) == f"{tmp_path}/opt/python/bin"
