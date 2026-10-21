# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""install.sh resolves ``python3`` to the image's own Python, and the kernel-agent steps to the one it selected."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_INSTALL_SH = _ASSETS / "install.sh"
_KERNEL_TOOLS_SH = _ASSETS / "install_kernel_tools.sh"


_PARENT_PYTHON_GUARD = 'if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then\n'


def _venv_loop(installer: Path) -> str:
    text = installer.read_text(encoding="utf-8")
    start = text.index("for _venv_bin in ")
    guard = text.rfind(_PARENT_PYTHON_GUARD, 0, start)
    if guard != -1:
        return text[guard : text.index("\nfi\n", start) + len("\nfi\n")]
    return text[start : text.index("\ndone\n", start) + len("\ndone\n")]


def _make_pythons(tmp_path: Path, images: list[str]) -> None:
    for image in images:
        bin_dir = tmp_path / image.lstrip("/")
        bin_dir.mkdir(parents=True, exist_ok=True)
        for name in ("python", "python3"):
            (bin_dir / name).write_text("#!/bin/sh\n", encoding="utf-8")
            (bin_dir / name).chmod(0o755)


def _python3_dir(tmp_path: Path, images: list[str]) -> str:
    _make_pythons(tmp_path, images)
    loop = _venv_loop(_INSTALL_SH).replace(" /opt/", f" {tmp_path}/opt/").replace(" /venv/", f" {tmp_path}/venv/")
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHON"}}
    env["PATH"] = "/usr/bin:/bin"
    script = f'{loop}p=$(command -v python3); printf %s "${{p%/*}}"'
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    return out.stdout


def test_rocm10_image_python_is_put_first(tmp_path: Path) -> None:
    assert _python3_dir(tmp_path, ["/opt/python/bin"]) == f"{tmp_path}/opt/python/bin"


def test_opt_venv_still_wins_over_opt_python(tmp_path: Path) -> None:
    images = ["/opt/venv/bin", "/opt/python/bin"]
    assert _python3_dir(tmp_path, images) == f"{tmp_path}/opt/venv/bin"


def test_no_image_python_leaves_path_unchanged(tmp_path: Path) -> None:
    assert _python3_dir(tmp_path, []) in {"/usr/bin", "/bin"}


def test_kernel_agent_steps_keep_the_python_install_sh_selected(tmp_path: Path) -> None:
    _make_pythons(tmp_path, ["/custom/venv/bin", "/opt/python/bin"])
    path_line = next(
        line for line in _KERNEL_TOOLS_SH.read_text(encoding="utf-8").splitlines() if line.startswith("export PATH=")
    )
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["PATH"] = f"{tmp_path}/opt/python/bin:/usr/bin:/bin"
    env["PYTHON"] = f"{tmp_path}/custom/venv/bin/python3"
    script = f'{path_line}\np=$(command -v python3); printf %s "${{p%/*}}"'
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert out.stdout == f"{tmp_path}/custom/venv/bin"
