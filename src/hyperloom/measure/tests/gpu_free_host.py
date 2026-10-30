# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make tests that run the shipped runner independent of the GPUs of the machine they run on.

The shipped runner reads the model from ``amd-smi`` and the VRAM in use from ``rocm-smi``. On a GPU host the real
tools answer and the runner compares them with the stand-in sheet, so a CPU test would see a hardware mismatch
instead of the outcome it asserts. Tools that exit non-zero ahead of the real ones on ``PATH`` give every host the
same answer: nothing readable, so no hardware check is made.

A conftest imports :func:`machine_without_readable_gpus` to apply it to a whole directory. A module that measures on
real cards sets ``USES_REAL_GPUS = True`` and keeps the host's own tools. A test that installs its own fake tools
prepends them to ``PATH`` inside the test, which runs after this fixture, so they come first.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

USES_REAL_GPUS_FLAG = "USES_REAL_GPUS"
_GPU_TOOLS = ("amd-smi", "rocm-smi")
_UNREADABLE_TOOL = "#!/bin/sh\nexit 1\n"
_DEVICE_ENV = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


@pytest.fixture(autouse=True)
def machine_without_readable_gpus(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if getattr(request.module, USES_REAL_GPUS_FLAG, False):
        return
    shadow = tmp_path / "gpu-tools-shadow"
    shadow.mkdir()
    for name in _GPU_TOOLS:
        tool = shadow / name
        tool.write_text(_UNREADABLE_TOOL)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{shadow}{os.pathsep}{os.environ.get('PATH', '')}")
    for name in _DEVICE_ENV:
        monkeypatch.delenv(name, raising=False)
