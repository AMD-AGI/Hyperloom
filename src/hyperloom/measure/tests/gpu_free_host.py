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

The runner also reads who holds its cards from the kernel's list (:mod:`hyperloom.sandbox.card_occupancy`), and the
card guard enforces by default: on a GPU host whose cards other work holds, every test that waits for a runner's
signed result would get ``foreign_process_on_card`` instead. A conftest can point the test process at an idle machine
of its own, but a runner is another process and reads the host's list. A test therefore starts its runner through
:data:`IDLE_CARDS_PYTHON` (or :func:`idle_cards_python`, for a template that names an interpreter): this module, run as
``python -m hyperloom.measure.tests.gpu_free_host -m <module> <args>``, points the kernel's list and ``/proc`` at an
idle machine in a temporary directory and then runs ``<module>`` as ``python -m`` would.
"""

from __future__ import annotations

import os
import runpy
import shlex
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.tests.kernel_tree import KernelTree

USES_REAL_GPUS_FLAG = "USES_REAL_GPUS"
_GPU_TOOLS = ("amd-smi", "rocm-smi")
_UNREADABLE_TOOL = "#!/bin/sh\nexit 1\n"
_DEVICE_ENV = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
_MODULE = "hyperloom.measure.tests.gpu_free_host"
#: The interpreter a test starts a runner module with, ``(*IDLE_CARDS_PYTHON, "-m", module, *args)``, so that the runner
#: reads an idle machine of its own instead of the host's cards.
IDLE_CARDS_PYTHON = (sys.executable, "-m", _MODULE)


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


def idle_cards_python(directory: Path) -> str:
    """An executable below ``directory`` that a local runner template can name as its interpreter (the template starts
    ``"$0" -m hyperloom.measure.runner run ...``): the module it is given runs on an idle machine of its own."""
    script = directory / "idle-cards-python"
    script.write_text(f'#!/bin/sh\nexec {shlex.join(IDLE_CARDS_PYTHON)} "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _run_on_idle_cards(argv: list[str]) -> None:
    """Run ``-m <module> <args>`` as ``python`` would, with the kernel's list and ``/proc`` of an idle machine."""
    if len(argv) < 2 or argv[0] != "-m":
        raise SystemExit(f"usage: python -m {_MODULE} -m <module> [args...]")
    module, args = argv[1], argv[2:]
    with tempfile.TemporaryDirectory(prefix="hl-idle-machine-") as root:
        machine = KernelTree(Path(root))
        card_occupancy.KFD_PATH, card_occupancy.PROC_PATH = machine.kfd, machine.procfs
        sys.argv = [module, *args]
        runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    _run_on_idle_cards(sys.argv[1:])
