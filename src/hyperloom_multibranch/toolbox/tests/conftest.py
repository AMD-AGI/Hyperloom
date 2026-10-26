# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here runs the same on a GPU host and on a CPU host, busy or idle.

The pool asks the kernel's list who holds a card before it hands it out, and a released lease waits until that list
shows no memory on its cards. A test that read the host's own list would fail wherever other work is on the cards, so
every test here reads an idle machine of its own (a test that lists holders injects its own reader on top). A module
that measures on real cards sets ``USES_REAL_GPUS = True`` and reads the host's list.
"""

from pathlib import Path

import pytest

from hyperloom.measure.tests.gpu_free_host import USES_REAL_GPUS_FLAG, machine_without_readable_gpus  # noqa: F401
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.tests.kernel_tree import KernelTree


@pytest.fixture(autouse=True)
def idle_card_list(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if getattr(request.module, USES_REAL_GPUS_FLAG, False):
        return
    tree = KernelTree(tmp_path / "idle-machine")
    monkeypatch.setattr(card_occupancy, "KFD_PATH", tree.kfd)
    monkeypatch.setattr(card_occupancy, "PROC_PATH", tree.procfs)
