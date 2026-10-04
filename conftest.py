# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repository-wide pytest bootstrap."""

from __future__ import annotations

import os
from pathlib import Path


def _pin_child_processes_to_this_tree() -> None:
    """Put this checkout's ``src`` first on the child ``PYTHONPATH``.

    ``pythonpath`` in ``pyproject.toml`` only reaches the pytest process. Tests
    that shell out inherit the ambient one, so a Hyperloom installed elsewhere
    on the host silently answers their imports instead of the tree under test.
    """
    src = str(Path(__file__).resolve().parent / "src")
    existing = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    os.environ["PYTHONPATH"] = os.pathsep.join([src, *(entry for entry in existing if entry != src)])


_pin_child_processes_to_this_tree()
