# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which interpreter runs a task's driver, when it is not the one running the loop."""

from __future__ import annotations

import os
import sys

#: Names another interpreter for driver scripts. Unset means the loop's own, which is the single-venv default.
ENV_VAR = "FORGE_DRIVER_PYTHON"


def driver_python() -> str:
    """The interpreter a driver script runs under.

    A driver imports the kernel together with its framework (``torch``, ``triton``, the framework package itself), so
    it has to run somewhere those are importable. On a single-venv install that is the interpreter running the loop,
    and this returns exactly what every caller used before. A host that keeps the orchestrator and the framework in
    separate venvs -- the shape a pre-built framework image leaves behind -- names the framework's interpreter here
    instead of installing a second copy of the stack beside the orchestrator.
    """
    configured = os.environ.get(ENV_VAR, "").strip()
    return configured or sys.executable
