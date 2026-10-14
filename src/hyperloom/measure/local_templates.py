# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local sandbox template that starts the measurement runner (the local stage of design 12.4).

The local sandbox provider maps no paths: a process inside it gets ``HYPERLOOM_SANDBOX_ROOT`` and finds ``in/``,
``out/`` and ``work/`` under it. The runner's defaults are global (``/in``, ``/out``, ``/scratch``), so concurrent
sandboxes would collide; this template names the sandbox's own directories explicitly.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Final

from hyperloom.sandbox.local import LocalTemplate

__all__ = ["RUNNER_SCRIPT", "local_runner_command", "local_runner_template"]

#: ``$0`` is the python interpreter (the first argument after the script), so the script itself holds no path.
RUNNER_SCRIPT: Final = (
    'exec "$0" -m hyperloom.measure.runner run --in "$HYPERLOOM_SANDBOX_ROOT/in"'
    ' --out "$HYPERLOOM_SANDBOX_ROOT/out" --scratch "$HYPERLOOM_SANDBOX_ROOT/work/scratch"'
    ' --spec-wait-sec "$1"'
)
_DEFAULT_SPEC_WAIT_SEC = 600


def local_runner_command(
    python: str = sys.executable, *, spec_wait_sec: int = _DEFAULT_SPEC_WAIT_SEC
) -> tuple[str, ...]:
    """The startup argv of a local runner: the shipped runner run on ``python`` in the sandbox's own directories."""
    if spec_wait_sec <= 0:
        raise ValueError("spec_wait_sec must be positive")
    return ("/bin/sh", "-c", RUNNER_SCRIPT, python, str(int(spec_wait_sec)))


def local_runner_template(
    name: str = "runner-1",
    *,
    gpus: int = 1,
    python: str = sys.executable,
    envs: Mapping[str, str] | None = None,
    spec_wait_sec: int = _DEFAULT_SPEC_WAIT_SEC,
) -> LocalTemplate:
    """A sealed, signing ``measurement`` template for ``gpus`` cards whose startup command is the runner."""
    return LocalTemplate(
        name,
        "measurement",
        command=local_runner_command(python, spec_wait_sec=spec_wait_sec),
        gpus=gpus,
        sealed=True,
        signing=True,
        envs=dict(envs or {}),
    )
