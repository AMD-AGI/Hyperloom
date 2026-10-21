# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local sandbox template that starts the measurement runner (the local stage of design 12.4).

The local sandbox provider maps no paths: a process inside it gets ``HYPERLOOM_SANDBOX_ROOT`` and finds ``in/``,
``out/`` and ``work/`` under it. The runner's defaults are global (``/in``, ``/out``, ``/scratch``), so concurrent
sandboxes would collide; this template names the sandbox's own directories explicitly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Final

from hyperloom.common.env_safety import is_secret_shaped_env_name
from hyperloom.measure.framework_base import (
    DEFAULT_FRAMEWORK_ROOTS,
    DEFAULT_FRAMEWORK_WATCH,
    FRAMEWORK_WATCH_ENV,
    SESSION_DIGEST_ENV,
    tree_digest,
)
from hyperloom.sandbox.local import LocalTemplate

__all__ = [
    "IMAGE_ENV_NAMES",
    "IMAGE_ENV_PREFIXES",
    "RUNNER_SCRIPT",
    "framework_envs",
    "image_env",
    "local_runner_command",
    "local_runner_template",
]

#: Variables a framework image sets for itself (kernel backends, allocator and compiler settings, library paths). A
#: hosted sandbox gets them from the image and its template; the local sandbox starts from an emptied environment, so
#: the local template carries them, and a test then runs with the settings it would have in the image.
IMAGE_ENV_PREFIXES: Final = ("SGLANG_", "HIP_", "HSA_", "AITER_", "ROCM", "PYTORCH_", "TORCH", "TRITON_")
IMAGE_ENV_NAMES: Final = frozenset({"LD_LIBRARY_PATH"})

#: ``$0`` is the python interpreter (the first argument after the script), so the script itself holds no path.
RUNNER_SCRIPT: Final = (
    'exec "$0" -m hyperloom.measure.runner run --in "$HYPERLOOM_SANDBOX_ROOT/in"'
    ' --out "$HYPERLOOM_SANDBOX_ROOT/out" --scratch "$HYPERLOOM_SANDBOX_ROOT/work/scratch"'
    ' --spec-wait-sec "$1"'
)
_DEFAULT_SPEC_WAIT_SEC = 600


def image_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The image's own variables in ``environ`` (the application's, which shares the image): a named list, never all of it.

    Credential-shaped names stay out even when a prefix matches.
    """
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if (name in IMAGE_ENV_NAMES or name.startswith(IMAGE_ENV_PREFIXES)) and not is_secret_shaped_env_name(name)
    }


def local_runner_command(
    python: str = sys.executable, *, spec_wait_sec: int = _DEFAULT_SPEC_WAIT_SEC
) -> tuple[str, ...]:
    """The startup argv of a local runner: the shipped runner run on ``python`` in the sandbox's own directories."""
    if spec_wait_sec <= 0:
        raise ValueError("spec_wait_sec must be positive")
    return ("/bin/sh", "-c", RUNNER_SCRIPT, python, str(int(spec_wait_sec)))


def framework_envs() -> dict[str, str]:
    """The variables that make the runner record the framework base: what to hash, and its hash as the session starts.

    Nothing on a machine that has none of the watched paths.
    """
    watched = [path for path in DEFAULT_FRAMEWORK_WATCH if os.path.exists(path)]
    if not watched:
        return {}
    return {FRAMEWORK_WATCH_ENV: os.pathsep.join(watched), SESSION_DIGEST_ENV: tree_digest(watched)}


def local_runner_template(
    name: str = "runner-1",
    *,
    gpus: int = 1,
    python: str = sys.executable,
    envs: Mapping[str, str] | None = None,
    spec_wait_sec: int = _DEFAULT_SPEC_WAIT_SEC,
) -> LocalTemplate:
    """A sealed, signing ``measurement`` template for ``gpus`` cards whose startup command is the runner.

    ``envs`` are the template's fixed variables; when omitted they are the image's variables of this process
    (:func:`image_env`), the local stand-in for what a hosted image and template set.

    Every test sees the image's framework installs that exist on this machine as they are now and writes only
    privately (``framework_roots``, see :class:`~hyperloom.sandbox.templates.LocalTemplate`); ``dataclasses.replace``
    the template with ``framework_roots=()`` to run tests on the shared tree, and their results then say no root was
    isolated.
    """
    return LocalTemplate(
        name,
        "measurement",
        command=local_runner_command(python, spec_wait_sec=spec_wait_sec),
        gpus=gpus,
        sealed=True,
        signing=True,
        envs={**(image_env() if envs is None else envs), **framework_envs()},
        framework_roots=tuple(root for root in DEFAULT_FRAMEWORK_ROOTS if os.path.isdir(root)),
    )
