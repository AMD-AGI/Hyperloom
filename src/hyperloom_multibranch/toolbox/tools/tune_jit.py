# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a ``kernel_tune`` job's tuner builds is the job's: an aiter JIT directory of its own.

aiter installs a module it builds into ``AITER_JIT_DIR``, or into the ``jit`` directory of the package it was imported
from when that is unset: a tuner pinned to the image's aiter would put each module it builds into the image, where every
later test on the node loads it. The tuner is told ``AITER_JIT_DIR`` of a directory below the job's, holding links to
the image's prebuilt modules (as a test's own directory does, :mod:`hyperloom.measure.runner.own_jit`), so it builds
only what the image lacks or what the tables it benchmarks make stale, and what it builds stays with the job.

The tuner keeps the temporary directory it inherits: Forge looks for the comparison files aiter's tuner writes below
``tempfile.gettempdir()/aiter_compare`` in the literal ``/tmp/aiter_compare``, so a ``TMPDIR`` of the job's own would
hide every tuned row from it.
"""

from __future__ import annotations

import os
from pathlib import Path

from hyperloom.measure.jit.session import link_prebuilt
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.runner.own_jit import OWN_AITER_JIT_DIR

__all__ = ["own_jit_dir"]


def own_jit_dir(configs: Path, workdir: Path) -> dict[str, str]:
    """``{AITER_JIT_DIR: workdir/aiter_jit}``, made with links to the modules of the image's ``jit`` beside ``configs``
    in it; the directory is kept with the job's other output."""
    jit = workdir / OWN_AITER_JIT_DIR
    jit.mkdir(parents=True, exist_ok=True)
    link_prebuilt(Path(os.path.abspath(configs)).parent / "jit", jit, frozenset())
    return {AITER_JIT.env_var: str(jit)}
