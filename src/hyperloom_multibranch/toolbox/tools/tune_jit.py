# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a ``kernel_tune`` job's tuner builds and compares is the job's: an aiter JIT directory and a temporary directory
of its own.

aiter installs a module it builds into ``AITER_JIT_DIR``, or into the ``jit`` directory of the package it was imported
from when that is unset: a tuner pinned to the image's aiter would put each module it builds into the image, where every
later test on the node loads it. The tuner is told ``AITER_JIT_DIR`` of a directory below the job's, holding links to
the image's prebuilt modules (as a test's own directory does, :mod:`hyperloom.measure.runner.own_jit`), so it builds
only what the image lacks or what the tables it benchmarks make stale, and what it builds stays with the job.

aiter's tuner writes its comparison files below ``tempfile.gettempdir()/aiter_compare``, and Forge reads them back from
the same place (it gives aiter's process its own ``TMPDIR``), taking the newest file of a tuner and chunk written since
the chunk started, whoever wrote it. Two jobs that share a ``TMPDIR`` can therefore take each other's tuned rows, so each
job's tuner is given a ``TMPDIR`` of its own (:func:`own_temp_dir`). It is a short path below the toolbox's own
temporary directory, not below the job's directory: that one lies on the session's shared storage, and can be longer
than the sockets Python's ``multiprocessing`` makes in a temporary directory allow (about a hundred bytes of path).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hyperloom.measure.jit.session import link_prebuilt
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.runner.own_jit import OWN_AITER_JIT_DIR

__all__ = ["TEMP_DIR_ENV", "own_jit_dir", "own_temp_dir"]

#: The variable ``tempfile.gettempdir()`` reads first, in the tuner and in the aiter process Forge starts.
TEMP_DIR_ENV = "TMPDIR"
_TEMP_PREFIX = "kernel_tune-"


def own_jit_dir(configs: Path, workdir: Path) -> dict[str, str]:
    """``{AITER_JIT_DIR: workdir/aiter_jit}``, made with links to the modules of the image's ``jit`` beside ``configs``
    in it; the directory is kept with the job's other output."""
    jit = workdir / OWN_AITER_JIT_DIR
    jit.mkdir(parents=True, exist_ok=True)
    link_prebuilt(Path(os.path.abspath(configs)).parent / "jit", jit, frozenset())
    return {AITER_JIT.env_var: str(jit)}


@contextmanager
def own_temp_dir() -> Iterator[dict[str, str]]:
    """``{TMPDIR: <a new directory>}`` for one job's tuner, removed with what is in it when the block ends; the tuner
    writes what it keeps (its table, report and profiles) to its output directory, not here."""
    with tempfile.TemporaryDirectory(prefix=_TEMP_PREFIX, ignore_cleanup_errors=True) as directory:
        yield {TEMP_DIR_ENV: directory}
