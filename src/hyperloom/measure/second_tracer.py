# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A second tracer beside the profiler of a profile test.

A profile test may run its server under a second tracer: a tool, installed as a package of its own, that attributes
GPU time to the modules that launched it. It runs on a best-effort basis: it never changes the test's return code, and
a profile whose traced attempt failed runs again without it. This package carries no tracer. A package declares one
under the entry-point group :data:`ENTRY_POINT_GROUP`, as ``name = "package.module:prepare"``: a callable that, given
a profile's points and the test's environment, returns a :class:`SecondTracer` made ready for it, or the note that
says why the profile runs without it.

What a tracer keeps goes to ``<output_dir>/second_tracer``, beside ``profile/`` and never in it: a ``.json`` below
``profile/`` would read as a trace. The report tool reads two files there. :data:`MODULE_SHARES_FILE` is a CSV of the
GPU time of each module by step kind, with the columns ``kind``, ``module``, ``gpu_ms``, ``share_percent`` and
``steps``; the module :data:`UNATTRIBUTED` holds the time no module was found for. :data:`CHECK_FILE` is a JSON object
whose ``ok`` says whether the tracer's own check passed.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

#: The entry-point group a package declares its tracer under.
ENTRY_POINT_GROUP = "hyperloom.measure.second_tracers"
#: Below a test's output directory, beside ``profile``.
OUTPUT_DIR = "second_tracer"
MODULE_SHARES_FILE = "module_shares.csv"
CHECK_FILE = "check.json"
#: The tail of the server log of a traced attempt that failed, kept when the profile ran again without the tracer.
ATTEMPT_LOG_FILE = "attempt_server.log"
#: The ``module`` of the row that holds the GPU time a tracer could not attribute to a module.
UNATTRIBUTED = "unattributed"
_LOG_TAIL_BYTES = 64 * 1024


class TracerError(Exception):
    """A second tracer could not be given a test's points; the message says which step failed and why."""


@dataclass(frozen=True)
class SecondTracer:
    """A second tracer made ready for one profile test.

    ``name`` starts the notes the test carries about it. ``traced_points(points, environ)`` gives the points whose
    servers launch under the tracer, the given ones left as they are; it raises :class:`TracerError`, ``OSError`` or
    ``ValueError`` when it cannot. ``report(output_dir, ranks)`` analyses what the traced server wrote into
    ``<output_dir>/second_tracer`` and returns the notes the test carries; it never raises, and a report that failed is
    a note. ``close()`` removes the tracer's own work.
    """

    name: str
    traced_points: Callable[[Sequence[Path], Mapping[str, str]], tuple[Path, ...]]
    report: Callable[[Path, int], tuple[str, ...]]
    close: Callable[[], None]


def prepare(points: Sequence[Path], environ: Mapping[str, str]) -> SecondTracer | str | None:
    """The installed tracer made ready for a profile of ``points``, the note that says why the profile runs without it,
    or ``None`` when no tracer is installed. Of several installed, the first by name is asked."""

    declared = sorted(metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda point: point.name)
    if not declared:
        return None
    return declared[0].load()(points, environ)


def keep_attempt(server_log: str | None, output_dir: Path) -> None:
    """Keep the tail of a failed traced attempt's server log beside the test's output, where it says why."""

    if not server_log:
        return
    kept = Path(output_dir) / OUTPUT_DIR
    kept.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        with open(server_log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
            (kept / ATTEMPT_LOG_FILE).write_bytes(handle.read())
