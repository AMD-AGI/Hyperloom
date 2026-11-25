# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The last lines the servers of a test timed their requests with, read for the specialist that asked for the test.

A specialist reads a test through its result, and in a sandbox it cannot open the test's landed logs. So the result
carries, for a server built from its checkout, the lines of each server log the rule pack names as its request timing
(:mod:`hyperloom.common.server_timing`), each led by the log's path below the test's output directory. A round that ran
on a server another round booted holds a copy of the stretch of that server's log it ran over, so a line is shown under
the first log that holds it and not again.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import BenchmarkPointError, read_benchmark_point
from hyperloom.common.server_timing import read_timing_lines
from hyperloom.inference_optimizer.framework_registry import native_component
from hyperloom_multibranch.toolbox.table_hits import server_logs

__all__ = ["read_test_timing_lines"]

_NOT_READ = "not read ({why})"


def read_test_timing_lines(points: Sequence[Path], output_dir: Path) -> tuple[str, ...]:
    """The timing lines of every server log below ``output_dir``, in the order of the logs' paths, for a test whose
    ``points`` are served by a framework built from its checkout; none for any other test, nor for one whose point
    cannot be read. A log that cannot be read is named with why, so that it is not taken for a server that timed
    nothing."""
    try:
        component = native_component(read_benchmark_point(points[0]).framework) if points else None
    except (OSError, yaml.YAMLError, BenchmarkPointError):
        return ()
    if component is None:
        return ()
    shown: list[str] = []
    seen: set[str] = set()
    for log in server_logs(output_dir):
        where = log.relative_to(output_dir)
        try:
            lines = read_timing_lines(log, component)
        except OSError as exc:
            shown.append(f"{where}: {_NOT_READ.format(why=type(exc).__name__)}")
            continue
        shown.extend(f"{where}: {line}" for line in lines if line not in seen)
        seen.update(lines)
    return tuple(shown)
