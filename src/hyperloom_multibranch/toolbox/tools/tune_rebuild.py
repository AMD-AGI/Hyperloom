# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which of the image's prebuilt aiter modules the rows of a kernel tune make stale.

A module generated from a tuned table holds only the kernels the image's table names; a test whose table differs builds
its own at the server's first prefill (``hyperloom.measure.framework.table_modules``), minutes before its first request
is served. ``kernel_tune`` says so beside the rows, so the specialist who puts them knows a rebuild comes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hyperloom.measure.framework.table_modules import stale_table_modules
from hyperloom.measure.jit.modules import config_path, read_config
from hyperloom.measure.runner.rebuild_watchdog import MODULE_BUILD_SEC
from hyperloom_multibranch.toolbox.tools.tune_tables import TuneTable

__all__ = ["rebuild_warning"]


def rebuild_warning(tables: Sequence[TuneTable], configs: Path) -> str | None:
    """The warning for the tables whose verified rows make a prebuilt module of the aiter at ``configs`` (its
    ``aiter/configs``) stale; ``None`` when none does, or the module config cannot be read (then nothing is claimed)."""
    package = Path(configs).parent
    try:
        config = read_config(config_path(package))
    except (OSError, ValueError):
        return None
    tuned = [(table.env, table.rows_file) for table in tables if table.verified and table.rows_file is not None]
    stale = sorted(stale_table_modules(config, package, package, [{env: str(path)} for env, path in tuned]))
    if not stale:
        return None
    return (
        f"These rows name kernels outside the image's prebuilt {', '.join(stale)}: a test that puts them rebuilds "
        f"{'that module' if len(stale) == 1 else 'those modules'} at its server's first prefill (estimated "
        f"{MODULE_BUILD_SEC} s each), and sglang's scheduler watchdog is raised for it."
    )
