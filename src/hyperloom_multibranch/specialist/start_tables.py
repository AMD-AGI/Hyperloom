# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel tables of the commit an autonomous branch starts on, laid out where its agent's servers read them.

A test loads each kernel artifact of a recipe through the variable the artifact names (``hyperloom.measure.compile``):
a rows-only table as the image's default tables for the variable, the rows' shapes taken out of them
(``hyperloom.measure.framework.rows``), then the rows; any other table as its file. The agent starts its own servers,
outside any test, so :func:`lay_start_tables` writes the same files under the agent's directory and the agent starts
with the variables naming them. The default tables are private copies, since aiter writes back into every table a
variable lists. A keep that leaves such a variable as it was given carries no table change of the agent's
(:func:`from_start_tables`); one that lists its own files beside the given ones loads the image's default tables, the
start's rows and its own, which :mod:`agent_import` reads as rows over the image's default tables.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.contracts.recipe import KernelArtifact
from hyperloom.measure.compile import (
    DEVICE_ORDINALS,
    PER_DEVICE_TABLE_ENVS,
    KernelLoadError,
    kernel_load_env,
    kernel_rows_only,
)
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER, RowsOnlyRefused, lay_rows_over
from hyperloom.measure.framework.table_modules import default_tables
from hyperloom.measure.framework.tables import read_variables
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.specialist.agent_import import KeptChange

#: Below the directory the tables are laid in: the start commit's files, at the paths its recipe names.
FILES_DIR = "files"
ENCODING = "utf-8"


class StartTableError(ValueError):
    """A kernel table of the start commit that cannot be laid out as a test loads it."""


@dataclass(frozen=True)
class StartTable:
    """What a variable is given for a table of the start commit: ``value``, and the private copies of the image's
    default tables it lists before the table's rows (none for a table that is not rows-only)."""

    value: str
    defaults: tuple[str, ...] = ()


async def lay_start_tables(
    repo: ExperimentRepo, commit: str, kernels: tuple[KernelArtifact, ...], directory: Path, configs: Path
) -> dict[str, StartTable]:
    """Write ``kernels``, the tables of ``commit``, under ``directory`` as a test loads them over the image's aiter
    tables at ``configs``; what each variable is given, by name.

    Raises :class:`StartTableError` for a table a test of ``commit`` could not load either.
    """

    if not kernels:
        return {}
    files = directory / FILES_DIR
    await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
    files.mkdir(parents=True)
    await repo.checkout(commit, files)
    try:
        laid = [await asyncio.to_thread(_lay, kernel, files / kernel.ref, directory, configs) for kernel in kernels]
    except (KernelLoadError, RowsOnlyRefused, OSError) as error:
        raise StartTableError(f"the start commit's kernel tables cannot be laid out: {error}") from error
    return dict(laid)


def _lay(kernel: KernelArtifact, file: Path, directory: Path, configs: Path) -> tuple[str, StartTable]:
    env = kernel_load_env(kernel)
    if env in PER_DEVICE_TABLE_ENVS:
        # PyTorch reads the file of each device with its ordinal before the extension.
        for ordinal in DEVICE_ORDINALS:
            shutil.copyfile(file, file.with_name(f"{file.stem}{ordinal}{file.suffix}"))
    if not kernel_rows_only(kernel):
        return env, StartTable(str(file))
    variable = read_variables(configs.parent / "jit" / "core.py").get(env)
    defaults = [] if variable is None else [Path(path) for path in default_tables(configs.parent, variable)]
    defaults = [path for path in defaults if path.is_file()]
    if variable is None or not defaults:
        raise StartTableError(f"{kernel.ref}: the aiter install at {configs.parent} has no default table for {env}")
    over = lay_rows_over(defaults, [file], configs, variable.tuned)
    copies = []
    for path in defaults:
        copy = directory / env / path.relative_to(configs)
        copy.parent.mkdir(parents=True, exist_ok=True)
        if path in over.kept:
            copy.write_bytes(over.kept[path])
        else:
            shutil.copyfile(path, copy)
        copies.append(str(copy))
    return env, StartTable(os.pathsep.join([*copies, str(file)]), tuple(copies))


def from_start_tables(kept: KeptChange, tables: Mapping[str, StartTable]) -> KeptChange:
    """``kept`` without each variable it runs with as it was given, and with the private default tables any other
    value of such a variable lists read as the image's default tables."""

    envs: dict[str, str] = {}
    renamed: dict[str, str] = {}
    for name, value in kept.envs.items():
        table = tables.get(name)
        if table is not None and value == table.value:
            continue
        envs[name] = value if table is None else _over_image(value, table.defaults)
        renamed[value] = envs[name]
    named = {renamed[value]: text for value, text in kept.tables.items() if value in renamed}
    return replace(kept, envs=envs, tables=named)


def _over_image(value: str, defaults: tuple[str, ...]) -> str:
    parts = [DEFAULT_TABLE_PLACEHOLDER if part in defaults else part for part in value.split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(parts))


def save_start_tables(tables: Mapping[str, StartTable], path: Path) -> None:
    document = {name: {"value": table.value, "defaults": list(table.defaults)} for name, table in tables.items()}
    path.write_text(json.dumps(document), encoding=ENCODING)


def load_start_tables(path: Path) -> dict[str, StartTable]:
    """The tables :func:`save_start_tables` wrote to ``path``; none when it wrote none there."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except FileNotFoundError:
        return {}
    return {name: StartTable(entry["value"], tuple(entry["defaults"])) for name, entry in document.items()}


__all__ = [
    "FILES_DIR",
    "StartTable",
    "StartTableError",
    "from_start_tables",
    "lay_start_tables",
    "load_start_tables",
    "save_start_tables",
]
