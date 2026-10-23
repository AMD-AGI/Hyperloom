# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A few tuned rows laid over the image's default table for one test, so a table need not be uploaded whole.

aiter merges every file an ``AITER_CONFIG_*`` variable lists (``AITER_CONFIG.update_config_files``), so a variable that
names the image's default table and a file of rows loads both with no merge of ours. What aiter does with a shape both
files carry is no merge: it rewrites the files it was given and raises, and the server never starts. A rows-only table
is meant to override, so the rows' shapes are taken out of the default table's copy before the test, and the test says
how many it took out. The columns that make two rows one shape are ``framework_tables.table_key``'s.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.measure.framework_tables import table_key

__all__ = ["DEFAULT_TABLE_PLACEHOLDER", "RowsOnlyRefused", "RowsOver", "lay_rows_over"]

#: Names, in the value of an ``AITER_CONFIG_*`` variable, the tables the image reads for it when it is unset.
DEFAULT_TABLE_PLACEHOLDER = "${HL_IMAGE_DEFAULT_TABLE}"
#: aiter infers it from ``cu_num`` for a table that has none, so a rows file may leave it out.
_INFERRED_COLUMN = "gfx"


class RowsOnlyRefused(ValueError):
    """The rows cannot be laid over the default table: they lack a column that identifies a shape, or repeat one."""


@dataclass(frozen=True)
class RowsOver:
    """What laying the rows over the default tables did: the shape key, the rows, and what each default lost."""

    key: tuple[str, ...] | None
    rows: int
    #: Each default table that lost rows, with its text as the test reads it.
    kept: dict[Path, bytes] = field(default_factory=dict)
    #: How many rows each of those tables lost, by file name.
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def overridden(self) -> int:
        return sum(self.dropped.values())

    def document(self) -> dict[str, Any]:
        """What ``framework_base.json`` says of the variable."""
        return {
            "key": None if self.key is None else list(self.key),
            "rows": self.rows,
            "overridden": self.overridden,
            "dropped": dict(self.dropped),
        }


@dataclass(frozen=True)
class _Table:
    header: list[str]
    rows: list[list[str]]


def _read(path: Path) -> _Table:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return _Table(header, [row for row in reader if row])


def _cell(text: str) -> str:
    """One cell as pandas, which aiter reads the tables with, compares it: ``64`` is ``64.0``, ``False`` is ``false``."""
    value = text.strip()
    if value.lower() in ("true", "false"):
        return value.lower()
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else repr(number)


def _project(row: list[str], indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(_cell(row[i]) if i < len(row) else "" for i in indices)


def _indices(header: list[str], columns: Sequence[str]) -> list[int]:
    return [header.index(name) for name in columns]


def _check(path: Path, table: _Table, key: tuple[str, ...]) -> None:
    """The rows must say every column but ``gfx`` that makes a shape, and name no shape twice."""
    missing = [name for name in key if name not in table.header and name != _INFERRED_COLUMN]
    if missing:
        raise RowsOnlyRefused(
            f"the rows {path.name} lack the column(s) {missing} that identify a shape in this table (key: "
            f"{list(key)}); aiter would not match them to the default table's rows, so they would never be read"
        )
    columns = [name for name in key if name in table.header]
    seen = Counter(_project(row, _indices(table.header, columns)) for row in table.rows)
    repeated = next((shape for shape, count in seen.items() if count > 1), None)
    if repeated is not None:
        raise RowsOnlyRefused(
            f"the rows {path.name} carry the shape {dict(zip(columns, repeated, strict=True))} more than once; "
            "aiter refuses a table that lists a shape twice, so give each shape one row"
        )


def _text(table: _Table, rows: list[list[str]]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(table.header)
    writer.writerows(rows)
    return out.getvalue().encode()


def lay_rows_over(defaults: Sequence[Path], rows: Sequence[Path], configs: Path, tuned: str | None) -> RowsOver:
    """The default tables as the test reads them once the shapes in ``rows`` are taken out of them.

    ``configs`` is the install's ``aiter/configs`` and ``tuned`` the name aiter merges the variable's tables under
    (``None`` for a variable the install does not declare). Where aiter checks no key there is nothing to take out.

    Raises:
        RowsOnlyRefused: When a rows file lacks a column of the key, or names one shape twice.
        OSError: When a table cannot be read.
    """
    loaded = {path: _read(path) for path in (*defaults, *rows)}
    count = sum(len(loaded[path].rows) for path in rows)
    columns = {name for table in loaded.values() for name in table.header}
    key = None if tuned is None else table_key(configs, tuned, columns)
    if key is None:
        return RowsOver(None, count)
    for path in rows:
        _check(path, loaded[path], key)
    kept: dict[Path, bytes] = {}
    dropped: dict[str, int] = {}
    for path in defaults:
        table = loaded[path]
        taken = _shapes_taken(table, [loaded[source] for source in rows], key)
        left = [row for index, row in enumerate(table.rows) if index not in taken]
        if taken:
            kept[path] = _text(table, left)
            dropped[path.name] = len(taken)
    return RowsOver(key, count, kept, dropped)


def _shapes_taken(default: _Table, rows: list[_Table], key: tuple[str, ...]) -> set[int]:
    """The positions of the default's rows whose shape one of the ``rows`` tables has, by the columns both carry."""
    taken: set[int] = set()
    for table in rows:
        shared = [name for name in key if name in table.header and name in default.header]
        wanted = {_project(row, _indices(table.header, shared)) for row in table.rows}
        by_default = _indices(default.header, shared)
        taken.update(index for index, row in enumerate(default.rows) if _project(row, by_default) in wanted)
    return taken
