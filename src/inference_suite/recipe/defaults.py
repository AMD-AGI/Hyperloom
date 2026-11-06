# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The default table: the fixed-workload launch a framework gets on an architecture when no recipe serves it (D5).

The table is ``defaults/table.yaml``, one row per framework and set of architectures, each naming the launcher lines
it was read from. A row carries no topology: the resolver writes the query's TP and EP in the framework's own spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml

from hyperloom_common.contract_fields import ContractFields
from inference_suite.argv import normalize_argv
from inference_suite.recipe.model import RecipeSourceError

TABLE = "defaults/table.yaml"
#: Where the table lives in the repository, as a default-table base names its source.
TABLE_PATH = f"src/inference_suite/recipe/{TABLE}"

_FIELDS = ContractFields(RecipeSourceError)


@dataclass(frozen=True)
class DefaultRow:
    """The launch the table gives ``framework`` on the architectures ``gfx``, read from ``origin``."""

    framework: str
    gfx: frozenset[str]
    args: tuple[str, ...]
    env: Mapping[str, str]
    origin: str


@dataclass(frozen=True)
class DefaultTable:
    """The table's rows, at most one per framework and architecture."""

    rows: tuple[DefaultRow, ...]

    def row(self, framework: str, gfx: str) -> DefaultRow | None:
        return next((row for row in self.rows if row.framework == framework and gfx in row.gfx), None)


def _row(document: object) -> DefaultRow:
    fields = _FIELDS.known(document, ("framework", "gfx", "args", "env", "origin"), "a default row")
    return DefaultRow(
        framework=_FIELDS.text(fields.get("framework"), "a default row's framework"),
        gfx=frozenset(_FIELDS.strings(fields.get("gfx"), "a default row's architectures")),
        args=normalize_argv(_FIELDS.strings(fields.get("args"), "a default row's args")),
        env=dict(_FIELDS.string_map(fields.get("env"), "a default row's env")),
        origin=_FIELDS.text(fields.get("origin"), "a default row's origin"),
    )


@cache
def default_table() -> DefaultTable:
    """The table shipped as :data:`TABLE` beside this module."""
    text = resources.files(__name__.rpartition(".")[0]).joinpath(TABLE).read_text(encoding="utf-8")
    fields = _FIELDS.known(yaml.safe_load(text), ("rows",), TABLE)
    rows = tuple(map(_row, _FIELDS.items(fields.get("rows"), f"the rows of {TABLE}")))
    claimed = [(row.framework, gfx) for row in rows for gfx in row.gfx]
    _FIELDS.unique(claimed, f"the framework and architecture of the rows of {TABLE}")
    return DefaultTable(rows)


__all__ = ["TABLE", "TABLE_PATH", "DefaultRow", "DefaultTable", "default_table"]
