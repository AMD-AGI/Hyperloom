# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The default table: the fixed-workload launch a framework gets on an architecture when no recipe serves it (D5).

The table is ``defaults/table.yaml``, one row per framework and set of architectures, each naming the launcher lines
it was read from. It also says how each framework spells its topology, so a launch the table or an operator's file
gives carries the query's TP and EP the way the framework reads them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml

from hyperloom_common.contract_fields import ContractFields
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value, normalize_argv
from inference_suite.contracts.base_launch import Topology
from inference_suite.recipe.model import RecipeSourceError

TABLE = "defaults/table.yaml"
#: Where the table lives in the repository, as a default-table base names its source.
TABLE_PATH = f"src/inference_suite/recipe/{TABLE}"

_FIELDS = ContractFields(RecipeSourceError)


def _size(value: str | None, what: str) -> int:
    try:
        return int(value or 1)
    except ValueError as error:
        raise RecipeSourceError(f"its {what} {value!r} is not a size") from error


@dataclass(frozen=True)
class TopologyFlags:
    """How a framework spells its topology: ``tp`` and a sized ``ep`` flag, or an ``ep_switch`` that spreads the
    experts over every rank (``tp`` times the ``dp`` replicas). The first spelling of each is the one written."""

    tp: tuple[str, ...]
    ep: tuple[str, ...] = ()
    ep_switch: tuple[str, ...] = ()
    dp: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _FIELDS.string_tuple(self.tp, "the tp spellings"):
            raise RecipeSourceError("a framework names at least one tp spelling")
        if bool(self.ep) == bool(self.ep_switch):
            raise RecipeSourceError("a framework sizes its experts with an ep flag or a switch, one of the two")

    @staticmethod
    def _names(spellings: Sequence[str]) -> frozenset[str]:
        return frozenset(map(canonical_flag_name, spellings))

    def read(self, args: Sequence[str]) -> Topology:
        """The topology ``args`` run, an unset size being 1."""
        occurrences = argv_flag_occurrences(args)
        tp = _size(last_alias_value(occurrences, self._names(self.tp)), "tp")
        if self.ep:
            return Topology(tp=tp, ep=_size(last_alias_value(occurrences, self._names(self.ep)), "ep"))
        if last_alias_value(occurrences, self._names(self.ep_switch)) is None:
            return Topology(tp=tp)
        return Topology(tp=tp, ep=tp * _size(last_alias_value(occurrences, self._names(self.dp)), "dp"))

    def argv(self, topology: Topology) -> tuple[str, ...] | str:
        """The flags that run ``topology`` on one replica, or why the framework cannot."""
        args = (self.tp[0], str(topology.tp))
        if topology.ep == 1:
            return args
        if self.ep:
            return (*args, self.ep[0], str(topology.ep))
        if topology.ep != topology.tp:
            return f"its experts span all {topology.tp} TP ranks, so it runs no ep {topology.ep}"
        return (*args, self.ep_switch[0])


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
    """Each framework's topology spellings and the table's rows; every row's framework spells its topology."""

    topology: Mapping[str, TopologyFlags]
    rows: tuple[DefaultRow, ...]

    def __post_init__(self) -> None:
        unspelled = sorted({row.framework for row in self.rows} - set(self.topology))
        if unspelled:
            raise RecipeSourceError(f"{TABLE} gives no topology spellings for {', '.join(unspelled)}")

    def row(self, framework: str, gfx: str) -> DefaultRow | None:
        return next((row for row in self.rows if row.framework == framework and gfx in row.gfx), None)


def _topology_flags(document: object, framework: str) -> TopologyFlags:
    keys = ("tp", "ep", "ep_switch", "dp")
    fields = _FIELDS.known(document, keys, f"the {framework} topology spellings")
    return TopologyFlags(*(_FIELDS.strings(fields.get(key, ()), f"the {framework} {key} spellings") for key in keys))


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
    fields = _FIELDS.known(yaml.safe_load(text), ("topology", "rows"), TABLE)
    topology = _FIELDS.mapping(fields.get("topology"), f"the topology spellings of {TABLE}")
    rows = tuple(map(_row, _FIELDS.items(fields.get("rows"), f"the rows of {TABLE}")))
    claimed = [(row.framework, gfx) for row in rows for gfx in row.gfx]
    _FIELDS.unique(claimed, f"the framework and architecture of the rows of {TABLE}")
    return DefaultTable({name: _topology_flags(value, name) for name, value in topology.items()}, rows)


__all__ = ["TABLE", "TABLE_PATH", "DefaultRow", "DefaultTable", "TopologyFlags", "default_table"]
