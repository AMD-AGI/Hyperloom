# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How many rows of each kernel table a test loaded its server actually used, and what that means (#109 item 4).

A tuned table that changes nothing a test ran gives a flat result, and a flat result looks the same whether the table
was read and did not matter or was never read. The result says which: for each kernel table the test's recipe loads,
how many of the table's rows aiter resolved a lookup in, counted from the test's server logs.

aiter logs a lookup that missed unconditionally and a lookup that hit only under ``AITER_LOG_TUNED_CONFIG``, so the
count has three outcomes and a reader must not fold them into two:

* ``exercised``: a hit line of the server names a row of the table;
* ``not_exercised``: every server logged its hits (every point sets the variable, or every server log holds a hit line of
  some table) and none is a row of this table, so the table changed nothing this test ran;
* ``not_observable``: the logs cannot say (no hit logging, no log, a log or a table this reader cannot read), so nothing
  is claimed. This is the outcome of any doubt: "not exercised" is only said when the evidence is complete.

A hit line names the row aiter found by the columns it prints (``AITER_TUNED_HIT`` in the rule pack; the shape is the
padded one it found, not the one it was asked for), and a row of the table is hit when every column the line prints is the
row's. A line is not counted for a table when it names the private copy of another table: the copy of a table lives at
``tables/<variable>/<digest>/<file>``, and a variable that lists several tables (a rows-only put over the image's
default table) is read from the merged view, which names no table and is told apart by its rows. Rows put with
``rows_only`` are the whole table here, so the image's own rows are never counted; a whole table put with the image's rows
in it counts a hit of one of them as a hit.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hyperloom.common.outside_facts import AITER_LOG_TUNED_CONFIG_ENV, AITER_TUNED_HIT, AITER_TUNED_HIT_MARKER
from hyperloom.contracts.measurement import (
    TABLE_EXERCISED,
    TABLE_NOT_EXERCISED,
    TABLE_NOT_OBSERVABLE,
    TableHits,
)
from hyperloom.contracts.recipe import KernelArtifact
from hyperloom.measure.compile import BlobError, CompiledEnvironment, LocalBlobStore, kernel_load_env
from hyperloom.measure.farm import server_environment
from hyperloom.measure.framework.rows import cell
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX

__all__ = ["describe", "read_table_hits", "runtime_findings"]

_HIT = re.compile(AITER_TUNED_HIT)
_HIT_MARKER = re.compile(AITER_TUNED_HIT_MARKER)
#: The group of ``AITER_TUNED_HIT`` that names the file aiter looked in; every other group is a column of the row it found.
_FILE_GROUP = "table"
#: The columns that make an aiter GEMM lookup: the table's rows are found by them, and a table without them is not one
#: whose hits this reader knows the lines of.
_SHAPE = ("M", "N", "K")
#: ``tables/<variable>/<digest>/<file>``: where a test's server finds the private copy of a table the test names
#: (``hyperloom.measure.framework.copies``). A path that does not match is another file, the merged view for one.
_PRIVATE_COPY = re.compile(rf"/tables/({AITER_VARIABLE_PREFIX}\w+)/([0-9a-f]+)/")
_SERVER_LOGS = "server*.log"
_UTF8 = "utf-8"

_WHY_PACK = "the rule pack's aiter_tuned_hit pattern has no group {groups}, so no hit line can be read"
_WHY_NOT_AITER = (
    "hits of a table loaded through {env} are not read: only the hits of an aiter GEMM table ({prefix}*) are logged in "
    "a form this reader knows"
)
_WHY_NO_SHAPE = "the table has no {columns} columns, so its hits cannot be told: only a dense GEMM table is read"
_WHY_UNREADABLE = "the table cannot be read as rows: {why}"
_WHY_NO_LOG = "the test landed no server log"
_WHY_LOG_UNREAD = "{count} server log(s) could not be read"
_WHY_LINES_UNREAD = "{count} log line(s) report a tuned lookup in a form this reader cannot read"
_WHY_NOT_LOGGED = (
    "not every server log of the test holds a hit line of any table (for instance, an engine that does not print "
    "aiter's tuned-lookup hits), so a table with no hit here may have been hit unseen"
)
_WHY_ELSEWHERE = "{count} aiter lookup(s) hit rows of other tables and none hit a row of this one"
_WHY_NO_HITS = "aiter logged no hit at all"


@dataclass(frozen=True)
class _Table:
    """One kernel table of the test, read as rows whose cells are as aiter compares them."""

    artifact: KernelArtifact
    env: str
    rows: tuple[Mapping[str, str], ...]
    #: The positions of the rows of each shape, so a hit line is looked up and not searched for.
    by_shape: Mapping[tuple[str, ...], tuple[int, ...]]

    @classmethod
    def of(cls, artifact: KernelArtifact, env: str, rows: tuple[Mapping[str, str], ...]) -> _Table:
        shapes: dict[tuple[str, ...], list[int]] = {}
        for position, row in enumerate(rows):
            shapes.setdefault(tuple(row.get(column, "") for column in _SHAPE), []).append(position)
        return cls(artifact, env, rows, {shape: tuple(positions) for shape, positions in shapes.items()})

    def hit_by(self, printed: Mapping[str, str], path: str) -> list[int]:
        """The rows of the table that a hit line of ``path`` printing ``printed`` says aiter resolved the lookup in."""
        copy = _PRIVATE_COPY.search(path)
        if copy is not None and (copy.group(1) != self.env or not self.artifact.sha256.startswith(copy.group(2))):
            return []
        shape = tuple(printed[column] for column in _SHAPE)
        return [
            index
            for index in self.by_shape.get(shape, ())
            if all(self.rows[index].get(column, value) == value for column, value in printed.items())
        ]


@dataclass
class _Tally:
    """What the logs said of one table."""

    rows: set[int] = field(default_factory=set)
    hit_lines: int = 0


@dataclass
class _Scan:
    """What a pass over the server logs found: per table (by artifact reference), and what no table is credited with.

    ``logs`` is how many server logs there were, ``logs_with_hits`` how many of them hold a hit line of some table, and
    ``requested`` whether every point of the test asks its server for hits.
    """

    tallies: dict[str, _Tally]
    logs: int
    requested: bool
    hit_lines: int = 0
    logs_with_hits: int = 0
    unread_lines: int = 0
    unread_logs: int = 0

    @property
    def hits_logged(self) -> bool:
        """Whether every server logged its hits: all were asked to, or each log shows a hit line of some table.

        One server that did not log its hits can have hit the table unseen, however many hits another logged.
        """
        return self.requested or (self.logs > 0 and self.logs_with_hits == self.logs)


def read_table_hits(environment: object, points: Sequence[Path], output_dir: Path) -> tuple[TableHits, ...]:
    """The hits of each kernel table the recipe of ``environment`` loaded, from the server logs below ``output_dir``.

    ``points`` are the test's measurement points: when every one sets ``AITER_LOG_TUNED_CONFIG`` its server logs its
    hits, so a table that no hit line names was not exercised; with fewer to go on (a point that does not set it, a
    server log with no hit line) it was not observed. Nothing here raises for a log or a table that cannot be read: that
    is ``not_observable``, with the reason.
    """
    if not isinstance(environment, CompiledEnvironment) or not environment.recipe.kernels:
        return ()
    if missing := _missing_groups(_HIT):
        why = _WHY_PACK.format(groups=" or ".join(missing))
        return tuple(_unobserved(kernel, kernel_load_env(kernel), why) for kernel in environment.recipe.kernels)
    read = [_load(artifact, environment.blob_root) for artifact in environment.recipe.kernels]
    tables = [item for item in read if isinstance(item, _Table)]
    scan = _scan(_logs(output_dir) if tables else [], tables, _hit_logging_requested(points))
    return tuple(_verdict(item, scan) if isinstance(item, _Table) else item for item in read)


def _missing_groups(pattern: re.Pattern[str]) -> list[str]:
    """The groups the reader needs of a hit pattern that it lacks: a pack that names none is wrong, not silent."""
    return sorted({_FILE_GROUP, *_SHAPE} - set(pattern.groupindex))


def _unobserved(artifact: KernelArtifact, env: str, reason: str, rows: int | None = None) -> TableHits:
    return TableHits(env, artifact.target, artifact.sha256, TABLE_NOT_OBSERVABLE, rows=rows, reason=reason)


def _load(artifact: KernelArtifact, blob_root: Path) -> _Table | TableHits:
    """The table ``artifact`` holds as rows, or the ``not_observable`` entry that says why it is not read."""
    env = kernel_load_env(artifact)
    if not env.startswith(AITER_VARIABLE_PREFIX):
        return _unobserved(artifact, env, _WHY_NOT_AITER.format(env=env, prefix=AITER_VARIABLE_PREFIX))
    try:
        data = LocalBlobStore(blob_root).get(artifact.sha256).decode(_UTF8)
        records = [record for record in csv.reader(io.StringIO(data)) if record]
    except (BlobError, OSError, UnicodeDecodeError, csv.Error) as exc:
        return _unobserved(artifact, env, _WHY_UNREADABLE.format(why=type(exc).__name__))
    header, *body = records or [[]]
    if not set(_SHAPE) <= set(header):
        return _unobserved(artifact, env, _WHY_NO_SHAPE.format(columns="/".join(_SHAPE)), rows=len(body))
    rows = tuple({name: cell(value) for name, value in zip(header, record, strict=False)} for record in body)
    return _Table.of(artifact, env, rows)


def _logs(output_dir: Path) -> list[Path]:
    try:
        return sorted(Path(output_dir).rglob(_SERVER_LOGS))
    except OSError:
        return []


def _hit_logging_requested(points: Sequence[Path]) -> bool:
    """Whether the server of every point is launched with ``AITER_LOG_TUNED_CONFIG`` on (aiter reads it as an integer)."""
    if not points:
        return False
    for point in points:
        try:
            config = yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}
            value = server_environment(config).get(AITER_LOG_TUNED_CONFIG_ENV)
            if int(str(value)) == 0:
                return False
        except (OSError, yaml.YAMLError, AttributeError, ValueError):
            return False
    return True


def _scan(logs: Sequence[Path], tables: Sequence[_Table], requested: bool) -> _Scan:
    scan = _Scan({table.artifact.ref: _Tally() for table in tables}, len(logs), requested)
    for log in logs:
        before = scan.hit_lines
        try:
            with log.open(encoding=_UTF8, errors="replace") as handle:
                for line in handle:
                    _count(line, tables, scan)
        except OSError:
            scan.unread_logs += 1
        scan.logs_with_hits += scan.hit_lines > before
    return scan


def _count(line: str, tables: Sequence[_Table], scan: _Scan) -> None:
    if _HIT_MARKER.search(line) is None:
        return
    match = _HIT.search(line)
    if match is None:
        scan.unread_lines += 1
        return
    scan.hit_lines += 1
    printed = {name: cell(value) for name, value in match.groupdict().items() if name != _FILE_GROUP and value}
    for table in tables:
        rows = table.hit_by(printed, match.group(_FILE_GROUP))
        if rows:
            tally = scan.tallies[table.artifact.ref]
            tally.hit_lines += 1
            tally.rows.update(rows)


def _verdict(table: _Table, scan: _Scan) -> TableHits:
    """What the scan says of ``table``: a hit is a fact, a miss is one only when nothing could have hidden a hit."""
    artifact, tally = table.artifact, scan.tallies[table.artifact.ref]
    if tally.rows:
        return TableHits(
            table.env,
            artifact.target,
            artifact.sha256,
            TABLE_EXERCISED,
            rows=len(table.rows),
            rows_hit=len(tally.rows),
            hit_lines=tally.hit_lines,
        )
    gaps = _gaps(scan)
    if gaps or not scan.hits_logged:
        reason = "; ".join(gaps) or _WHY_NOT_LOGGED
        return _unobserved(artifact, table.env, reason, rows=len(table.rows))
    reason = _WHY_ELSEWHERE.format(count=scan.hit_lines) if scan.hit_lines else _WHY_NO_HITS
    return TableHits(
        table.env,
        artifact.target,
        artifact.sha256,
        TABLE_NOT_EXERCISED,
        rows=len(table.rows),
        rows_hit=0,
        hit_lines=0,
        reason=reason,
    )


def _gaps(scan: _Scan) -> list[str]:
    """Why a table with no hit may still have been hit: what the logs hold that was not read, or no log at all."""
    if not scan.logs:
        return [_WHY_NO_LOG]
    gaps = []
    if scan.unread_logs:
        gaps.append(_WHY_LOG_UNREAD.format(count=scan.unread_logs))
    if scan.unread_lines:
        gaps.append(_WHY_LINES_UNREAD.format(count=scan.unread_lines))
    return gaps


def describe(hits: TableHits) -> str:
    """One sentence for what ``hits`` means to the specialist that reads a flat result."""
    name = f"kernel table {hits.target} ({hits.env})"
    if hits.status == TABLE_EXERCISED:
        never = (
            f"; {hits.rows - hits.rows_hit} never hit" if hits.rows is not None and hits.rows_hit is not None else ""
        )
        return f"{name}: exercised, {hits.rows_hit} of {hits.rows} rows hit in {hits.hit_lines} log lines{never}"
    if hits.status == TABLE_NOT_EXERCISED:
        return (
            f"{name}: NOT EXERCISED, none of its {hits.rows} rows was hit ({hits.reason}); the table changed nothing "
            "this test ran, so a flat result says nothing about its rows"
        )
    return f"{name}: not observable, {hits.reason}; whether its rows were hit is unknown"


def runtime_findings(table_hits: Sequence[TableHits]) -> tuple[str, ...]:
    """The finding of each table, in the order of ``table_hits``."""
    return tuple(describe(hits) for hits in table_hits)
