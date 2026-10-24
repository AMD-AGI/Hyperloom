# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tuned rows of one Forge tuner, as the rows-only table ``put_kernel_artifact`` takes, each checked against the
reference and none handed over unchecked.

A row is a measurement: a kernel, its parameters and the microseconds it took. aiter's tuner compares the output of every
candidate kernel with the output of a reference GEMM of the same shape (``checkAllclose``: element by element within
``rtol`` and ``atol``, 1e-2 unless the tuner sets its own, a catastrophic deviation counting as every element) and writes the fraction of elements
outside the tolerance as the row's ``errRatio`` (``err_ratio`` in the bf16 table). Forge drops the rows whose ratio is
above its limit and refuses a table whose rows carry none. This module does not rely on that alone: a row is handed over
only when its table has the column and its own figure is a finite number within :data:`ERR_RATIO_LIMIT` (or, for the one
kind of row described below, its relative-norm error is within :data:`REL_NORM_ERR_LIMIT`). A row that fails or cannot be
checked is returned as such, with the reason, and is never in the CSV.

The CSV is keyed as aiter keys the variable's table (``framework.tables.table_key``): one row to a shape (the fastest,
if the tuner left several), in the numeric order of the key, so the same rows are the same bytes and the same artifact
whichever branch puts them.

One kind of row is checked by another figure. An asm kernel at a splitK above 1 accumulates its partial results in bf16 with
atomics, which puts more elements outside the reference's tolerance than any other row has while the output as a whole stays
accurate, so Forge measures such a row by the relative norm of its error as well and writes it to the table's ``relNormErr``
column. A row of that kind whose own column holds a finite figure within :data:`REL_NORM_ERR_LIMIT` is verified by it
(``check`` is :data:`CHECK_REL_NORM_ERR`, and both figures are on the row); a row with the figure blank, not a number or above
the limit, and every other row, is judged by its ratio alone. The column is not one of the image's table, so it is left out
of the rows handed over.

A call may allow a higher ratio (up to :data:`MAX_ERR_RATIO_LIMIT`, aiter's own filter). The rows Forge then took out
of the table for a ratio above the default and within the allowed one are not handed over as checked: they come back in
a CSV of their own, each marked ``verified`` false with the reason :data:`RELAXED_REASON` and its ratio, and never in the
default CSV. Forge's report names such a row by shape, libtype, splitK, microseconds and ratio, not by kernel, so the
full row is read back from the profile of every candidate the tuner measured; a row that cannot be found there, or
matches several candidates, stays rejected. Whether such rows stand is for an end-to-end accuracy check to say.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import (
    ASM_LIBTYPE,
    FORGE_REL_NORM_ERR_COLUMN,
    TUNED_LIBTYPE_COLUMN,
    TUNED_SPLITK_COLUMN,
)
from hyperloom.measure.framework.rows import cell
from hyperloom.measure.framework.tables import read_variables, table_key
from hyperloom_multibranch.toolbox.kernel_artifacts import shape_order

__all__ = [
    "CHECK_ERR_RATIO",
    "CHECK_REL_NORM_ERR",
    "ERR_COLUMNS",
    "ERR_RATIO_LIMIT",
    "MAX_ERR_RATIO_LIMIT",
    "RELAXED_REASON",
    "REL_NORM_ERR_LIMIT",
    "Relax",
    "RowsRefused",
    "TunedRows",
    "dropped_by_forge",
    "from_artifact",
    "split_dropped",
]

#: The columns that hold aiter's mismatch ratio of a row, by the tables that name it so.
ERR_COLUMNS = ("errRatio", "err_ratio")
#: The most a row's mismatch ratio may be: Forge's own limit for a dense table (aiter's tuner filters at 0.05).
ERR_RATIO_LIMIT = 0.01
#: The most a call may allow: aiter's tuner filters its candidates at it, so no row above it was ever measured as usable.
MAX_ERR_RATIO_LIMIT = 0.05
#: The most a relative-norm error may be for it to stand in for a mismatch ratio above the limit: Forge's own bound.
REL_NORM_ERR_LIMIT = 0.01
#: What a verified row says decided it, and the name of the figure that did: its mismatch ratio, or its relative-norm error.
CHECK_ERR_RATIO = "err_ratio"
CHECK_REL_NORM_ERR = "rel_norm_err"
#: Why a relaxed row is not verified.
RELAXED_REASON = "err_ratio_above_default"
_US = "us"
#: How close a profile row's microseconds and ratio must be to the figures Forge's report names for the same row.
_US_TOLERANCE = 1e-3
_RATIO_TOLERANCE = 1e-6
_SHAPE = ("M", "N", "K")
_TUNED_SUFFIX = "_tuned_gemm"
_UTF8 = "utf-8"


class RowsRefused(ValueError):
    """The tuner's table cannot be handed over as rows of the variable's table."""


@dataclass(frozen=True)
class TunedRows:
    """What one tuner's table comes to: the rows-only CSV, and what was checked of every row it holds or left out.

    ``verified`` are the rows of ``csv``, each as the table has it plus ``verified`` (true), its ``err_ratio`` and the
    ``check`` that verified it (:data:`CHECK_ERR_RATIO`, or :data:`CHECK_REL_NORM_ERR` with its ``rel_norm_err`` too);
    ``refused`` are the rows left out, each with ``verified`` false and the ``reason``. ``target`` is the target the
    table is put under (:func:`suggested_target`). ``relaxed`` are the rows of ``relaxed_csv``, each with ``verified``
    false, its ``err_ratio`` and :data:`RELAXED_REASON`; ``relax_limit`` is the highest ratio the call allowed.
    """

    csv: str
    verified: tuple[Mapping[str, Any], ...]
    refused: tuple[Mapping[str, Any], ...]
    target: str
    relaxed_csv: str = ""
    relaxed: tuple[Mapping[str, Any], ...] = ()
    relax_limit: float = ERR_RATIO_LIMIT


@dataclass(frozen=True)
class _Row:
    record: Mapping[str, str]
    ratio: float
    #: The relative-norm error that verified the row, when its ratio above the limit did not.
    rel_norm: float | None = None

    @property
    def check(self) -> str:
        return CHECK_ERR_RATIO if self.rel_norm is None else CHECK_REL_NORM_ERR

    @property
    def us(self) -> float:
        try:
            return float(self.record.get(_US, ""))
        except ValueError:
            return math.inf


@dataclass
class _Checked:
    kept: list[_Row] = field(default_factory=list)
    refused: list[Mapping[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Relax:
    """What a call allowed beyond the default ratio: the highest ratio it takes, the rows Forge took out of the table for a
    ratio above the default and within it (as :func:`dropped_by_forge` lists them), and the files of the profile of every
    candidate the tuner measured (each tuner names its own), which the full rows are read back from."""

    limit: float
    dropped: tuple[Mapping[str, Any], ...]
    profiles: tuple[Path, ...]


def suggested_target(tuned: str) -> str:
    """The target a table is put under, from the name aiter merges the variable's tables under (``gemm_bf16``)."""
    return "gemm_" + tuned.removesuffix(_TUNED_SUFFIX)


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open(newline="", encoding=_UTF8) as handle:
            header, *rows = csv.reader(handle)
    except (OSError, ValueError, csv.Error) as exc:
        raise RowsRefused(f"the tuned table {path.name} cannot be read: {exc}") from None
    return header, [row for row in rows if row]


def _shape_of(record: Mapping[str, str]) -> dict[str, str]:
    return {name: record[name] for name in _SHAPE if name in record}


def _refusal(record: Mapping[str, str], why: str, ratio: float | None = None) -> Mapping[str, Any]:
    return {**_shape_of(record), "verified": False, "err_ratio": ratio, "reason": why}


def _asm_split_k(record: Mapping[str, str]) -> bool:
    """Whether an asm kernel at a splitK above 1 runs the row: the rows whose ratio bf16 accumulation moves."""
    try:
        split_k = int(float(record.get(TUNED_SPLITK_COLUMN, "")))
    except (ValueError, OverflowError):
        return False
    return record.get(TUNED_LIBTYPE_COLUMN, "").strip().lower() == ASM_LIBTYPE.lower() and split_k > 1


def _rel_norm(record: Mapping[str, str]) -> float | None:
    """The relative-norm error that may stand in for the row's ratio: the finite figure, from 0 to
    :data:`REL_NORM_ERR_LIMIT`, in the column of an asm row at a splitK above 1. None for another row, and for a column
    that is missing, blank, not a number or above the limit."""
    if not _asm_split_k(record):
        return None
    try:
        figure = float(record[FORGE_REL_NORM_ERR_COLUMN])
    except (KeyError, ValueError):
        return None
    return figure if math.isfinite(figure) and 0 <= figure <= REL_NORM_ERR_LIMIT else None


def _rel_norm_note(record: Mapping[str, str]) -> str:
    """What a refusal says of the figure a row that could have been verified by it holds, if it holds one."""
    figure = record.get(FORGE_REL_NORM_ERR_COLUMN, "").strip()
    if not figure or not _asm_split_k(record):
        return ""
    held = f"its relative-norm error {figure[:32]!r}"
    return f"; {held} is not a figure from 0 to {REL_NORM_ERR_LIMIT:g}, so it does not stand in"


def _judge(record: Mapping[str, str], column: str) -> _Row | Mapping[str, Any]:
    """The row, kept with its figure, or the refusal that says why its ratio does not let it through."""
    try:
        ratio = float(record[column])
    except (KeyError, ValueError):
        ratio = math.nan
    if not math.isfinite(ratio) or ratio < 0:
        return _refusal(record, f"the row's {column} is not a ratio from 0 up: the row was not checked")
    if ratio <= ERR_RATIO_LIMIT:
        return _Row(record, ratio)
    if (figure := _rel_norm(record)) is not None:
        return _Row(record, ratio, figure)
    why = f"{ratio:.4g} of the output elements are outside the reference's tolerance, above {ERR_RATIO_LIMIT:g}"
    return _refusal(record, why + _rel_norm_note(record), ratio)


def _check(header: Sequence[str], rows: Sequence[Sequence[str]]) -> _Checked:
    """Every row with a finite mismatch ratio within the limit, or an asm splitK row whose relative-norm error is within
    its own, is kept; the others are refused, each with its reason."""
    column = next((name for name in ERR_COLUMNS if name in header), None)
    checked = _Checked()
    for row in rows:
        record = dict(zip(header, row, strict=False))
        if column is None:
            checked.refused.append(
                _refusal(
                    record, f"the table has no accuracy column ({' or '.join(ERR_COLUMNS)}): the row was not checked"
                )
            )
            continue
        judged = _judge(record, column)
        if isinstance(judged, _Row):
            checked.kept.append(judged)
        else:
            checked.refused.append(judged)
    return checked


def _fastest(kept: Sequence[_Row], key: Sequence[str]) -> dict[tuple[str, ...], _Row]:
    """The fastest row of each shape (a table that lists a shape twice is one aiter refuses to load)."""
    best: dict[tuple[str, ...], _Row] = {}
    for row in kept:
        shape = tuple(cell(row.record.get(name, "")) for name in key)
        if shape not in best or row.us < best[shape].us:
            best[shape] = row
    return best


def _text(header: Sequence[str], rows: Sequence[_Row]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([row.record.get(name, "") for name in header] for row in rows)
    return out.getvalue()


def dropped_by_forge(run: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The rows Forge took out of the table of one entry of its ``tuners_run`` because aiter's reference check found them
    wrong, as rejected rows (they are not in the table any more, and were measured, so they are said)."""
    dropped = run.get("dropped_inaccurate")
    return tuple(
        {
            **{name: row.get(name) for name in (*_SHAPE, TUNED_LIBTYPE_COLUMN, TUNED_SPLITK_COLUMN, _US)},
            "verified": False,
            "err_ratio": row.get("err_ratio"),
            "reason": "aiter's reference check measured it wrong and Forge removed it from the table",
        }
        for row in (dropped if isinstance(dropped, list) else ())
        if isinstance(row, Mapping)
    )


def split_dropped(
    dropped: Sequence[Mapping[str, Any]], limit: float
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """The rows Forge dropped, as (those whose ratio is above the default and within ``limit``, all the others)."""
    within = tuple(row for row in dropped if _within(row.get("err_ratio"), limit))
    return within, tuple(row for row in dropped if row not in within)


def _within(ratio: Any, limit: float) -> bool:
    return isinstance(ratio, (int, float)) and math.isfinite(ratio) and ERR_RATIO_LIMIT < ratio <= limit


def _profile(paths: Sequence[Path]) -> list[dict[str, str]]:
    """The candidates the tuner measured, one record each; a file that is missing or cannot be read has none."""
    records: list[dict[str, str]] = []
    for path in paths:
        try:
            with path.open(newline="", encoding=_UTF8) as handle:
                records.extend(csv.DictReader(handle))
        except (OSError, ValueError, csv.Error):
            continue
    return records


def _same_row(record: Mapping[str, str], dropped: Mapping[str, Any], column: str) -> bool:
    """Whether the profile ``record`` is the row Forge's report names: the shape, libtype and splitK, and the microseconds
    and ratio to the tolerance a rounded figure keeps."""
    try:
        same_shape = all(
            int(float(record[name])) == int(float(dropped[name])) for name in (*_SHAPE, TUNED_SPLITK_COLUMN)
        )
        lib = dropped.get(TUNED_LIBTYPE_COLUMN)
        same_lib = lib is None or record[TUNED_LIBTYPE_COLUMN].strip() == str(lib).strip()
        same_us = abs(float(record[_US]) - float(dropped[_US])) <= _US_TOLERANCE
        same_ratio = abs(float(record[column]) - float(dropped["err_ratio"])) <= _RATIO_TOLERANCE
    except (KeyError, ValueError, TypeError):
        return False
    return same_shape and same_lib and same_us and same_ratio


def _read_back(
    dropped: Mapping[str, Any], profile: Sequence[Mapping[str, str]], header: Sequence[str], column: str
) -> _Row | str:
    """The full row of the profile that the dropped row is, or why it cannot be told."""
    if not column:
        return f"the table has no accuracy column ({' or '.join(ERR_COLUMNS)})"
    if not profile:
        return "the tuner's profile of its candidates is missing or cannot be read"
    found = [dict(record) for record in profile if _same_row(record, dropped, column)]
    distinct = [record for index, record in enumerate(found) if record not in found[:index]]
    if not distinct:
        return "no candidate of the tuner's profile has its shape, libtype, splitK, microseconds and ratio"
    if len(distinct) > 1:
        return f"{len(distinct)} different candidates of the tuner's profile have its shape, libtype, splitK, microseconds and ratio"
    if missing := [name for name in header if name not in distinct[0]]:
        return f"the tuner's profile row lacks the table's column(s) {missing}"
    return _Row({name: distinct[0][name] for name in header}, float(dropped["err_ratio"]))


def _relaxed(relax: Relax, header: Sequence[str], column: str | None) -> _Checked:
    """The rows within the allowed ratio, read back from the profile; the ones that cannot be read back are refused."""
    checked = _Checked()
    profile = _profile(relax.profiles) if relax.dropped else []
    for dropped in relax.dropped:
        found = _read_back(dropped, profile, header, column or "")
        if isinstance(found, _Row):
            checked.kept.append(found)
            continue
        checked.refused.append(
            {
                **dropped,
                "reason": f"its ratio {dropped['err_ratio']:.4g} is within the allowed {relax.limit:g}, but the row "
                f"cannot be read back, so it is not handed over: {found}",
            }
        )
    return checked


def _verified(row: _Row) -> Mapping[str, Any]:
    figure = {} if row.rel_norm is None else {CHECK_REL_NORM_ERR: row.rel_norm}
    return {**row.record, "verified": True, "check": row.check, CHECK_ERR_RATIO: row.ratio, **figure}


def _flagged(rows: Sequence[_Row]) -> tuple[Mapping[str, Any], ...]:
    return tuple({**row.record, "verified": False, "err_ratio": row.ratio, "reason": RELAXED_REASON} for row in rows)


def from_artifact(artifact: Path, env: str, configs: Path, relax: Relax | None = None) -> TunedRows:
    """The rows of the tuned table ``artifact``, which tunes the variable ``env`` of the aiter install at ``configs``, and,
    when the call allowed a higher ratio (``relax``), the rows within it that Forge took out of the table.

    Raises:
        RowsRefused: When the table cannot be read, or the install declares no shape key for ``env`` (aiter checks none
            there, so rows cannot be laid over its default table).
    """
    header, rows = _read(artifact)
    variable = read_variables(configs.parent / "jit" / "core.py").get(env)
    key = None if variable is None else table_key(configs, variable.tuned, header)
    if variable is None or not key:
        raise RowsRefused(
            f"the aiter install at {configs.parent} declares no shape key for {env} (no untuned table to read it from), "
            "so the rows cannot be laid over the variable's default table"
        )
    checked = _check(header, rows)
    verified = _sorted_fastest(checked.kept, key)
    column = next((name for name in ERR_COLUMNS if name in header), None)
    columns = [name for name in header if name != FORGE_REL_NORM_ERR_COLUMN]
    loose = _relaxed(relax, columns, column) if relax is not None else _Checked()
    relaxed = _sorted_fastest(loose.kept, key)
    return TunedRows(
        _text(columns, verified),
        tuple(_verified(row) for row in verified),
        (*checked.refused, *loose.refused),
        suggested_target(variable.tuned),
        _text(columns, relaxed) if relaxed else "",
        _flagged(relaxed),
        ERR_RATIO_LIMIT if relax is None else relax.limit,
    )


def _sorted_fastest(kept: Sequence[_Row], key: Sequence[str]) -> list[_Row]:
    """The fastest row of each shape, in the numeric order of the key."""
    best = _fastest(kept, key)
    return [best[shape] for shape in sorted(best, key=shape_order)]
