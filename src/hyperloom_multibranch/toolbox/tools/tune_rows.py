# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tuned rows of one Forge tuner, as the rows-only table ``put_kernel_artifact`` takes, each checked against the
reference and none handed over unchecked.

A row is a measurement: a kernel, its parameters and the microseconds it took. aiter's tuner compares the output of every
candidate kernel with the output of a reference GEMM of the same shape (``checkAllclose``: element by element within
``rtol`` and ``atol``, 1e-2 unless the tuner sets its own, a catastrophic deviation counting as every element) and writes the fraction of elements
outside the tolerance as the row's ``errRatio`` (``err_ratio`` in the bf16 table). Forge drops the rows whose ratio is
above its limit and refuses a table whose rows carry none. This module does not rely on that alone: a row is handed over
only when its table has the column and its own figure is a finite number within :data:`ERR_RATIO_LIMIT`. A row that fails
or cannot be checked is returned as such, with the reason, and is never in the CSV.

The CSV is keyed as aiter keys the variable's table (``framework.tables.table_key``): one row to a shape (the fastest,
if the tuner left several), in the numeric order of the key, so the same rows are the same bytes and the same artifact
whichever branch puts them.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.measure.framework.rows import cell
from hyperloom.measure.framework.tables import read_variables, table_key
from hyperloom_multibranch.toolbox.kernel_artifacts import shape_order

__all__ = [
    "ERR_COLUMNS",
    "ERR_RATIO_LIMIT",
    "RowsRefused",
    "TunedRows",
    "dropped_by_forge",
    "from_artifact",
]

#: The columns that hold aiter's mismatch ratio of a row, by the tables that name it so.
ERR_COLUMNS = ("errRatio", "err_ratio")
#: The most a row's mismatch ratio may be: Forge's own limit for a dense table (aiter's tuner filters at 0.05).
ERR_RATIO_LIMIT = 0.01
_US = "us"
_SHAPE = ("M", "N", "K")
_TUNED_SUFFIX = "_tuned_gemm"
_UTF8 = "utf-8"


class RowsRefused(ValueError):
    """The tuner's table cannot be handed over as rows of the variable's table."""


@dataclass(frozen=True)
class TunedRows:
    """What one tuner's table comes to: the rows-only CSV, and what was checked of every row it holds or left out.

    ``verified`` are the rows of ``csv``, each as the table has it plus ``verified`` (true) and its ``err_ratio``;
    ``refused`` are the rows left out, each with ``verified`` false and the ``reason``. ``target`` is the target the
    table is put under (:func:`suggested_target`).
    """

    csv: str
    verified: tuple[Mapping[str, Any], ...]
    refused: tuple[Mapping[str, Any], ...]
    target: str


@dataclass(frozen=True)
class _Row:
    record: Mapping[str, str]
    ratio: float

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


def _check(header: Sequence[str], rows: Sequence[Sequence[str]]) -> _Checked:
    """Every row with a finite mismatch ratio within the limit is kept; the others are refused, each with its reason."""
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
        try:
            ratio = float(record[column])
        except (KeyError, ValueError):
            ratio = math.nan
        if not math.isfinite(ratio) or ratio < 0:
            checked.refused.append(
                _refusal(record, f"the row's {column} is not a ratio from 0 up: the row was not checked")
            )
        elif ratio > ERR_RATIO_LIMIT:
            checked.refused.append(
                _refusal(
                    record,
                    f"{ratio:.4g} of the output elements are outside the reference's tolerance, above {ERR_RATIO_LIMIT:g}",
                    ratio,
                )
            )
        else:
            checked.kept.append(_Row(record, ratio))
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
            **{name: row.get(name) for name in (*_SHAPE, "libtype", "splitK", _US)},
            "verified": False,
            "err_ratio": row.get("err_ratio"),
            "reason": "aiter's reference check measured it wrong and Forge removed it from the table",
        }
        for row in (dropped if isinstance(dropped, list) else ())
        if isinstance(row, Mapping)
    )


def from_artifact(artifact: Path, env: str, configs: Path) -> TunedRows:
    """The rows of the tuned table ``artifact``, which tunes the variable ``env`` of the aiter install at ``configs``.

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
    best = _fastest(checked.kept, key)
    chosen = [best[shape] for shape in sorted(best, key=shape_order)]
    verified = tuple({**row.record, "verified": True, "err_ratio": row.ratio} for row in chosen)
    return TunedRows(_text(header, chosen), verified, tuple(checked.refused), suggested_target(variable.tuned))
