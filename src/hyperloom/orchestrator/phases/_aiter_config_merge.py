# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merging a GEMM tuning candidate's rows into the aiter table the runtime dispatches from.

A candidate names only the shapes it tuned. Serving it alone would un-tune every other
shape the runtime already had a row for, so a candidate reaches an e2e run only as the
complete table the two produce together.
"""

from __future__ import annotations

import csv
import logging as _logging
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

log = _logging.getLogger(__name__)

#: What an aiter dispatch table means by a column a source table never wrote.
_AITER_COLUMN_DEFAULTS: dict[str, str] = {"xbf16": "0", "run_1stage": "0", "ksplit": "0"}

#: How a tuned table's stem names the untuned table that defines its dispatch keys.
_TUNED_STEM_SUFFIX = re.compile(r"(?:_)?tuned$")


class _DispatchTable(NamedTuple):
    """A dispatch table as it is read off disk: its columns, and its rows."""

    columns: list[str]
    rows: list[dict[str, str]]


class _MergeSources(NamedTuple):
    """The complete runtime tables a candidate is merged into, and the directory they came from."""

    paths: list[Path]
    config_dir: Path | None


def _read_table(path: Path) -> _DispatchTable:
    """Read a dispatch table off disk."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return _DispatchTable(list(reader.fieldnames or []), [dict(row) for row in reader])


def _write_table(path: Path, columns: list[str], rows: list[dict[str, str]]) -> Path:
    """Write a dispatch table to ``path``, and return where it landed."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def untuned_table_stem(tuned_stem: str) -> str:
    """The stem of the untuned table that pairs with a tuned one."""
    if _TUNED_STEM_SUFFIX.search(tuned_stem):
        return _TUNED_STEM_SUFFIX.sub("_untuned", tuned_stem)
    return tuned_stem.replace("tuned", "untuned")


def runtime_config_sources(config_dirs: list[Path], *, filename: str, tuned_stem: str) -> _MergeSources:
    """The complete runtime tables the candidate has to be merged into before it can be served."""
    cache_dir = Path(os.environ.get("INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR", "/tmp/aiter_configs"))
    cached = cache_dir / filename
    if cached.is_file():
        return _MergeSources([cached], None)
    for config_dir in config_dirs:
        base_path = config_dir / filename
        model_paths = sorted(
            path
            for path in (config_dir / "model_configs").glob(f"*{tuned_stem}*.csv")
            if path.is_file() and "untuned" not in path.name
        )
        paths = ([base_path] if base_path.is_file() else []) + model_paths
        if paths:
            return _MergeSources(paths, config_dir)
    return _MergeSources([], None)


def _untuned_header(config_dir: Path | None, config_dirs: list[Path], untuned_stem: str) -> list[str]:
    """The columns the untuned table names, which is what makes a dispatch key authoritative."""
    if config_dir is None:
        config_dir = next((d for d in config_dirs if (d / f"{untuned_stem}.csv").is_file()), None)
    if config_dir is None:
        return []
    untuned_path = config_dir / f"{untuned_stem}.csv"
    return _read_table(untuned_path).columns if untuned_path.is_file() else []


def _merged_columns(source_columns: list[list[str]], candidate_columns: list[str]) -> list[str]:
    """Every column the merged table carries, with columns new to it placed before ``tflops``."""
    all_columns = list(source_columns[0])
    for columns in [*source_columns[1:], candidate_columns]:
        for column in columns:
            if column not in all_columns:
                insert_at = all_columns.index("tflops") if "tflops" in all_columns else len(all_columns)
                all_columns.insert(insert_at, column)
    return all_columns


def _column_value(row: Mapping[str, str], column: str) -> str:
    """A row's value for ``column``, defaulted where the table it came from has no such field."""
    value = row.get(column)
    return _AITER_COLUMN_DEFAULTS.get(column, "0") if value is None else value


def _rows_over_columns(rows: list[dict[str, str]], columns: list[str]) -> list[dict[str, str]]:
    """``rows`` restated over ``columns``, so tables with different fields can be concatenated."""
    return [{column: _column_value(row, column) for column in columns} for row in rows]


def _dispatch_key_columns(all_columns: list[str], untuned_header: list[str]) -> list[str]:
    """The columns that together say which dispatch a row tunes."""
    key_cols = [column for column in untuned_header if column in all_columns]
    if not key_cols:
        key_cols = [column for column in ("M", "N", "K") if column in all_columns]
    for column in ("gfx", "cu_num", "_tag"):
        if column in all_columns and column not in key_cols:
            key_cols.append(column)
    return key_cols


def _dispatch_key(row: Mapping[str, str], key_cols: list[str]) -> tuple[str, ...]:
    """The values that identify which dispatch a row tunes."""
    return tuple(row.get(column, "") for column in key_cols)


def _measured_microseconds(row: Mapping[str, str]) -> float:
    """A row's measured time, or infinity when it carries none a comparison could use."""
    try:
        return float(row.get("us", ""))
    except (TypeError, ValueError):
        return math.inf


def _fastest_row_per_dispatch(rows: list[dict[str, str]], key_cols: list[str]) -> list[dict[str, str]]:
    """Keep the fastest row per dispatch key; ties keep the first row seen, unreadable times sort last."""
    best: dict[tuple[str, ...], dict[str, str]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = _dispatch_key(row, key_cols)
        if key not in best:
            best[key] = row
            order.append(key)
        elif _measured_microseconds(row) < _measured_microseconds(best[key]):
            best[key] = row
    return [best[key] for key in order]


def _deduplicated_dispatch_rows(
    rows: list[dict[str, str]], key_cols: list[str], *, label: str, env_var: str, timed: bool
) -> list[dict[str, str]] | None:
    """One row per dispatch key, or ``None`` when duplicates cannot be resolved by measurement."""
    keys = [_dispatch_key(row, key_cols) for row in rows]
    if len(set(keys)) == len(keys):
        return rows
    if not timed:
        log.warning(
            "gemm E2E: %s has duplicate dispatch keys for %s but no 'us' column to select the fastest row",
            label,
            env_var,
        )
        return None
    deduplicated = _fastest_row_per_dispatch(rows, key_cols)
    log.info(
        "gemm E2E: removed %d duplicate %s row(s) for %s",
        len(rows) - len(deduplicated),
        label,
        env_var,
    )
    return deduplicated


def merge_candidate_into_runtime(
    candidate_path: Path, sources: _MergeSources, *, env_var: str, untuned_stem: str, config_dirs: list[Path]
) -> str | None:
    """Write the table the candidate's rows and the runtime's complete config produce together."""
    candidate = _read_table(candidate_path)
    source_tables = [_read_table(path) for path in sources.paths]
    all_columns = _merged_columns([table.columns for table in source_tables], candidate.columns)
    runtime_rows: list[dict[str, str]] = []
    for table in source_tables:
        runtime_rows.extend(_rows_over_columns(table.rows, all_columns))
    candidate_rows = _rows_over_columns(candidate.rows, all_columns)
    key_cols = _dispatch_key_columns(all_columns, _untuned_header(sources.config_dir, config_dirs, untuned_stem))
    if not key_cols:
        log.warning("gemm E2E: cannot derive dispatch keys for %s", env_var)
        return None
    timed = "us" in all_columns
    kept = _deduplicated_dispatch_rows(runtime_rows, key_cols, label="runtime config", env_var=env_var, timed=timed)
    tuned = _deduplicated_dispatch_rows(candidate_rows, key_cols, label="candidate", env_var=env_var, timed=timed)
    if kept is None or tuned is None:
        return None
    # Drop rows from runtime that the candidate improves, then concat.
    tuned_keys = {_dispatch_key(row, key_cols) for row in tuned}
    merged_rows = [row for row in kept if _dispatch_key(row, key_cols) not in tuned_keys] + tuned
    merged_path = _write_table(candidate_path.parent / f"merged_{candidate_path.name}", all_columns, merged_rows)
    log.info(
        "gemm E2E: merged %d candidate rows into %d rows from %d aiter config file(s) -> %d total (%s)",
        len(tuned),
        len(kept),
        len(sources.paths),
        len(merged_rows),
        merged_path,
    )
    return str(merged_path)
