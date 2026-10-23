# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One table of a ``kernel_tune`` answer: what a Forge run for one aiter table came to.

A table is ``tuned`` when it has verified rows (and then carries their rows-only CSV), ``no_rows`` when the tuner ran and
none of its rows is handed over (nothing beat the default, or every row failed the reference check), ``failed`` when the run
did not give a usable table and ``skipped`` when the time did not reach it. Whatever the status, every row the reference
check refused is listed with ``verified`` false and its reason; none is in a CSV.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hyperloom_multibranch.toolbox.tools.tune_rows import RowsRefused, TunedRows, dropped_by_forge, from_artifact

__all__ = ["ROWS_FILE", "TableStatus", "TuneTable", "failed", "read_run", "skipped"]

#: The keyed rows-only CSV of a table, written beside the tuner's own output.
ROWS_FILE = "rows_only.csv"
_RESULT = "result.json"
_TAIL_CHARS = 600


class TableStatus(str, Enum):
    TUNED = "tuned"
    NO_ROWS = "no_rows"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class TuneTable:
    """The answer for one table: the variable it is for, the tuner, and what came of the run."""

    env: str
    tuner: str
    status: TableStatus
    reason: str
    rows: TunedRows | None = None
    rows_file: Path | None = None
    refused: tuple[Mapping[str, Any], ...] = ()
    forge: Mapping[str, Any] | None = None

    @property
    def verified(self) -> int:
        return 0 if self.rows is None else len(self.rows.verified)

    def document(self) -> dict[str, Any]:
        """The table as the answer says it: what a specialist needs to put it, and to know what was left out."""
        table: dict[str, Any] = {
            "env": self.env,
            "kind": "gemm",
            "rows_only": True,
            "tuner": self.tuner,
            "status": self.status.value,
            "reason": self.reason,
            "rejected_rows": list(self.refused),
            "forge": self.forge,
        }
        if self.rows is not None:
            table |= {
                "target": self.rows.target,
                "rows": self.verified,
                "rows_csv": self.rows.csv if self.verified else "",
                "rows_file": str(self.rows_file) if self.rows_file is not None else "",
                "verified_rows": list(self.rows.verified),
            }
        return table


def failed(demand: Mapping[str, Any], reason: str) -> TuneTable:
    return TuneTable(demand["env_var"], demand["tuner"], TableStatus.FAILED, reason)


def skipped(demand: Mapping[str, Any], reason: str) -> TuneTable:
    return TuneTable(demand["env_var"], demand["tuner"], TableStatus.SKIPPED, reason)


def _skipped_or_failed(report: Mapping[str, Any]) -> str:
    entries = [*(report.get("failed_tuners") or []), *(report.get("tuners_skipped") or [])]
    return "; ".join(
        str(entry.get("error") or entry.get("skip_reason") or "") for entry in entries if isinstance(entry, Mapping)
    )


def _row_reason(rows: TunedRows, rejected: int) -> str:
    if rows.verified:
        return f"{len(rows.verified)} row(s) checked against the reference" + (
            f", {rejected} rejected" if rejected else ""
        )
    if rejected:
        return f"none of the {rejected} row(s) the tuner measured passed the reference check"
    return "the tuner left no row"


def _write_rows(out: Path, rows: TunedRows) -> Path:
    path = out / ROWS_FILE
    path.write_text(rows.csv, encoding="utf-8")
    return path


def read_run(demand: Mapping[str, Any], out: Path, done: subprocess.CompletedProcess[str], configs: Path) -> TuneTable:
    """The table a finished Forge run left in ``out``: its verified rows, the rows it rejected, or why there is none.

    ``configs`` is the aiter install's ``configs`` directory the table's shape key is read from.
    """
    tuner = demand["tuner"]
    try:
        report = json.loads((out / _RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = (done.stderr or done.stdout or "")[-_TAIL_CHARS:].strip()
        return failed(demand, f"the tuner exited {done.returncode} and left no result: {tail}")
    run = next((r for r in report.get("tuners_run") or [] if r.get("tuner") == tuner), {})
    forge = {key: report.get(key) for key in ("status", "micro_decision", "total_elapsed_s", "error", "error_class")}
    forge["run"] = run
    refused = dropped_by_forge(run)
    artifact = (report.get("artifacts") or {}).get(tuner)

    def without_rows(status: TableStatus, reason: str) -> TuneTable:
        return TuneTable(demand["env_var"], tuner, status, reason, refused=refused, forge=forge)

    if not artifact:
        why = report.get("error") or _skipped_or_failed(report) or "no row beat the default of the shapes it tuned"
        return without_rows(TableStatus.FAILED if report.get("status") == "failed" else TableStatus.NO_ROWS, str(why))
    path = Path(artifact)
    if not path.is_relative_to(out):
        return without_rows(TableStatus.FAILED, f"the tuner names a table outside its output directory: {artifact}")
    try:
        rows = from_artifact(path, demand["env_var"], configs)
    except RowsRefused as exc:
        return without_rows(TableStatus.FAILED, str(exc))
    refused = (*refused, *rows.refused)
    status = TableStatus.TUNED if rows.verified else TableStatus.NO_ROWS
    rows_file = _write_rows(out, rows) if rows.verified else None
    return TuneTable(demand["env_var"], tuner, status, _row_reason(rows, len(refused)), rows, rows_file, refused, forge)


def tables_summary(tables: Sequence[TuneTable]) -> str:
    """One line for the inbox: the rows handed over, and what was refused or failed; or why no table has rows."""
    tuned = [table for table in tables if table.status is TableStatus.TUNED]
    if not tuned:
        return "; ".join(f"{table.env}: {table.status.value}: {table.reason}" for table in tables)
    rejected = sum(len(table.refused) for table in tables)
    broken = sum(table.status is TableStatus.FAILED for table in tables)
    return (
        f"{sum(table.verified for table in tuned)} reference-checked row(s) for {', '.join(t.env for t in tuned)}"
        + (f"; {rejected} rejected" if rejected else "")
        + (f"; {broken} table(s) failed" if broken else "")
    )
