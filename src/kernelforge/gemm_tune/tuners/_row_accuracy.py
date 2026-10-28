# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which rows of a tuned aiter dense GEMM table are accurate enough to ship.

aiter records each row's errRatio, the fraction of output elements its own check found wrong; a row above
``_MAX_ERR_RATIO`` is dropped, except the rows :func:`_judged_by_rel_norm` selects, which ship when their
relative-norm error against an fp32 dequantized matmul is measured and within ``_MAX_REL_NORM_ERR``.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Fraction of output elements aiter's own accuracy check found wrong.
_MAX_ERR_RATIO = 0.01
_ERR_RATIO_COLUMNS = ("err_ratio", "errRatio")
# ||out - ref|| / ||ref|| against the fp32 dequantized matmul, for the rows _judged_by_rel_norm selects.
_MAX_REL_NORM_ERR = 0.01
REL_NORM_COLUMN = "relNormErr"

#: (m, n, k, split_k, kernel_name) -> relative-norm error, or None when it could not be measured.
RelNormFn = Callable[[int, int, int, int, str], float | None]


def _row_err_ratio(row: dict[str, str]) -> float | None:
    """The accuracy figure aiter recorded for a row, or None if it recorded none."""
    for col in _ERR_RATIO_COLUMNS:
        if col in row:
            try:
                ratio = float(row[col])
            except (TypeError, ValueError):
                return None
            # NaN compares false against the limit, so an unrecorded figure would read as an accurate row.
            return ratio if math.isfinite(ratio) else None
    return None


def _judged_by_rel_norm(row: dict[str, str]) -> bool:
    """asm rows with splitK > 1: their bf16 atomic accumulation fails aiter's per-element errRatio while the
    output as a whole stays accurate."""
    try:
        split_k = int(row.get("splitK") or 0)
    except ValueError:
        return False
    return str(row.get("libtype", "")).strip().lower() == "asm" and split_k > 1


@dataclass(frozen=True)
class _Verdict:
    """Whether a row may ship, and the relative-norm error that decided it when one was measured."""

    accurate: bool
    rel_norm_err: float | None = None


def _judge_row(row: dict[str, str], rel_norm: RelNormFn | None) -> _Verdict | None:
    """The accuracy verdict on one tuned row, or None when aiter recorded no figure for it."""
    err_ratio = _row_err_ratio(row)
    if err_ratio is None:
        return None
    if err_ratio <= _MAX_ERR_RATIO:
        return _Verdict(True)
    return _rel_norm_verdict(row, rel_norm)


def _rel_norm_verdict(row: dict[str, str], rel_norm: RelNormFn | None) -> _Verdict:
    """The verdict on a row above the errRatio limit: accurate only when its relative-norm error is measured and passes."""
    if rel_norm is None or not _judged_by_rel_norm(row):
        return _Verdict(False)
    measured = rel_norm(int(row["M"]), int(row["N"]), int(row["K"]), int(row["splitK"]), row.get("kernelName", ""))
    return _Verdict(measured is not None and measured <= _MAX_REL_NORM_ERR, measured)


@dataclass(frozen=True)
class AccuracyFilter:
    """Whether aiter's accuracy figures were applied to the artifact, and which rows they removed."""

    completed: bool
    dropped: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""


def _read_dict_rows(tuned_csv: Path) -> list[dict[str, str]]:
    with tuned_csv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r]


def _rewrite_rows(tuned_csv: Path, header: list[str], rows: list[dict[str, str]]) -> str:
    """Write ``rows`` beside the artifact and rename over it; the error, or "" on success."""
    tmp = tuned_csv.with_name(tuned_csv.name + ".filtered.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, tuned_csv)
    except (OSError, csv.Error) as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove the partial file %s", tmp)
        return str(exc)
    return ""


def drop_inaccurate_rows(tuned_csv: Path, rel_norm: RelNormFn | None = None) -> AccuracyFilter:
    """Remove rows measured as numerically wrong, in place.

    ``rel_norm`` measures the rows :func:`_judged_by_rel_norm` selects once
    their errRatio is above the limit; the figure lands in ``REL_NORM_COLUMN``
    of the row, kept or dropped. Without it errRatio decides every row.
    """
    try:
        if not tuned_csv.is_file():
            return AccuracyFilter(completed=False, reason=f"{tuned_csv} does not exist")
        rows = _read_dict_rows(tuned_csv)
    except (OSError, csv.Error) as exc:
        return AccuracyFilter(completed=False, reason=f"{tuned_csv} could not be read: {exc}")
    if not rows:
        return AccuracyFilter(completed=True)
    header = list(rows[0].keys())
    if not any(c in header for c in _ERR_RATIO_COLUMNS):
        return AccuracyFilter(
            completed=False,
            reason=f"{tuned_csv} carries no accuracy column (aiter schema drift?)",
        )

    keep, dropped, unmeasured = _judge_rows(rows, rel_norm)
    if unmeasured:
        return AccuracyFilter(
            completed=False,
            dropped=dropped,
            reason=f"{unmeasured} row(s) in {tuned_csv} carry no readable accuracy figure",
        )
    if not dropped and not any(REL_NORM_COLUMN in row for row in keep):
        return AccuracyFilter(completed=True)

    if REL_NORM_COLUMN not in header and any(REL_NORM_COLUMN in row for row in rows):
        header.append(REL_NORM_COLUMN)
    error = _rewrite_rows(tuned_csv, header, keep)
    if error:
        return AccuracyFilter(
            completed=False,
            dropped=dropped,
            reason=f"{len(dropped)} inaccurate row(s) could not be removed from {tuned_csv}: {error}",
        )
    _log_dropped(dropped, tuned_csv.name)
    return AccuracyFilter(completed=True, dropped=dropped)


def _judge_rows(
    rows: list[dict[str, str]], rel_norm: RelNormFn | None
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    """Split ``rows`` into kept and dropped, recording each measured relative-norm error on its row.

    A row with no readable accuracy figure is kept and counted in the third element.
    """
    keep: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    unmeasured = 0
    for row in rows:
        verdict = _judge_row(row, rel_norm)
        unmeasured += verdict is None
        if verdict is not None and verdict.rel_norm_err is not None:
            row[REL_NORM_COLUMN] = f"{verdict.rel_norm_err:.6g}"
        (keep if verdict is None or verdict.accurate else dropped).append(row)
    return keep, dropped, unmeasured


def _row_rel_norm_err(row: dict[str, str]) -> float | None:
    """The relative-norm error recorded on a row, or None when it was not measured."""
    figure = row.get(REL_NORM_COLUMN)
    return float(figure) if figure else None


def dropped_row_record(row: dict[str, str]) -> dict[str, object]:
    """What a tuner result reports of a row the filter dropped."""
    return {
        "M": row.get("M"),
        "N": row.get("N"),
        "K": row.get("K"),
        "libtype": row.get("libtype"),
        "splitK": row.get("splitK"),
        "us": row.get("us"),
        "err_ratio": _row_err_ratio(row),
        "rel_norm_err": _row_rel_norm_err(row),
    }


def _log_dropped(dropped: list[dict[str, str]], table: str) -> None:
    for row in dropped:
        log.error(
            "dropping M=%s N=%s K=%s (%s, splitK=%s, %sus) from %s -- aiter measured err_ratio=%s, above the %.2f "
            "limit; relative-norm error %s",
            row.get("M"),
            row.get("N"),
            row.get("K"),
            row.get("libtype"),
            row.get("splitK"),
            row.get("us"),
            table,
            _row_err_ratio(row),
            _MAX_ERR_RATIO,
            row.get(REL_NORM_COLUMN) or "not measured",
        )
