# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-rank the decode rows of a tuned aiter dense table by in-graph, cache-cold timing.

For every row sglang serves inside a CUDA graph, the tuner's pick and the fastest few eligible profile candidates
(:mod:`._serve_safe`) are timed the way decode runs them, and the fastest ships. A row whose timing fails keeps the
tuner's pick, and the failure is logged and reported with the rows that changed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Any

from ..aiter_graph_retime import Kernel, RetimeError, RetimeFn, Timing, make_retime_fn
from ..dense_shapes import _SGLANG_CUDAGRAPH_BS
from ._row_accuracy import REL_NORM_COLUMN, _judge_row, _read_dict_rows, _rewrite_rows
from ._serve_safe import _SPLITK_FORWARDING_LIBTYPES, _Candidate, _serve_safe_candidates, _SplitKLimit

log = logging.getLogger(__name__)

# Rows above sglang's largest graph batch run eagerly, where the tuner's eager timing is the right one.
_GRAPH_MAX_M = max(_SGLANG_CUDAGRAPH_BS)
# Candidates timed besides the tuner's pick, fastest by the tuner's own figure first.
_TOP_BY_TUNER = 3


@dataclass(frozen=True)
class RetimedRow:
    """One decode row the re-time changed or could not fully time."""

    shape: tuple[int, int, int]
    tuner_pick: Kernel
    tuner_pick_us: float | None  # None when the tuner's pick could not be timed, so it ships unchanged
    shipped: Kernel
    shipped_us: float | None
    errors: tuple[str, ...] = ()
    replacement: dict[str, str] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        m, n, k = self.shape
        record: dict[str, Any] = {
            "M": m,
            "N": n,
            "K": k,
            "tuner_pick": _describe(self.tuner_pick, self.tuner_pick_us),
            "shipped": _describe(self.shipped, self.shipped_us),
        }
        if self.tuner_pick_us and self.shipped_us:
            record["graph_speedup"] = round(self.tuner_pick_us / self.shipped_us, 4)
        if self.errors:
            record["errors"] = list(self.errors)
        return record


def _describe(kernel: Kernel, graph_us: float | None) -> dict[str, Any]:
    return {
        "libtype": kernel.libtype,
        "kernelName": kernel.kernel_name,
        "splitK": kernel.split_k,
        "graph_us": None if graph_us is None else round(graph_us, 3),
    }


def retime_table(
    artifact: str,
    profile_csv: Path,
    limit: _SplitKLimit,
    script_key: str,
    gpu_ids: str,
) -> list[dict[str, Any]]:
    """Ship each decode row's fastest candidate in a graph; the rows it changed or could not time.

    It dispatches on the tuner's card like the split-K trial, so the same switch turns it off.
    """
    if os.environ.get("FORGE_SPLITK_TRIAL", "1") == "0":
        return []
    time_fn = make_retime_fn(script_key, gpu_ids, profile_csv.parent / "graph_retime")
    libtypes = _SPLITK_FORWARDING_LIBTYPES.get(script_key, frozenset())
    return [r.to_dict() for r in retime_decode_rows(Path(artifact), profile_csv, limit, libtypes, time_fn)]


def retime_decode_rows(
    tuned_csv: Path,
    profile_csv: Path,
    limit: _SplitKLimit,
    forwarding_libtypes: frozenset[str],
    time_fn: RetimeFn | None,
) -> list[RetimedRow]:
    """Ship the fastest in-graph eligible candidate on every decode row of ``tuned_csv``, in place.

    ``limit`` and ``forwarding_libtypes`` are what the splitK cap and accuracy gate already held the table to, so a
    replacement is a row they would ship. Returns the rows that changed or could not be fully timed; nothing when
    ``time_fn`` is None.
    """
    if time_fn is None:
        return []
    rows = _read_dict_rows(tuned_csv)
    header = list(rows[0].keys()) if rows else []
    job_columns = [c for c in header if c != REL_NORM_COLUMN]
    by_shape = _serve_safe_candidates(profile_csv, job_columns)
    # Grouped by (N, K) so the timer reuses one set of weight copies per projection.
    decode = sorted(
        (i for i, row in enumerate(rows) if int(row["M"]) <= _GRAPH_MAX_M),
        key=lambda i: (int(rows[i]["N"]), int(rows[i]["K"]), int(rows[i]["M"])),
    )
    plans = [_Contenders.of(i, rows[i], by_shape, limit, forwarding_libtypes) for i in decode]
    jobs = [{c: row.get(c, "") for c in job_columns} for plan in plans for row in plan.rows]
    try:
        timings = time_fn(jobs) if jobs else []
    except RetimeError as exc:
        log.error("in-graph re-time of %s failed, every decode row keeps the tuner's pick: %s", tuned_csv, exc)
        return [plan.untimed(str(exc)) for plan in plans]
    retimed, offset = [], 0
    for plan in plans:
        outcome = plan.choose(timings[offset : offset + len(plan.rows)], limit)
        offset += len(plan.rows)
        if outcome.replacement is not None:
            rows[plan.index] = outcome.replacement
        if outcome.replacement is not None or outcome.errors:
            retimed.append(outcome)
    if any(r.replacement is not None for r in retimed):
        return _rewrite(tuned_csv, header, rows, retimed)
    return retimed


def _rewrite(
    tuned_csv: Path, header: list[str], rows: list[dict[str, str]], retimed: list[RetimedRow]
) -> list[RetimedRow]:
    """Write the re-timed rows over ``tuned_csv``; when that fails, report every replacement as not shipped."""
    if any(REL_NORM_COLUMN in row for row in rows) and REL_NORM_COLUMN not in header:
        header.append(REL_NORM_COLUMN)
    error = _rewrite_rows(tuned_csv, header, rows)
    if not error:
        return retimed
    log.error("in-graph re-time: could not rewrite %s, the tuner's picks ship: %s", tuned_csv, error)
    note = f"table not rewritten: {error}"
    return [
        replace(r, shipped=r.tuner_pick, shipped_us=r.tuner_pick_us, errors=(*r.errors, note), replacement=None)
        for r in retimed
    ]


@dataclass(frozen=True)
class _Contenders:
    """A decode row's tuner pick (first) and the eligible candidates timed against it, as the rows that would ship."""

    index: int
    shape: tuple[int, int, int]
    rows: list[dict[str, str]]
    candidates: list[_Candidate | None]

    @classmethod
    def of(
        cls,
        index: int,
        row: dict[str, str],
        by_shape: dict[tuple[str, str, str], list[_Candidate]],
        limit: _SplitKLimit,
        forwarding_libtypes: frozenset[str],
    ) -> _Contenders:
        shape = (int(row["M"]), int(row["N"]), int(row["K"]))
        contenders: dict[Kernel, _Candidate | None] = {Kernel.from_row(row): None}
        candidates = by_shape.get((row["M"], row["N"], row["K"]), [])
        for candidate in _eligible(candidates, limit, limit.ceiling(shape), forwarding_libtypes):
            contenders.setdefault(Kernel.from_row(candidate.record), candidate)
        shipped_rows = [row if c is None else c.record for c in contenders.values()]
        return cls(index, shape, shipped_rows, list(contenders.values()))

    @property
    def pick(self) -> Kernel:
        return Kernel.from_row(self.rows[0])

    def untimed(self, error: str) -> RetimedRow:
        return RetimedRow(self.shape, self.pick, None, self.pick, None, (error,))

    def choose(self, timings: Sequence[Timing], limit: _SplitKLimit) -> RetimedRow:
        """The row's outcome given each contender's in-graph time, the replacement row when another kernel won."""
        kernels = [Kernel.from_row(row) for row in self.rows]
        errors = tuple(
            f"{k.libtype} {k.kernel_name} splitK={k.split_k}: {t.error}" for k, t in zip(kernels, timings) if t.error
        )
        if errors:
            log.warning("in-graph re-time of M=%d N=%d K=%d could not time: %s", *self.shape, "; ".join(errors))
        pick_us = timings[0].us
        if pick_us is None:
            return RetimedRow(self.shape, self.pick, None, self.pick, None, errors)
        timed = [(t.us, i) for i, t in enumerate(timings) if t.us is not None]
        best_us, best = min(timed)
        winner = self.candidates[best]
        if winner is None:
            return RetimedRow(self.shape, self.pick, pick_us, self.pick, pick_us, errors)
        log.info(
            "in-graph re-time of M=%d N=%d K=%d: %s splitK=%d %.2fus replaces %s splitK=%d %.2fus",
            *self.shape,
            kernels[best].libtype,
            kernels[best].split_k,
            best_us,
            self.pick.libtype,
            self.pick.split_k,
            pick_us,
        )
        replacement = _replacement_row(self.rows[0], winner, limit)
        return RetimedRow(self.shape, self.pick, pick_us, kernels[best], best_us, errors, replacement)


def _eligible(
    candidates: Sequence[_Candidate],
    limit: _SplitKLimit,
    ceiling: int,
    forwarding_libtypes: frozenset[str],
) -> list[_Candidate]:
    """The fastest candidates by tuner time that production serves on this shape and the accuracy gate would keep."""

    def servable(candidate: _Candidate) -> bool:
        forwards = candidate.record["libtype"].strip().lower() in forwarding_libtypes
        return candidate.splitk <= (ceiling if forwards else 0)

    ranked = sorted(filter(servable, candidates), key=lambda c: c.us)
    return list(islice((c for c in ranked if limit.accurate(c)), _TOP_BY_TUNER))


def _replacement_row(row: dict[str, str], winner: _Candidate, limit: _SplitKLimit) -> dict[str, str]:
    """The deployed row for ``winner``, carrying the relative-norm figure the accuracy gate records."""
    replacement = {column: winner.record.get(column, "") for column in row}
    verdict = _judge_row(winner.record, limit.rel_norm)
    if verdict is not None and verdict.rel_norm_err is not None:
        replacement[REL_NORM_COLUMN] = f"{verdict.rel_norm_err:.6g}"
    return replacement
