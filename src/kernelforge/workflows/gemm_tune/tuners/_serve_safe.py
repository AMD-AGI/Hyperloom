# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which profile candidates a tuned aiter dense table may ship in place of the tuner's pick.

A candidate is eligible when production serves its splitK on that shape and it would survive the accuracy gate of
:mod:`._row_accuracy`; whatever replaces the tuner's pick chooses among these.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from ._row_accuracy import _MAX_ERR_RATIO, RelNormFn, _judged_by_rel_norm, _rel_norm_verdict

log = logging.getLogger(__name__)

# Which (op, libtype) pairs reach a production wrapper that actually forwards
# the tuned row's ``splitK``. A tuned row is a measurement plus a set of kernel
# parameters, and the measurement is only honoured if every parameter survives
# the trip to the serving call -- a dropped ``splitK`` deploys a config that was
# benchmarked with split-K and runs without it, i.e. slower than measured while
# every engagement gate still reports the artifact as served. Read off
# ``aiter/ops/gemm_op_a8w8.py``:
#
#   gemm_a8w8_blockscale               ck -> splitK=splitK   cktile -> splitK=splitK
#   gemm_a8w8_blockscale_bpreshuffle   ck -> kernelName only  cktile -> kernelName only
#                                      asm -> splitK=splitK   opus/flydsl -> no splitK
#
# Fail closed: an op absent from this table is treated as forwarding nothing, so
# adding a tuner (or a new libtype) can only under-claim, never silently ship a
# row whose splitK the runtime discards. Verified against aiter
# d9e5ef7ce08ee7045d583aed768cff41aa9210fe; re-check on an aiter bump.
_SPLITK_FORWARDING_LIBTYPES: dict[str, frozenset[str]] = {
    "a8w8_blockscale": frozenset({"ck", "cktile"}),
    "a8w8_blockscale_bpreshuffle": frozenset({"asm"}),
}


class _Candidate(NamedTuple):
    """A profile row the cap may deploy, with the figures selection reads."""

    us: float
    splitk: int
    row: list[str]
    err_ratio: float
    record: dict[str, str]


@dataclass(frozen=True)
class _SplitKLimit:
    """The split-K a shape may ship, and the check that clears an asm split-K replacement above the errRatio limit.

    ``support_fn`` trials the per-shape ceiling (None keeps ``max_splitk``); without ``rel_norm`` errRatio decides
    every replacement, as :func:`drop_inaccurate_rows` would after the cap.
    """

    max_splitk: int
    support_fn: Callable[[int, int, int], int | None] | None = None
    rel_norm: RelNormFn | None = None

    def ceiling(self, shape: tuple[int, int, int]) -> int:
        """The highest splitK this shape may ship, trialled per shape where a trial exists."""
        if self.support_fn is None:
            return self.max_splitk
        supported = self.support_fn(*shape)
        return self.max_splitk if supported is None else int(supported)

    def accurate(self, candidate: _Candidate) -> bool:
        """Whether the candidate would survive :func:`drop_inaccurate_rows` on the deployed table."""
        return candidate.err_ratio <= _MAX_ERR_RATIO or _rel_norm_verdict(candidate.record, self.rel_norm).accurate


def _serve_safe_candidates(
    profile_csv: Path,
    header: list[str],
) -> dict[tuple[str, str, str], list[_Candidate]]:
    """Index the profile's usable candidates per shape.

    The splitK cap and the relative-norm check are applied per shape at
    selection rather than here, so ``support_fn`` can keep a splitK above the
    static ceiling where the kernel supports it, and only the candidates
    selection reaches are measured.
    """
    by_shape: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    schema_ok = True
    try:
        with profile_csv.open() as handle:
            for record in csv.DictReader(handle):
                # A candidate must carry every column the deployed CSV has, or the rewritten row would get empty cells
                # and a renamed/absent errRatio would silently disable the correctness filter.
                if any(column not in record for column in header):
                    schema_ok = False
                    continue
                try:
                    micros, splitk = float(record["us"]), int(record["splitK"])
                    err_ratio = float(record.get("errRatio") or 0)  # absent -> 0 (no KeyError)
                except (KeyError, ValueError, TypeError):
                    continue
                if micros <= 0 or not _admissible_replacement(record, err_ratio):
                    continue
                by_shape[(record["M"], record["N"], record["K"])].append(
                    _Candidate(micros, splitk, [record[column] for column in header], err_ratio, record)
                )
    except OSError:
        by_shape = defaultdict(list)
    if not schema_ok:
        log.warning(
            "splitK cap: profile %s lacks columns present in the tuned CSV; some "
            "serve-safe candidates were skipped (possible aiter schema drift)",
            profile_csv,
        )
    return by_shape


def _admissible_replacement(record: dict[str, str], err_ratio: float) -> bool:
    """A profile row the cap may deploy in place of an unservable one.

    A row :func:`_judged_by_rel_norm` selects is indexed above the errRatio
    limit; :meth:`_SplitKLimit.accurate` measures it when selection reaches it.
    """
    return err_ratio <= _MAX_ERR_RATIO or _judged_by_rel_norm(record)


def _fastest_serve_safe(candidates: Sequence[_Candidate], max_splitk: int, limit: _SplitKLimit) -> _Candidate | None:
    """The fastest accurate candidate whose splitK the deployment can serve, or None when there is none."""
    servable = sorted((c for c in candidates if c.splitk <= max_splitk), key=lambda c: c.us)
    return next((c for c in servable if limit.accurate(c)), None)


@dataclass(frozen=True)
class _DeployedColumns:
    """Where the deployed CSV keeps the columns the splitK cap reads."""

    m: int
    n: int
    k: int
    splitk: int
    libtype: int | None

    def shape_key(self, row: list[str]) -> tuple[str, str, str]:
        """The ``(M, N, K)`` this row describes, as the profile index is keyed."""
        return row[self.m], row[self.n], row[self.k]


def _deployed_columns(header: list[str]) -> _DeployedColumns | None:
    """Locate the columns the cap reads, or None when the header lacks one.

    Case-insensitive: were the deployed header's case ever to stop matching
    exactly, the cap would return early and pass unsafe splitK rows through
    unchanged -> serve crash.
    """
    index = {str(name).strip().lower(): position for position, name in enumerate(header)}
    try:
        return _DeployedColumns(index["m"], index["n"], index["k"], index["splitk"], index.get("libtype"))
    except KeyError:
        return None


def _forwards_splitk(row: list[str], libtype: int | None, forwarding_libtypes: frozenset[str]) -> bool:
    """Whether this row's libtype has a production wrapper that forwards splitK."""
    if libtype is None or libtype >= len(row):
        # No libtype column to check against a contract that is keyed on it;
        # treat as non-forwarding, matching the fail-closed default.
        return False
    return str(row[libtype]).strip().lower() in forwarding_libtypes


def _cap_splitk_to_serve_safe(
    artifact_csv: Path,
    profile_csv: Path | None,
    limit: _SplitKLimit,
    *,
    forwarding_libtypes: frozenset[str],
) -> tuple[int, bool]:
    """Rewrite deployed rows whose splitK production cannot dispatch or forward.

    Two hazards, same remedy: a splitK the production kernel cannot dispatch
    (crashes engine init), and a splitK on a libtype whose wrapper takes no such
    argument (no crash, just a kernel slower than the one that won the
    benchmark, with every engagement gate still green). Either way the row is
    replaced by the fastest serve-safe candidate from the profile that would
    pass the accuracy gate, and a shape with no such candidate is dropped.

    ``forwarding_libtypes`` is the set of libtypes whose wrapper forwards splitK
    for this op (see ``_SPLITK_FORWARDING_LIBTYPES``); a row on any other
    libtype is capped at 0 regardless of dispatch support.
    """
    try:
        with artifact_csv.open() as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return 0, False
    if len(rows) < 2:
        return 0, False
    hdr = rows[0]
    columns = _deployed_columns(hdr)
    if columns is None:
        return 0, False
    by_shape = _serve_safe_candidates(profile_csv, hdr) if profile_csv else {}

    out, changed, has_splitk = [hdr], 0, False
    dropped_unforwarded = 0
    for row in rows[1:]:
        try:
            sk = int(row[columns.splitk])
        except (ValueError, IndexError):
            out.append(row)
            continue
        if sk == 0:
            # splitK=0 is the default dispatch: always serve-safe, and its keep decision never depends on the
            # per-shape max, so skip the (GPU-dispatching) trial entirely for these rows.
            out.append(row)
            continue
        if not _forwards_splitk(row, columns.libtype, forwarding_libtypes):
            # The wrapper for this libtype takes no splitK, so the only
            # honourable value is 0 -- fall through to candidate replacement
            # with maxsk=0 rather than shipping a measurement production cannot
            # reproduce.
            dropped_unforwarded += 1
            try:
                key = columns.shape_key(row)
            except IndexError:
                out.append(row)
                continue
            safe = _fastest_serve_safe(by_shape.get(key, ()), 0, limit)
            if safe is not None:
                out.append(safe.row)
            changed += 1
            continue
        try:
            key = columns.shape_key(row)
            shape = (int(row[columns.m]), int(row[columns.n]), int(row[columns.k]))
        except (ValueError, IndexError):
            out.append(row)
            continue
        maxsk = limit.ceiling(shape)
        if sk <= maxsk:
            out.append(row)
            has_splitk = has_splitk or sk > 0
            continue
        safe = _fastest_serve_safe(by_shape.get(key, ()), maxsk, limit)
        if safe is not None:
            out.append(safe.row)
            has_splitk = has_splitk or safe.splitk > 0
        # else: drop the row -> serve falls back to the aiter default
        changed += 1
    if changed:
        with artifact_csv.open("w", newline="") as f:
            csv.writer(f).writerows(out)
    _warn_unforwarded_splitk(dropped_unforwarded, forwarding_libtypes)
    return changed, has_splitk


def _warn_unforwarded_splitk(dropped_unforwarded: int, forwarding_libtypes: frozenset[str]) -> None:
    """Say how many rows lost a splitK their libtype's production wrapper cannot forward."""
    if dropped_unforwarded:
        log.warning(
            "splitK forwarding: %d row(s) carried splitK>0 on a libtype whose "
            "production wrapper takes no splitK (forwarding set: %s); replaced "
            "with a splitK=0 candidate or dropped, because serving them would "
            "have run a config slower than the one benchmarked",
            dropped_unforwarded,
            sorted(forwarding_libtypes),
        )
