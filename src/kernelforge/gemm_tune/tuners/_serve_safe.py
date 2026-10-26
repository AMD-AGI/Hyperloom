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
