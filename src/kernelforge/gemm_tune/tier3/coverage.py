# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runtime asked for that no tuner can serve."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Reasons a tuner did not run that say nothing about coverage.
_NOT_A_COVERAGE_GAP = (
    "already at peak performance",
    "not supported",
    "unavailable on",
    # The fp4/gfx942 skip words it this way, and it is the strongest form of
    # "not our problem" there is: aiter ships no fp4 kernel for this card, so
    # there is nothing for any tuner, generated or not, to select between.
    "unsupported on",
    "no gemm shapes available",
    "requires --tunableop-input",
    "is not moe",
    "num_experts",
    "intermediate size",
    "moe_intermediate_size",
)


# Why a demanded table went untuned. Three of these argue for writing a tuner
# and one does not: `not_selected` is a routing bug, and generating a second
# tuner would paper over it.
KIND_NO_TUNER = "no_tuner"  # nothing implements this at all
KIND_SKIPPED = "skipped"  # a tuner exists and declined, for a reason
KIND_EMPTY = "empty"  # a tuner exists, ran, and produced nothing landable
KIND_NOT_SELECTED = "not_selected"  # a tuner exists and routing did not pick it

# The kinds a generated tuner is a legitimate answer to. ``skipped`` is in here
# only because the reasons that are *not* a gap have already been filtered out
# by ``_is_coverage_gap`` before a gap of that kind is ever built.
_WARRANTS = frozenset({KIND_NO_TUNER, KIND_SKIPPED, KIND_EMPTY})


@dataclass
class CoverageGap:
    """A demanded table that went untuned, and why."""

    table: str
    # Both absent is the strongest form of gap: nothing owns this table at all.
    tuner: str | None = None
    env_var: str | None = None
    key_schema: list[str] = field(default_factory=list)
    logged_fields: list[str] = field(default_factory=list)
    miss_count: int = 0
    distinct_keys: int = 0
    reason: str = ""
    kind: str = KIND_NO_TUNER

    @property
    def warrants_generated_tuner(self) -> bool:
        """An absent result does. A routing miss is a routing bug.

        Absent, not unimplemented: a tuner that owns the table and hands back
        nothing landable leaves the runtime no better off than one that was
        never written. What a generated tuner cannot help with is a table whose
        owner was simply never selected -- there the fix is to select it.
        """
        return self.kind in _WARRANTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "tuner": self.tuner,
            "env_var": self.env_var,
            "kind": self.kind,
            "warrants_generated_tuner": self.warrants_generated_tuner,
            "key_schema": list(self.key_schema),
            "logged_fields": list(self.logged_fields),
            "miss_count": self.miss_count,
            "distinct_keys": self.distinct_keys,
            "reason": self.reason,
        }


def _is_coverage_gap(skip_reason: str) -> bool:
    low = (skip_reason or "").lower()
    return not any(marker in low for marker in _NOT_A_COVERAGE_GAP)


def _gap(entry: dict[str, Any], table: str, tuner: str | None, *, kind: str, reason: str) -> CoverageGap:
    return CoverageGap(
        table=table,
        tuner=tuner,
        env_var=entry.get("env_var"),
        key_schema=list(entry.get("key_schema") or []),
        logged_fields=list(entry.get("logged_fields") or []),
        miss_count=int(entry.get("miss_count") or 0),
        distinct_keys=int(entry.get("distinct_keys") or 0),
        reason=reason,
        kind=kind,
    )


def _landed(results: list[Any] | None) -> set[str] | None:
    """Tuners that finished with something the integrate lane could apply.

    ``None`` when no results were supplied, which is how the pre-run caller says
    "nobody has run yet" rather than "nobody produced anything" -- the two must
    not collapse, or every selected tuner would read as empty before it started.

    Built from ``per_tuner_candidates`` rather than from ``status``, so this
    agrees with the promotion rule the report itself uses. A tuner that reports
    ``ok`` but wrote no artifact and named no env var is not landable, and the
    runtime will keep missing exactly the keys it was asked about.
    """
    if results is None:
        return None
    from ..candidates import per_tuner_candidates

    return {c.tuner for c in per_tuner_candidates(results)}


def coverage_gaps(
    demand_report: dict[str, Any] | None,
    tuner_specs: list[Any],
    results: list[Any] | None = None,
) -> list[CoverageGap]:
    """Demanded tables that no selected tuner will write.

    Passing ``results`` is what turns "a tuner owns this" into "a tuner covered
    this": an owner that came back empty-handed yields ``KIND_EMPTY`` instead of
    silently counting as coverage. Omit it before the tuners run.
    """
    demands = (demand_report or {}).get("demands") or []
    if not demands:
        return []

    will_run = {str(getattr(s, "name", "")) for s in tuner_specs if getattr(s, "should_run", False)}
    skipped = {
        str(getattr(s, "name", "")): str(getattr(s, "skip_reason", "") or "")
        for s in tuner_specs
        if not getattr(s, "should_run", True)
    }
    landed = _landed(results)

    gaps: list[CoverageGap] = []
    for entry in demands:
        tuner = entry.get("tuner")
        table = str(entry.get("table") or "")
        if tuner and tuner in will_run:
            if landed is None or tuner in landed:
                continue
            gaps.append(
                _gap(
                    entry,
                    table,
                    tuner,
                    kind=KIND_EMPTY,
                    reason=f"{tuner} ran and produced nothing landable for {table}",
                )
            )
            continue
        if tuner is None:
            kind = KIND_NO_TUNER
            reason = f"no tuner is registered for {table}"
        elif tuner in skipped:
            if not _is_coverage_gap(skipped[tuner]):
                continue
            kind = KIND_SKIPPED
            reason = f"{tuner} skipped: {skipped[tuner]}"
        else:
            kind = KIND_NOT_SELECTED
            reason = f"{tuner} owns {table} but was not selected for this run"
        gaps.append(_gap(entry, table, tuner, kind=kind, reason=reason))

    gaps.sort(key=lambda g: -g.miss_count)
    for gap in gaps:
        log.warning(
            "tuning coverage gap [%s]: %s (%d misses over %d keys) -- %s",
            gap.kind,
            gap.table,
            gap.miss_count,
            gap.distinct_keys,
            gap.reason,
        )
    return gaps
