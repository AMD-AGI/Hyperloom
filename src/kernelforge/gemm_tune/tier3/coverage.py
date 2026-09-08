# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runtime asked for that no tuner can serve.

The trigger for writing a tuner is "aiter ships no script **and** forge has no
implementation", and until now nothing measured whether that ever happens. The
skip reasons that would answer it are prose, spread across the router, and never
collected -- so the question was settled by argument instead of by the fleet.

This turns it into a record: one entry per demanded table that ended with no
tuner able to write it, carrying the reason and enough of the key schema to say
what a tuner would have to produce. Running it over a campaign's demand files is
what says whether a generated tuner has a real target, and it is also the input
:mod:`.mandate` needs if one does.

Note what is deliberately *not* a gap: a table whose tuner exists but was skipped
for a reason of its own -- an unsupported dtype on this architecture, a missing
shape source, a kernel that cannot serve this checkpoint. Those are answers, not
absences, and a generated tuner would not change any of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Reasons a tuner did not run that say nothing about coverage. A generated tuner
# is not the answer to any of them: the capability exists, this run could not use
# it. Matched case-insensitively as substrings of the router's skip reason.
_NOT_A_COVERAGE_GAP = (
    "already at peak performance",
    "not supported",
    "unavailable on",
    "no gemm shapes available",
    "requires --tunableop-input",
    "is not moe",
    "num_experts",
    "intermediate size",
    "moe_intermediate_size",
)


# Why a demanded table went untuned. Only the first is an argument for writing a
# tuner; the other two are arguments for fixing something that already exists,
# and treating them alike would manufacture demand for the third tier. A real
# production log made the distinction immediately: a vLLM run missed 122 bf16
# keys with `sglang_dense_bf16` -- the tuner that owns that very table -- simply
# not selected by the framework branch. Nothing about that calls for a new tuner.
KIND_NO_TUNER = "no_tuner"  # nothing implements this: the Tier-3 case
KIND_SKIPPED = "skipped"  # a tuner exists and declined, for a reason
KIND_NOT_SELECTED = "not_selected"  # a tuner exists and routing did not pick it


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
        """Only an absent capability does. A routing miss is a routing bug."""
        return self.kind == KIND_NO_TUNER

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


def coverage_gaps(
    demand_report: dict[str, Any] | None,
    tuner_specs: list[Any],
) -> list[CoverageGap]:
    """Demanded tables that no selected tuner will write.

    Args:
        demand_report: A parsed serving log (``evidence.parse_log``). Without one
            there is no demand, and therefore nothing to be missing.
        tuner_specs: What the router chose, including the skipped ones -- a
            skipped tuner still tells us the capability exists.

    Returns:
        One entry per uncovered table, most-demanded first.
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

    gaps: list[CoverageGap] = []
    for entry in demands:
        tuner = entry.get("tuner")
        table = str(entry.get("table") or "")
        if tuner and tuner in will_run:
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
        gaps.append(
            CoverageGap(
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
        )

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
