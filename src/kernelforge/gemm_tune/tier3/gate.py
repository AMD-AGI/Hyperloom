# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a generated tuner may be attempted at all.

Four conditions, and every one has to hold. They are separate on purpose: each
answers a different question, and collapsing them into one switch would let a
single misconfiguration open the whole path.

1. **Nobody turned it off.** On by default, with a kill switch and an optional
   table list. This was the reverse until it was pointed out that condition 2
   already restricts it to the cases where nothing else can do anything at
   all: when no tuner owns the table, the time a generated one spends is not
   time taken from a tuner that would have covered it, because there is none.
   Keeping it shut then buys nothing and costs the one case it exists for.
2. **Nothing else can do the job.** Only a ``no_tuner`` gap qualifies. A tuner
   that exists and was skipped, or exists and was not routed to, is a bug in
   routing or a legitimate refusal -- generating a second tuner would paper over
   the first.
3. **There is enough demand to be worth it.** A table asked for twice is not a
   reason to write code; the floor keeps machine time proportional to what the
   runtime actually wants.
4. **The keys are describable.** A mandate with no shapes and no key schema
   cannot be written against, and asking anyway produces a plausible script for
   an imagined problem.

The decision is returned with its reasons rather than as a boolean, because the
useful artefact when this says no is *why* -- that is what tells you whether to
fix routing, widen the whitelist, or leave it alone.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .coverage import CoverageGap

log = logging.getLogger(__name__)

# Comma-separated table names to restrict generation to. Empty -- the default --
# means every table that clears the other three conditions.
ALLOW_ENV = "FORGE_TIER3_ALLOW"
# The kill switch. Set to 1/true/yes to stop generation being attempted at all,
# without having to know which tables are in play.
DISABLE_ENV = "FORGE_TIER3_DISABLE"
MIN_MISSES_ENV = "FORGE_TIER3_MIN_MISSES"
DEFAULT_MIN_MISSES = 25


@dataclass
class GateDecision:
    """Whether to attempt a generated tuner, and what decided it."""

    allowed: bool
    gap: CoverageGap | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "table": self.gap.table if self.gap else None,
            "reasons": list(self.reasons),
        }


def _allowed_tables() -> set[str]:
    raw = os.environ.get(ALLOW_ENV, "").strip()
    return {t.strip() for t in raw.split(",") if t.strip()}


def _disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def should_generate(gaps: list[CoverageGap]) -> GateDecision:
    """Pick the one gap worth generating a tuner for, if any."""
    if _disabled():
        return GateDecision(False, None, [f"{DISABLE_ENV} is set; generation is off"])
    allow = _allowed_tables()

    try:
        floor = int(os.environ.get(MIN_MISSES_ENV, "").strip() or DEFAULT_MIN_MISSES)
    except ValueError:
        floor = DEFAULT_MIN_MISSES
    floor = max(floor, 1)

    reasons: list[str] = []
    for gap in sorted(gaps, key=lambda g: -g.miss_count):
        if not gap.warrants_generated_tuner:
            reasons.append(
                f"{gap.table}: {gap.kind} -- a tuner for this exists, so the fix is there and not a generated one"
            )
            continue
        if allow and "*" not in allow and gap.table not in allow:
            reasons.append(f"{gap.table}: {ALLOW_ENV} is set and does not list it")
            continue
        if gap.miss_count < floor:
            reasons.append(f"{gap.table}: {gap.miss_count} misses is below the floor of {floor}")
            continue
        if not gap.key_schema:
            reasons.append(f"{gap.table}: no key schema to write a tuner against")
            continue
        log.warning(
            "tier3: generating a tuner for %s (%d misses over %d keys) -- %s",
            gap.table,
            gap.miss_count,
            gap.distinct_keys,
            gap.reason,
        )
        return GateDecision(True, gap, [f"{gap.table}: {gap.reason}"])

    if not gaps:
        reasons.append("no coverage gaps in this run")
    return GateDecision(False, None, reasons)
