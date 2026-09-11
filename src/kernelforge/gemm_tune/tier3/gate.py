# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide whether a generated tuner may run.

The gate requires enablement, an uncovered result, sufficient demand, and
describable keys. It returns reasons so callers can distinguish routing,
coverage, and configuration failures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .coverage import CoverageGap

log = logging.getLogger(__name__)

# Comma-separated table names to restrict generation to.
ALLOW_ENV = "FORGE_TIER3_ALLOW"
# The kill switch.
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
                f"{gap.table}: {gap.kind} -- a tuner for this exists and was not "
                f"routed to, so the fix is there and not a generated one"
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
