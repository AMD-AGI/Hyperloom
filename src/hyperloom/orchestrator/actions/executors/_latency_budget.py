# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Latency budget — the constraint that makes a throughput-only KEEP safe.

The optimizer maximizes ``output_throughput`` and nothing else. Latency is
measured, reported and fed to the prompts, but no latency number has ever
blocked a KEEP. For every lever that was in the search space when that was
decided the omission is survivable, because a serving-config flag that doubles
throughput rarely destroys per-request latency.

Compute partitioning is not like that. Splitting a card raises aggregate
throughput precisely *by* making each stream slower, so a throughput-only gate
does not merely tolerate a latency regression -- it selects for the largest one
available. On one MI355X the ladder tops out at CPX with two streams per
partition: ~20% more throughput than the best SPX configuration, with
per-request latency going from 183 ms to 1211 ms. A gate reading only throughput
calls that a win and reports +20%.

So a session that opts into partition modes should also state what latency it
can live with. The budget is off by default and absolute rather than relative: an
SLA is a fixed number the workload owner already knows, and a percentage cap
against a baseline would silently ratchet as the baseline improves.

The gate fails closed. An operator who names a budget has asserted the
constraint matters, and a candidate whose latency was never measured cannot be
shown to satisfy it -- so it is refused rather than admitted on the assumption
that unmeasured means acceptable. The reason string says which of the two
happened, because the remedy differs: one needs a different candidate, the other
needs the benchmark to report latency at all.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

log = logging.getLogger(__name__)

#: Env override for the session latency budget, in milliseconds. Non-positive or
#: unparseable disables the gate, matching the CLI default.
LATENCY_BUDGET_ENV = "HYPERLOOM_MAX_LATENCY_MS"

#: Journal/ledger reasons. Distinct so a report can tell "too slow" apart from
#: "never timed", which are different bugs with different owners.
REASON_OVER_BUDGET = "latency_budget_exceeded"
REASON_UNMEASURED = "latency_unmeasured_under_budget"


def _finite_positive(value: Any) -> float | None:
    """Return ``value`` as a finite positive float, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if not math.isfinite(val) or val <= 0:
        return None
    return val


def resolve_latency_budget_ms(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
) -> float:
    """Resolve the session's latency budget in milliseconds.

    Precedence is most-specific-first: an explicit task parameter, then the
    session state the CLI seeded, then the environment. Anything non-positive or
    unparseable means no budget, which is the default and leaves KEEP behaviour
    exactly as it was.

    Args:
        params: Task params, which may carry ``latency_budget_ms``.
        shared_state: Live SharedState, which may carry ``latency_budget_ms``.

    Returns:
        The budget in ms, or ``0.0`` when the gate is off.
    """
    for candidate in (
        (params or {}).get("latency_budget_ms"),
        getattr(shared_state, "latency_budget_ms", None),
        os.environ.get(LATENCY_BUDGET_ENV),
    ):
        if candidate in (None, ""):
            continue
        try:
            val = float(candidate)
        except (TypeError, ValueError):
            log.warning("ignoring unparseable latency budget %r", candidate)
            continue
        if val > 0:
            return val
    return 0.0


def latency_keep_block(
    observed_ms: Any,
    *,
    budget_ms: float,
) -> tuple[bool, str]:
    """Decide whether the latency budget blocks a KEEP.

    Args:
        observed_ms: The candidate's mean end-to-end latency, or ``None`` when
            the benchmark reported none.
        budget_ms: The session budget; ``<= 0`` disables the gate.

    Returns:
        ``(blocked, reason)``. ``reason`` is empty when nothing blocks.
    """
    budget = _finite_positive(budget_ms)
    if budget is None:
        return False, ""
    observed = _finite_positive(observed_ms)
    if observed is None:
        return True, (
            f"{REASON_UNMEASURED}: a {budget:.0f} ms budget is set but this "
            f"candidate reported no end-to-end latency, so the constraint "
            f"cannot be shown to hold. Have the benchmark report mean_e2el_ms."
        )
    if observed > budget:
        return True, (
            f"{REASON_OVER_BUDGET}: {observed:.0f} ms exceeds the {budget:.0f} ms budget ({observed / budget:.2f}x)"
        )
    return False, ""


def describe_latency_budget(budget_ms: float) -> str:
    """Render the budget for a log line or report header."""
    budget = _finite_positive(budget_ms)
    return f"latency budget {budget:.0f} ms" if budget else "no latency budget"


__all__ = [
    "LATENCY_BUDGET_ENV",
    "REASON_OVER_BUDGET",
    "REASON_UNMEASURED",
    "describe_latency_budget",
    "latency_keep_block",
    "resolve_latency_budget_ms",
]
