# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Latency budget — the constraint that makes a throughput-only KEEP safe.

The optimizer maximizes ``output_throughput`` and nothing else. Latency is
measured, reported and fed to the prompts, but no latency number has ever
blocked a KEEP. For most levers in the search space that omission is
survivable, because a serving-config flag that raises throughput rarely
destroys per-request latency at the same time.

It stops being survivable as soon as a lever buys throughput *by* making each
stream slower. Then a throughput-only gate does not merely tolerate a latency
regression, it selects for the largest one on offer. Measured case: one MI355X
running a 1.26B-parameter vision model at six views per forward pass, with the
card split into eight partitions and two streams on each. Aggregate throughput
went up about 20% against the best unsplit configuration while per-request
latency went from 183 ms to 1211 ms. A gate reading only throughput calls that
a win and reports +20%.

So a session should be able to state what latency it can live with. The budget
is off by default and absolute rather than relative: an SLA is a fixed number
the workload owner already knows, and a percentage cap against a baseline would
silently ratchet as the baseline improves.

The gate fails closed. An operator who names a budget has asserted the
constraint matters, and a candidate whose latency was never measured cannot be
shown to satisfy it -- so it is refused rather than admitted on the assumption
that unmeasured means acceptable. The reason string says which of the two
happened, because the remedy differs: one needs a different candidate, the
other needs the benchmark to report latency at all.

Enforcement lives at two depths. Explore's decision round applies it per
variant, so an over-budget candidate never earns a rebench round. Every KEEP,
whatever produced it, passes :func:`latency_keep_block` again at the single
choke point that writes ``current_best``, because explore is not the only path
that promotes -- kernel, framework, specialist and integrate winners all land
there too, and a gate wired only into explore would let them through.
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


def read_session_budget() -> float:
    """Read the launcher-published budget for seeding SharedState.

    The CLI publishes the resolved flag into :data:`LATENCY_BUDGET_ENV` before
    the orchestrator starts, and this is the one parser of that variable. Seeding
    from the env rather than re-reading argv keeps the persisted session manifest
    equal to what the executors are actually handed.

    Returns:
        The budget in ms, or ``0.0`` when unset, unparseable or non-positive.
    """
    raw = os.environ.get(LATENCY_BUDGET_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        log.warning("ignoring unparseable %s=%r", LATENCY_BUDGET_ENV, raw)
        return 0.0
    return val if val > 0 else 0.0


#: Spellings of mean end-to-end latency in flight across the executors. The
#: benchmark path normalizes to ``e2el_mean_ms``, the raw harness payloads use
#: ``mean_e2el_ms``, and the GEAK lanes carry ``e2el_ms`` next to ``ttft_ms``.
#: Order is preference order.
_LATENCY_KEYS: tuple[str, ...] = ("e2el_mean_ms", "mean_e2el_ms", "e2el_ms")


def latency_from_result(result: Any) -> float | None:
    """Pull mean end-to-end latency out of a result or variant payload.

    Every promotion path has to answer the same question for the gate, and each
    reaches the field under a different name. Centralizing the aliases keeps a
    fail-closed gate from refusing a candidate that did report latency, just not
    under the spelling the caller happened to check.

    Args:
        result: A result, variant or lift payload; non-mappings return ``None``.

    Returns:
        The latency in ms, or ``None`` when no key holds a usable number.
    """
    if not isinstance(result, dict):
        return None
    for key in _LATENCY_KEYS:
        found = _finite_positive(result.get(key))
        if found is not None:
            return found
    return None


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
    "latency_from_result",
    "latency_keep_block",
    "read_session_budget",
    "resolve_latency_budget_ms",
]
