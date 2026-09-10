# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX benchmark timeout derivations shared by baseline and grid launches."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

log = logging.getLogger(__name__)

# An AgentX baseline does not fit either of the caps above, and the cold-start detector cannot see why.
AGENTX_BASELINE_OVERHEAD_SEC = 7200  # setup + corpus + warmup + first-compile
AGENTX_DEFAULT_DURATION_SEC = 3600  # mirrors aiperf_client.sh's default

# The warmup share of that overhead is not a constant either, and it is the share that actually varies by model:
# aiperf_client.sh bounds the warmup drain with AGENTX_WARMUP_GRACE_PERIOD, so a model whose warmup runs long is a
# model whose operator has already had to raise that knob for the round to complete at all.
AGENTX_CANON_WARMUP_GRACE_SEC = 1800  # aiperf_client.sh's CANON_WARMUP_GRACE
_AGENTX_NON_WARMUP_OVERHEAD_SEC = AGENTX_BASELINE_OVERHEAD_SEC - AGENTX_CANON_WARMUP_GRACE_SEC

# ...and the warmup share does not only vary by model, it varies by CONCURRENCY, which the grace knob cannot express
# because it is one flat number.
AGENTX_CANON_WARMUP_CONC = 8

# Seen (warning, scaling) payloads, so a conc sweep does not reprint them once per rung per arm.
_AGENTX_SAID: set[tuple] = set()


def _say_once(emit, key: tuple) -> None:
    """Call ``emit`` the first time this exact payload appears in the process."""
    if key in _AGENTX_SAID:
        return
    _AGENTX_SAID.add(key)
    emit()


def _agentx_positive_int(src: "Mapping[str, str]", name: str) -> int:
    """Read a positive integer from ``src``; 0 when unset or unusable."""
    raw = (src.get(name) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        try:
            as_float = float(raw)
        except ValueError:
            return 0
        if not as_float.is_integer():
            return 0
        value = int(as_float)
    return value if value > 0 else 0


def _agentx_conc(src: "Mapping[str, str]") -> int:
    """Concurrency for the round, from CONC; 0 when unset/unparseable."""
    return _agentx_positive_int(src, "CONC")


def agentx_warmup_grace_conc(env: "Mapping[str, str] | None" = None) -> int:
    """The concurrency ``AGENTX_WARMUP_GRACE_PERIOD`` was measured at."""
    src = os.environ if env is None else env
    return _agentx_positive_int(src, "AGENTX_WARMUP_GRACE_CONC") or AGENTX_CANON_WARMUP_CONC


def agentx_warmup_grace_sec(env: "Mapping[str, str] | None" = None) -> int:
    """The warmup bound for this round: the operator's grace, scaled by CONC."""
    src = os.environ if env is None else env
    measured = _agentx_positive_int(src, "AGENTX_WARMUP_GRACE_PERIOD")
    if not measured:
        # Nothing was measured, so there is nothing to scale.
        return AGENTX_CANON_WARMUP_GRACE_SEC
    grace = measured
    # Scaling requires the anchor to be DECLARED, not assumed.
    if not _agentx_positive_int(src, "AGENTX_WARMUP_GRACE_CONC"):
        return grace
    # The client's warmup is linear in CONC by construction (per-lane requests x CONC lanes), but the grace knob is a
    # flat number, so a grace measured at one concurrency under-budgets every higher one.
    anchor = agentx_warmup_grace_conc(src)
    conc = _agentx_conc(src)
    if conc <= anchor:
        return grace
    scaled = (grace * conc) // anchor
    _say_once(
        lambda: log.info(
            "agentx_warmup_grace_sec: scaling the warmup share %ds -> %ds for CONC=%d "
            "(warmup work is linear in CONC; the grace is declared as measured at "
            "CONC=%d via AGENTX_WARMUP_GRACE_CONC). The floor only raises the bound -- "
            "an over-large one costs a longer wait on a hung round, an under-sized one "
            "kills a warmup that would have finished.",
            grace,
            scaled,
            conc,
            anchor,
        ),
        ("grace-scaled", grace, scaled, conc, anchor),
    )
    return scaled


def agentx_baseline_timeout_sec(env: "Mapping[str, str] | None" = None) -> int:
    """Resolve the AgentX baseline cap: explicit, else duration + overhead."""
    src = os.environ if env is None else env

    # One parser for every knob in this module.
    def _int(name: str, default: int) -> int:
        return _agentx_positive_int(src, name) or default

    def _is_valid_override(name: str) -> bool:
        return _agentx_positive_int(src, name) > 0

    explicit = _int("AGENTX_BASELINE_TIMEOUT_SEC", 0)
    if explicit:
        return explicit

    # Same validity bar as `_int` itself (parses to a positive int) rather than "non-empty string" -- otherwise an
    # invalid override (e.g. "abc" or "-1") both silently falls back to the default AND suppresses the warning meant
    # to flag exactly that case.
    if _is_valid_override("AGENTX_BASELINE_OVERHEAD_SEC"):
        overhead = _int("AGENTX_BASELINE_OVERHEAD_SEC", AGENTX_BASELINE_OVERHEAD_SEC)
        grace = None
    else:
        # Derive the warmup share from the same knob that bounds it in the client, via the same helper the client's
        # value is exported from, so the cap and the client's --warmup-grace-period cannot drift apart.
        grace = agentx_warmup_grace_sec(src)
        overhead = _AGENTX_NON_WARMUP_OVERHEAD_SEC + grace
        if not _is_valid_override("AGENTX_WARMUP_GRACE_PERIOD"):
            # Nothing has been tuned for this model at all.
            _say_once(
                lambda: log.warning(
                    "agentx_baseline_timeout_sec: neither AGENTX_BASELINE_OVERHEAD_SEC nor "
                    "AGENTX_WARMUP_GRACE_PERIOD is set, so the overhead falls back to the "
                    "canonical %ds (= %ds non-warmup + %ds canonical warmup grace). That "
                    "grace is calibrated on GLM-5.2/Qwen3.8 and may be far too small for a "
                    "long-context or slow-prefill model -- a raw aiperf run against Kimi-K3 "
                    "at concurrency=64 measured warmup alone taking ~12075s. Raise "
                    "AGENTX_WARMUP_GRACE_PERIOD (the client honours it too, so the warmup "
                    "and this cap stay consistent) or pin AGENTX_BASELINE_OVERHEAD_SEC.",
                    overhead,
                    _AGENTX_NON_WARMUP_OVERHEAD_SEC,
                    AGENTX_CANON_WARMUP_GRACE_SEC,
                ),
                ("untuned-overhead", overhead),
            )
    duration = _int("AGENTX_DURATION", AGENTX_DEFAULT_DURATION_SEC)
    total = duration + overhead
    # Log every input, so a timeout in the field can be read back to the value that produced it instead of guessing
    # which knob was in play.
    _say_once(
        lambda: log.info(
            "agentx_baseline_timeout_sec: %ds = duration %ds + overhead %ds (%s)",
            total,
            duration,
            overhead,
            "explicit AGENTX_BASELINE_OVERHEAD_SEC"
            if grace is None
            else f"{_AGENTX_NON_WARMUP_OVERHEAD_SEC}s non-warmup + {grace}s warmup grace",
        ),
        ("baseline-timeout", total, duration, overhead, grace),
    )
    return total
