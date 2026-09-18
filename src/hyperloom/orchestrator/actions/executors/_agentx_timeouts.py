# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX client warmup parameters, independent of the benchmark process cap."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

log = logging.getLogger(__name__)

AGENTX_CANON_WARMUP_GRACE_SEC = 1800  # aiperf_client.sh's CANON_WARMUP_GRACE

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
