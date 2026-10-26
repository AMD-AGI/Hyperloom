# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How long a test's short-prompt check may take, sized from what this session measured.

The check boots its fresh server processes one after another and asks each the prompts, so the time it needs is about
the number of processes times (a server boot plus the time to answer the prompts). The rule pack's ``phase_cap_sec`` is
one fixed time for every model: a model whose server boots slowly fits only one or two processes in it and the check
ends incomplete, and on a fast one a hung process holds the cards long after the others are done.

:class:`CheckTimes` keeps the longest server boot the session measured (a test's own boot, and the boot of each process of
a check) and the longest time a process of a check took to answer the prompts. Once a boot is measured, a check's cap is
``processes * (boot + prompts) * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC``, the prompt time counting as nothing until a
check has measured it; before any boot is measured the cap is the pack's. The longest, not the latest, is kept: the check
stops early once its next process would not fit, so a cap that is too long costs nothing unless a process hangs, and one
that is too short makes the check incomplete.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from hyperloom.measure.short_prompts.outputs import OK, OutputsError, ProcessEntry, parse_outputs, read_outputs_document
from hyperloom.measure.short_prompts.plan import ShortPromptPlan

__all__ = ["CAP_MARGIN_FRACTION", "CAP_MARGIN_SEC", "CheckTimes", "landed_processes"]

#: The margin a sized cap leaves over the measured time: a share of it, and a fixed time on top.
CAP_MARGIN_FRACTION = 0.25
CAP_MARGIN_SEC = 120.0
_CAP_DIGITS = 1

_LOG = logging.getLogger(__name__)


def _measured(seconds: float | None) -> float | None:
    return seconds if seconds is not None and math.isfinite(seconds) and seconds > 0 else None


class CheckTimes:
    """The session's longest measured server boot and prompt time, and the cap of a check sized from them."""

    def __init__(self) -> None:
        self._boot: float | None = None
        self._prompts: float | None = None

    def saw_boot(self, seconds: float | None) -> None:
        """A server booted in ``seconds`` (``None``, or a value that is not a positive time, is no measurement)."""
        measured = _measured(seconds)
        if measured is not None:
            self._boot = max(self._boot or 0.0, measured)

    def saw_check(self, processes: Iterable[ProcessEntry]) -> None:
        """The processes of a check that landed: each that answered every prompt measured a boot and a prompt time."""
        for entry in processes:
            if entry.status != OK:
                continue
            self.saw_boot(entry.boot_seconds)
            measured = _measured(entry.query_seconds)
            if measured is not None:
                self._prompts = max(self._prompts or 0.0, measured)

    def sized(self, plan: ShortPromptPlan | None) -> ShortPromptPlan | None:
        """``plan`` with its cap sized from the measured times; as it is when there is no plan or no boot measured."""
        if plan is None or self._boot is None:
            return plan
        each = self._boot + (self._prompts or 0.0)
        cap = round(plan.processes * each * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC, _CAP_DIGITS)
        if cap != plan.phase_cap_sec:
            _LOG.info(
                "short-prompt check capped at %.0f s: %d processes of %.0f s boot and %.0f s prompts measured this session "
                "(the rule pack's cap is %.0f s)",
                cap,
                plan.processes,
                self._boot,
                self._prompts or 0.0,
                plan.phase_cap_sec,
            )
        return replace(plan, phase_cap_sec=cap)


def landed_processes(directory: Path) -> tuple[ProcessEntry, ...]:
    """The processes of the check whose outputs landed in ``directory``; none when there is no readable outputs file."""
    try:
        return parse_outputs(read_outputs_document(directory)).processes
    except OutputsError:
        return ()
