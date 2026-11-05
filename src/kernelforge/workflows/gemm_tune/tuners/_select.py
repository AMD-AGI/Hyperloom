# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a tuner run asks aiter for the lowest-energy accurate kernel per shape (``--select power``).

aiter's power mode keeps, among the kernels that pass its accuracy gate, the one with the lowest package energy
(``us x PPT watts``) and stamps those rows ``select=power``. It writes no row for a shape whose watts it could not
measure. A power table is therefore usually slower than a latency one, and only an E2E measurement can say whether it
is worth deploying.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from ..script_probe import probe_script
from .base import LATENCY, POWER

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectPlan:
    """The ``--select`` arguments one tuner run gets, and why power was not asked for when it was wanted."""

    args: tuple[str, ...]
    note: str = ""

    @property
    def power(self) -> bool:
        return bool(self.args)


def select_plan(script: Path, select: str) -> SelectPlan:
    """Ask ``script`` for power selection only when its ``--help`` shows it accepts ``--select``.

    Every aiter tuner of a release with power selection lists ``--select`` (the parser is shared), and Hyperloom's
    tuners all opt in to it. An older aiter rejects the flag, and a script whose ``--help`` could not be read is not
    assumed to accept it, so both tune for latency instead of failing.
    """
    if select != POWER:
        return SelectPlan(())
    # Not from the probe cache: it is keyed on the script's bytes, and --select comes from aiter's shared parser, which
    # an aiter upgrade changes without touching the script.
    surface = probe_script(script, use_cache=False)
    if not surface.probed:
        return SelectPlan((), f"{script.name} --help could not be read ({surface.reason}); tuned for latency")
    if "--select" not in surface.flags:
        return SelectPlan((), f"{script.name} has no --select (aiter predates power selection); tuned for latency")
    return SelectPlan(("--select", POWER))


def power_row_count(table: Path) -> int:
    """Rows of ``table`` that aiter stamped ``select=power``."""
    with table.open(newline="", encoding="utf-8", errors="replace") as handle:
        return sum(1 for row in csv.DictReader(handle) if (row.get("select") or "").strip().lower() == POWER)


def wrote_power_rows(plan: SelectPlan, table: Path | str) -> bool:
    """Whether a run that asked for power left ``table`` holding rows aiter stamped ``select=power``."""
    return plan.power and bool(table) and Path(table).is_file() and power_row_count(Path(table)) > 0


def selected(plan: SelectPlan, table: Path | str) -> tuple[str, str]:
    """``(select, note)`` for a finished run: power only when the deployed table holds power rows.

    A note means power was asked for and the table holds latency picks instead, so it is also logged as a warning.
    """
    if not plan.power:
        select, note = LATENCY, plan.note
    elif not table or not Path(table).is_file():
        select, note = LATENCY, "the tuner left no table"
    elif power_row_count(Path(table)) == 0:
        select, note = LATENCY, "aiter measured package power for no shape, so the table has no power row"
    else:
        select, note = POWER, ""
    if note:
        log.warning("power selection not applied: %s", note)
    return select, note


def kept(select: str, note: str, table: Path | str) -> tuple[str, str]:
    """``(select, note)`` after Hyperloom's own filters ran on the table: one they emptied deploys no power pick."""
    if select != POWER or power_row_count(Path(table)) > 0:
        return select, note
    note = "every power row was dropped as unservable, inaccurate or unverified"
    log.warning("power selection not applied: %s", note)
    return LATENCY, note
