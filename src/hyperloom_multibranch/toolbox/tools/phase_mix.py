# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The op shares of a graded wave: the kernel shares of a trace's decode window and of its prefill window, weighted by the
share of the wave's time each phase takes.

TraceLens' splitter writes three steady-state windows of a serving trace. Its ``mixed`` window holds a few prefill steps
among decode steps and takes its time from whatever steps it holds: in a Qwen3-14B-FP8 profile (ISL 1024, OSL 1024,
concurrency 64) 2 prefill steps of 32 are 52% of the window, while a graded wave spends 11% of its time in prefill, so
paged attention read 14.1% of the mixed window and 28.5% of the decode window. The brief turns the shares into Amdahl
ceilings of the graded metric, so they are the graded wave's: the ``decode_only`` window's kernel shares and the
``prefilldecode`` window's (the longest run of prefill steps), each as :mod:`~.kernel_shares` and :mod:`~.kernel_launchers`
make them (the profiler's misreported events left out, a kernel name that several modules launch split by module), and

    op share = decode share x f_decode + prefill share x f_prefill

where ``f_*`` are the time fractions of a wave of ``conc`` requests (the point's ``CONC``, ``ISL`` and ``OSL``) that start
together, taken from the trace's own measured step times:

    prefill time = (conc x ISL / prefill tokens per step) x prefill step time
    decode time  = OSL x decode step time
    f_prefill = prefill time / (prefill time + decode time),  f_decode = 1 - f_prefill

A step's time is the window's kernel time over its steps, its prefill tokens per step the prompt tokens of the splitter's
step list (``execution_details.json``) over its steps. A range ratio below 1 shortens ISL and OSL alike, so it cancels. A
decode step is taken to give one token per request (no speculative decoding), and the decode window is one stretch of the
wave, so the step time is that stretch's.

A window the splitter wrote no file for, whose kernels fail the checks of :mod:`~.kernel_shares`, or whose steps the
splitter does not list gives no shares; the other window is then the op shares alone, and the result says so. With neither
the shares are degraded, and so they are when both are there but the request names no ``CONC``, ``ISL`` and ``OSL`` to
weight them by: a mix is never guessed.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_multibranch.toolbox.tools.kernel_launchers import Relabelled, read_launched_window
from hyperloom_multibranch.toolbox.tools.kernel_shares import GpuEvent, KernelWindow, kernel_window, read_gpu_events
from hyperloom_multibranch.toolbox.tools.op_shares import (
    EXCLUDED_CLAIMED_KEY,
    EXCLUDED_EVENTS_KEY,
    KERNEL_SHARES_KEY,
    NAME_COLUMN,
    OWN_DEGRADED_KEY,
    PHASE_NOTE_KEY,
    SHARE_COLUMN,
)

PREFILL, DECODE = "prefill", "decode"
#: The splitter window each phase is read from: its longest run of decode steps, its longest run of prefill steps.
PHASE_MODES = {PREFILL: "prefilldecode", DECODE: "decode_only"}
#: The splitter's list of what it extracted: per window file, the steps with their token counts.
EXECUTION_DETAILS = "execution_details.json"
#: The request keys of the point's workload the fractions are made of.
CONC_KEY, ISL_KEY, OSL_KEY = "split_conc", "split_isl", "split_osl"
#: The result's weight of each phase, and its windows with their facts.
PHASE_FRACTIONS_KEY, PHASE_WINDOWS_KEY = "phase_fractions", "phase_windows"
#: The columns of the mix table after the name and the weighted share: each phase's own share.
_PHASE_COLUMNS = {DECODE: "decode_percent", PREFILL: "prefill_percent"}
_ENCODING = "utf-8"
_FRACTION_DIGITS = 4
_TIME_DIGITS = 3


@dataclass(frozen=True)
class Workload:
    """The graded point's shape: concurrent requests, prompt tokens and output tokens of one."""

    conc: float
    isl: float
    osl: float

    @classmethod
    def from_hints(cls, hints: Mapping[str, str]) -> Workload | None:
        """The workload the request's hints name; ``None`` when one is missing."""
        try:
            conc, isl, osl = (float(hints[key]) for key in (CONC_KEY, ISL_KEY, OSL_KEY))
        except (KeyError, ValueError):
            return None
        return cls(conc, isl, osl) if all(math.isfinite(v) and v > 0 for v in (conc, isl, osl)) else None


@dataclass(frozen=True)
class StepCounts:
    """What the splitter lists of a window's steps: how many, and the prompt tokens in them."""

    steps: int
    tokens: int


@dataclass(frozen=True)
class PhaseWindow:
    """One phase's splitter window: its file, kernels, steps and prompt tokens, and why it gives no shares (empty when it does)."""

    phase: str
    chunk: str
    kernels: KernelWindow | None
    launched: Relabelled | None
    steps: int
    tokens: int
    reason: str

    @property
    def usable(self) -> bool:
        return not self.reason

    @property
    def step_ms(self) -> float:
        return self.kernels.kernel_ms / self.steps if self.kernels and self.steps else 0.0

    def facts(self) -> dict[str, Any]:
        """What the result says of the window."""
        kernels, launched = self.kernels, self.launched
        return {
            "chunk": self.chunk,
            "used": self.usable,
            "steps": self.steps,
            "tokens": self.tokens,
            "step_ms": round(self.step_ms, _TIME_DIGITS),
            "window_kernel_ms": round(kernels.kernel_ms, _TIME_DIGITS) if kernels else 0.0,
            "window_gpu_busy_percent": round(kernels.busy_percent, 2) if kernels else 0.0,
            "excluded_events": kernels.excluded_events if kernels else 0,
            "excluded_claimed_ms": round(kernels.excluded_claimed_ms, 2) if kernels else 0.0,
            "excluded_estimated_ms": round(kernels.excluded_estimated_ms, _TIME_DIGITS) if kernels else 0.0,
            "launcher_labelled_kernels": launched.launched if launched else 0,
            "graph_kernels_renamed": launched.renamed if launched else 0,
            "launcher_notes": list(launched.notes) if launched else [],
            "degraded_reason": self.reason,
        }


@dataclass(frozen=True)
class MixRow:
    """One kernel of the mix: its weighted share and the share it has in each phase's window."""

    name: str
    percent: float
    by_phase: Mapping[str, float]


@dataclass(frozen=True)
class GradedMix:
    """The shares of a graded wave: the weight of each phase (empty when degraded), the rows, why there are none, and a note
    when one window stands for the wave alone."""

    fractions: Mapping[str, float]
    rows: tuple[MixRow, ...]
    degraded_reason: str
    note: str


def rank_trace_reader(trace: Path) -> Callable[[], list[GpuEvent]]:
    """A reader of the rank trace's GPU events that reads the file once, however many windows price excluded events from it."""
    events: list[list[GpuEvent]] = []

    def read() -> list[GpuEvent]:
        if not events:
            events.append(read_gpu_events(trace))
        return events[0]

    return read


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def split_steps(split_dir: Path) -> dict[str, StepCounts]:
    """The steps and prompt tokens of each window file the splitter wrote to ``split_dir``, by the file's name; empty when
    its ``execution_details.json`` is missing or unreadable."""
    try:
        details = json.loads((split_dir / EXECUTION_DETAILS).read_text(encoding=_ENCODING))
    except (OSError, ValueError):
        return {}
    found: dict[str, StepCounts] = {}
    for entry in details if isinstance(details, list) else []:
        steps = entry.get("steps") if isinstance(entry, dict) else None
        if isinstance(steps, list) and isinstance(entry.get("output_path"), str):
            context = sum(_int(s.get("context_sum")) for s in steps if isinstance(s, dict))
            found[Path(entry["output_path"]).name] = StepCounts(len(steps), context)
    return found


def read_phase_window(
    phase: str,
    chunk: Path | None,
    counts: Mapping[str, StepCounts],
    capture: Path | None,
    reference: Callable[[], Iterable[GpuEvent]],
) -> PhaseWindow:
    """``phase``'s window: the kernels of ``chunk`` (``None``: the splitter wrote none) and its steps from ``counts``.

    Raises :class:`~.kernel_shares.KernelTraceError` when the file is not a readable trace.
    """
    if chunk is None:
        return PhaseWindow(
            phase, "", None, None, 0, 0, f"the splitter wrote no {PHASE_MODES[phase]} steady-state window"
        )
    launched = read_launched_window(chunk, capture)
    kernels = kernel_window(launched.events, reference)
    listed = counts.get(chunk.name, StepCounts(0, 0))
    reason = kernels.degraded_reason
    if not reason and not listed.steps:
        reason = f"the splitter's {EXECUTION_DETAILS} lists no steps for {chunk.name}"
    if not reason and phase == PREFILL and not listed.tokens:
        reason = f"the splitter's {EXECUTION_DETAILS} lists no prompt tokens in the steps of {chunk.name}"
    return PhaseWindow(phase, chunk.name, kernels, launched, listed.steps, listed.tokens, reason)


def wave_ms(window: PhaseWindow, workload: Workload) -> float:
    """The time the phase takes in a wave of ``workload``, in ms, from the window's step time (see the module docstring)."""
    if window.phase == PREFILL:
        per_step = window.tokens / window.steps
        return workload.conc * workload.isl / per_step * window.step_ms
    return workload.osl * window.step_ms


def _time_fractions(usable: Sequence[PhaseWindow], workload: Workload) -> dict[str, float]:
    """The share of a wave's time each of the ``usable`` windows' phases takes; empty when the times are not finite."""
    times = {w.phase: wave_ms(w, workload) for w in usable}
    total = math.fsum(times.values())
    return {phase: ms / total for phase, ms in times.items()} if math.isfinite(total) and total > 0 else {}


def _rows(usable: Sequence[PhaseWindow], weights: Mapping[str, float]) -> tuple[MixRow, ...]:
    shares = {w.phase: w.kernels.shares for w in usable if w.kernels}
    names = {name for phase_shares in shares.values() for name in phase_shares}
    rows = []
    for name in names:
        by_phase = {phase: shares[phase].get(name, 0.0) for phase in weights}
        rows.append(MixRow(name, math.fsum(weights[phase] * share for phase, share in by_phase.items()), by_phase))
    return tuple(sorted(rows, key=lambda row: (-row.percent, row.name)))


def graded_mix(windows: Sequence[PhaseWindow], workload: Workload | None) -> GradedMix:
    """The graded wave's shares from the phase ``windows`` that give any; degraded when none, or the two cannot be weighted."""
    usable = [w for w in windows if w.usable]
    if not usable:
        return GradedMix({}, (), "; ".join(f"the {w.phase} window: {w.reason}" for w in windows), "")
    note = ""
    if len(usable) == 1:
        weights = {usable[0].phase: 1.0}
        left = next(w for w in windows if not w.usable)
        note = f"the {usable[0].phase} window stands for the whole wave: the {left.phase} window gives none ({left.reason})"
    elif workload is None:
        reason = f"the request names no {CONC_KEY}, {ISL_KEY} and {OSL_KEY} to weight the prefill and decode windows by"
        return GradedMix({}, (), reason, "")
    else:
        weights = _time_fractions(usable, workload)
        if not weights:
            return GradedMix(
                {}, (), "the time of a wave of this workload is not a finite number from the windows' step times", ""
            )
    return GradedMix({phase: weights.get(phase, 0.0) for phase in PHASE_MODES}, _rows(usable, weights), "", note)


def write_mix_table(mix: GradedMix, path: Path) -> None:
    """The mix's kernels with their weighted shares, and each phase's own, the table the op shares are read from."""
    with path.open("w", encoding=_ENCODING, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((NAME_COLUMN, SHARE_COLUMN, *_PHASE_COLUMNS.values()))
        for row in mix.rows:
            writer.writerow(
                (row.name, f"{row.percent:.6f}", *(f"{row.by_phase.get(phase, 0.0):.6f}" for phase in _PHASE_COLUMNS))
            )


def mix_fields(windows: Sequence[PhaseWindow], mix: GradedMix, table: Path) -> dict[str, Any]:
    """What a result says of its op shares: where the table is (empty when degraded), the weight of each phase, the windows
    with their facts, the events left out of the shares, why there are none, and the note on a window that stands alone."""
    used = [w.kernels for w in windows if w.usable and w.kernels]
    return {
        KERNEL_SHARES_KEY: "" if mix.degraded_reason else str(table),
        PHASE_FRACTIONS_KEY: {phase: round(fraction, _FRACTION_DIGITS) for phase, fraction in mix.fractions.items()},
        PHASE_WINDOWS_KEY: {w.phase: w.facts() for w in windows},
        EXCLUDED_EVENTS_KEY: sum(k.excluded_events for k in used),
        EXCLUDED_CLAIMED_KEY: round(math.fsum(k.excluded_claimed_ms for k in used), 2),
        OWN_DEGRADED_KEY: mix.degraded_reason,
        PHASE_NOTE_KEY: mix.note,
    }


__all__ = [
    "DECODE",
    "EXECUTION_DETAILS",
    "PHASE_MODES",
    "PREFILL",
    "GradedMix",
    "MixRow",
    "PhaseWindow",
    "StepCounts",
    "Workload",
    "graded_mix",
    "mix_fields",
    "rank_trace_reader",
    "read_phase_window",
    "split_steps",
    "wave_ms",
    "write_mix_table",
]
