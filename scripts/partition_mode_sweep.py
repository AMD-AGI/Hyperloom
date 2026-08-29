#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sweep the AMD compute-partition modes and report which one the workload wants.

Sets each requested mode on one card in turn, runs the same fixed benchmark on
every partition that mode creates, and sums the result. The comparison it prints
is the one an operator needs before committing a multi-hour optimization
session: under which shape does this workload go fastest, and what does that
cost in per-request latency.

**The fan-out is the whole point.** A partition gets a fraction of the card, so
a benchmark that loads one partition and ignores the rest measures a fraction of
the card. Measured on this node, one MI355X in ``CPX`` presents eight 32-CU
partitions -- a single-partition run reports roughly an eighth of the card's
throughput, making ``CPX`` look catastrophic when in aggregate it may well win.
Every figure here is therefore the sum over a mode's partitions with all of them
loaded at once, and a mode whose partitions cannot all be loaded is reported as
unmeasured rather than as slow.

**Why the privileged set lives here and not in the optimizer.** Changing the
mode is a card-wide privileged operation that evicts every process holding a GPU
context. That is reasonable between benchmarks in an operator-run script and
unreasonable inside an optimization loop that also runs agent-authored code, so
``hyperloom`` itself only ever *reads* the mode -- see
``hyperloom.common.gpu_partition`` -- and refuses at launch any session whose
streams will not fit one partition. This script is the other half of that
split: the boundary that establishes the shape the optimizer then treats as
fixed for the whole session.

**Device indices are not portable between tools, so partitions are matched on CU
count.** Measured on an 8-card MI355X node with card 0 in ``CPX``: ``amd-smi``
orders by PCI address and calls the eight partitions devices 0-7, while HSA/HIP
enumerates whole cards first and calls them devices 7-14. A device list computed
with one tool and handed to the other is wrong, and wrong invisibly -- the
benchmark runs to completion, on the wrong silicon. Partitions are selected here
the way :func:`~hyperloom.common.gpu_partition.partition_device_predicate`
documents: by matching the expected CU count, in the HIP index space the
benchmark itself will use, and narrowed to the card being swept so that an
identical CU count on a neighbouring card cannot be mistaken for a partition.

Only the target card is repartitioned; the rest of the node is left alone. That
keeps total silicon constant across the sweep, which is what makes the modes
comparable to each other.

Usage::

    # print the plan, touch nothing
    python3 scripts/partition_mode_sweep.py --benchmark-config bench.yaml --dry-run

    # sweep the modes the card reports it supports
    python3 scripts/partition_mode_sweep.py \\
        --benchmark-config /path/to/benchmark.yaml \\
        --modes SPX,DPX,QPX,CPX \\
        --output-dir /shared/partition-sweep

    # arbitrary workload; {device} and {output_dir} are substituted per partition
    python3 scripts/partition_mode_sweep.py \\
        --benchmark-command 'my_bench --gpu {device} --out {output_dir}' \\
        --per-stream-gib 20.7

Exit codes:

    0  the sweep completed and a winner was reported
    1  the sweep ran but no mode produced a valid measurement
    2  the request was refused before anything was changed
    3  the sweep finished but the card could not be restored to its entry mode
    4  the sweep stopped on an error it does not model, after reporting whatever
       it had already measured and restoring the card

Every path out of a started sweep goes through the restore and the report,
including an unexpected exception, so ``3`` stays reachable and the modes
measured before a failure are never lost to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess  # nosec B404 - fixed argv, never a shell.
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperloom.common.coerce import to_float  # noqa: E402
from hyperloom.common.gpu_partition import (  # noqa: E402
    MODE_PARTITION_COUNTS,
    PARTITION_COUNT_ENV,
    PARTITION_CU_ENV,
    PARTITION_MODE_ENV,
    PARTITION_STREAMS_ENV,
    PARTITION_TOTAL_STREAMS_ENV,
    PartitionError,
    PartitionLayout,
    fits_in_partition,
    layout_for,
    parse_mode,
    partition_device_predicate,
    read_device_gib,
)

#: Read timeout for an ``amd-smi`` query. Generous because a card mid-transition
#: answers slowly, and a spurious timeout here reads as a hardware fault.
_READ_TIMEOUT_S = 30.0

#: Timeout for the set itself.
_SET_TIMEOUT_S = 120.0

#: How long to keep retrying a set the card refuses because it still holds
#: processes. The condition clears on its own once the previous benchmark's
#: contexts are torn down, so the correct response is to wait rather than fail:
#: this script's own predecessor run is the most likely holder.
_DRAIN_TIMEOUT_S = 120.0
_DRAIN_POLL_S = 2.0

#: Pause between a successful set and trusting the enumeration. The set returns
#: before the new devices appear; measured here at one to two seconds, so this
#: is that with margin. Without it the first device scan after a set sees the
#: old topology and the run is attributed to the wrong mode.
_SETTLE_S = 5.0

#: Substrings marking a refusal as "busy, try again" rather than "wrong".
#: Matched narrowly so that permanent failures -- unknown mode, missing binary,
#: denied permission -- fail fast instead of retrying for two minutes.
#: ``AMDSMI_STATUS_BUSY`` is the authoritative one; the human-readable half of
#: the message is not stable enough to match on alone.
_BUSY_MARKERS = ("amdsmi_status_busy", "device busy", "resident process", "try again")

#: What ``amd-smi process`` puts in ``process_list`` when a GPU is idle: a bare
#: string where a caller would reasonably expect a list of dicts. Parsed
#: explicitly because treating it as a process would make every sweep refuse to
#: start on a perfectly free node.
_NO_PROCESS_MARKER = "no running processes"

#: Throughput fields summed across partitions, in preference order for the
#: headline figure. Named to match ``benchmark_result.extract_benchmark_measurement``.
_THROUGHPUT_FIELDS = ("output_throughput", "total_token_throughput", "request_throughput")

#: Per-request latency fields, in the same preference order the optimizer's
#: latency budget uses.
_LATENCY_FIELDS = ("e2el_mean_ms", "mean_e2el_ms", "e2el_ms")


class SweepError(RuntimeError):
    """Raised when the sweep cannot proceed and nothing has been left changed."""


@dataclass(frozen=True)
class HsaAgent:
    """One GPU agent as HSA/HIP enumerates it.

    Attributes:
        index: Position among GPU agents, which is the index
            ``ROCR_VISIBLE_DEVICES`` and HIP both use.
        cu: Compute units the agent reports.
        bus: PCI bus byte, identifying the physical card. Partitions of one card
            share it and differ only in PCI function.
        device: PCI device number.
        function: PCI function number.
    """

    index: int
    cu: int
    bus: int
    device: int
    function: int

    @property
    def bdf(self) -> str:
        """Human-readable ``bus:device.function``."""
        return f"{self.bus:02x}:{self.device:02x}.{self.function}"


@dataclass
class PartitionRun:
    """One benchmark process on one partition."""

    partition_index: int
    hsa_device: int
    output_dir: Path
    returncode: int | None = None
    measurement: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error and bool(self.measurement)


@dataclass
class ModeResult:
    """What one mode produced, or why it produced nothing."""

    mode: str
    layout: PartitionLayout | None = None
    runs: list[PartitionRun] = field(default_factory=list)
    skipped: str = ""
    error: str = ""

    @property
    def measured(self) -> bool:
        """Whether every partition returned a usable measurement.

        Deliberately all-or-nothing. A mode with six of eight partitions
        reporting is not a slower mode, it is an unmeasured one, and summing the
        six would understate it by a quarter while looking like a result.
        """
        return bool(self.runs) and all(run.ok for run in self.runs)

    def total(self, fields: Sequence[str] = _THROUGHPUT_FIELDS) -> float | None:
        """Sum the first available throughput field across partitions."""
        for name in fields:
            values = [to_float(run.measurement.get(name)) for run in self.runs]
            if values and all(v is not None for v in values):
                return sum(v for v in values if v is not None)
        return None

    def worst_latency_ms(self) -> float | None:
        """The slowest partition's mean latency.

        The worst partition rather than the average of them, because a request
        landing on the slow one is not consoled by the mean.
        """
        for name in _LATENCY_FIELDS:
            values = [to_float(run.measurement.get(name)) for run in self.runs]
            present = [v for v in values if v is not None]
            if present and len(present) == len(values):
                return max(present)
        return None


# ------------------------------------------------------------------ amd-smi


def _sudo_prefix(use_sudo: bool) -> list[str]:
    """Return the privilege prefix for a mutating call.

    ``-n`` matters: an unattended sweep that stops at an interactive password
    prompt hangs until someone notices, with nothing in the output saying why.
    """
    return ["sudo", "-n"] if use_sudo else []


def _amd_smi_json(args: Sequence[str], timeout_s: float = _READ_TIMEOUT_S, *, sudo: bool = False) -> object:
    """Run an ``amd-smi`` subcommand with ``--json`` and parse its output.

    Args:
        args: Subcommand and flags.
        timeout_s: Per-call timeout.
        sudo: Route through ``sudo -n``. Needed for the profile query, which
            degrades every field to ``"N/A"`` rather than failing when it lacks
            privilege.

    Raises:
        SweepError: If ``amd-smi`` is missing, fails, times out, or returns
            output that is not JSON.
    """
    cmd = [*_sudo_prefix(sudo), "amd-smi", *args, "--json"]
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SweepError("amd-smi not found; this sweep needs it on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SweepError(f"amd-smi {' '.join(args)} timed out after {timeout_s:.0f}s") from exc
    if proc.returncode != 0:
        raise SweepError(f"amd-smi {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise SweepError(f"amd-smi {' '.join(args)} returned unparseable JSON") from exc


def _is_busy(detail: str) -> bool:
    """Whether a failed set was refused because the card was still occupied."""
    low = (detail or "").lower()
    return any(marker in low for marker in _BUSY_MARKERS)


def current_modes(payload: object) -> dict[int, str]:
    """Extract GPU id to mode from an ``amd-smi partition --json`` payload."""
    rows: list[dict] = []
    if isinstance(payload, dict):
        raw = payload.get("current_partition")
        if isinstance(raw, list):
            rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
    modes: dict[int, str] = {}
    for row in rows:
        try:
            gpu_id = int(row.get("gpu_id"))
        except (TypeError, ValueError):
            continue
        mode = str(row.get("accelerator_type") or "").strip().upper().rstrip("*")
        if mode and mode != "N/A":
            modes[gpu_id] = mode
    return modes


def read_mode(gpu_id: int) -> str:
    """Read one card's live compute-partition mode."""
    modes = current_modes(_amd_smi_json(["partition"]))
    if gpu_id not in modes:
        raise SweepError(f"amd-smi reported no compute-partition state for GPU {gpu_id}")
    return modes[gpu_id]


def supported_modes(payload: object) -> tuple[str, ...]:
    """Extract the modes a card reports it can enter, in profile order.

    An empty result means "the card did not say", not "the card supports
    nothing": the profile query returns ``"N/A"`` for everything when run
    without privilege, and those two cases must not collapse into one answer.
    """
    rows: list[dict] = []
    if isinstance(payload, dict):
        raw = payload.get("partition_profiles")
        if isinstance(raw, list):
            rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
    seen: list[str] = []
    for row in rows:
        # The table is sparse: a profile's first row names it, and the rows
        # after it continue the same profile with its other resources under
        # blank identity fields. Only named rows describe a profile.
        mode = str(row.get("accelerator_type") or "").strip().upper().rstrip("*")
        if mode and mode != "N/A" and mode in MODE_PARTITION_COUNTS and mode not in seen:
            seen.append(mode)
    return tuple(seen)


def _live_processes(listing: object, *, gpu_id: int) -> int:
    """Count the entries of one card's ``process_list`` that are real processes.

    The idle sentinel turns up at either depth -- as the whole ``process_list``,
    or as the lone entry's ``process_info`` -- and a process turns up either
    wrapped in ``process_info`` or as the entry itself. All four were observed.
    Anything else raises, because a shape this function does not recognise is
    indistinguishable from an idle card once it has been counted as zero.
    """
    if isinstance(listing, str):
        entries: list[object] = [listing]
    elif isinstance(listing, list):
        entries = list(listing)
    else:
        raise SweepError(
            f"amd-smi process gave GPU {gpu_id} a process_list of type "
            f"{type(listing).__name__}; expected a list or the idle string"
        )
    live = 0
    for entry in entries:
        info = entry.get("process_info", entry) if isinstance(entry, dict) else entry
        if isinstance(info, str):
            # Only the known idle sentinel means idle. An unrecognised string
            # counts as a process: over-counting refuses a sweep, under-counting
            # evicts somebody's work.
            if _NO_PROCESS_MARKER not in info.lower():
                live += 1
        elif isinstance(info, dict):
            if str(info.get("name") or "").strip():
                live += 1
        else:
            raise SweepError(
                f"amd-smi process gave GPU {gpu_id} a process entry of type "
                f"{type(info).__name__}; expected a mapping or a string"
            )
    return live


def resident_processes(payload: object) -> dict[int, int]:
    """Count real processes holding a context on each GPU.

    ``amd-smi`` reports an idle GPU as a ``process_list`` holding the *string*
    ``"No running processes detected"`` rather than an empty list, so a naive
    length check finds one process on every idle card and this sweep would
    refuse to start on a free node.

    Every departure from the documented shape raises instead of being skipped.
    This count is the only thing between a payload this parser does not
    understand and an ``amd-smi set`` that evicts whatever is running, and a
    parser that answers ``{}`` for a payload it cannot read reports a busy node
    as a free one -- the single wrong answer here that destroys work. Refusing
    costs an operator one ``--allow-busy``; guessing costs somebody a job.

    Raises:
        SweepError: If the payload is not a list of per-GPU rows, or a row is
            missing ``gpu`` or ``process_list``, or either field has a type
            this parser does not model.
    """
    if not isinstance(payload, list):
        raise SweepError(
            f"amd-smi process returned {type(payload).__name__}, not the expected list of "
            f"per-GPU rows, so which cards are in use cannot be read from it"
        )
    counts: dict[int, int] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise SweepError(f"amd-smi process returned a {type(row).__name__} where a GPU row was expected")
        missing = [key for key in ("gpu", "process_list") if key not in row]
        if missing:
            raise SweepError(f"amd-smi process returned a GPU row without {' or '.join(missing)}")
        try:
            gpu_id = int(row["gpu"])
        except (TypeError, ValueError) as exc:
            raise SweepError(f"amd-smi process reported a GPU id of {row['gpu']!r}, which is not a number") from exc
        counts[gpu_id] = _live_processes(row["process_list"], gpu_id=gpu_id)
    return counts


def card_bus(payload: object, gpu_id: int) -> int:
    """Read the PCI bus byte of one card from an ``amd-smi list`` payload.

    The bus identifies the physical card and does not change when it is
    repartitioned, so it is captured once and used afterwards to tell this
    card's partitions from an identically-shaped neighbour.
    """
    rows = payload if isinstance(payload, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("gpu")) != gpu_id:
                continue
        except (TypeError, ValueError):
            continue
        bdf = str(row.get("bdf") or "")
        match = re.search(r"(?:[0-9a-fA-F]{4}:)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.(\d)", bdf)
        if match:
            return int(match.group(1), 16)
    raise SweepError(f"amd-smi list reported no PCI address for GPU {gpu_id}")


def set_mode(
    gpu_id: int,
    mode: str,
    *,
    sudo: bool,
    drain_timeout_s: float = _DRAIN_TIMEOUT_S,
    settle_s: float = _SETTLE_S,
) -> str:
    """Set one card's compute-partition mode and verify it took effect.

    The verification is the point. ``amd-smi set`` reports success for a change
    that has only been staged, and it exits zero on some permission failures, so
    a caller trusting the exit code goes on to benchmark the old topology while
    labelling the results with the new mode -- a wrong number with a reassuring
    log line above it.

    Args:
        gpu_id: Card to reconfigure.
        mode: Target mode.
        sudo: Whether to route the set through ``sudo -n``.
        drain_timeout_s: How long to keep retrying while the card reports
            resident processes. Zero fails on the first refusal.
        settle_s: Pause before reading back, so the new devices have appeared.

    Returns:
        The mode read back from the card, equal to ``mode`` on success.

    Raises:
        SweepError: If the mode is unknown, the set fails, or the read-back
            disagrees with what was asked for.
    """
    canonical = parse_mode(mode)
    if canonical not in MODE_PARTITION_COUNTS:
        raise SweepError(f"unknown compute-partition mode {mode!r}")
    if read_mode(gpu_id) == canonical:
        return canonical

    cmd = [*_sudo_prefix(sudo), "amd-smi", "set", "-g", str(gpu_id), "--compute-partition", canonical]
    deadline = time.monotonic() + max(0.0, drain_timeout_s)
    waited = False
    while True:
        try:
            proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
                cmd,
                capture_output=True,
                text=True,
                timeout=_SET_TIMEOUT_S,
                # Some builds ask for confirmation; answer rather than block.
                input="Y\n",
                check=False,
            )
        except FileNotFoundError as exc:
            raise SweepError("amd-smi not found; this sweep needs it on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SweepError(f"setting {canonical} on GPU {gpu_id} timed out") from exc
        if proc.returncode == 0:
            break
        detail = (proc.stderr or proc.stdout).strip()
        if _is_busy(detail) and time.monotonic() < deadline:
            if not waited:
                print(f"  GPU {gpu_id} busy; waiting for it to drain before setting {canonical}")
                waited = True
            time.sleep(_DRAIN_POLL_S)
            continue
        hint = (
            f" The card still held processes after {drain_timeout_s:.0f}s."
            if _is_busy(detail)
            else " A card with resident processes refuses to repartition; stop them first."
        )
        raise SweepError(f"setting {canonical} on GPU {gpu_id} failed ({proc.returncode}): {detail}.{hint}")

    time.sleep(max(0.0, settle_s))
    observed = read_mode(gpu_id)
    if observed != canonical:
        raise SweepError(
            f"GPU {gpu_id} reports {observed} after being set to {canonical}. The command "
            f"returned success but the mode did not change, so any measurement now would be "
            f"attributed to the wrong mode. Most often this is a permission failure that "
            f"amd-smi did not report as one -- try --sudo."
        )
    return observed


# ------------------------------------------------------- HIP enumeration


def parse_hsa_agents(text: str) -> tuple[HsaAgent, ...]:
    """Parse ``rocminfo`` output into GPU agents in HIP index order.

    HIP indices are what the benchmark will use, and they are not ``amd-smi``
    indices: on this node under ``CPX``, ``amd-smi`` calls card 0's partitions
    devices 0-7 while HSA calls them 7-14, because HSA enumerates whole cards
    first. Reading the order from HSA is the only way to hand the benchmark a
    device it agrees with.

    Within an agent block ``BDFID`` appears *before* ``Compute Unit``, so a
    line-at-a-time parser that prints on ``BDFID`` attributes every agent the
    previous one's CU count. Fields are therefore collected per block and only
    interpreted once the block ends.
    """
    agents: list[HsaAgent] = []
    block: dict[str, str] = {}
    index = 0

    def _flush(block: dict[str, str]) -> None:
        nonlocal index
        if block.get("Device Type") != "GPU":
            return
        try:
            cu = int(block.get("Compute Unit", ""))
            bdfid = int(block.get("BDFID", ""))
        except ValueError:
            return
        agents.append(
            HsaAgent(
                index=index,
                cu=cu,
                bus=(bdfid >> 8) & 0xFF,
                device=(bdfid >> 3) & 0x1F,
                function=bdfid & 0x7,
            )
        )
        index += 1

    for line in text.splitlines():
        if re.match(r"^Agent \d+", line):
            _flush(block)
            block = {}
            continue
        match = re.match(r"\s+(Device Type|Compute Unit|BDFID):\s*(.+?)\s*$", line)
        if match and match.group(1) not in block:
            block[match.group(1)] = match.group(2)
    _flush(block)
    return tuple(agents)


def read_hsa_agents() -> tuple[HsaAgent, ...]:
    """Run ``rocminfo`` and parse its GPU agents."""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=_READ_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SweepError("rocminfo not found; it is how HIP device order is read") from exc
    except subprocess.TimeoutExpired as exc:
        raise SweepError("rocminfo timed out") from exc
    if proc.returncode != 0:
        raise SweepError(f"rocminfo failed ({proc.returncode}): {proc.stderr.strip()}")
    return parse_hsa_agents(proc.stdout)


def partition_cu_on_bus(agents: Sequence[HsaAgent], bus: int) -> int:
    """The CU count the swept card's devices report, once they agree.

    Every GPU device on the swept card's bus is one of its partitions, and a
    mode's partitions are identical, so a disagreement means the enumeration was
    read while the card was still transitioning. Nothing measured against a
    half-applied topology is worth keeping, so that is an error rather than a
    figure to pick from.
    """
    counts = {a.cu for a in agents if a.bus == bus}
    if not counts:
        raise SweepError(f"HIP reports no GPU on bus {bus:02x}")
    if len(counts) > 1:
        raise SweepError(
            f"devices on bus {bus:02x} report different CU counts ({sorted(counts)}), which a "
            f"settled card does not do -- the enumeration was read mid-transition"
        )
    return counts.pop()


def card_total_gib(device_gib: float | None, device_mode: str) -> float | None:
    """Scale one device's HBM back up to the whole card's.

    The reading is per device, so under a split mode it is one partition's share
    and has to be multiplied by the partition count to describe the card. Doing
    that once at entry means each mode's per-partition memory is a division of
    the same known total, rather than a fresh probe whose device-index mapping
    changes with every set.
    """
    if not device_gib or device_gib <= 0:
        return None
    return device_gib * MODE_PARTITION_COUNTS.get(parse_mode(device_mode), 1)


def select_partition_devices(
    agents: Sequence[HsaAgent],
    layout: PartitionLayout,
    *,
    bus: int,
) -> tuple[int, ...]:
    """Return the HIP indices of the swept card's partitions.

    Selection is by CU count, never by index, and is narrowed to one PCI bus.
    Both halves matter: the CU match is what distinguishes a partition from a
    whole card, and the bus match is what stops seven untouched 256-CU
    neighbours being mistaken for partitions when the mode under test is
    ``SPX``, whose "partition" is a whole card.

    Args:
        agents: GPU agents in HIP order.
        layout: The mode's expected shape.
        bus: PCI bus of the card being swept.

    Returns:
        HIP indices, ascending, one per partition.

    Raises:
        SweepError: If the number found is not the number the mode implies,
            which is what a set that did not really take looks like from here.
    """
    is_partition = partition_device_predicate(layout.cu_per_partition)
    found = tuple(a.index for a in agents if a.bus == bus and is_partition(a.cu))
    if len(found) != layout.partitions:
        on_bus = [f"{a.bdf}={a.cu}CU" for a in agents if a.bus == bus]
        raise SweepError(
            f"{layout.mode} implies {layout.partitions} partitions of "
            f"{layout.cu_per_partition} CU on bus {bus:02x}, but HIP reports "
            f"{len(found)}: {', '.join(on_bus) or 'nothing on that bus'}. "
            f"Benchmarking now would measure the wrong silicon."
        )
    return found


# ------------------------------------------------------------- the fan-out


def build_partition_command(
    template: Sequence[str],
    *,
    device: int,
    output_dir: Path,
    layout: PartitionLayout,
    partition_index: int,
) -> list[str]:
    """Substitute per-partition values into a benchmark command template.

    Substitution is per already-split token, so a path containing a space cannot
    turn into two arguments and no shell is involved at any point.
    """
    values = {
        "device": str(device),
        "output_dir": str(output_dir),
        "mode": layout.mode,
        "partitions": str(layout.partitions),
        "partition_index": str(partition_index),
        "cu": str(layout.cu_per_partition),
    }
    out: list[str] = []
    for token in template:
        for key, value in values.items():
            token = token.replace("{" + key + "}", value)
        out.append(token)
    return out


def magpie_command(benchmark_config: Path, python_exe: str = sys.executable) -> list[str]:
    """The in-tree benchmark invocation, with placeholders for the fan-out."""
    return [
        python_exe,
        "-m",
        "Magpie",
        "-v",
        "benchmark",
        "--benchmark-config",
        str(benchmark_config),
        "--output-dir",
        "{output_dir}",
        "--run-mode",
        "local",
    ]


def partition_env(
    base: dict[str, str],
    layout: PartitionLayout,
    *,
    device: int,
    streams_per_partition: int,
) -> dict[str, str]:
    """Environment for one partition's benchmark process.

    Pins the process to its partition and publishes the session shape in the
    same variables the optimizer publishes, so a benchmark entrypoint written
    against that contract behaves identically whether it was launched by a
    session or by this sweep.

    ``HIP_VISIBLE_DEVICES`` and ``CUDA_VISIBLE_DEVICES`` are removed rather than
    set. Leaving an inherited one alongside ``ROCR_VISIBLE_DEVICES`` means two
    masks apply in sequence, and the second is interpreted as an index into the
    first -- so a stale ``HIP_VISIBLE_DEVICES=0`` silently redirects every
    partition's work onto whichever device the first mask selected.
    """
    env = dict(base)
    env["ROCR_VISIBLE_DEVICES"] = str(device)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env[PARTITION_MODE_ENV] = layout.mode
    env[PARTITION_COUNT_ENV] = str(layout.partitions)
    env[PARTITION_CU_ENV] = str(layout.cu_per_partition)
    env[PARTITION_STREAMS_ENV] = str(streams_per_partition)
    env[PARTITION_TOTAL_STREAMS_ENV] = str(layout.partitions * streams_per_partition)
    return env


def read_measurement(output_dir: Path) -> dict[str, Any]:
    """Parse one partition's benchmark report using the in-tree extractor.

    Imported here rather than at module scope: the extractor pulls in the
    orchestrator, and the pure logic in this file -- command construction,
    device selection, mode gating -- is worth testing without that weight.
    """
    reports = sorted(output_dir.rglob("benchmark_report.json"))
    if not reports:
        return {}
    try:
        payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    import hyperloom.orchestrator.actions.executors._grid_runner  # noqa: F401
    from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

    return extract_benchmark_measurement(payload if isinstance(payload, dict) else None, workspace=output_dir)


def run_mode(
    layout: PartitionLayout,
    devices: Sequence[int],
    *,
    template: Sequence[str],
    output_root: Path,
    streams_per_partition: int,
    timeout_s: float,
    base_env: dict[str, str] | None = None,
) -> list[PartitionRun]:
    """Run the benchmark on every partition at once and collect each result.

    All partitions are launched before any is waited on, which is the only
    arrangement that measures what partitioning is for. Running them in sequence
    would measure one partition at a time on an otherwise idle card and report a
    fraction of the mode's throughput.
    """
    base = dict(os.environ if base_env is None else base_env)
    runs: list[PartitionRun] = []
    procs: list[tuple[PartitionRun, subprocess.Popen[str] | None, Any]] = []

    for position, device in enumerate(devices):
        out_dir = output_root / layout.mode.lower() / f"partition{position}"
        out_dir.mkdir(parents=True, exist_ok=True)
        run = PartitionRun(partition_index=position, hsa_device=device, output_dir=out_dir)
        runs.append(run)
        cmd = build_partition_command(
            template,
            device=device,
            output_dir=out_dir,
            layout=layout,
            partition_index=position,
        )
        env = partition_env(base, layout, device=device, streams_per_partition=streams_per_partition)
        log = None
        try:
            log = (out_dir / "benchmark.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(  # nosec B603 - argv built from a template, no shell.
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, ValueError) as exc:
            if log is not None:
                log.close()
            run.error = f"could not launch: {exc}"
            procs.append((run, None, None))
            continue
        procs.append((run, proc, log))

    deadline = time.monotonic() + max(1.0, timeout_s)
    for run, proc, log in procs:
        if proc is None:
            continue
        remaining = max(1.0, deadline - time.monotonic())
        try:
            run.returncode = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            run.error = f"exceeded the {timeout_s:.0f}s budget"
            run.returncode = -1
        finally:
            if log is not None:
                log.close()
        if run.error:
            continue
        if run.returncode != 0:
            run.error = f"exited {run.returncode}; see {run.output_dir / 'benchmark.log'}"
            continue
        run.measurement = read_measurement(run.output_dir)
        if not run.measurement:
            run.error = f"no benchmark_report.json under {run.output_dir}"
    return runs


# ------------------------------------------------------------- reporting


def render(results: Sequence[ModeResult], *, entry_mode: str) -> list[str]:
    """Render the comparison an operator reads to choose a mode."""
    lines = ["", "=" * 78, "Compute-partition sweep", "=" * 78, ""]
    measured = [r for r in results if r.measured and r.total() is not None]
    baseline = next((r for r in measured if r.mode == entry_mode), None)
    best = max(measured, key=lambda r: r.total() or 0.0, default=None)

    lines.append(f"{'mode':<6} {'parts':>5} {'CU':>5} {'throughput':>13} {'vs entry':>9} {'worst lat':>11}")
    lines.append("-" * 78)
    for result in results:
        if not result.measured or result.total() is None:
            why = result.skipped or result.error or "not measured"
            lines.append(f"{result.mode:<6} {'-':>5} {'-':>5} {'-':>13} {'-':>9} {'-':>11}   {why}")
            continue
        layout = result.layout
        total = result.total() or 0.0
        rel = ""
        if baseline is not None and (baseline.total() or 0.0) > 0:
            rel = f"{total / (baseline.total() or 1.0):.2f}x"
        latency = result.worst_latency_ms()
        lines.append(
            f"{result.mode:<6} {layout.partitions if layout else '-':>5} "
            f"{layout.cu_per_partition if layout else '-':>5} "
            f"{total:>13.1f} {rel:>9} {(f'{latency:.1f} ms' if latency else '-'):>11}"
        )
    lines.append("")

    if best is None:
        lines.append("No mode produced a measurement, so there is nothing to choose between.")
        lines.append("")
        return lines

    lines.append(f"Fastest measured mode: {best.mode} at {best.total():.1f} aggregate throughput.")
    if baseline is not None and best.mode != baseline.mode:
        gain = (best.total() or 0.0) / (baseline.total() or 1.0)
        best_lat = best.worst_latency_ms()
        base_lat = baseline.worst_latency_ms()
        lines.append(
            f"That is {gain:.2f}x the {baseline.mode} the card was in when this started."
            + (
                f" Worst-partition latency moves from {base_lat:.1f} ms to {best_lat:.1f} ms."
                if best_lat and base_lat
                else ""
            )
        )
    lines.append("")
    lines.append(
        "Throughput is the sum over a mode's partitions, all loaded together; a mode is\n"
        "reported only when every one of its partitions returned a measurement. The\n"
        "latency column is the worst partition's mean, not the average of them.\n"
        "\n"
        "This chooses a mode. It does not tune one: run the optimizer in the winning\n"
        "mode to do that, and pass --compute-partition-mode so the session refuses to\n"
        "start if the card is not actually in it."
    )
    lines.append("")
    return lines


def summary_json(results: Sequence[ModeResult], *, entry_mode: str, gpu_id: int) -> dict[str, Any]:
    """Machine-readable form of the same comparison."""
    return {
        "gpu": gpu_id,
        "entry_mode": entry_mode,
        "modes": [
            {
                "mode": r.mode,
                "measured": r.measured,
                "skipped": r.skipped,
                "error": r.error,
                "partitions": r.layout.partitions if r.layout else None,
                "cu_per_partition": r.layout.cu_per_partition if r.layout else None,
                "gib_per_partition": r.layout.gib_per_partition if r.layout else None,
                "aggregate_throughput": r.total(),
                "worst_partition_latency_ms": r.worst_latency_ms(),
                "partition_runs": [
                    {
                        "partition_index": run.partition_index,
                        "hsa_device": run.hsa_device,
                        "returncode": run.returncode,
                        "error": run.error,
                        "output_throughput": to_float(run.measurement.get("output_throughput")),
                        "e2el_mean_ms": to_float(run.measurement.get("e2el_mean_ms")),
                    }
                    for run in r.runs
                ],
            }
            for r in results
        ],
    }


# ------------------------------------------------------------------- main


def resolve_modes(requested: str | None, available: Sequence[str]) -> tuple[str, ...]:
    """Parse the requested mode list, defaulting to what the card reports.

    Raises:
        SweepError: If a requested mode is not a mode, or the card says it
            cannot enter it. A card that reported no profiles at all is not
            second-guessed -- the request stands and the set will judge it.
    """
    if not (requested or "").strip():
        if available:
            return tuple(available)
        raise SweepError(
            "the card reported no partition profiles, so there is no mode list to "
            "default to; name the modes with --modes (and try --sudo, since the "
            "profile query needs privilege)"
        )
    modes: list[str] = []
    for raw in str(requested).replace(",", " ").split():
        # parse_mode refuses an unknown spelling with PartitionError. Converted
        # here so that every way of mistyping --modes leaves by the same door as
        # the other usage errors, rather than as a traceback.
        try:
            mode = parse_mode(raw)
        except PartitionError as exc:
            raise SweepError(f"unknown compute-partition mode {raw!r}") from exc
        if mode not in MODE_PARTITION_COUNTS:
            raise SweepError(f"unknown compute-partition mode {raw!r}")
        if mode not in modes:
            modes.append(mode)
    if available:
        rejected = [m for m in modes if m not in available]
        if rejected:
            raise SweepError(
                f"the card does not report support for {', '.join(rejected)}; it lists {', '.join(available)}"
            )
    return tuple(modes)


def _restore_entry_mode(gpu_id: int, entry_mode: str, *, sudo: bool) -> bool:
    """Put the card back in the mode it was found in. True if it could not be.

    Raises nothing, because it is called from a ``finally``: an exception here
    would replace whatever sent the sweep into the restore and would take the
    report down with it, losing both the diagnosis and the modes already
    measured.

    A read-back that fails is treated as "mode unknown" and the set is attempted
    regardless. The alternative is skipping the restore because the check that
    would have proved it necessary is the thing that broke, which leaves a card
    in a shape nobody asked for.
    """
    try:
        if read_mode(gpu_id) == entry_mode:
            return False
    except Exception as exc:
        print(f"note: could not read GPU {gpu_id}'s mode ({exc}); attempting the restore anyway")
    try:
        print(f"\nrestoring {entry_mode} on GPU {gpu_id}")
        set_mode(gpu_id, entry_mode, sudo=sudo)
    except Exception as exc:
        print(
            f"ERROR: could not restore {entry_mode} on GPU {gpu_id}: {exc}\n"
            f"The card is NOT in the mode it started in. Anything that runs on it now "
            f"will be measured under a shape nobody asked for.",
            file=sys.stderr,
        )
        return True
    return False


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", type=int, default=0, help="Card to repartition and sweep (default 0).")
    ap.add_argument(
        "--modes",
        default="",
        help="Comma-separated modes to try, in order. Default: every mode the card reports.",
    )
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--benchmark-config", type=Path, help="Run the in-tree Magpie benchmark with this config.")
    source.add_argument(
        "--benchmark-command",
        help="Command to run per partition. {device} {output_dir} {mode} {partitions} "
        "{partition_index} {cu} are substituted.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("partition-sweep"),
        help="Where per-partition benchmark output and the summary go.",
    )
    ap.add_argument(
        "--streams-per-partition",
        type=int,
        default=2,
        help="Concurrent streams the entrypoint should place on each partition (default 2).",
    )
    ap.add_argument(
        "--per-stream-gib",
        type=float,
        default=0.0,
        help="Per-stream HBM footprint. When given, modes whose partitions provably cannot hold "
        "the streams are skipped instead of being run into an out-of-memory failure.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Per-mode budget in seconds for the whole fan-out (default 3600).",
    )
    ap.add_argument("--sudo", action="store_true", help="Route privileged amd-smi calls through 'sudo -n'.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit without changing anything.")
    ap.add_argument(
        "--allow-busy",
        action="store_true",
        help="Proceed even though processes hold a context on the swept card; the set evicts them. "
        "Skips the check entirely, so it is also the way past an amd-smi process payload this "
        "script cannot parse.",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    if args.benchmark_command:
        template: list[str] = shlex.split(args.benchmark_command)
    elif args.benchmark_config:
        template = magpie_command(args.benchmark_config)
    else:
        print("ERROR: one of --benchmark-config or --benchmark-command is required.", file=sys.stderr)
        return 2
    if args.streams_per_partition < 1:
        print("ERROR: --streams-per-partition must be at least 1.", file=sys.stderr)
        return 2

    try:
        entry_mode = read_mode(args.gpu)
        bus = card_bus(_amd_smi_json(["list"]), args.gpu)
        card_gib = card_total_gib(read_device_gib(args.gpu), entry_mode)
        try:
            available = supported_modes(_amd_smi_json(["partition", "-a", "-g", str(args.gpu)], sudo=args.sudo))
        except SweepError as exc:
            print(f"note: could not read partition profiles ({exc}); not restricting the mode list")
            available = ()
        modes = resolve_modes(args.modes, available)
        # Scoped to the swept card because that is the only card a set touches.
        # A neighbour's benchmark is not a reason to refuse, and refusing on one
        # left --allow-busy as the only way forward -- which drops the guard on
        # the target card too, the one card it exists to protect. Skipped
        # entirely when no set will follow, so a payload it cannot read never
        # blocks a caller it was not protecting.
        if not (args.allow_busy or args.dry_run):
            busy = resident_processes(_amd_smi_json(["process"]))
            if args.gpu not in busy:
                raise SweepError(
                    f"amd-smi process listed no GPU {args.gpu}, so whether the card is in use "
                    f"is unknown. Repartitioning evicts every context on it, so this refuses "
                    f"rather than assume it is idle. Pass --allow-busy to sweep anyway."
                )
            if busy[args.gpu]:
                raise SweepError(
                    f"{busy[args.gpu]} process(es) still hold a context on GPU {args.gpu}. "
                    f"Repartitioning evicts them, so this refuses rather than killing "
                    f"someone's work. Stop them, or pass --allow-busy if they are yours."
                )
    except (SweepError, PartitionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Nothing has been set at this point, so there is no card to restore --
        # but the exit code still has to mean something. 2 is for a refusal this
        # script decided on, so an unmodelled failure gets its own code rather
        # than borrowing that one.
        traceback.print_exc()
        print("ERROR: unexpected error while reading the card; nothing was changed.", file=sys.stderr)
        return 4

    print(f"GPU {args.gpu} on PCI bus {bus:02x} is in {entry_mode}; sweeping {', '.join(modes)}.")
    if args.dry_run:
        print("\n--dry-run: nothing will be set. Planned per mode:")
        for mode in modes:
            partitions = MODE_PARTITION_COUNTS[mode]
            print(f"  {mode:<4} set GPU {args.gpu}, then {partitions} concurrent benchmark(s), one per partition")
        print(f"\nCommand template: {' '.join(template)}")
        print(f"Would restore {entry_mode} on exit.")
        return 0

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ModeResult] = []
    unexpected: str | None = None
    interrupted = {"flag": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        interrupted["flag"] = True
        print(f"\nsignal {signum} received; finishing the current mode, then restoring {entry_mode}.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    try:
        for mode in modes:
            if interrupted["flag"]:
                results.append(ModeResult(mode=mode, skipped="interrupted before this mode ran"))
                continue
            result = ModeResult(mode=mode)
            results.append(result)
            print(f"\n--- {mode} ---")
            try:
                set_mode(args.gpu, mode, sudo=args.sudo)
                agents = read_hsa_agents()
                cu = partition_cu_on_bus(agents, bus)
                gib = None if card_gib is None else card_gib / MODE_PARTITION_COUNTS[mode]
                layout = layout_for(mode, cu_per_partition=cu, gib_per_partition=gib)
                result.layout = layout
                if args.per_stream_gib > 0 and not fits_in_partition(
                    args.per_stream_gib, layout, args.streams_per_partition
                ):
                    result.skipped = (
                        f"{args.streams_per_partition} x {args.per_stream_gib:.1f} GiB will not fit a "
                        f"{layout.gib_per_partition:.1f} GiB partition"
                    )
                    print(f"  skipped: {result.skipped}")
                    continue
                devices = select_partition_devices(agents, layout, bus=bus)
                print(
                    f"  {layout.partitions} partition(s) of {layout.cu_per_partition} CU at HIP devices {list(devices)}"
                )
                result.runs = run_mode(
                    layout,
                    devices,
                    template=template,
                    output_root=output_root,
                    streams_per_partition=args.streams_per_partition,
                    timeout_s=args.timeout,
                )
                ok = sum(1 for r in result.runs if r.ok)
                print(f"  {ok}/{len(result.runs)} partition(s) reported")
                for run in result.runs:
                    if run.error:
                        print(f"    partition {run.partition_index} (HIP {run.hsa_device}): {run.error}")
            except (SweepError, PartitionError) as exc:
                result.error = str(exc)
                print(f"  failed: {exc}")
            except Exception as exc:
                # Everything this script anticipates arrives as a SweepError or a
                # PartitionError. Anything else is a bug here or an amd-smi
                # behaviour not modelled, which means the assumptions driving
                # privileged sets no longer hold -- so stop sweeping. Stopping by
                # breaking rather than propagating is the point: the card still
                # gets restored, and the modes already measured still get
                # reported. Letting it escape lost the table, the summary file,
                # and the exit code that says the card was left wrong.
                unexpected = f"{type(exc).__name__}: {exc}"
                result.error = f"unexpected error: {unexpected}"
                print(f"  aborted: {result.error}", file=sys.stderr)
                traceback.print_exc()
                break
    finally:
        restore_failed = _restore_entry_mode(args.gpu, entry_mode, sudo=args.sudo)

    summary = output_root / "sweep_summary.json"
    try:
        print("\n".join(render(results, entry_mode=entry_mode)))
        summary.write_text(
            json.dumps(summary_json(results, entry_mode=entry_mode, gpu_id=args.gpu), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {summary}")
    except Exception as exc:
        # A full disk or an unrenderable result must not cost the caller the exit
        # code, which is the one thing it cannot reconstruct for itself -- least
        # of all the code saying the card was left in the wrong mode.
        traceback.print_exc()
        print(f"ERROR: could not write the report to {summary}: {exc}", file=sys.stderr)
        unexpected = unexpected or f"{type(exc).__name__}: {exc}"

    # A card left in the wrong shape outranks everything else: it mislabels
    # whatever runs on the node next, not just this sweep.
    if restore_failed:
        return 3
    if unexpected:
        print(f"ERROR: sweep stopped on an unexpected error ({unexpected}).", file=sys.stderr)
        return 4
    return 0 if any(r.measured for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
