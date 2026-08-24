# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD compute-partition modes (SPX/DPX/QPX/CPX) as a hardware-level lever.

Every other lever in the optimizer is an environment variable or a server flag:
it travels in a ``GridVariant``, applies by materializing a benchmark YAML, and
reverts by not setting it again. A compute-partition mode is none of those
things. It is privileged, it is global to the card, it evicts every process
resident on that card, and it renumbers the devices underneath a running
process. So it gets its own module rather than a row in an existing table.

What the modes buy, and what they cost: partitioning a card only ever gives a
single stream *fewer* CUs, so it cannot improve single-stream latency and will
always make it worse. It pays only in aggregate, when there are at least as many
concurrent streams as partitions -- which is why ``streams_per_partition``
belongs next to the mode in any configuration that sets one. Measured on one
MI355X with a 1.26B-parameter vision model, CPX at two streams per partition
carried ~20% more aggregate throughput than the best SPX configuration while
per-request latency went from 183 ms to 1211 ms. That trade is a decision for
whoever owns the SLA, never a default.

Two invariants this module exists to enforce:

* **A set is not a set until it reads back.** ``amd-smi set`` reports success
  for a memory-partition change that has only been staged pending a driver
  reload, so trusting the exit code silently measures the old topology as if it
  were the new one. Every mutation here re-reads the mode and compares.
* **Partitions are not identified by device index.** HIP enumerates whole GPUs
  before partitions, so with one card of eight split into DPX the partitions are
  devices 7 and 8, not 0 and 1. Selecting by index measures a full card and
  reports it as a partition -- a wrong number with no error attached. Callers
  get :func:`partition_device_predicate` for a CU-count test instead.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

from .gpu_identity import AMD_GPU_DISPATCH_IDENTITIES

log = logging.getLogger(__name__)

#: Compute-partition mode -> partitions per card. Both gfx942 and gfx950 boards
#: carry eight XCDs, so the ladder is the same width on each; the CU count per
#: partition differs and is derived from the board's own total.
MODE_PARTITION_COUNTS: dict[str, int] = {
    "SPX": 1,
    "DPX": 2,
    "QPX": 4,
    "CPX": 8,
}

#: Mode a session is restored to. SPX is the only mode every board supports and
#: the only one compatible with NPS1, so it is the safe terminal state.
DEFAULT_MODE = "SPX"

_SET_TIMEOUT_S = 120.0
_READ_TIMEOUT_S = 30.0

#: How long a set waits out a card that still holds processes, and how often it
#: retries. The card genuinely refuses to repartition while a process is
#: resident, but teardown is not synchronous: ``docker rm -f`` returns before
#: the runtime has released the device, so a repartition issued immediately
#: afterwards races it. Treating that refusal as fatal cost a sweep two
#: configurations -- the restore after one run failed and left a shared card in
#: QPX, and the next run then could not set its own mode. The condition clears
#: on its own in seconds, so the correct response is to wait for it.
_DRAIN_TIMEOUT_S = 120.0
_DRAIN_POLL_S = 2.0

#: Substrings that mark a refusal as "busy, try again" rather than "wrong".
#: Matching narrowly keeps permanent failures -- an unknown mode, a missing
#: binary, a denied permission -- fast instead of retrying them for two minutes.
#: ``AMDSMI_STATUS_BUSY`` is the authoritative one and the reason the others are
#: only a safety net: the library reports the refusal as status code 30, and the
#: human-readable half of the message ("Device busy") is not stable enough to
#: match on alone.
_BUSY_MARKERS = ("amdsmi_status_busy", "device busy", "resident process", "try again")

#: Opt-in elevation for the set path. Reading the partition state is
#: unprivileged; changing it is not, so a session running as an ordinary user
#: needs a way to say how. Set to ``1`` to route the set through
#: ``sudo -n``, which requires a NOPASSWD sudoers entry for ``amd-smi``.
#:
#: Off by default and never inferred. Escalating privilege because a command
#: failed is not a fallback, it is a decision, and it belongs to the operator.
PARTITION_SUDO_ENV = "HYPERLOOM_PARTITION_SUDO"


def _is_busy(detail: str) -> bool:
    """Whether a failed set was refused because the card was still occupied."""
    low = (detail or "").lower()
    return any(marker in low for marker in _BUSY_MARKERS)


class PartitionError(RuntimeError):
    """Raised when a partition mode cannot be read, set, or verified."""


@dataclass(frozen=True)
class PartitionLayout:
    """What one compute-partition mode does to one card.

    Attributes:
        mode: Canonical mode name (``SPX``/``DPX``/``QPX``/``CPX``).
        partitions: Independent devices the card presents in this mode.
        cu_per_partition: Compute units each partition gets.
        gib_per_partition: HBM each partition gets, or ``None`` when the caller
            did not supply the card's capacity.
    """

    mode: str
    partitions: int
    cu_per_partition: int
    gib_per_partition: float | None = None

    def describe(self) -> str:
        """Return a one-line summary for logs and reports."""
        mem = f", {self.gib_per_partition:.0f} GiB" if self.gib_per_partition else ""
        return f"{self.mode} ({self.partitions} x {self.cu_per_partition} CU{mem})"


def parse_modes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse an operator-supplied mode list into canonical order-preserving modes.

    Args:
        raw: Comma-separated string (``"spx,dpx"``) or an already-split
            sequence. Empty or ``None`` means the lever is off.

    Returns:
        Canonical upper-case mode names, deduplicated, in the order given.

    Raises:
        PartitionError: If any entry is not a known mode. Refused at parse time
            rather than at apply time, because the apply site is a privileged
            hardware mutation partway through a session.
    """
    if not raw:
        return ()
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    modes: list[str] = []
    for item in items:
        name = str(item).strip().upper()
        if not name:
            continue
        if name not in MODE_PARTITION_COUNTS:
            raise PartitionError(
                f"unknown compute-partition mode {name!r}; expected one of {', '.join(MODE_PARTITION_COUNTS)}"
            )
        if name not in modes:
            modes.append(name)
    return tuple(modes)


def layout_for(gpu_type: str | None, mode: str, hbm_gib: float | None = None) -> PartitionLayout:
    """Describe what ``mode`` does to a ``gpu_type`` card.

    CU counts come from :data:`AMD_GPU_DISPATCH_IDENTITIES` rather than a second
    table, so a board added there is described here without a further edit.

    Args:
        gpu_type: Board name (``mi300x``, ``mi355x``, ...).
        mode: Compute-partition mode.
        hbm_gib: The card's total HBM, when known. Supplied by the caller from a
            live device rather than tabled here, since capacity varies across
            boards that share an ISA.

    Returns:
        The resulting layout.

    Raises:
        PartitionError: If the mode is unknown or the board is unrecognised.
    """
    canonical = str(mode or "").strip().upper()
    partitions = MODE_PARTITION_COUNTS.get(canonical)
    if partitions is None:
        raise PartitionError(f"unknown compute-partition mode {mode!r}")
    identity = AMD_GPU_DISPATCH_IDENTITIES.get(str(gpu_type or "").strip().lower())
    if identity is None:
        raise PartitionError(f"unknown gpu_type {gpu_type!r}; cannot size partitions without the board's CU count")
    cu_total = identity[1]
    return PartitionLayout(
        mode=canonical,
        partitions=partitions,
        cu_per_partition=cu_total // partitions,
        gib_per_partition=(hbm_gib / partitions) if hbm_gib else None,
    )


def partition_device_predicate(cu_per_partition: int):
    """Return a predicate selecting partition devices by CU count.

    The index-based alternative is what makes a partitioning measurement quietly
    wrong: HIP enumerates whole cards first, so under DPX on one card of eight
    ``device 0`` is a full 256-CU GPU and the partitions are devices 7 and 8.
    Callers apply this to ``torch.cuda.get_device_properties(i).multi_processor_count``
    (kept out of this module so it stays importable without torch).

    Args:
        cu_per_partition: CU count a real partition must report.

    Returns:
        A callable taking a device's CU count and returning whether it is a
        partition of the requested shape.
    """

    def _is_partition(device_cu: int) -> bool:
        return int(device_cu) == int(cu_per_partition)

    return _is_partition


def fits_in_partition(
    required_gib: float,
    layout: PartitionLayout,
    streams_per_partition: int = 1,
) -> bool:
    """Report whether the streams sharing one partition fit in its memory.

    This is the real criterion behind "partitioning helps smaller models": a
    partition gets a fraction of the card's HBM, and every stream on it holds
    its own copy of the weights plus activations. A model needing the whole card
    cannot be partitioned at all, however throughput-bound it is.

    ``streams_per_partition`` is not a refinement -- it is the question. A mode
    is only worth setting at two streams per partition, and one stream fitting
    says nothing about two: a 20 GiB footprint fits a 36 GiB CPX partition alone
    and exhausts it in pairs. Gating on the single-stream figure is how a
    configuration gets declared feasible and then dies at the second worker.

    Args:
        required_gib: Peak footprint of one stream, weights included.
        layout: The layout under consideration.
        streams_per_partition: Concurrent streams intended per partition.

    Returns:
        ``True`` when they fit, or when capacity is unknown -- an unknown is
        reported by the caller that has the number, not guessed at here.
    """
    if not layout.gib_per_partition or required_gib <= 0:
        return True
    return required_gib * max(1, int(streams_per_partition)) <= layout.gib_per_partition


def _amd_smi_json(args: Sequence[str], timeout_s: float) -> object:
    """Run an ``amd-smi`` subcommand with ``--json`` and parse its output."""
    cmd = ["amd-smi", *args, "--json"]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PartitionError("amd-smi not found; compute partitioning needs it on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise PartitionError(f"amd-smi {' '.join(args)} timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        raise PartitionError(f"amd-smi {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise PartitionError(f"amd-smi {' '.join(args)} returned unparseable JSON") from exc


def _set_prefix() -> list[str]:
    """Return the command prefix for the privileged set (``sudo -n`` or none).

    ``-n`` matters: an automated optimization loop that stops at an interactive
    password prompt hangs until its budget expires, with no indication why.
    """
    if os.environ.get(PARTITION_SUDO_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return ["sudo", "-n"]
    return []


def read_partition_modes() -> dict[int, str]:
    """Read the live compute-partition mode of every GPU.

    Returns:
        Mapping of GPU id to mode name.

    Raises:
        PartitionError: If ``amd-smi`` is absent, fails, or reports no cards.
    """
    payload = _amd_smi_json(["partition"], _READ_TIMEOUT_S)
    rows: list[dict] = []
    if isinstance(payload, dict):
        raw_rows = payload.get("current_partition")
        if isinstance(raw_rows, list):
            rows = [r for r in raw_rows if isinstance(r, dict)]
    modes: dict[int, str] = {}
    for row in rows:
        try:
            gpu_id = int(row.get("gpu_id"))
        except (TypeError, ValueError):
            continue
        mode = str(row.get("accelerator_type") or "").strip().upper()
        if mode:
            modes[gpu_id] = mode
    if not modes:
        raise PartitionError("amd-smi partition reported no compute-partition state")
    return modes


def read_partition_mode(gpu_id: int) -> str:
    """Read one GPU's compute-partition mode.

    Raises:
        PartitionError: If that GPU is not present in the report.
    """
    modes = read_partition_modes()
    if gpu_id not in modes:
        raise PartitionError(f"amd-smi partition reported no state for GPU {gpu_id}")
    return modes[gpu_id]


def set_partition_mode(gpu_id: int, mode: str, drain_timeout_s: float = _DRAIN_TIMEOUT_S) -> str:
    """Set one GPU's compute-partition mode and verify it took effect.

    The verification is the point of this function. ``amd-smi set`` reports
    success for a change that has only been staged, so a caller trusting the
    exit code proceeds to benchmark the old topology while labelling the results
    with the new mode -- a wrong number with a reassuring log line above it.

    A card still holding processes refuses to repartition, which is a race
    rather than a wall: see :data:`_DRAIN_TIMEOUT_S`.

    Args:
        gpu_id: GPU to reconfigure.
        mode: Target mode.
        drain_timeout_s: How long to keep retrying while the card reports
            resident processes. Zero fails on the first refusal.

    Returns:
        The mode read back from the hardware, which equals ``mode`` on success.

    Raises:
        PartitionError: If the mode is unknown, the set fails, or the read-back
            disagrees with what was requested.
    """
    canonical = str(mode or "").strip().upper()
    if canonical not in MODE_PARTITION_COUNTS:
        raise PartitionError(f"unknown compute-partition mode {mode!r}")

    current = read_partition_mode(gpu_id)
    if current == canonical:
        return current

    cmd = [*_set_prefix(), "amd-smi", "set", "-g", str(gpu_id), "--compute-partition", canonical]
    deadline = time.monotonic() + max(0.0, drain_timeout_s)
    waited_for_drain = False
    while True:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                cmd,
                capture_output=True,
                text=True,
                timeout=_SET_TIMEOUT_S,
                # Some builds prompt for confirmation; answer it rather than block.
                input="Y\n",
                check=False,
            )
        except FileNotFoundError as exc:
            raise PartitionError("amd-smi not found; compute partitioning needs it on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise PartitionError(f"setting {canonical} on GPU {gpu_id} timed out") from exc
        if proc.returncode == 0:
            break
        detail = (proc.stderr or proc.stdout).strip()
        if _is_busy(detail) and time.monotonic() < deadline:
            if not waited_for_drain:
                log.info("GPU %d busy; waiting for it to drain before setting %s", gpu_id, canonical)
                waited_for_drain = True
            time.sleep(_DRAIN_POLL_S)
            continue
        hint = (
            f" The card still held processes after {drain_timeout_s:.0f}s."
            if _is_busy(detail)
            else " A card with resident processes refuses to repartition; stop them first."
        )
        raise PartitionError(f"setting {canonical} on GPU {gpu_id} failed ({proc.returncode}): {detail}.{hint}")

    observed = read_partition_mode(gpu_id)
    if observed != canonical:
        raise PartitionError(
            f"GPU {gpu_id} reports {observed} after being set to {canonical}. The "
            f"command returned success but the mode did not change, so any "
            f"measurement now would be attributed to the wrong mode. Two known "
            f"causes: amd-smi exits 0 on a permission failure (set "
            f"{PARTITION_SUDO_ENV}=1 for a NOPASSWD sudo path), and a memory "
            f"partition change needs an amdgpu reload to take effect."
        )
    log.info("GPU %d compute partition: %s -> %s", gpu_id, current, canonical)
    return observed


@contextmanager
def partitioned(gpu_id: int, mode: str, restore_to: str | None = None) -> Iterator[str]:
    """Hold a GPU in ``mode`` for the duration of the block, then restore it.

    Restoration runs on the way out of both the happy and the failing path,
    because the alternative is leaving a shared card in a mode the next tenant
    did not ask for. A failure to restore is logged rather than raised, so it
    cannot mask the exception that caused the exit.

    Args:
        gpu_id: GPU to reconfigure.
        mode: Mode to hold during the block.
        restore_to: Mode to return to; defaults to whatever was observed on
            entry, falling back to :data:`DEFAULT_MODE`.

    Yields:
        The mode the hardware confirmed.
    """
    try:
        entry_mode = read_partition_mode(gpu_id)
    except PartitionError:
        entry_mode = DEFAULT_MODE
    target_restore = str(restore_to or entry_mode or DEFAULT_MODE).strip().upper()
    observed = set_partition_mode(gpu_id, mode)
    try:
        yield observed
    finally:
        if target_restore != observed:
            try:
                set_partition_mode(gpu_id, target_restore)
            except PartitionError as exc:
                log.error(
                    "GPU %d left in %s: restore to %s failed: %s",
                    gpu_id,
                    observed,
                    target_restore,
                    exc,
                )


__all__ = [
    "DEFAULT_MODE",
    "MODE_PARTITION_COUNTS",
    "PARTITION_SUDO_ENV",
    "PartitionError",
    "PartitionLayout",
    "fits_in_partition",
    "layout_for",
    "parse_modes",
    "partition_device_predicate",
    "partitioned",
    "read_partition_mode",
    "read_partition_modes",
    "set_partition_mode",
]
