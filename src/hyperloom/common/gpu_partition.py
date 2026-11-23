# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD compute-partition modes (SPX/DPX/QPX/CPX) as an observed session property."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from hyperloom_common.gpu_card import GpuCard

log = logging.getLogger(__name__)

#: Compute-partition mode -> partitions per card. Both gfx942 and gfx950 boards
#: carry eight XCDs, so the ladder is the same width on each. Used to interpret a
#: mode name; the per-partition CU count and memory are the session card's.
MODE_PARTITION_COUNTS: dict[str, int] = {
    "SPX": 1,
    "DPX": 2,
    "QPX": 4,
    "CPX": 8,
}

#: The observed shape, published once at launch for the benchmark entrypoint to
#: fan out across and for the provenance record to quote. Named here, in
#: ``common``, because the platform fingerprint reads them and must not import
#: the orchestrator -- and must not spawn a probe of its own, since it also runs
#: on the crash path where a subprocess is the last thing wanted.
PARTITION_MODE_ENV = "HYPERLOOM_PARTITION_MODE"
PARTITION_COUNT_ENV = "HYPERLOOM_PARTITION_COUNT"
PARTITION_CU_ENV = "HYPERLOOM_PARTITION_CU"
PARTITION_STREAMS_ENV = "HYPERLOOM_PARTITION_STREAMS_PER_PARTITION"
PARTITION_TOTAL_STREAMS_ENV = "HYPERLOOM_PARTITION_TOTAL_STREAMS"

_READ_TIMEOUT_S = 30.0


class PartitionError(RuntimeError):
    """Raised when a partition mode cannot be read or interpreted."""


@dataclass(frozen=True)
class PartitionLayout:
    """What one compute-partition mode means for one card."""

    mode: str
    partitions: int
    cu_per_partition: int
    gib_per_partition: float | None = None

    @property
    def partitioned(self) -> bool:
        """Whether this mode splits the card at all."""
        return self.partitions > 1

    @property
    def capacity_known(self) -> bool:
        """Whether :attr:`gib_per_partition` is a figure worth checking against."""
        return self.gib_per_partition is not None and self.gib_per_partition > 0

    def describe(self) -> str:
        """Return a one-line summary for logs and reports."""
        mem = f", {self.gib_per_partition:.0f} GiB" if self.gib_per_partition else ""
        return f"{self.mode} ({self.partitions} x {self.cu_per_partition} CU{mem})"


@dataclass(frozen=True)
class PartitionModel:
    """The model a partition shape must fit: its path and precision, and the session state holding its identity."""

    model_path: str | None = None
    precision: str | None = None
    shared_state: Any = None


def parse_mode(raw: str | None) -> str:
    """Parse an operator-supplied mode name into its canonical form."""
    name = str(raw or "").strip().upper()
    if not name:
        return ""
    if name not in MODE_PARTITION_COUNTS:
        raise PartitionError(
            f"unknown compute-partition mode {raw!r}; expected one of {', '.join(MODE_PARTITION_COUNTS)}"
        )
    return name


def _amd_smi_json(args: Sequence[str], timeout_s: float = _READ_TIMEOUT_S) -> object:
    """Run a read-only ``amd-smi`` subcommand with ``--json`` and parse it."""
    cmd = ["amd-smi", *args, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PartitionError("amd-smi not found; reading compute partitioning needs it on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise PartitionError(f"amd-smi {' '.join(args)} timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        raise PartitionError(f"amd-smi {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise PartitionError(f"amd-smi {' '.join(args)} returned unparseable JSON") from exc


def read_partition_modes() -> dict[int, str]:
    """Read the live compute-partition mode of every GPU."""
    payload = _amd_smi_json(["partition"])
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


def read_partition_mode(gpu_id: int = 0) -> str:
    """Read one GPU's compute-partition mode."""
    modes = read_partition_modes()
    if gpu_id not in modes:
        raise PartitionError(f"amd-smi partition reported no state for GPU {gpu_id}")
    return modes[gpu_id]


def layout_for(mode: str, *, cu_per_partition: int, gib_per_partition: float | None = None) -> PartitionLayout:
    """Describe what ``mode`` means; a partitioned card's device is one partition."""
    canonical = parse_mode(mode)
    if not canonical:
        raise PartitionError("no compute-partition mode given")
    return PartitionLayout(
        mode=canonical,
        partitions=MODE_PARTITION_COUNTS[canonical],
        cu_per_partition=int(cu_per_partition),
        gib_per_partition=gib_per_partition,
    )


def observe_partition(gpu_id: int = 0, *, card: GpuCard | None) -> PartitionLayout | None:
    """Read the live partition topology of one card, sized by ``card``; ``None`` when either is unknown."""
    try:
        mode = read_partition_mode(gpu_id)
    except PartitionError as exc:
        log.debug("no compute-partition state for GPU %d: %s", gpu_id, exc)
        return None
    if mode not in MODE_PARTITION_COUNTS:
        log.warning("GPU %d reports compute-partition mode %r, which this build does not know", gpu_id, mode)
        return None
    if card is None or card.compute_units is None:
        log.warning(
            "GPU %d is in %s, but the session read no compute-unit count to size its partitions by", gpu_id, mode
        )
        return None
    return layout_for(mode, cu_per_partition=card.compute_units, gib_per_partition=card.memory_mib / 1024)


def published_shape(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Read the partition shape a launch published, without touching the device."""
    source = os.environ if env is None else env
    mode = str(source.get(PARTITION_MODE_ENV, "") or "").strip().upper()
    if not mode:
        return None

    def _int(key: str) -> int | None:
        try:
            return int(str(source.get(key, "")).strip())
        except (TypeError, ValueError):
            return None

    shape: dict[str, Any] = {"mode": mode}
    for key, name in (
        (PARTITION_COUNT_ENV, "partitions"),
        (PARTITION_CU_ENV, "cu_per_partition"),
        (PARTITION_STREAMS_ENV, "streams_per_partition"),
    ):
        value = _int(key)
        if value is not None:
            shape[name] = value
    return shape


def partition_device_predicate(cu_per_partition: int):
    """Return the partition CU predicate; No caller in this repository uses it directly."""

    def _is_partition(device_cu: int) -> bool:
        return int(device_cu) == int(cu_per_partition)

    return _is_partition


def fits_in_partition(
    required_gib: float,
    layout: PartitionLayout,
    streams_per_partition: int = 1,
) -> bool:
    """Report whether the streams sharing one partition fit in its memory."""
    if not layout.capacity_known or required_gib <= 0:
        return True
    # ``capacity_known`` has already established this is a positive float; the coercion is only so the shared
    # predicate can carry the decision without the arithmetic having to re-assert the type.
    capacity_gib = float(layout.gib_per_partition or 0.0)
    return required_gib * max(1, int(streams_per_partition)) <= capacity_gib


# The probe helpers behind observe_partition -- read_partition_mode(s) -- are
# deliberately absent. observe_partition's docstring calls itself "the single entry point a caller needs", and listing
# the steps it takes here would contradict that: they are the call graph, not the interface.
__all__ = [
    "MODE_PARTITION_COUNTS",
    "PARTITION_COUNT_ENV",
    "PARTITION_CU_ENV",
    "PARTITION_MODE_ENV",
    "PARTITION_STREAMS_ENV",
    "PARTITION_TOTAL_STREAMS_ENV",
    "PartitionError",
    "PartitionLayout",
    "PartitionModel",
    "fits_in_partition",
    "layout_for",
    "observe_partition",
    "parse_mode",
    "partition_device_predicate",
    "published_shape",
]
