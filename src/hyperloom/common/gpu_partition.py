# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD compute-partition modes (SPX/DPX/QPX/CPX) as an observed session property.

The mode a card is in is a property of the *card*, not of a process: changing it
is privileged, it evicts every process resident on the card, and it renumbers
the devices underneath anything already running. Nothing here changes it. The
mode must be established before ``optimize`` starts -- by the operator or the
provisioning platform, since the shape is checked and recorded at launch, before
any benchmark process exists to establish it -- and this module reads it,
describes it, and answers one question about it before a session commits three
hours to a shape that cannot work.

That division is deliberate and matches how the rest of the repo treats
high-risk host state: NPS and the CPU governor are probed and warned about, not
set, and BIOS-level knobs are deferred to ``scripts/platform_audit.py``. An
optimization loop that runs agent-authored code is the wrong place to hold a
privileged hardware mutation, so every entry point here is a read.

What the modes buy, and what they cost: partitioning a card only ever gives a
single stream *fewer* CUs, so it cannot improve single-stream latency and will
always make it worse. It pays only in aggregate, when there are at least as many
concurrent streams as partitions -- which is why ``streams_per_partition``
belongs next to the mode in any configuration that names one. Measured on one
MI355X with a 1.26B-parameter vision model at six views per forward pass, CPX at
two streams per partition carried ~20% more aggregate throughput than the best
SPX configuration while per-request latency went from 183 ms to 1211 ms. That
trade belongs to whoever owns the SLA, which is another reason it is an input
here rather than something the optimizer decides.

Two streams per partition is where every mode peaked on that workload, but it is
a ceiling only a light footprint can reach: the same model at 25 views per pass
peaks at 20.7 GiB per stream, which two streams cannot hold in a 36 GiB CPX
partition. Whether a session can run in the mode it was given is therefore a
memory question, and :func:`fits_in_partition` is what answers it -- at launch,
where a refusal costs seconds instead of hours.

Two invariants this module exists to enforce:

* **Partitions are not identified by device index.** HIP enumerates whole GPUs
  before partitions, so with one card of eight split into DPX the partitions are
  devices 7 and 8, not 0 and 1. Selecting by index measures a full card and
  reports it as a partition -- a wrong number with no error attached. Callers
  get :func:`partition_device_predicate` for a CU-count test instead.
* **Ask the device, do not derive from a table.** Under a split mode a device
  *is* a partition, so ``amd-smi`` reports that partition's own CU count and
  memory. Reading them is exact and needs no assumption about how a board
  divides; deriving them from a per-board table means a board whose entry is
  stale produces a plausible wrong answer. The table remains only as a fallback
  for when the device cannot be reached at all, and says so when it is used.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from .gpu_identity import AMD_GPU_DISPATCH_IDENTITIES

log = logging.getLogger(__name__)

#: Compute-partition mode -> partitions per card. Both gfx942 and gfx950 boards
#: carry eight XCDs, so the ladder is the same width on each. Used to interpret a
#: mode name; the per-partition CU count and memory are read from the device.
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
    """What one compute-partition mode means for one card.

    Attributes:
        mode: Canonical mode name (``SPX``/``DPX``/``QPX``/``CPX``).
        partitions: Independent devices the card presents in this mode.
        cu_per_partition: Compute units each partition gets.
        gib_per_partition: HBM each partition gets, or ``None`` when it could
            not be determined.
        probed: ``True`` when ``cu_per_partition`` came from the device,
            ``False`` when it was derived from the per-board table. Carried so a
            caller can say which, rather than presenting a fallback as a
            measurement.
    """

    mode: str
    partitions: int
    cu_per_partition: int
    gib_per_partition: float | None = None
    probed: bool = False

    @property
    def partitioned(self) -> bool:
        """Whether this mode splits the card at all."""
        return self.partitions > 1

    @property
    def capacity_known(self) -> bool:
        """Whether :attr:`gib_per_partition` is a figure worth checking against.

        One predicate for the two places that ask, so an unreadable capacity
        cannot be treated as unknown by the caller that reports it and as a real
        limit by the arithmetic that acts on it. A non-positive capacity is
        unknown rather than tiny: no partition has zero memory, so the reading
        is wrong rather than restrictive.
        """
        return self.gib_per_partition is not None and self.gib_per_partition > 0

    def describe(self) -> str:
        """Return a one-line summary for logs and reports."""
        mem = f", {self.gib_per_partition:.0f} GiB" if self.gib_per_partition else ""
        source = "" if self.probed else " (derived)"
        return f"{self.mode} ({self.partitions} x {self.cu_per_partition} CU{mem}){source}"


def parse_mode(raw: str | None) -> str:
    """Parse an operator-supplied mode name into its canonical form.

    Args:
        raw: A mode name (``"cpx"``, ``"CPX"``). Empty or ``None`` means the
            operator named no expectation.

    Returns:
        The canonical upper-case mode name, or ``""`` when nothing was named.

    Raises:
        PartitionError: If the value is not one of the known modes. Refused at
            parse time so a typo is a usage error rather than a session that
            runs to completion under an assertion that could never hold.
    """
    name = str(raw or "").strip().upper()
    if not name:
        return ""
    if name not in MODE_PARTITION_COUNTS:
        raise PartitionError(
            f"unknown compute-partition mode {raw!r}; expected one of {', '.join(MODE_PARTITION_COUNTS)}"
        )
    return name


def _amd_smi_json(args: Sequence[str], timeout_s: float = _READ_TIMEOUT_S) -> object:
    """Run a read-only ``amd-smi`` subcommand with ``--json`` and parse it.

    Every call this module makes is unprivileged. The one query that would need
    elevation -- ``amd-smi partition -a``, which enumerates the profiles a board
    *could* enter -- is deliberately absent: it exists to validate a mode before
    setting one, and nothing here sets one.

    Args:
        args: Subcommand and its flags.
        timeout_s: Per-call timeout.

    Returns:
        The parsed JSON payload.

    Raises:
        PartitionError: If ``amd-smi`` is missing, fails, times out, or returns
            output that is not JSON.
    """
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
    """Read the live compute-partition mode of every GPU.

    Returns:
        Mapping of GPU id to mode name.

    Raises:
        PartitionError: If ``amd-smi`` is absent, fails, or reports no cards.
    """
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
    """Read one GPU's compute-partition mode.

    Raises:
        PartitionError: If that GPU is not present in the report.
    """
    modes = read_partition_modes()
    if gpu_id not in modes:
        raise PartitionError(f"amd-smi partition reported no state for GPU {gpu_id}")
    return modes[gpu_id]


#: Field names ``amd-smi static --asic`` has used for a device's compute-unit
#: count. Tried in order; the first that parses wins. Several are accepted
#: because the schema has moved between releases and a missing CU count silently
#: sends the caller to the per-board table, which is the answer this probe
#: exists to avoid.
_CU_KEYS: tuple[str, ...] = ("num_compute_units", "compute_units", "num_cu", "cu_count")


def read_device_cu(gpu_id: int = 0) -> int | None:
    """Read the compute-unit count of one device.

    Under a split mode a device *is* a partition, so this is that partition's CU
    count -- exactly the number :func:`partition_device_predicate` matches on,
    with no assumption about how the board divides. In an unpartitioned mode it
    is the whole card's.

    Args:
        gpu_id: GPU to interrogate.

    Returns:
        The device's CU count, or ``None`` when it could not be read.
    """
    try:
        payload = _amd_smi_json(["static", "-g", str(gpu_id), "--asic"])
    except PartitionError as exc:
        # Warned, not debugged: falling back to the per-board table is a
        # downgrade in accuracy, and the consequence lands far from here as a
        # benchmark that finds no device of the expected width.
        log.warning("could not read GPU %d compute units, will fall back to the board table: %s", gpu_id, exc)
        return None
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    candidates = rows if isinstance(rows, list) else [payload]
    for row in candidates:
        if not isinstance(row, dict):
            continue
        asic = row.get("asic") if isinstance(row.get("asic"), dict) else row
        for key in _CU_KEYS:
            try:
                value = int(asic[key])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0:
                return value
    log.warning("GPU %d reported no compute-unit count, will fall back to the board table", gpu_id)
    return None


#: GiB per unit amd-smi may label a VRAM size with. The binary divisor is the
#: right one for its "MB": an MI355X with 288 GiB reports 294896 "MB", which is
#: mebibytes. "GB" is treated the same way for consistency with that.
_VRAM_UNIT_GIB: dict[str, float] = {
    "B": 1.0 / (1024**3),
    "KB": 1.0 / (1024**2),
    "KIB": 1.0 / (1024**2),
    "MB": 1.0 / 1024,
    "MIB": 1.0 / 1024,
    "GB": 1.0,
    "GIB": 1.0,
    "TB": 1024.0,
    "TIB": 1024.0,
}


def read_device_gib(gpu_id: int = 0) -> float | None:
    """Read the HBM available to one device, in GiB.

    ``amd-smi`` reports VRAM per device, and under a split mode a device is a
    partition -- so this is the memory one partition has, which is the figure
    :func:`fits_in_partition` needs. Read rather than derived by dividing a card
    total, which is what made the same field ambiguous when the mode was
    something this process changed underneath itself.

    Args:
        gpu_id: GPU to interrogate.

    Returns:
        The device's HBM in GiB, or ``None`` when unreadable or unparseable.
    """
    try:
        payload = _amd_smi_json(["static", "-g", str(gpu_id), "--vram"])
    except PartitionError as exc:
        # Warned rather than debugged: this figure is the only input to the
        # feasibility check, so losing it means the session proceeds without one
        # -- a quieter log than the drop it is meant to prevent would invert the
        # severities.
        log.warning("could not read HBM capacity for GPU %d; feasibility will not be checked: %s", gpu_id, exc)
        return None

    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        log.warning("GPU %d returned no VRAM rows; feasibility will not be checked", gpu_id)
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        vram = row.get("vram")
        size = vram.get("size") if isinstance(vram, dict) else None
        if not isinstance(size, dict):
            continue
        try:
            value = float(size.get("value"))
        except (TypeError, ValueError):
            continue
        scale = _VRAM_UNIT_GIB.get(str(size.get("unit") or "").strip().upper())
        if scale is None or value <= 0:
            continue
        return value * scale
    log.warning("GPU %d reported no usable VRAM size; feasibility will not be checked", gpu_id)
    return None


def layout_for(
    mode: str,
    *,
    gpu_type: str | None = None,
    cu_per_partition: int | None = None,
    gib_per_partition: float | None = None,
) -> PartitionLayout:
    """Describe what ``mode`` means, preferring probed numbers to tabled ones.

    Args:
        mode: Compute-partition mode.
        gpu_type: Board name, used only for the fallback CU derivation.
        cu_per_partition: The partition's CU count as read from the device. When
            given, it is used as-is and no board table is consulted.
        gib_per_partition: The partition's HBM as read from the device.

    Returns:
        The layout, with ``probed`` recording which source the CU count came
        from.

    Raises:
        PartitionError: If the mode is unknown, or if no CU count was probed and
            the board cannot be sized from the table.
    """
    canonical = parse_mode(mode)
    if not canonical:
        raise PartitionError("no compute-partition mode given")
    partitions = MODE_PARTITION_COUNTS[canonical]
    if cu_per_partition and int(cu_per_partition) > 0:
        return PartitionLayout(
            mode=canonical,
            partitions=partitions,
            cu_per_partition=int(cu_per_partition),
            gib_per_partition=gib_per_partition,
            probed=True,
        )

    # Fallback: derive from the board's total. This is the path that made a
    # stale table dangerous, so it is only reached when the device could not be
    # asked, and the result is marked as derived.
    identity = AMD_GPU_DISPATCH_IDENTITIES.get(str(gpu_type or "").strip().lower())
    if identity is None:
        raise PartitionError(
            f"cannot size {canonical} partitions: GPU {gpu_type!r} reported no CU count and is not in the board table"
        )
    cu_total = identity[1]
    if cu_total % partitions:
        # Flooring here would be silent and then fatal much later: partition
        # devices are selected by matching this exact CU count, so a floored
        # value matches nothing and the benchmark reports "mode did not take
        # effect" -- true, but about the wrong cause.
        raise PartitionError(
            f"{gpu_type} has {cu_total} CU by the board table, which does not divide into "
            f"{partitions} {canonical} partitions; the per-partition CU count would be wrong "
            f"and device selection matches on it exactly"
        )
    return PartitionLayout(
        mode=canonical,
        partitions=partitions,
        cu_per_partition=cu_total // partitions,
        gib_per_partition=gib_per_partition,
        probed=False,
    )


def observe_partition(gpu_id: int = 0, *, gpu_type: str | None = None) -> PartitionLayout | None:
    """Read the live partition topology of one card.

    The single entry point a caller needs to answer "what shape is this card in,
    and how big is one partition". Everything comes from the device except the
    CU fallback, which is marked as such.

    Args:
        gpu_id: GPU to interrogate.
        gpu_type: Board name, used only if the CU probe fails.

    Returns:
        The live layout, or ``None`` when the mode itself could not be read --
        which is the ordinary case on a host without ``amd-smi``, and means the
        caller should proceed without a partitioning opinion rather than fail.
    """
    try:
        mode = read_partition_mode(gpu_id)
    except PartitionError as exc:
        log.debug("no compute-partition state for GPU %d: %s", gpu_id, exc)
        return None
    if mode not in MODE_PARTITION_COUNTS:
        log.warning("GPU %d reports compute-partition mode %r, which this build does not know", gpu_id, mode)
        return None
    try:
        return layout_for(
            mode,
            gpu_type=gpu_type,
            cu_per_partition=read_device_cu(gpu_id),
            gib_per_partition=read_device_gib(gpu_id),
        )
    except PartitionError as exc:
        log.warning("GPU %d is in %s but its partitions could not be sized: %s", gpu_id, mode, exc)
        return None


def published_shape(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Read the partition shape a launch published, without touching the device.

    The provenance record needs to state which topology produced a number, and
    it is written in places where spawning ``amd-smi`` is not acceptable: the
    crash-safe ``final.json`` writer, and unit tests that must not run a probe.
    The launch already established the shape and put it in the environment, so
    this reads that rather than asking again.

    Args:
        env: Environment to read; defaults to the process environment.

    Returns:
        The shape, or ``None`` when this session published none -- which is the
        ordinary case, and must stay distinguishable from an unpartitioned card.
    """
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
    """Return a predicate selecting partition devices by CU count.

    **No caller in this repository.** It is the reference implementation of a
    rule the out-of-tree benchmark entrypoint is most likely to get wrong, kept
    beside the documentation that states the rule rather than left to be
    re-derived downstream; do not go looking for the consumer.

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
        ``True`` when they fit, or when either figure is unknown -- an unknown is
        reported by the caller that has the number, not guessed at here.

    The unknown-capacity guard shares :attr:`PartitionLayout.capacity_known`
    with :func:`~hyperloom.orchestrator.actions.executors._partition_shape.validate_session_shape`,
    which tests it first so it can warn -- something a bool return cannot do.
    That makes this guard unreachable from the in-tree caller by design, and it
    stays because a predicate that silently multiplies by ``None`` for anyone
    else is worse than one redundant test.
    """
    if not layout.capacity_known or required_gib <= 0:
        return True
    # ``capacity_known`` has already established this is a positive float; the
    # coercion is only so the shared predicate can carry the decision without
    # the arithmetic having to re-assert the type.
    capacity_gib = float(layout.gib_per_partition or 0.0)
    return required_gib * max(1, int(streams_per_partition)) <= capacity_gib


# The probe helpers behind observe_partition -- read_partition_mode(s),
# read_device_cu, read_device_gib -- are deliberately absent. observe_partition's
# docstring calls itself "the single entry point a caller needs", and listing the
# steps it takes here would contradict that: they are the call graph, not the
# interface. They stay importable for the tests that exercise each payload shape.
__all__ = [
    "MODE_PARTITION_COUNTS",
    "PARTITION_COUNT_ENV",
    "PARTITION_CU_ENV",
    "PARTITION_MODE_ENV",
    "PARTITION_STREAMS_ENV",
    "PARTITION_TOTAL_STREAMS_ENV",
    "PartitionError",
    "PartitionLayout",
    "fits_in_partition",
    "layout_for",
    "observe_partition",
    "parse_mode",
    "partition_device_predicate",
    "published_shape",
]
