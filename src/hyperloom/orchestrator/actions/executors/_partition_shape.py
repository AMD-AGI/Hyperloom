# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The compute-partition shape a session runs in, checked once at launch.

The mode is fixed for the session and established outside it. This module does
three things with that fact, and nothing else:

* **Records it.** Whatever mode the card is in becomes part of the session's
  platform fingerprint, so a number is never filed under a topology it was not
  measured on. Two runs of the same configuration on the same card in SPX and in
  CPX are different experiments, and without this they are indistinguishable in
  the history.
* **Checks it.** A session given a mode whose partitions cannot hold the
  workload is going to fail, and it is going to take hours to find out. The
  arithmetic that says so costs milliseconds, so it runs at launch.
* **Publishes it.** The benchmark entrypoint already has to fan work out across
  partitions. It gets the shape -- mode, partition count, CU per partition,
  streams -- so it can place work and, crucially, verify what it is running on.

What it deliberately does not do is change the mode. The card must already be in
its mode before ``optimize`` starts: this runs at launch, and the entrypoint that
places work across partitions does not start until the first benchmark, long
after the shape has been checked and recorded. So the apply belongs to the
operator or the provisioning platform, not to anything downstream of here and
not to an optimization loop running agent-authored code. Nothing here needs
privilege.

The check fails closed on the one thing it can be wrong about. If the operator
declares an expected mode and the card cannot be read, the session is refused
rather than run: the declaration exists precisely to catch an external set that
did not take, and an unverifiable assertion is not a satisfied one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.gpu_partition import (
    PARTITION_COUNT_ENV,
    PARTITION_CU_ENV,
    PARTITION_MODE_ENV,
    PARTITION_STREAMS_ENV,
    PARTITION_TOTAL_STREAMS_ENV,
    PartitionLayout,
    fits_in_partition,
    observe_partition,
)

log = logging.getLogger(__name__)

#: GPU whose partition state describes this session. Single-card: this
#: session's numbers describe partitions of that one physical GPU.
PARTITION_GPU_ENV = "HYPERLOOM_PARTITION_GPU"

#: Streams per partition when unset. Two is where every mode measured on MI355X
#: peaked: one leaves each partition idle through the fixed per-pass cost, a
#: second fills it, a third only adds queueing. A default rather than a constant
#: because it is a property of the workload, not of the hardware.
DEFAULT_STREAMS_PER_PARTITION = 2


@dataclass(frozen=True)
class ShapeVerdict:
    """The launch-time verdict on a session's partition shape.

    Attributes:
        layout: The live topology, or ``None`` when it could not be read.
        refusal: Why the session must not start. Empty when it may.
        warnings: Things the operator should know that do not stop the run.
        notes: Lines describing the shape, for the launch banner.
    """

    layout: PartitionLayout | None = None
    refusal: str = ""
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Whether the session may proceed."""
        return not self.refusal


def partition_gpu_id() -> int:
    """Return the GPU whose partition state describes this session (default 0).

    An unusable value is warned about rather than quietly replaced. Reading card
    0 in silence is the exact mislabelling this module exists to prevent: the
    session would file card 0's topology as its own while the benchmark ran on
    the card the operator meant to name.
    """
    raw = os.environ.get(PARTITION_GPU_ENV, "").strip()
    if not raw:
        return 0
    try:
        gpu = int(raw)
    except ValueError:
        gpu = -1
    if gpu < 0:
        log.warning(
            "%s=%r is not a usable GPU id; reading compute partitioning from GPU 0 instead, "
            "whose topology may not be this session's.",
            PARTITION_GPU_ENV,
            raw,
        )
        return 0
    return gpu


def per_stream_footprint_gib(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
) -> tuple[float, str]:
    """Resolve the per-stream HBM footprint to size partitions against.

    In practice today there is one source: the checkpoint's weight bytes.

    * **Weights.** ``weight_bytes`` read byte-exact from the checkpoint's
      safetensors index. A *lower bound*, and deliberately used as one: each
      stream holds its own copy of the weights, so the true footprint is never
      smaller. That makes a "does not fit" verdict from this source a proof and
      a "fits" verdict no evidence at all -- which is exactly the asymmetry a
      refusal needs, since it only ever acts on the former.
    * **Measured.** ``peak_gib_per_stream`` would be the real footprint --
      weights, activations and workspace -- and is the only thing that could
      rule out a mode the weights alone fit. Nothing in this repository writes
      it: the branch below reads it from task params and from ``current_best``
      so a harness that starts reporting it is honoured without a change here,
      but no in-tree producer exists, so the weights bound is what every
      refusal is actually made on. Do not read the two bullets as a fallback
      chain that is exercised.

    Args:
        params: Task params, consulted for an explicit override.
        shared_state: Live SharedState, consulted for a prior measurement and
            the model identity.

    Returns:
        ``(gib, source)``, or ``(0.0, "")`` when neither source can answer.
        ``source`` names the origin for the message that reports a refusal,
        since "too big by the weights alone" and "too big as measured" call for
        different responses from an operator.
    """
    measured = (params or {}).get("peak_gib_per_stream")
    if measured is None:
        best = getattr(shared_state, "current_best", None)
        if isinstance(best, dict):
            measured = best.get("peak_gib_per_stream")
    try:
        if measured is not None and float(measured) > 0:
            return float(measured), "measured"
    except (TypeError, ValueError):
        # A report carrying the field as something other than a number is
        # treated as not carrying it: fall through to the weight-bytes bound
        # rather than fail, since the contract is "refuses nothing when the
        # footprint is unknown" and a malformed reading is unknown.
        pass

    model_path = str((params or {}).get("model_path") or getattr(shared_state, "model_path", "") or "").strip()
    if not model_path:
        return 0.0, ""
    # Params first, then state: the launch path has resolved flags but no
    # SharedState yet, and it is the path where a refusal is worth the most.
    precision = str((params or {}).get("precision") or getattr(shared_state, "precision", "") or "")
    # Lazy: the kernel package pulls in the analytical stack, and this module is
    # imported whether a partition shape is in play or not.
    from hyperloom.orchestrator.kernel.roofline_ceiling import load_model_meta

    try:
        meta = load_model_meta(model_path, precision_hint=precision)
    except Exception as exc:  # noqa: BLE001 — an unreadable checkpoint is "unknown", not fatal
        log.debug("cannot size partitions from %s: %s", model_path, exc)
        return 0.0, ""
    if meta is None or meta.weight_bytes <= 0:
        return 0.0, ""
    return meta.weight_bytes / float(1024**3), "weights"


def validate_session_shape(
    *,
    declared_mode: str = "",
    streams: int | None = None,
    gpu_type: str | None = None,
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
    gpu_id: int | None = None,
    fanout_expected: bool = True,
) -> ShapeVerdict:
    """Check, at launch, that this session can run in the shape it was given.

    Args:
        declared_mode: The mode the operator asserts the card is in. Empty means
            no assertion, in which case an unreadable card is merely unrecorded
            rather than a refusal.
        streams: Concurrent streams intended per partition. ``None`` means the
            caller named none and takes the default; a value below one is
            refused, not quietly replaced by it. Tested against ``None`` rather
            than falsiness for that reason -- ``0 or DEFAULT`` is ``DEFAULT``,
            which would honour the one value most likely to be a mistake.
        gpu_type: Board name, used only if the CU probe fails.
        params: Task params, for the footprint resolution.
        shared_state: Live SharedState, for the footprint resolution.
        gpu_id: GPU to interrogate; defaults to the configured one.
        fanout_expected: Whether anything in this session will actually place
            streams on partitions. ``False`` suppresses the footprint refusal,
            because the question it answers does not arise: see below.

    Returns:
        The verdict. ``refusal`` is non-empty only for a shape that cannot work:
        a declared mode that does not match the card, and a workload whose
        streams provably do not fit one partition.

    The footprint check is gated on ``fanout_expected`` rather than run
    unconditionally because "N streams share one partition" is a premise, not a
    fact about the card. Without a fan-out, nothing puts a second stream on a
    partition, and worse, nothing pins the benchmark to a partition at all --
    HIP enumerates whole cards before partitions, so on a node where one card of
    eight is split, device 0 is a *whole* card. Refusing such a session on
    ``streams x footprint`` would be arithmetic about a shape it was never going
    to run in. The mode is still observed, recorded and published; only the
    refusal is withheld.
    """
    warnings: list[str] = []
    notes: list[str] = []
    try:
        requested_streams = DEFAULT_STREAMS_PER_PARTITION if streams is None else int(streams)
    except (TypeError, ValueError):
        requested_streams = 0
    if requested_streams < 1:
        # Refused rather than defaulted, and refused before the card is touched:
        # this is a usage error about the request, not a fact about the hardware,
        # so it needs no probe to decide and must not differ from the CLI's own
        # verdict on the same value.
        return ShapeVerdict(
            refusal=(
                f"streams per partition must be >= 1, got {streams!r}. One stream per "
                f"partition is the floor: a partition with no stream on it measures nothing, "
                f"and a mode is only worth setting at two."
            ),
        )
    streams = requested_streams
    gpu = partition_gpu_id() if gpu_id is None else gpu_id

    layout = observe_partition(gpu, gpu_type=gpu_type)
    if layout is None:
        if declared_mode:
            return ShapeVerdict(
                refusal=(
                    f"--compute-partition-mode {declared_mode} was declared, but GPU {gpu}'s "
                    f"compute-partition state could not be read, so the claim cannot be "
                    f"checked. The flag asserts what the card is already in -- it does not "
                    f"set it. Drop the flag to run without the assertion."
                ),
            )
        # Nothing declared and nothing readable is the ordinary case on a host
        # without amd-smi. The session runs exactly as it always has.
        return ShapeVerdict()

    if declared_mode and layout.mode != declared_mode:
        return ShapeVerdict(
            layout=layout,
            refusal=(
                f"GPU {gpu} is in {layout.mode}, not the declared {declared_mode}. Nothing in "
                f"the optimizer changes the mode, so this session would measure {layout.mode} "
                f"and record it under a name the operator did not intend. Set the card to "
                f"{declared_mode} before launching, or drop the flag."
            ),
        )

    notes.append(f"Compute partitioning : {layout.describe()}")
    if not layout.probed:
        warnings.append(
            f"GPU {gpu}'s CU count came from the built-in board table, not the device. "
            f"Partition devices are selected by matching that count exactly, so if the "
            f"table is wrong for this board the benchmark will find no partitions."
        )

    if not layout.partitioned:
        # An unpartitioned card is the default and needs no further checking:
        # one partition is the whole card, which is what every other session
        # already assumes.
        return ShapeVerdict(layout=layout, warnings=tuple(warnings), notes=tuple(notes))

    if not fanout_expected:
        # Record the shape, refuse nothing. This is the card someone else left
        # split, met by a session that will not place work per partition.
        warnings.append(
            f"GPU {gpu} is in {layout.mode}, but nothing in this session places work on "
            f"individual partitions, so the shape is recorded and no per-partition memory "
            f"check is made. Which device the benchmark lands on is not decided here -- HIP "
            f"enumerates whole cards before partitions -- so treat the numbers as belonging "
            f"to an unknown fraction of the card until the fan-out is known."
        )
        return ShapeVerdict(layout=layout, warnings=tuple(warnings), notes=tuple(notes))

    notes.append(f"Streams per partition: {streams} ({streams * layout.partitions} total)")

    footprint_gib, source = per_stream_footprint_gib(params, shared_state)
    if footprint_gib <= 0:
        warnings.append(
            f"Cannot size this workload against a {layout.mode} partition: the checkpoint's "
            f"weight bytes could not be read. The session will run, but a partition too small "
            f"for it will surface as an out-of-memory failure rather than a refusal here."
        )
        return ShapeVerdict(layout=layout, warnings=tuple(warnings), notes=tuple(notes))

    if not layout.capacity_known:
        # Shares its predicate with fits_in_partition rather than testing
        # `is None` here and falsiness there, which sent a zero capacity down
        # opposite paths. Checked here, ahead of the arithmetic, because only
        # this side can say so out loud.
        warnings.append(
            f"GPU {gpu} reported no usable per-partition memory, so the {footprint_gib:.1f} GiB "
            f"per stream could not be checked against a {layout.mode} partition."
        )
        return ShapeVerdict(layout=layout, warnings=tuple(warnings), notes=tuple(notes))

    capacity_gib = float(layout.gib_per_partition or 0.0)
    needed = footprint_gib * streams
    if not fits_in_partition(footprint_gib, layout, streams):
        return ShapeVerdict(
            layout=layout,
            refusal=(
                f"this workload does not fit GPU {gpu}'s {layout.mode} partitions: "
                f"{streams} x {footprint_gib:.1f} GiB = {needed:.1f} GiB needed per partition, "
                f"{capacity_gib:.1f} GiB available "
                f"({layout.partitions} x {layout.cu_per_partition} CU). "
                f"The {footprint_gib:.1f} GiB is "
                + (
                    "a measured per-stream peak."
                    if source == "measured"
                    else "the weights alone, so the real footprint is larger."
                )
                + " Use a wider partition mode, fewer streams per partition, or a smaller model."
            ),
            warnings=tuple(warnings),
            notes=tuple(notes),
        )

    notes.append(
        f"Per-partition memory : {needed:.1f} GiB needed of {capacity_gib:.1f} GiB "
        f"({streams} x {footprint_gib:.1f} GiB, from {source})"
    )
    return ShapeVerdict(layout=layout, warnings=tuple(warnings), notes=tuple(notes))


def runtime_env(layout: PartitionLayout, streams: int, *, fanout: bool = True) -> dict[str, str]:
    """Build the env published at launch for the shape this session runs in.

    Two readers, and they want different things, which is why ``fanout`` splits
    the block rather than suppressing it:

    * **The provenance record.** ``platform_fingerprint`` reads the observed mode,
      partition count and CU per partition back out of here, because it is
      written on the crash path where spawning ``amd-smi`` is not acceptable.
      Those three describe the card and are true whatever the benchmark does, so
      they are always published -- withholding them is how a CPX number gets
      filed as though it were SPX, which is the failure this module exists to
      prevent.
    * **The fan-out instruction.** Streams per partition and the total
      concurrency are directions to a benchmark that places work on each
      partition. Only a scriptable entrypoint does that, so publishing them to a
      serving session would state a concurrency nothing was going to drive.

    The entrypoint is given the shape rather than a device list on purpose. HIP
    enumerates whole cards before partitions, so an index list computed here
    would be wrong in the one case that matters and wrong invisibly; the process
    holding the GPU context is the one positioned to check a device's CU count
    against :data:`PARTITION_CU_ENV` and refuse what does not match.

    Args:
        layout: The observed topology.
        streams: Streams to place on each partition.
        fanout: Whether this session's benchmark places work per partition.

    Returns:
        Env mapping for the benchmark process.
    """
    streams = max(1, int(streams))
    env = {
        PARTITION_MODE_ENV: layout.mode,
        PARTITION_COUNT_ENV: str(layout.partitions),
        PARTITION_CU_ENV: str(layout.cu_per_partition),
    }
    if fanout:
        env[PARTITION_STREAMS_ENV] = str(streams)
        env[PARTITION_TOTAL_STREAMS_ENV] = str(layout.partitions * streams)
    return env


def session_shape_summary(
    layout: PartitionLayout,
    streams: int,
    *,
    fanout_expected: bool = True,
) -> dict[str, Any]:
    """Summarize the session's partition shape for the report and the manifest.

    A layout is required. An unknown shape is ``{}`` at the call site, not a
    record of zeroes here: this used to accept ``None`` and answer with a
    four-key mapping missing ``cu_probed``, ``gib_per_partition`` and
    ``fanout_expected``, so a consumer that met it saw a second schema for the
    same field -- one whose absent provenance key reads as a positive claim.

    Args:
        layout: The observed topology.
        streams: Streams placed on each partition.
        fanout_expected: Whether this session's benchmark places work on each
            partition. Recorded because it decides whether the throughput can be
            an aggregate at all, which the report has to state and cannot infer.

    Returns:
        A JSON-safe mapping, always carrying every key.
    """
    return {
        "mode": layout.mode,
        "partitions": layout.partitions,
        "cu_per_partition": layout.cu_per_partition,
        "gib_per_partition": layout.gib_per_partition,
        "streams_per_partition": max(1, int(streams)),
        "cu_probed": layout.probed,
        "fanout_expected": bool(fanout_expected),
    }


__all__ = [
    "DEFAULT_STREAMS_PER_PARTITION",
    "PARTITION_COUNT_ENV",
    "PARTITION_CU_ENV",
    "PARTITION_GPU_ENV",
    "PARTITION_MODE_ENV",
    "PARTITION_STREAMS_ENV",
    "PARTITION_TOTAL_STREAMS_ENV",
    "ShapeVerdict",
    "partition_gpu_id",
    "per_stream_footprint_gib",
    "runtime_env",
    "session_shape_summary",
    "validate_session_shape",
]
