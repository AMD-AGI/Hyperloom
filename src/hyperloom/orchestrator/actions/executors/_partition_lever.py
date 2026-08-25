# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Applying a compute-partition mode around one benchmark run.

The mode travels as an ordinary variant env, ``HYPERLOOM_PARTITION_MODE``. That
is deliberate: a candidate identified by its envs is already fingerprinted,
deduplicated in the explore ledger, graded by the KEEP gate and recorded in the
journal, so expressing the mode that way inherits all of it instead of growing a
parallel search axis that each of those would have to learn about.

What cannot be inherited is the application. Every other env is delivered *to*
the benchmark process; this one has to change the card before that process
starts, and change it back afterwards. Hence this module, called from the
scriptable runner rather than from the env materializer.

Two things it deliberately refuses to do:

* **Guess.** If the mode cannot be set -- no privilege, no ``amd-smi``, a card
  with resident processes -- the run fails. Falling back to the current topology
  would produce a perfectly good measurement labelled with the wrong mode, which
  is worse than no measurement, because it is indistinguishable from a real one.
* **Enumerate devices for the benchmark.** The runner publishes the shape it
  established (how many partitions, how many CUs each) and the benchmark selects
  its own devices by CU count. Index-based selection is the classic way to
  measure a whole card and report it as a partition, and the process that owns
  the GPU context is the one positioned to check.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Sequence

from hyperloom.common.gpu_partition import (
    MODE_PARTITION_COUNTS,
    PartitionError,
    PartitionLayout,
    fits_in_partition,
    layout_for,
    parse_modes,
    partitioned,
    read_hbm_gib,
    read_partition_mode,
)

log = logging.getLogger(__name__)

#: Per-variant selector. Present => this run is measured under that mode.
PARTITION_MODE_ENV = "HYPERLOOM_PARTITION_MODE"

#: Session-level lever published by the CLI.
PARTITION_MODES_ENV = "HYPERLOOM_COMPUTE_PARTITION_MODES"
STREAMS_PER_PARTITION_ENV = "HYPERLOOM_STREAMS_PER_PARTITION"

#: GPU whose partition mode is managed. Single-card today: the measurement is
#: aggregate throughput on one physical GPU, and repartitioning a whole node to
#: measure that is a much larger blast radius for no extra signal.
PARTITION_GPU_ENV = "HYPERLOOM_PARTITION_GPU"

#: Published to the benchmark so its entrypoint can fan out and, crucially,
#: verify what it is running on.
RUNTIME_MODE_ENV = "HYPERLOOM_PARTITION_MODE"
RUNTIME_COUNT_ENV = "HYPERLOOM_PARTITION_COUNT"
RUNTIME_CU_ENV = "HYPERLOOM_PARTITION_CU"
RUNTIME_STREAMS_ENV = "HYPERLOOM_PARTITION_STREAMS_PER_PARTITION"
RUNTIME_TOTAL_STREAMS_ENV = "HYPERLOOM_PARTITION_TOTAL_STREAMS"


def partition_gpu_id() -> int:
    """Return the GPU whose mode this session manages (default 0)."""
    try:
        return max(0, int(os.environ.get(PARTITION_GPU_ENV, "0").strip() or 0))
    except ValueError:
        return 0


def streams_per_partition() -> int:
    """Return the configured streams per partition (default 2).

    Two is where every mode measured on MI355X peaked. It is a default rather
    than a constant because it is a property of the workload's fixed per-pass
    cost, not of the hardware.
    """
    try:
        return max(1, int(os.environ.get(STREAMS_PER_PARTITION_ENV, "2").strip() or 2))
    except ValueError:
        return 2


def read_session_lever() -> tuple[tuple[str, ...], int, float]:
    """Read the session lever back out of the env the CLI published.

    One reader for the several places that need it -- seeding the manifest,
    persisting across a resume, reporting the session's shape -- because that
    env is the canonical post-validation form of these three flags. Parsing it
    independently at each site is that many chances for the manifest to record
    a contract the executors were never handed.

    Returns:
        ``(modes, streams_per_partition, latency_budget_ms)``. Empty modes and a
        zero budget each mean off.
    """
    from ._latency_budget import LATENCY_BUDGET_ENV

    modes = tuple(m.strip() for m in os.environ.get(PARTITION_MODES_ENV, "").split(",") if m.strip())
    try:
        budget = max(0.0, float(os.environ.get(LATENCY_BUDGET_ENV, "").strip() or 0.0))
    except ValueError:
        budget = 0.0
    return modes, streams_per_partition(), budget


def resolve_session_modes(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
) -> tuple[str, ...]:
    """Resolve the modes this session is allowed to explore.

    Precedence is most-specific-first, matching
    :func:`_latency_budget.resolve_latency_budget_ms`: an explicit task
    parameter, then the session state the CLI seeded, then the environment.

    Args:
        params: Task params, which may carry ``compute_partition_modes``.
        shared_state: Live SharedState, which may carry the persisted list.

    Returns:
        Canonical modes, empty when the lever is off.
    """
    for candidate in (
        (params or {}).get("compute_partition_modes"),
        getattr(shared_state, "compute_partition_modes", None),
        os.environ.get(PARTITION_MODES_ENV),
    ):
        if not candidate:
            continue
        try:
            modes = parse_modes(candidate)
        except PartitionError as exc:
            log.warning("ignoring unusable compute-partition modes %r: %s", candidate, exc)
            continue
        if modes:
            return modes
    return ()


def per_stream_footprint_gib(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
) -> tuple[float, str]:
    """Resolve the per-stream HBM footprint to size partitions against.

    Two sources, tightest first:

    * **Measured.** ``peak_gib_per_stream`` from the baseline run, when the
      harness reported it. This is the real footprint -- weights, activations
      and workspace -- so it is the only source that can rule out a mode the
      weights alone would fit.
    * **Weights.** ``weight_bytes`` read byte-exact from the checkpoint's
      safetensors index. A *lower bound*, and deliberately used as one: each
      stream holds its own copy of the weights, so the true footprint is never
      smaller. That makes a "does not fit" verdict from this source a proof and
      a "fits" verdict no evidence at all -- which is exactly the asymmetry
      pruning needs, since it only ever acts on the former.

    Args:
        params: Task params, consulted for an explicit override.
        shared_state: Live SharedState, consulted for the baseline measurement
            and the model identity.

    Returns:
        ``(gib, source)``, or ``(0.0, "")`` when neither source can answer.
        ``source`` names the origin for the log line that reports a drop, since
        "too big by the weights alone" and "too big as measured" call for
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
        pass

    model_path = str((params or {}).get("model_path") or getattr(shared_state, "model_path", "") or "").strip()
    if not model_path:
        return 0.0, ""
    # Lazy: the kernel package pulls in the analytical stack, and this module is
    # imported by the executors on every run whether the lever is on or not.
    from hyperloom.orchestrator.kernel.roofline_ceiling import load_model_meta

    try:
        meta = load_model_meta(
            model_path,
            precision_hint=str(getattr(shared_state, "precision", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 — an unreadable checkpoint is "unknown", not fatal
        log.debug("cannot size partitions from %s: %s", model_path, exc)
        return 0.0, ""
    if meta is None or meta.weight_bytes <= 0:
        return 0.0, ""
    return meta.weight_bytes / float(1024**3), "weights"


def prune_infeasible_modes(
    modes: Sequence[str],
    *,
    gpu_type: str | None,
    hbm_gib: float | None,
    footprint_gib: float,
    streams: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Drop modes whose partitions provably cannot hold the streams placed on them.

    A partition gets its fraction of the card's HBM while every stream on it
    keeps a full copy of the weights, so the narrow modes run out of memory
    first. Measuring that is expensive and uninformative: the run OOMs, the
    variant is scored as a failure, and the operator learns from a stack trace
    what arithmetic could have said for free.

    Silent when it cannot compute -- an unknown capacity or an unknown footprint
    yields no drops rather than a guess, because the cost of wrongly dropping a
    mode is an optimization that never considers the configuration that would
    have won.

    Args:
        modes: Canonical modes the operator asked for, in their order.
        gpu_type: Board name, needed to size partitions.
        hbm_gib: Card capacity, or ``None`` when unknown.
        footprint_gib: Per-stream footprint; ``0`` when unknown.
        streams: Streams that will share each partition.

    Returns:
        ``(kept, reasons)``. ``kept`` preserves the input order; ``reasons``
        holds one human-readable line per dropped mode.
    """
    if not hbm_gib or footprint_gib <= 0:
        return tuple(modes), ()
    kept: list[str] = []
    reasons: list[str] = []
    for mode in modes:
        try:
            layout = layout_for(gpu_type, mode, hbm_gib=hbm_gib)
        except PartitionError as exc:
            # Undescribable here means undescribable at apply time too, but this
            # is not the place that gets to refuse it: keep the mode and let the
            # apply path raise with its own context.
            log.debug("not sizing %s: %s", mode, exc)
            kept.append(mode)
            continue
        if fits_in_partition(footprint_gib, layout, streams):
            kept.append(mode)
            continue
        reasons.append(
            f"{mode}: {streams} x {footprint_gib:.1f} GiB = {footprint_gib * streams:.1f} GiB "
            f"needed per partition, {layout.gib_per_partition:.1f} GiB available "
            f"({layout.partitions} x {layout.cu_per_partition} CU)"
        )
    return tuple(kept), tuple(reasons)


def partition_lever_grid(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
    *,
    framework: str | None,
    hbm_gib: float | None = None,
) -> list[dict[str, Any]]:
    """Expand the session's mode list into one explore variant per mode.

    Each variant is env-only, carrying nothing but
    :data:`PARTITION_MODE_ENV`. That keeps the mode a plain point in the search
    space: it is fingerprinted, deduplicated, gated and journalled by the same
    code as every other variant, and a mode that loses is reverted like any
    other losing knob.

    A mode that may already match the card is still emitted: it is worth a
    measurement under the same harness as its rivals rather than an inherited
    baseline number taken on trust.

    A mode whose partitions cannot hold the streams destined for them is not
    emitted, because that variant has only one outcome and it is an OOM. See
    :func:`prune_infeasible_modes`; the arithmetic needs a card capacity and a
    per-stream footprint, and drops nothing when either is unknown.

    Only scriptable frameworks get variants. The mode is applied by
    :func:`plan_partition_run`, which the scriptable runner calls and the
    serving path does not, so emitting these for a server framework would
    deliver the env, change nothing, and record the result under a mode the
    card was never in -- the exact mislabelling this module refuses elsewhere.
    The CLI rejects that combination at launch; this is the second line.

    Args:
        params: Task params, consulted for the mode list.
        shared_state: Live SharedState, consulted for the persisted list.
        framework: Framework this round runs under.
        hbm_gib: Card capacity for the feasibility check. Read from the managed
            GPU when omitted; injectable so the decision is testable without a
            card and overridable by a caller that already knows.

    Returns:
        Variant payload dicts ready for ``_grid_variants_from_payload``, or
        ``[]`` when the lever is off, the framework cannot apply it, or no
        requested mode has room for the workload.
    """
    from hyperloom.inference_optimizer.framework_registry import is_scriptable

    modes = resolve_session_modes(params, shared_state)
    if not modes:
        return []
    if not is_scriptable(framework):
        log.warning(
            "compute-partition modes %s ignored: framework %r does not apply "
            "partition modes, so the variants would measure the card's current "
            "topology under another mode's name",
            ",".join(modes),
            framework or "",
        )
        return []

    streams = streams_per_partition()
    footprint_gib, footprint_source = per_stream_footprint_gib(params, shared_state)
    # Capacity is only read once a footprint exists to compare it against. With
    # no footprint the check cannot reach a verdict either way, and this spares
    # every mode-less-of-a-workload caller an amd-smi subprocess.
    if hbm_gib is None and footprint_gib > 0:
        hbm_gib = read_hbm_gib(partition_gpu_id())
    feasible, dropped = prune_infeasible_modes(
        modes,
        gpu_type=str((params or {}).get("gpu_type") or getattr(shared_state, "gpu_type", "") or ""),
        hbm_gib=hbm_gib,
        footprint_gib=footprint_gib,
        streams=streams,
    )
    for reason in dropped:
        # Warned rather than debugged: the operator asked for this mode by name,
        # and a list that quietly comes back shorter than it went in is the kind
        # of thing that gets rediscovered as a bug.
        log.warning(
            "compute-partition %s dropped at %d stream(s)/partition, footprint from %s",
            reason,
            streams,
            footprint_source,
        )
    if not feasible:
        log.warning(
            "no requested compute-partition mode has room for a %.1f GiB/stream "
            "footprint (%s) at %d stream(s)/partition; the lever contributes no "
            "variants this round",
            footprint_gib,
            footprint_source,
            streams,
        )
        return []
    return [
        {
            "name": f"partition_{mode.lower()}",
            "extra_args": "",
            "extra_envs": {PARTITION_MODE_ENV: mode},
            "note": f"compute-partition {mode}",
            "provenance": "partition_lever",
        }
        for mode in feasible
    ]


def requested_mode(envs: dict[str, Any] | None) -> str:
    """Return the partition mode this run asks for, or ``""`` when none.

    The variant's own env wins over the process environment, so a grid can hold
    one variant at SPX while the session lever lists several modes.
    """
    for source in (envs or {}, os.environ):
        value = str((source or {}).get(PARTITION_MODE_ENV) or "").strip().upper()
        if value:
            return value
    return ""


def runtime_env(layout: PartitionLayout, streams: int) -> dict[str, str]:
    """Build the env describing an established partition shape.

    Args:
        layout: The layout the hardware confirmed.
        streams: Streams to place on each partition.

    Returns:
        Env mapping for the benchmark subprocess.
    """
    return {
        RUNTIME_MODE_ENV: layout.mode,
        RUNTIME_COUNT_ENV: str(layout.partitions),
        RUNTIME_CU_ENV: str(layout.cu_per_partition),
        RUNTIME_STREAMS_ENV: str(streams),
        RUNTIME_TOTAL_STREAMS_ENV: str(layout.partitions * streams),
    }


def _refuse_split_card_for_unpartitioned_run() -> None:
    """Refuse a mode-less run on a card that is still split.

    :func:`partitioned` logs a failed restore instead of raising, so it cannot
    mask the exception that caused the exit. That is right, but it means a
    session can carry on with the card left in a mode nobody asked for -- and
    the runs that request *no* mode are the ones with nothing to notice it. They
    would measure a split card and be recorded as the unpartitioned baseline,
    which is the mislabelling this module refuses everywhere else, arrived at
    from the one direction that had no check.

    Only speaks when the session lever is engaged. With the lever off nothing
    here has touched the hardware, so a split card is the operator's own
    arrangement and none of this module's business.

    Raises:
        PartitionError: If the managed card is in a split mode. An unreadable
            mode is left alone: "cannot tell" is not evidence of a problem, and
            the apply path already refuses what it cannot verify.
    """
    if not resolve_session_modes():
        return
    gpu_id = partition_gpu_id()
    try:
        current = read_partition_mode(gpu_id)
    except PartitionError as exc:
        log.debug("cannot check GPU %d topology before an unpartitioned run: %s", gpu_id, exc)
        return
    if MODE_PARTITION_COUNTS.get(current) == 1:
        return
    raise PartitionError(
        f"GPU {gpu_id} is in {current}, but this run requested no partition mode. "
        f"Measuring it now would file a split card's number as the unpartitioned "
        f"baseline. A restore this session failed, or the card was already split "
        f"when the session started; return it to a single partition before "
        f"continuing."
    )


def plan_partition_run(
    envs: dict[str, Any] | None,
    *,
    gpu_type: str | None,
) -> tuple[str, dict[str, str]]:
    """Resolve the mode for this run and the env describing it.

    Args:
        envs: The materialized benchmark envs, which may select a mode.
        gpu_type: Board name, needed to size the partitions.

    Returns:
        ``(mode, env)``. ``("", {})`` when the lever is not engaged, which is
        the default and leaves the run exactly as it was.

    Raises:
        PartitionError: If a mode was requested but cannot be described -- an
            unknown mode or an unrecognised board. Raised rather than ignored:
            the request was explicit, so silently not honouring it would
            mislabel the measurement. Also if no mode was requested but the
            managed card is still split; see
            :func:`_refuse_split_card_for_unpartitioned_run`.
    """
    mode = requested_mode(envs)
    if not mode:
        _refuse_split_card_for_unpartitioned_run()
        return "", {}
    layout = layout_for(gpu_type, mode)
    streams = streams_per_partition()
    return mode, runtime_env(layout, streams)


@contextmanager
def hold_partition_mode(mode: str, *, gpu_type: str | None) -> Iterator[PartitionLayout | None]:
    """Hold the managed GPU in ``mode`` for the duration of the block.

    Yields ``None`` when no mode was requested, so callers can wrap
    unconditionally.

    Raises:
        PartitionError: If the mode cannot be established or verified.
    """
    if not mode:
        yield None
        return
    layout = layout_for(gpu_type, mode)
    gpu_id = partition_gpu_id()
    log.info("benchmark: holding GPU %d at %s", gpu_id, layout.describe())
    with partitioned(gpu_id, mode):
        yield layout


def maybe_hold_partition_mode(mode: str, *, gpu_type: str | None):
    """Return a context manager for ``mode``, or a no-op when the lever is off."""
    if not mode:
        return nullcontext(None)
    return hold_partition_mode(mode, gpu_type=gpu_type)


__all__ = [
    "PARTITION_GPU_ENV",
    "PARTITION_MODES_ENV",
    "PARTITION_MODE_ENV",
    "PartitionError",
    "RUNTIME_COUNT_ENV",
    "RUNTIME_CU_ENV",
    "RUNTIME_MODE_ENV",
    "RUNTIME_STREAMS_ENV",
    "RUNTIME_TOTAL_STREAMS_ENV",
    "STREAMS_PER_PARTITION_ENV",
    "hold_partition_mode",
    "maybe_hold_partition_mode",
    "partition_gpu_id",
    "partition_lever_grid",
    "per_stream_footprint_gib",
    "plan_partition_run",
    "prune_infeasible_modes",
    "requested_mode",
    "resolve_session_modes",
    "runtime_env",
    "streams_per_partition",
]
