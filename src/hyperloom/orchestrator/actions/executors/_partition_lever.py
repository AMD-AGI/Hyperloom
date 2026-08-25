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
from typing import Any, Iterator

from hyperloom.common.gpu_partition import (
    PartitionError,
    PartitionLayout,
    layout_for,
    parse_modes,
    partitioned,
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


def partition_lever_grid(
    params: dict[str, Any] | None = None,
    shared_state: Any = None,
    *,
    framework: str | None,
) -> list[dict[str, Any]]:
    """Expand the session's mode list into one explore variant per mode.

    Each variant is env-only, carrying nothing but
    :data:`PARTITION_MODE_ENV`. That keeps the mode a plain point in the search
    space: it is fingerprinted, deduplicated, gated and journalled by the same
    code as every other variant, and a mode that loses is reverted like any
    other losing knob.

    Every listed mode is emitted, including one that may already match the
    card. This function stays pure -- no hardware read -- and a mode the
    operator named explicitly is worth a measurement under the same harness as
    its rivals rather than an inherited baseline number taken on trust.

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

    Returns:
        Variant payload dicts ready for ``_grid_variants_from_payload``, or
        ``[]`` when the lever is off or the framework cannot apply it.
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
    return [
        {
            "name": f"partition_{mode.lower()}",
            "extra_args": "",
            "extra_envs": {PARTITION_MODE_ENV: mode},
            "note": f"compute-partition {mode}",
            "provenance": "partition_lever",
        }
        for mode in modes
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
            mislabel the measurement.
    """
    mode = requested_mode(envs)
    if not mode:
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
    "plan_partition_run",
    "requested_mode",
    "resolve_session_modes",
    "runtime_env",
    "streams_per_partition",
]
