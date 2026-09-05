# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The advisory projection: what the gate may say about resources it does not own.

An in-memory snapshot of resource facts ``PolicyGate`` cannot read
synchronously, refreshed out of band. A denial derived from it is labelled
advisory and dated, because the snapshot is not the authority on the resource
-- the acquire is (``RoundStore.open``, the task registry's unique idempotency
key, ``SpecialistGpuPool.try_acquire``, ``TaskRegistry.extend_lease``). It is
still a denial: a rule that fired on the last snapshot fires on the next one
too, and only a refreshed snapshot or the round's own holder lets the attempt
through.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..bus.gpu_pool import resolve_gpu_specialist_devices, resolve_whole_machine_devices
from ..state.round_store import Round

#: A bring-up round holds the machine.
RULE_ROUND_IN_FLIGHT = "enablement_round_in_flight"

#: The GPU specialist pool is configured to zero cards.
RULE_GPU_POOL_DISABLED = "specialist_gpu_pool_disabled"

#: More cards were asked for than the pool holds.
RULE_GPU_EXCEEDS_CAPACITY = "specialist_gpu_request_exceeds_capacity"

#: The lease named by an extension is not one the projection saw running.
RULE_LEASE_NOT_LIVE = "extend_lease_task_not_live"

__all__ = [
    "RULE_ROUND_IN_FLIGHT",
    "AdvisoryDenial",
    "AdvisoryLedger",
    "ResourceProjection",
    "baseline_advisory",
    "effective_gpu_specialist_pool_size",
    "extend_lease_advisory",
    "gpu_specialist_ceiling",
    "serving_tp_for_policy",
    "specialist_gpu_advisory",
    "whole_machine_pool_size",
]


def gpu_specialist_ceiling(shared_state: Any | None = None) -> int:
    """Configured GPU specialist capacity (0 disables ``needs_gpu=true`` dispatch).

    Args:
        shared_state: Optional SharedState whose ``gpu_specialist_capacity`` is
            read first; ``None`` falls back to the
            ``INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY`` env var.

    Returns:
        int: The configured capacity, 0 when unset or unparseable.
    """
    if shared_state is not None:
        try:
            return max(0, int(getattr(shared_state, "gpu_specialist_capacity", 0)))
        except (TypeError, ValueError):
            return 0
    try:
        return max(0, int(os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0")))
    except ValueError:
        return 0


def serving_tp_for_policy(shared_state: Any | None = None) -> int:
    """Resolve serving TP the way ``Coordinator._resolve_serving_tp`` does.

    Args:
        shared_state: Optional SharedState carrying ``tp``.

    Returns:
        int: Serving TP, 0 when unknown.
    """
    if shared_state is not None:
        try:
            tp = int(getattr(shared_state, "tp", 0))
        except (TypeError, ValueError):
            tp = 0
        if tp > 0:
            return tp
    try:
        return max(0, int(os.environ.get("TP", "0")))
    except ValueError:
        return 0


def effective_gpu_specialist_pool_size(shared_state: Any | None = None) -> int:
    """Size of the serving-disjoint GPU specialist pool after the serving carve.

    Args:
        shared_state: Optional SharedState carrying capacity and TP.

    Returns:
        int: Number of cards a non-bench ``needs_gpu`` specialist can be given.
    """
    ceiling = gpu_specialist_ceiling(shared_state)
    if ceiling <= 0:
        return 0
    return len(resolve_gpu_specialist_devices(ceiling, serving_tp=serving_tp_for_policy(shared_state)))


def whole_machine_pool_size() -> int:
    """Size of the whole-machine (framework / bench) GPU pool.

    Every visible card, with no serving carve and no specialist-capacity gate:
    a whole-machine specialist time-shares with serving.

    Returns:
        int: Number of visible cards.
    """
    return len(resolve_whole_machine_devices())


@dataclass(frozen=True)
class ResourceProjection:
    """A snapshot of the resource facts the gate is not allowed to read itself.

    Attributes:
        taken_unix: When the snapshot was taken; carried into every denial
            derived from it.
        excluding_round_id: The bring-up round denying an acquire, if any.
        excluding_round_holder: The task holding that round.
        excluding_round_permanent: Whether that round excludes for good.
        serving_tp: Tensor parallelism the served model runs at.
        gpu_specialist_capacity: Configured GPU specialist capacity.
        gpu_specialist_pool: Cards available to a serving-disjoint specialist.
        whole_machine_pool: Cards available to a whole-machine specialist.
        live_task_ids: Tasks observed running, or ``None`` when the snapshot
            never looked; an unknown set denies nothing.
    """

    taken_unix: float = 0.0
    excluding_round_id: str = ""
    excluding_round_holder: str = ""
    excluding_round_permanent: bool = False
    serving_tp: int = 0
    gpu_specialist_capacity: int = 0
    gpu_specialist_pool: int = 0
    whole_machine_pool: int = 0
    live_task_ids: frozenset[str] | None = None

    @property
    def round_excludes(self) -> bool:
        """bool: Whether a bring-up round was excluding when the snapshot was taken."""
        return bool(self.excluding_round_id)

    @classmethod
    def of(
        cls,
        shared_state: Any | None,
        *,
        now_unix: float = 0.0,
        rounds: Sequence[Round] = (),
        live_task_ids: Iterable[str] | None = None,
    ) -> "ResourceProjection":
        """Build a snapshot from live sources.

        Args:
            shared_state: SharedState to read the session's own facts from.
            now_unix: The instant the snapshot describes.
            rounds: Rounds excluding at ``now_unix``, from
                ``RoundStore.excluding``. The first is reported.
            live_task_ids: Tasks observed running; ``None`` leaves the set
                unknown.

        Returns:
            ResourceProjection: The snapshot.
        """
        excluding = rounds[0] if rounds else None
        return cls(
            taken_unix=float(now_unix),
            excluding_round_id="" if excluding is None else excluding.round_id,
            excluding_round_holder="" if excluding is None else excluding.holder_task_id,
            excluding_round_permanent=False if excluding is None else bool(excluding.exclusion_permanent),
            serving_tp=serving_tp_for_policy(shared_state),
            gpu_specialist_capacity=gpu_specialist_ceiling(shared_state),
            gpu_specialist_pool=effective_gpu_specialist_pool_size(shared_state),
            whole_machine_pool=whole_machine_pool_size(),
            live_task_ids=None if live_task_ids is None else frozenset(str(t) for t in live_task_ids),
        )


@dataclass(frozen=True)
class AdvisoryDenial:
    """One advisory refusal, and the snapshot it was derived from.

    Attributes:
        rule: The rule that fired.
        reason: What the snapshot showed.
        hint: What the emitting role should do instead.
        taken_unix: When the snapshot was taken.
    """

    rule: str
    reason: str
    hint: str = ""
    taken_unix: float = 0.0

    @property
    def message(self) -> str:
        """str: The refusal, stated as advisory and dated."""
        return (
            f"{self.reason} [advisory: read from a resource projection taken "
            f"at unix={self.taken_unix:.3f}; the resource is held or refused "
            f"at the acquire, which this does not replace]"
        )


class AdvisoryLedger:
    """The snapshot the resource rules judge against, swappable under them."""

    def __init__(self, projection: ResourceProjection | None = None):
        """Initialise the ledger.

        Args:
            projection: The snapshot to judge against; ``None`` installs an
                empty one, which refuses nothing.
        """
        self._projection = projection or ResourceProjection()

    @property
    def projection(self) -> ResourceProjection:
        """ResourceProjection: The snapshot currently in force."""
        return self._projection

    def refresh(self, projection: ResourceProjection) -> None:
        """Install a newer snapshot.

        Args:
            projection: The new snapshot.
        """
        self._projection = projection

    def deny(self, rule: str, reason: str, *, hint: str = "") -> AdvisoryDenial:
        """Refuse an attempt, dated to the snapshot the refusal came from.

        Args:
            rule: The rule that fired.
            reason: What the snapshot showed.
            hint: What the emitting role should do instead.

        Returns:
            AdvisoryDenial: The refusal.
        """
        return AdvisoryDenial(rule=rule, reason=reason, hint=hint, taken_unix=self._projection.taken_unix)


def baseline_advisory(ledger: AdvisoryLedger, *, task_id: str = "") -> AdvisoryDenial | None:
    """Refuse a baseline while a bring-up round holds the machine.

    The excluding round's own holder is admitted: it already won the acquire.

    Args:
        ledger: The snapshot to judge against.
        task_id: The task this baseline would run under, when the caller knows
            it. Empty for a proposal that has not been given a row yet.

    Returns:
        AdvisoryDenial | None: The refusal, if the snapshot supports one.
    """
    projection = ledger.projection
    if not projection.round_excludes:
        return None
    if task_id and task_id == projection.excluding_round_holder:
        return None
    permanence = (
        "its lease ran out and nothing ever confirmed the holder dead"
        if projection.excluding_round_permanent
        else "its lease is still live"
    )
    return ledger.deny(
        RULE_ROUND_IN_FLIGHT,
        (
            f"baseline: bring-up round {projection.excluding_round_id!r} "
            f"(holder={projection.excluding_round_holder!r}) holds the "
            f"machine -- {permanence}"
        ),
        hint="Let the round settle; a second bring-up would fight it for the same cards and ports.",
    )


def specialist_gpu_advisory(
    ledger: AdvisoryLedger,
    *,
    gpu_count: int,
    whole_machine: bool,
) -> AdvisoryDenial | None:
    """Judge a specialist's GPU request against the pool the snapshot saw.

    Args:
        ledger: The snapshot to judge against.
        gpu_count: Cards the dispatch would ask for.
        whole_machine: Whether the dispatch time-shares the whole machine
            rather than leasing from the serving-disjoint pool.

    Returns:
        AdvisoryDenial | None: The refusal, if the snapshot supports one.
    """
    projection = ledger.projection
    if projection.gpu_specialist_capacity <= 0 and not (whole_machine and projection.whole_machine_pool > 0):
        return ledger.deny(
            RULE_GPU_POOL_DISABLED,
            "delegate{action='specialist'}: needs_gpu=true but the GPU specialist pool is disabled",
            hint=(
                "Start the session with --gpu-specialist-capacity > 0 or set "
                "INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY before dispatching GPU specialists."
            ),
        )
    if whole_machine:
        pool_size = projection.whole_machine_pool
        pool_desc = "whole-machine GPU pool"
    else:
        pool_size = projection.gpu_specialist_pool
        pool_desc = "serving-disjoint GPU specialist pool"
    if gpu_count <= pool_size:
        return None
    return ledger.deny(
        RULE_GPU_EXCEEDS_CAPACITY,
        (
            f"delegate{{action='specialist'}}: effective gpu_count={gpu_count} "
            f"exceeds {pool_desc} size={pool_size} (configured "
            f"capacity={projection.gpu_specialist_capacity}, "
            f"serving_tp={projection.serving_tp})"
        ),
        hint=(
            "Lower params.gpu_count for non-bench probes, omit it for bench "
            "specialists only when the pool has at least serving TP free "
            "cards, or start a session with a larger GPU pool."
        ),
    )


def extend_lease_advisory(ledger: AdvisoryLedger, *, task_id: str) -> AdvisoryDenial | None:
    """Refuse a lease extension for a task the snapshot saw finished.

    Args:
        ledger: The snapshot to judge against.
        task_id: The task whose lease would move.

    Returns:
        AdvisoryDenial | None: The refusal, if the snapshot supports one.
    """
    live = ledger.projection.live_task_ids
    if live is None or task_id in live:
        return None
    return ledger.deny(
        RULE_LEASE_NOT_LIVE,
        f"extend_lease: task {task_id!r} was not running when the projection was taken",
        hint="Re-read get_running_tasks; a finished task's lease cannot be extended.",
    )
