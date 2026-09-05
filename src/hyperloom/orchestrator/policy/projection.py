# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resource facts ``PolicyGate`` needs but cannot read where it is asked.

Two reasons a fact lands here rather than being read in the rule that wants it:
the round store and the task registry answer asynchronously and the gate is
synchronous, and the pool sizes reach ``rocm-smi`` on a host with no
visible-device mask, which is not a call to make once per validated intent.

The repair pass updates this in place at the top of every tick, before anything
is admitted, and it already reads both sources. It is not the authority on any
of these resources -- the acquire is (``RoundStore.open``, the task registry's
unique idempotency key, ``SpecialistGpuPool.try_acquire``,
``TaskRegistry.extend_lease``) -- so a rule reading it can be wrong about a
resource that changed since; being wrong here denies an attempt that the
acquire would have refused anyway.
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

#: The lease named by an extension is not one the pass saw running.
RULE_LEASE_NOT_LIVE = "extend_lease_task_not_live"

__all__ = [
    "RULE_GPU_EXCEEDS_CAPACITY",
    "RULE_GPU_POOL_DISABLED",
    "RULE_LEASE_NOT_LIVE",
    "RULE_ROUND_IN_FLIGHT",
    "ResourceFacts",
    "effective_gpu_specialist_pool_size",
    "gpu_specialist_ceiling",
    "serving_tp_for_policy",
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


@dataclass
class ResourceFacts:
    """What the resource rules read, updated in place once per tick.

    A default instance refuses nothing: an unknown fact denies nothing, so a
    gate constructed without a repair pass behind it sends every attempt to its
    acquire.

    Attributes:
        read: Whether anything has read these from their sources. False denies
            nothing: a gate with no repair pass behind it must not refuse a
            dispatch on a pool size nobody looked up.
        excluding_round_id: The bring-up round denying an acquire, if any.
        excluding_round_holder: The task holding that round.
        serving_tp: Tensor parallelism the served model runs at.
        gpu_specialist_capacity: Configured GPU specialist capacity.
        gpu_specialist_pool: Cards available to a serving-disjoint specialist.
        whole_machine_pool: Cards available to a whole-machine specialist.
        live_task_ids: Tasks observed running, or ``None`` when nothing has
            looked yet.
    """

    read: bool = False
    excluding_round_id: str = ""
    excluding_round_holder: str = ""
    serving_tp: int = 0
    gpu_specialist_capacity: int = 0
    gpu_specialist_pool: int = 0
    whole_machine_pool: int = 0
    live_task_ids: frozenset[str] | None = None

    @property
    def round_excludes(self) -> bool:
        """bool: Whether a bring-up round was holding the machine at the update."""
        return bool(self.excluding_round_id)

    def update(
        self,
        shared_state: Any | None,
        *,
        rounds: Sequence[Round] = (),
        live_task_ids: Iterable[str] | None = None,
    ) -> None:
        """Re-read every fact from its source.

        Args:
            shared_state: SharedState to read the session's own facts from.
            rounds: Rounds excluding now, from ``RoundStore.excluding``. The
                first is reported.
            live_task_ids: Tasks observed running; ``None`` leaves the set
                unknown.
        """
        self.read = True
        excluding = rounds[0] if rounds else None
        self.excluding_round_id = "" if excluding is None else excluding.round_id
        self.excluding_round_holder = "" if excluding is None else excluding.holder_task_id
        self.serving_tp = serving_tp_for_policy(shared_state)
        self.gpu_specialist_capacity = gpu_specialist_ceiling(shared_state)
        self.gpu_specialist_pool = effective_gpu_specialist_pool_size(shared_state)
        self.whole_machine_pool = whole_machine_pool_size()
        self.live_task_ids = None if live_task_ids is None else frozenset(str(t) for t in live_task_ids)
