# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the local backend: cards out of the measurement pool and a directory of their own.

An autonomous branch runs its own agent, which serves its own models on cards of its own. The lease takes those cards
from the provider's :class:`~hyperloom.sandbox.gpu_pool.GpuPool`, the pool every measurement sandbox takes its cards
from, so a test never runs on a leased card and a leased card is never measured on. The cards are promised from the
moment the request is checked, while it waits for running tests to give some back, so the toolbox starts no test the
cards would not run. The lease owns only the cards and the directory: there is no PID namespace on this backend, and
confining the agent's processes is its sandbox wrapper's job.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from hyperloom.common.env_safety import GPU_MASK_ENV_NAMES, is_secret_shaped_env_name, scrub_benchmark_process_env
from hyperloom.measure.sandboxes import DEFAULT_RUNNER_START_TIMEOUT_SEC
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.toolbox import BranchLease, MeasurementCapacity, RequestRejected

__all__ = ["BranchLeases", "LEASE_WAIT_SECONDS"]

#: As long as a measurement sandbox waits for its cards.
LEASE_WAIT_SECONDS = float(DEFAULT_RUNNER_START_TIMEOUT_SEC)
_POLL_SECONDS = 0.2
_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"


class BranchLeases:
    """The leases on the cards of one local provider, at most one per branch.

    ``area`` is the directory the leases' working directories are made in, ``cards_per_test`` the cards one test of
    the session needs, ``image_env`` the environment the session's tests start from. All state is guarded by one lock,
    so a lease and its release may race from any task or thread: a lease's cards go back exactly once.
    """

    def __init__(
        self, provider: LocalSandboxProvider, area: Path, cards_per_test: int, image_env: Mapping[str, str]
    ) -> None:
        self._provider = provider
        self._area = area
        self.cards_per_test = cards_per_test
        self._env = _agent_env(image_env)
        self._guard = threading.Lock()
        self._promised: dict[str, int] = {}
        self._granted: dict[str, BranchLease] = {}
        self._closed = False

    def capacity(self) -> MeasurementCapacity:
        """The provider's cards, less the ones lost for good, with the promised ones counted as leased."""

        with self._guard:
            return self._capacity()

    def tests_at_once(self) -> int:
        """How many tests the cards left for measurement run at once, at least one so that a toolbox which lost its
        cards fails its tests instead of parking them."""

        return max(1, self.capacity().measurement_cards // self.cards_per_test)

    async def lease(self, branch_id: str, cards: int) -> BranchLease:
        """Hold ``cards`` cards and a working directory for ``branch_id``, waiting for running tests to give cards back.

        Raises:
            RequestRejected: ``lease_exists`` when the branch holds a lease, ``lease_leaves_no_measurement`` when
                fewer than one test's cards would be left, ``invalid_request`` for fewer than one card.
            SandboxUnavailable: When the cards were not free within :data:`LEASE_WAIT_SECONDS` or the toolbox closed.
        """

        self._promise(branch_id, cards)
        gpu_ids: tuple[int, ...] = ()
        try:
            gpu_ids = await self._take(cards)
            return self._grant(branch_id, gpu_ids)
        except BaseException:
            self._forget(branch_id, gpu_ids)
            raise

    def release(self, lease_id: str) -> None:
        """Return a lease's cards to the pool; an unknown or released lease changes nothing."""

        with self._guard:
            lease = self._granted.pop(lease_id, None)
            if lease is not None:
                del self._promised[lease.branch_id]
                self._provider.pool.release(lease.gpu_ids)

    def close(self) -> None:
        """Release every lease and refuse the waiting ones their cards."""

        with self._guard:
            self._closed = True
            granted = tuple(self._granted)
        for lease_id in granted:
            self.release(lease_id)

    def _capacity(self) -> MeasurementCapacity:
        total = self._provider.pool.total
        usable = total - len(self._provider.quarantined)
        return MeasurementCapacity(total, usable, leased_cards=min(sum(self._promised.values()), usable))

    def _promise(self, branch_id: str, cards: int) -> None:
        if cards < 1:
            raise RequestRejected(f"a lease needs at least one card, not {cards}")
        with self._guard:
            if branch_id in self._promised:
                raise RequestRejected(f"branch {branch_id!r} already holds a sandbox lease", "lease_exists")
            measuring = self._capacity().measurement_cards
            if measuring - cards < self.cards_per_test:
                raise RequestRejected(
                    f"leasing {cards} of the {measuring} cards tests run on would leave fewer than the "
                    f"{self.cards_per_test} one test needs",
                    "lease_leaves_no_measurement",
                )
            self._promised[branch_id] = cards

    async def _take(self, cards: int) -> tuple[int, ...]:
        pool = self._provider.pool
        until = time.monotonic() + LEASE_WAIT_SECONDS
        while not self._closed and time.monotonic() < until:
            try:
                return pool.acquire(cards, 0.0)
            except SandboxUnavailable:
                await asyncio.sleep(_POLL_SECONDS)
        reason = "the toolbox is closing" if self._closed else f"not freed within {LEASE_WAIT_SECONDS:g} s"
        raise SandboxUnavailable(f"no {cards} cards for the branch sandbox lease: {reason}")

    def _grant(self, branch_id: str, gpu_ids: tuple[int, ...]) -> BranchLease:
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        root = self._area / lease_id
        root.mkdir(parents=True)
        env = {**self._env, _VISIBLE_DEVICES: ",".join(map(str, gpu_ids))}
        with self._guard:
            if self._closed:
                raise SandboxUnavailable("the toolbox is closing")
            lease = BranchLease(
                lease_id, branch_id, gpu_ids, root, MappingProxyType(env), self._capacity().measurement_cards
            )
            self._granted[lease_id] = lease
        return lease

    def _forget(self, branch_id: str, gpu_ids: tuple[int, ...]) -> None:
        with self._guard:
            del self._promised[branch_id]
            if gpu_ids:
                self._provider.pool.release(gpu_ids)


def _agent_env(image_env: Mapping[str, str]) -> dict[str, str]:
    """``image_env`` without credentials, hooks that run code at start-up, or a mask of the visible cards."""

    kept = {
        name: value
        for name, value in image_env.items()
        if name not in GPU_MASK_ENV_NAMES and not is_secret_shaped_env_name(name)
    }
    return scrub_benchmark_process_env(kept)
