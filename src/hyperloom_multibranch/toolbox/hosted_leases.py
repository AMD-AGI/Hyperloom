# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the hosted backend: one ``dev`` sandbox of the session's card count.

On the hosted backend the cards are not the application's to hand out: the sandbox service gives a sandbox the cards of
its template (``<template>-g<cards>``, one template per card count) and counts the sandboxes, cards and GPU hours of
every role against one grant. The lease creates the ``dev`` sandbox with the ``dev`` key and hands the caller the
sandbox itself (:class:`HostedBranchLease`): commands and files go through it, so the key stays with the toolbox and a
specialist never holds it. The application does the accounting per role: a lease's cards count as leased, and the rest
of the grant's GPUs are what tests may run on, exactly as on the local backend.

A lease is the session's own card count, never another size (a caller picks a template, not a size). A create refused
for the grant's quota (429 ``grant_quota_exceeded``) is back-pressure: it is repeated with a growing wait until the
quota clears or :data:`~hyperloom_multibranch.toolbox.branch_leases.LEASE_WAIT_SECONDS` has passed. The cards are
promised from the moment the request is checked, while it waits, so no test is started that they would starve.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from hyperloom.measure.sandboxes import BRANCH_ID_KEY, PURPOSE_KEY, RUN_ID_KEY
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    CreateOptions,
    Role,
    Sandbox,
    SandboxError,
    SandboxGone,
    SandboxProvider,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)
from hyperloom_multibranch.contracts.toolbox import BranchLease, MeasurementCapacity, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import LEASE_WAIT_SECONDS, refuse_unfit_cards

__all__ = ["DevSession", "GrantLimits", "HostedBranchLease", "HostedBranchLeases", "LeaseTuning", "templates_by_cards"]

log = logging.getLogger(__name__)

DEV_ROLE: Role = "dev"
#: The value of ``hyperloom.purpose`` on a lease's sandbox, which with the run's id (``hyperloom.run-id``) is how the
#: sandboxes of this run's leases are found again.
LEASE_PURPOSE = "dev-lease"
DEFAULT_LEASE_LIFETIME_SEC = 3600
_CLOSING = "the toolbox is closing"
_CARDS_SUFFIX = re.compile(r"-g(\d+)$")
_POLL_SEC = 0.2


@dataclass(frozen=True)
class GrantLimits:
    """The totals of the grant the session runs under, one set for all roles (``HYPERLOOM_GRANT_LIMITS``).

    ``max_sandboxes`` and ``max_gpus`` bound what runs at once, ``gpu_hours`` the GPU time all sandboxes may use and
    ``max_runtime_sec`` the lifetime of one sandbox. The platform enforces all four; the lease reads the first two to
    keep a test's room and the last to bound a sandbox's lifetime.
    """

    max_sandboxes: int
    max_gpus: int
    gpu_hours: float
    max_runtime_sec: int

    def __post_init__(self) -> None:
        if min(self.max_sandboxes, self.max_gpus, self.max_runtime_sec) < 1 or self.gpu_hours <= 0:
            raise ValueError("a grant has at least one sandbox, one GPU, one second and some GPU hours")


def templates_by_cards(names: Iterable[str]) -> dict[int, str]:
    """Card count to template name, for the names a grant issued: ``<template>-g<cards>``.

    Raises:
        ValueError: A name that does not end in ``-g<cards>``, or two names for one card count.
    """
    found: dict[int, str] = {}
    for name in names:
        suffix = _CARDS_SUFFIX.search(name)
        if suffix is None:
            raise ValueError(f"template {name!r} does not name its cards (<template>-g<cards>)")
        cards = int(suffix.group(1))
        if found.setdefault(cards, name) != name:
            raise ValueError(f"templates {found[cards]!r} and {name!r} both have {cards} cards")
    return found


@dataclass(frozen=True)
class DevSession:
    """What a lease's sandboxes are made from: the run, the session's card count and the ``dev`` templates by cards.

    Attributes:
        run_id: The run's id, the first part of every idempotency key.
        tp: The session's tensor-parallel size, which is also the cards of one test and of every lease.
        templates: Card count to ``dev`` template name; it must have the session's ``tp``.
    """

    run_id: str
    tp: int
    templates: Mapping[int, str]

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("the lease needs the run id for its idempotency keys")
        if self.tp < 1 or self.tp not in self.templates:
            raise ValueError(
                f"no dev template has the session's {self.tp} cards; templates serve {sorted(self.templates)}"
            )

    @property
    def template(self) -> str:
        return self.templates[self.tp]


@dataclass(frozen=True)
class LeaseTuning:
    """How long a lease's sandbox lives and how long and how fast a refused create is repeated.

    ``lifetime_sec`` is cut to the grant's ``max_runtime_sec``; ``workdir`` is the directory of the sandbox a branch
    works in.
    """

    lifetime_sec: int = DEFAULT_LEASE_LIFETIME_SEC
    wait_sec: float = LEASE_WAIT_SECONDS
    first_backoff_sec: float = 0.5
    max_backoff_sec: float = 10.0
    workdir: str = "/work"


@dataclass(frozen=True)
class HostedBranchLease(BranchLease):
    """A :class:`BranchLease` whose cards are a ``dev`` sandbox's.

    ``gpu_ids`` are the cards' numbers inside the sandbox, ``root`` is the directory the branch works in there (a
    path of the sandbox, not of this machine) and ``env`` is empty: the sandbox has the image's own environment.
    ``sandbox`` runs commands and reads and writes its files; end the lease with ``release_branch_sandbox``, which
    ends the sandbox and gives its cards back.
    """

    template: str
    sandbox: Sandbox = field(repr=False, compare=False)


class HostedBranchLeases:
    """The leases of one run's ``dev`` sandboxes, at most one per branch.

    Args:
        provider: The sandbox provider; it must hold the ``dev`` role's key.
        grant: The grant's totals.
        session: The run, the card count and the templates.
        tuning: Lifetime and waits, see :class:`LeaseTuning`.

    All state is guarded by one lock, so a lease and its release may race from any task or thread: a lease's sandbox
    is ended exactly once and its cards go back once. ``on_release`` is called, on the thread that gave them back,
    after cards went back (a release, a lease that failed, a close), so the toolbox can start the tests they now let run.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        grant: GrantLimits,
        session: DevSession,
        tuning: LeaseTuning = LeaseTuning(),
    ) -> None:
        self._provider = provider
        self._grant = grant
        self._session = session
        self._tuning = tuning
        self._lifetime = min(tuning.lifetime_sec, grant.max_runtime_sec)
        self._guard = threading.Lock()
        self._promised: dict[str, int] = {}
        self._granted: dict[str, HostedBranchLease] = {}
        self._closed = False
        self.on_release: Callable[[], None] = _nothing

    def capacity(self) -> MeasurementCapacity:
        """The grant's GPUs, with the cards promised to leases counted as leased."""

        with self._guard:
            return self._capacity()

    def tests_at_once(self) -> int:
        """How many tests the cards and sandboxes left for measurement run at once, at least one so that a toolbox
        which lost them fails its tests instead of parking them."""

        with self._guard:
            by_cards = self._capacity().measurement_cards // self._session.tp
            by_sandboxes = self._grant.max_sandboxes - len(self._promised)
        return max(1, min(by_cards, by_sandboxes))

    async def lease(self, branch_id: str, cards: int) -> HostedBranchLease:
        """Create ``branch_id``'s ``dev`` sandbox, waiting while the grant has no room for it.

        Raises:
            RequestRejected: ``invalid_request`` for fewer than one card or a number other than the session's,
                ``lease_exists`` when the branch holds a lease, ``lease_leaves_no_measurement`` when fewer than one
                test's cards or no sandbox for a test would be left.
            SandboxUnavailable: When the quota did not clear within :attr:`LeaseTuning.wait_sec`, there is no capacity
                or the toolbox closed.
            SandboxDenied: When the service refuses the template, the key or the budget.
        """

        if cards >= 1 and cards != self._session.tp:
            raise RequestRejected(f"a dev sandbox has the session's {self._session.tp} cards, not {cards}")
        self._promise(branch_id, cards)
        made: Sandbox | None = None
        try:
            made = await self._create(branch_id)
            return self._grant_lease(branch_id, made)
        except BaseException:
            self._forget(branch_id)
            if made is not None:
                await asyncio.shield(asyncio.to_thread(_end, made))
            raise

    async def release(self, lease_id: str) -> None:
        """End a lease's sandbox and give its cards back; an unknown or released lease changes nothing."""

        with self._guard:
            lease = self._granted.pop(lease_id, None)
        if lease is None:
            return
        try:
            await asyncio.shield(asyncio.to_thread(_end, lease.sandbox))
        finally:
            self._forget(lease.branch_id)

    def close(self) -> None:
        """End every lease's sandbox and refuse the waiting ones; blocks, so call it from a thread.

        A sandbox of this run's leases that no lease holds (a create still in flight when it was cancelled, a lease of a
        process that died) is ended too; no other run's sandbox is.
        """

        with self._guard:
            self._closed = True
            granted = list(self._granted.values())
            self._granted.clear()
        for lease in granted:
            _end(lease.sandbox)
        try:
            leftovers = self._provider.find(
                {PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: self._session.run_id}, role=DEV_ROLE
            )
        except SandboxError:
            log.exception("the dev sandboxes of leases could not be listed")
            leftovers = []
        for info in leftovers:
            try:
                _end(self._provider.connect(info.sandbox_id, role=DEV_ROLE))
            except SandboxGone:
                pass
            except SandboxError:
                log.exception("sandbox %s could not be reached to end it", info.sandbox_id)
        with self._guard:
            self._promised.clear()
        self.on_release()

    def _capacity(self) -> MeasurementCapacity:
        total = self._grant.max_gpus
        return MeasurementCapacity(total, total, leased_cards=min(sum(self._promised.values()), total))

    def _promise(self, branch_id: str, cards: int) -> None:
        with self._guard:
            if self._closed:
                raise SandboxUnavailable(_CLOSING)
            if branch_id in self._promised:
                raise RequestRejected(f"branch {branch_id!r} already holds a sandbox lease", "lease_exists")
            refuse_unfit_cards(cards, self._capacity().measurement_cards, self._session.tp)
            if self._grant.max_sandboxes - len(self._promised) - 1 < 1:
                raise RequestRejected(
                    f"the grant's {self._grant.max_sandboxes} sandboxes would not leave one for a test",
                    "lease_leaves_no_measurement",
                )
            self._promised[branch_id] = cards

    def _forget(self, branch_id: str) -> None:
        with self._guard:
            self._promised.pop(branch_id, None)
        self.on_release()

    def _grant_lease(self, branch_id: str, sandbox: Sandbox) -> HostedBranchLease:
        with self._guard:
            if self._closed:
                raise SandboxUnavailable(_CLOSING)
            lease = HostedBranchLease(
                f"lease-{uuid.uuid4().hex[:12]}",
                branch_id,
                tuple(range(self._session.tp)),
                Path(self._tuning.workdir),
                MappingProxyType({}),
                self._capacity().measurement_cards,
                self._session.template,
                sandbox,
            )
            self._granted[lease.lease_id] = lease
        return lease

    async def _create(self, branch_id: str) -> Sandbox:
        """Create the sandbox with one idempotency key for every attempt, repeating a create the quota refused.

        A create the caller abandons (cancelled while the provider call is in flight) still ends: the sandbox it makes
        is ended when the call returns.
        """

        tuning = self._tuning
        metadata = {
            IDEMPOTENCY_KEY: f"{self._session.run_id}:{LEASE_PURPOSE}:{branch_id}:{uuid.uuid4().hex[:12]}",
            PURPOSE_KEY: LEASE_PURPOSE,
            RUN_ID_KEY: self._session.run_id,
            BRANCH_ID_KEY: branch_id,
        }
        until = time.monotonic() + tuning.wait_sec
        delay = tuning.first_backoff_sec
        while True:
            options = CreateOptions(wait_sec=max(until - time.monotonic(), 1.0))
            attempt = asyncio.create_task(
                asyncio.to_thread(
                    self._provider.create,
                    self._session.template,
                    role=DEV_ROLE,
                    timeout_sec=self._lifetime,
                    metadata=metadata,
                    options=options,
                )
            )
            try:
                return await asyncio.shield(attempt)
            except asyncio.CancelledError:
                attempt.add_done_callback(_end_when_made)
                raise
            except SandboxQuotaExceeded:
                if self._closed or until - time.monotonic() <= delay:
                    reason = _CLOSING if self._closed else f"no room in the grant within {tuning.wait_sec:g} s"
                    raise SandboxUnavailable(f"no dev sandbox for branch {branch_id!r}: {reason}") from None
                await self._pause(delay)
                delay = min(delay * 2, tuning.max_backoff_sec)

    async def _pause(self, seconds: float) -> None:
        """Sleep, but wake when the toolbox closes."""

        until = time.monotonic() + seconds
        while not self._closed and time.monotonic() < until:
            await asyncio.sleep(min(_POLL_SEC, max(until - time.monotonic(), 0.0)))


def _nothing() -> None:
    """The release hook of a toolbox that has nothing to start when cards come back."""


def _end(sandbox: Sandbox) -> None:
    """Kill ``sandbox``; one the service cannot end is left to its own timeout."""

    try:
        sandbox.kill()
    except SandboxGone:
        pass
    except SandboxError:
        log.exception("sandbox %s could not be ended; its own timeout will end it", sandbox.sandbox_id)


def _end_when_made(attempt: asyncio.Future[Sandbox]) -> None:
    """Done-callback of an abandoned create: end the sandbox it made, if it made one."""

    if attempt.cancelled() or attempt.exception() is not None:
        return
    threading.Thread(target=_end, args=(attempt.result(),), name="hl-lease-cleanup", daemon=True).start()
