# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the hosted backend: one ``dev`` sandbox of the session's card count.

On the hosted backend the cards are not the application's to hand out: the sandbox service gives a sandbox the cards of
its template (one template per card count: the count is the template's key in the host's template list, never read
from its opaque name) and counts the sandboxes, cards and GPU hours of every role against one grant. The lease creates the ``dev`` sandbox with the ``dev`` key and hands the caller the
sandbox itself (:class:`HostedBranchLease`): commands and files go through it, so the key stays with the toolbox and a
specialist never holds it. The application does the accounting per role: a lease's cards count as leased, and the rest
of the grant's GPUs are what tests may run on, exactly as on the local backend.

A lease is the session's own card count, never another size (a caller picks a template, not a size). Its sandbox lives
:data:`LIFETIME_MARGIN_SEC` longer than its holder says it needs the cards, or, for a holder that runs until it is
stopped, that margin less than the grant lets one sandbox live (:meth:`HostedBranchLeases.lifetime_sec`). A create refused
for the grant's quota (429 ``grant_quota_exceeded``) is back-pressure: it is repeated with a growing wait until the
quota clears or :data:`~hyperloom_multibranch.toolbox.branch_leases.LEASE_WAIT_SECONDS` has passed. The cards are
promised from the moment the request is checked, while it waits, so no test is started that they would starve.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping
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
from hyperloom_multibranch.contracts.blackboard.records import LeaseStart
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, MeasurementCapacity, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import LEASE_WAIT_SECONDS, lease_wait, refuse_unfit_cards
from hyperloom_multibranch.toolbox.lease_ledger import LeaseLedger

__all__ = ["LIFETIME_MARGIN_SEC", "DevSession", "GrantLimits", "HostedBranchLease", "HostedBranchLeases", "LeaseTuning"]

log = logging.getLogger(__name__)

DEV_ROLE: Role = "dev"
#: The value of ``hyperloom.purpose`` on a lease's sandbox, which with the run's id (``hyperloom.run-id``) is how the
#: sandboxes of this run's leases are found again.
LEASE_PURPOSE = "dev-lease"
#: Between a holder's own end, or the grant's runtime, and the end of its sandbox: the holder's stop, its processes'
#: grace and the release that ends the sandbox come first.
LIFETIME_MARGIN_SEC = 300
_CLOSING = "the toolbox is closing"
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
    """How long and how fast a refused create is repeated.

    ``workdir`` is the directory of the sandbox a branch works in. A ``dev`` sandbox's envd serves files from ``/app``,
    so ``/app/work`` is where its file API and its processes see the same files.
    """

    wait_sec: float = LEASE_WAIT_SECONDS
    first_backoff_sec: float = 0.5
    max_backoff_sec: float = 10.0
    workdir: str = "/app/work"


@dataclass(frozen=True)
class HostedBranchLease(BranchLease):
    """A :class:`BranchLease` whose cards are a ``dev`` sandbox's.

    ``gpu_ids`` are the cards' numbers inside the sandbox, ``root`` is the directory the branch works in there (a
    path of the sandbox, not of this machine) and ``env`` is empty: the sandbox has the image's own environment.
    ``sandbox`` runs commands and reads and writes its files; end the lease with ``release_branch_sandbox``, which
    ends the sandbox and gives its cards back. ``lifetime_sec`` is how long the sandbox lives from its create.
    """

    template: str
    lifetime_sec: int
    sandbox: Sandbox = field(repr=False, compare=False)


class HostedBranchLeases:
    """The leases of one run's ``dev`` sandboxes, at most one per branch.

    Args:
        provider: The sandbox provider; it must hold the ``dev`` role's key.
        grant: The grant's totals.
        session: The run, the card count and the templates.
        tuning: Waits and the working directory, see :class:`LeaseTuning`.

    All state is guarded by one lock, so a lease and its release may race from any task or thread: a lease's sandbox
    is ended exactly once and its cards go back once. ``on_release`` is called, on the thread that gave them back,
    after cards went back (a release, a lease that failed, a close), so the toolbox can start the tests they now let run.
    ``ledger`` records each lease's sandbox in the blackboard when it is granted and when it ends.
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
        self._guard = threading.Lock()
        self._promised: dict[str, int] = {}
        self._granted: dict[str, HostedBranchLease] = {}
        self._closed = False
        self.on_release: Callable[[], None] = _nothing
        self.ledger = LeaseLedger()

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

    async def lease(
        self,
        branch_id: str,
        cards: int,
        *,
        charge: LeaseCharge,
        wait_sec: float | None = None,
        lifetime_sec: float | None = None,
    ) -> HostedBranchLease:
        """Create ``branch_id``'s ``dev`` sandbox, waiting while the grant has no room for it.

        ``branch_id`` is the holder: a branch, or ``tool:<job id>``. ``charge`` goes into the blackboard's record of the
        grant, written before the lease is returned and started when the create was, since the cards are the lease's
        while its sandbox comes up; a grant the blackboard did not record ends the sandbox and raises as
        :meth:`~hyperloom_multibranch.toolbox.lease_ledger.LeaseLedger.started` does.

        ``wait_sec`` bounds that wait (:func:`~hyperloom_multibranch.toolbox.branch_leases.lease_wait`): ``None`` is
        :attr:`LeaseTuning.wait_sec`, ``0`` does not wait out a refusal of the quota. It does not bound the time a
        sandbox the grant has room for takes to come up, which is :attr:`LeaseTuning.wait_sec` at least.
        ``lifetime_sec`` is how long the holder needs the cards, ``None`` until it is stopped (:meth:`lifetime_sec`): a
        holder bounded by the session's deadline says how long is left of it, so its sandbox never outlives the run.

        Raises:
            RequestRejected: ``invalid_request`` for fewer than one card or a number other than the session's, or a
                ``wait_sec`` that is negative or not a number, ``lease_exists`` when the branch holds a lease,
                ``lease_leaves_no_measurement`` when fewer than one test's cards or no sandbox for a test would be left.
            SandboxUnavailable: When the quota did not clear within the wait, there is no capacity or the toolbox
                closed.
            SandboxDenied: When the service refuses the template, the key or the budget.
        """

        wait = lease_wait(wait_sec, self._tuning.wait_sec)
        if cards >= 1 and cards != self._session.tp:
            raise RequestRejected(f"a dev sandbox has the session's {self._session.tp} cards, not {cards}")
        self._promise(branch_id, cards)
        made: Sandbox | None = None
        try:
            lifetime, requested_at = self.lifetime_sec(lifetime_sec), self.ledger.clock()
            made = await self._create(branch_id, wait, lifetime)
            lease = self._grant_lease(branch_id, made, lifetime)
            await self._recorded(lease, charge, requested_at)
            return lease
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
            self.ledger.returned(lease_id)

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
        self.end_leftovers()
        with self._guard:
            self._promised.clear()
        self.on_release()

    def end_leftovers(self) -> None:
        """End every sandbox of this run's leases that no lease of this object holds; blocks, so call it from a thread.

        At a start these are the sandboxes an earlier process of the run left behind when it died, whose cards the
        grant would otherwise count until their own timeout; at a close, creates still in flight when they were
        cancelled. No other run's sandbox is touched.
        """

        try:
            found = self._provider.find({PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: self._session.run_id}, role=DEV_ROLE)
        except SandboxError:
            log.exception("the dev sandboxes of leases could not be listed")
            return
        with self._guard:
            held = {lease.sandbox.sandbox_id for lease in self._granted.values()}
        for info in found:
            if info.sandbox_id in held:
                continue
            try:
                _end(self._provider.connect(info.sandbox_id, role=DEV_ROLE))
            except SandboxGone:
                pass
            except SandboxError:
                log.exception("sandbox %s could not be reached to end it", info.sandbox_id)

    def lifetime_sec(self, needed_sec: float | None) -> int:
        """How long a lease's sandbox lives: what its holder needs, ``needed_sec``, or the grant's runtime for a holder
        that runs until it is stopped (``None``), and a margin; never past the grant's runtime less the margin."""

        longest = max(1, self._grant.max_runtime_sec - LIFETIME_MARGIN_SEC)
        return longest if needed_sec is None else min(longest, math.ceil(needed_sec) + LIFETIME_MARGIN_SEC)

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

    async def _recorded(self, lease: HostedBranchLease, charge: LeaseCharge, requested_at: float) -> None:
        """Record ``lease``'s grant; a lease the blackboard did not record is withdrawn, its sandbox left to the caller."""

        start = LeaseStart(
            lease_id=lease.lease_id,
            holder=lease.branch_id,
            branch_id=charge.branch,
            kind=charge.kind,
            cards=len(lease.gpu_ids),
            started_at=requested_at,
            tool=charge.tool,
            sandbox_id=lease.sandbox.sandbox_id,
        )
        try:
            await self.ledger.started(start)
        except BaseException:
            with self._guard:
                self._granted.pop(lease.lease_id, None)
            raise

    def _grant_lease(self, branch_id: str, sandbox: Sandbox, lifetime: int) -> HostedBranchLease:
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
                lifetime,
                sandbox,
            )
            self._granted[lease.lease_id] = lease
        return lease

    async def _create(self, branch_id: str, wait: float, lifetime: int) -> Sandbox:
        """Create the sandbox with one idempotency key for every attempt, repeating a create the quota refused.

        The quota's refusals are repeated for ``wait`` seconds, not at all when it is ``0``. A sandbox the grant has
        room for may take as long to come up as the tuning allows, however short the wait for room is.

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
        started = time.monotonic()
        until, comes_up_by = started + wait, started + max(wait, tuning.wait_sec)
        delay = tuning.first_backoff_sec
        while True:
            options = CreateOptions(wait_sec=max(comes_up_by - time.monotonic(), 1.0))
            attempt = asyncio.create_task(
                asyncio.to_thread(
                    self._provider.create,
                    self._session.template,
                    role=DEV_ROLE,
                    timeout_sec=lifetime,
                    metadata=metadata,
                    options=options,
                )
            )
            try:
                return await asyncio.shield(attempt)
            except asyncio.CancelledError:
                attempt.add_done_callback(_end_when_made)
                raise
            except SandboxQuotaExceeded as refused:
                if self._closed or until - time.monotonic() <= delay:
                    # The service's own words name the limit (e.g. "dev 0/0 gpus"); a wait cannot clear every one.
                    reason = _CLOSING if self._closed else f"no room in the grant within {wait:g} s ({refused})"
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
