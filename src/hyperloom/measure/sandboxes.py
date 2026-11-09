# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool of authoritative sandboxes the hosted measurement backend measures in.

One test gets one fresh sandbox. :class:`SandboxPool` creates it through the ``sandbox`` socket with the ``measurement``
key, never lets more tests run at once than ``max_parallel_tests``, hands back what the platform said about the sandbox
(its signing public key, the node it landed on, whether the node could not be avoided), and closes everything on
shutdown. Which runner template serves a card count, what the sandbox's lifetime is, what metadata it carries and what
environment it is created with are decided here, so the backend never builds a ``create`` call itself.

Every sandbox of a pool carries the run's id (``hyperloom.run-id``), and the pool finds and closes only sandboxes that
carry it: a pool never ends a measurement sandbox of another run that the same key can see.

No wait of :meth:`SandboxPool.acquire` is silent or open-ended: the wait for a slot, for room in the grant's quota and for
the sandbox to run each log when they begin and every :data:`WAIT_LOG_EVERY_SEC` while they last, and together they end
when the test's own timeout has passed (:class:`AcquireTimedOut`, naming what was waited for). A rerun whose sandbox can
never start thus ends its test with that reason instead of holding it, unlogged, until something else gives up.

Make-before-break (:class:`Handover`): on a cluster that queues sandboxes, a sandbox that ends gives its node back to
the queue, and the next test's sandbox asks again behind every other job of its priority. A test that is done with its
sandbox may therefore hand it over (:meth:`SandboxPool.hand_over`) instead of releasing it: while another test of the
session waits for a sandbox (:meth:`SandboxPool.note_upcoming`) and the grant has room for one more sandbox and its
GPUs beside the sandboxes the pool holds and those it is creating, the idle sandbox is kept until the next test's
sandbox runs, then closed. It is kept for one test at a time,
never takes a slot of ``max_parallel_tests``, and is closed anyway after :attr:`Handover.timeout_sec`, when the test
that would take its place is gone, when that test's create fails or is stopped, and when the pool closes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from hyperloom.sandbox.socket import (
    AVOID_NODES_KEY,
    IDEMPOTENCY_KEY,
    NODE_NAME_KEY,
    PRIORITY_KEY,
    SAME_NODE_KEY,
    SIGNING_PUBLIC_KEY,
    CreateOptions,
    Role,
    Sandbox,
    SandboxError,
    SandboxGone,
    SandboxProvider,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)

log = logging.getLogger(__name__)

__all__ = [
    "AcquireTimedOut",
    "DEFAULT_HANDOVER_TIMEOUT_SEC",
    "DEFAULT_MAX_PARALLEL_TESTS",
    "DEFAULT_RUNNER_START_TIMEOUT_SEC",
    "DEFAULT_TIMEOUT_SLACK_SEC",
    "HANDOVER_TIMER_NAME",
    "Handover",
    "MEASUREMENT_ROLE",
    "NoRunnerTemplate",
    "PoolClosed",
    "RUN_ID_KEY",
    "RunnerSandbox",
    "SandboxPool",
    "WAIT_LOG_EVERY_SEC",
    "sandbox_priority",
]

MEASUREMENT_ROLE: Role = "measurement"
DEFAULT_MAX_PARALLEL_TESTS = 8
DEFAULT_RUNNER_START_TIMEOUT_SEC = 3600
DEFAULT_TIMEOUT_SLACK_SEC = 900
#: How long a handed-over sandbox is kept for the next test's sandbox to run before it is closed anyway.
DEFAULT_HANDOVER_TIMEOUT_SEC = 600.0
#: The name of the thread that closes a handed-over sandbox when its timeout passes.
HANDOVER_TIMER_NAME = "sandbox-handover-timeout"
REQUEST_ID_KEY = "hyperloom.request-id"
PURPOSE_KEY = "hyperloom.purpose"
BRANCH_ID_KEY = "hyperloom.branch-id"
RUN_ID_KEY = "hyperloom.run-id"
_SLOT_POLL_SEC = 0.5
#: How long a create refused for the quota waits for one of this pool's sandboxes to end before it asks again: the
#: sandboxes that fill the quota may be another role's or another process's, and those end without telling the pool.
_QUOTA_RETRY_SEC = 10.0
#: How often a wait of :meth:`SandboxPool.acquire` says that it still waits (it also says so when it begins).
WAIT_LOG_EVERY_SEC = 60.0
#: How many more keys an attempt tries when the service says the one it holds is spent.
_MAX_SPENT_KEYS = 8
_PRIORITY_TIERS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
_DEFAULT_PRIORITY = 2


class NoRunnerTemplate(LookupError):
    """No runner template serves the card count a test needs."""


class PoolClosed(RuntimeError):
    """The pool was closed, or the caller's stop signal fired, while a test waited for a slot."""


class AcquireTimedOut(SandboxUnavailable):
    """The test's own timeout passed while it waited for a slot or for room in the grant's quota, before any sandbox
    was made for it; the message names what it waited for and what held it."""


def sandbox_priority(priority: int | None) -> int:
    """The ``agent-sandbox.io/priority`` tier of a test priority: 0 and 1 map to 0, 2 and 3 to 1, 4 to 2.

    A test without a priority is the lowest tier's neighbour, the default exploration priority.
    """
    tier = _PRIORITY_TIERS.get(_DEFAULT_PRIORITY if priority is None else priority)
    if tier is None:
        raise ValueError(f"a test priority is 0 to 4, got {priority!r}")
    return tier


@dataclass(frozen=True)
class RunnerSandbox:
    """An authoritative sandbox and what the platform reported about it.

    ``public_key_pem`` is ``None`` when the platform reported no signing key; the backend decides what that means.
    """

    sandbox_id: str
    public_key_pem: str | None
    node_name: str | None
    same_node: bool
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)
    handle: Sandbox | None = field(default=None, compare=False, repr=False)


def _nothing_reserved() -> tuple[int, int]:
    return 0, 0


@dataclass(frozen=True)
class Handover:
    """When a sandbox a test is done with is kept until the next test's sandbox runs (make-before-break).

    ``max_sandboxes`` and ``max_gpus`` are the grant's: the idle sandbox is kept only while the grant has room for it,
    every other sandbox of the pool (held or still being created) and the next one, with their GPUs. ``reserved`` says how many sandboxes and GPUs of
    the grant are held for other uses (the branches' ``dev`` sandboxes); it is asked from the threads that measure.
    ``timeout_sec`` is how long the idle sandbox waits for the next one to run before it is closed anyway.
    """

    max_sandboxes: int
    max_gpus: int
    timeout_sec: float = DEFAULT_HANDOVER_TIMEOUT_SEC
    reserved: Callable[[], tuple[int, int]] = field(default=_nothing_reserved, compare=False)


@dataclass(frozen=True)
class PoolLimits:
    """How many sandboxes a pool holds and how long it waits for them.

    ``max_parallel_tests``: how many sandboxes may be held at once; ``acquire`` waits for a free slot.
    ``timeout_slack_sec``: added to a test's own timeout to give the sandbox its lifetime.
    ``request_timeout_sec``: how long ``create`` may wait for the sandbox to run.
    ``handover``: when a sandbox a test is done with waits for the next one (:class:`Handover`); ``None`` never keeps one.
    """

    max_parallel_tests: int = DEFAULT_MAX_PARALLEL_TESTS
    timeout_slack_sec: int = DEFAULT_TIMEOUT_SLACK_SEC
    request_timeout_sec: float = DEFAULT_RUNNER_START_TIMEOUT_SEC
    handover: Handover | None = None


@dataclass(frozen=True)
class AttemptTag:
    """Which attempt of which request a sandbox is for, the nodes it must avoid and the test's own metadata."""

    request_id: str
    attempt: int
    avoid_nodes: tuple[str, ...] = ()
    meta: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _Hold:
    """A sandbox the pool holds: its GPUs (``None`` when not known, as for one re-attached) and the test it is for."""

    gpus: int | None
    request_id: str | None


@dataclass
class _Parked:
    """The sandbox a finished test handed over, until the next test's sandbox runs or it is closed anyway."""

    runner: RunnerSandbox
    since: float
    timer: threading.Timer | None = None
    claimed_by: str | None = None  # the test whose create it waits for, once one began


class SandboxPool:
    """Creates, re-attaches and closes the authoritative sandboxes of one run.

    Args:
        provider: The ``sandbox`` socket implementation.
        templates: Card count to runner template name.
        run_id: The run's id, the first part of every idempotency key and the value of the ``hyperloom.run-id`` metadata
            every sandbox of the pool carries.
        limits: The slot cap and the two waits, see :class:`PoolLimits`.
        create_envs: The environment every sandbox is created with (``CreateOptions.envs``); a sealed template refuses
            the names it does not accept, so the service's runner template must list them.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        templates: Mapping[int, str],
        *,
        run_id: str,
        limits: PoolLimits = PoolLimits(),
        create_envs: Mapping[str, str] | None = None,
    ) -> None:
        max_parallel_tests = limits.max_parallel_tests
        if max_parallel_tests < 1:
            raise ValueError("max_parallel_tests must be at least 1")
        if not run_id:
            raise ValueError("the pool needs the run id for its idempotency keys")
        self._provider = provider
        self._templates = {int(count): name for count, name in templates.items()}
        self._run_id = run_id
        self._create_envs = dict(create_envs or {})
        self._max = max_parallel_tests
        self._slack = limits.timeout_slack_sec
        self._request_timeout = limits.request_timeout_sec
        self._handover = limits.handover
        self._cond = threading.Condition()
        self._held: dict[str, _Hold] = {}
        self._parked: _Parked | None = None
        self._upcoming: frozenset[str] = frozenset()
        self._claims = 0
        # The creates and re-attaches that took a slot and whose sandbox is not held yet, with their GPUs (``None`` when
        # not known): the grant's room counts them, since their sandboxes will be.
        self._creating: dict[object, int | None] = {}
        self._closed = False
        self._ended = 0  # how many held sandboxes have ended: a create waiting for quota wakes on a change

    @property
    def active(self) -> int:
        """How many tests hold a slot right now (a slot is taken before ``create`` returns)."""
        with self._cond:
            return self._claims

    @property
    def parked(self) -> str | None:
        """The sandbox a finished test handed over and that waits for the next test's sandbox to run, if any."""
        with self._cond:
            return None if self._parked is None else self._parked.runner.sandbox_id

    def template_for(self, gpu_count: int) -> str:
        """The runner template for a card count; :class:`NoRunnerTemplate` when none serves it."""
        try:
            return self._templates[int(gpu_count)]
        except KeyError:
            known = sorted(self._templates)
            raise NoRunnerTemplate(f"no runner template for {gpu_count} GPU(s); templates serve {known}") from None

    def idempotency_key(self, request_id: str, attempt: int) -> str:
        return f"{self._run_id}:{request_id}:{attempt}"

    def acquire(
        self,
        gpu_count: int,
        tag: AttemptTag,
        *,
        timeout_sec: float,
        remaining_sec: float | None = None,
        stop: threading.Event | None = None,
    ) -> RunnerSandbox:
        """Create the authoritative sandbox of one test attempt, after a slot is free.

        ``tag`` names the attempt; ``timeout_sec`` is the test's own timeout; the sandbox lives that long plus the slack, but never past
        ``remaining_sec`` (what is left of the run). ``stop`` ends the wait for a slot with :class:`PoolClosed`. The
        waits for a slot, for room in the quota and for the sandbox to run share one deadline, ``timeout_sec`` (or
        ``remaining_sec`` when less) from the call, and each logs when it begins and while it lasts. A sandbox a
        finished test handed over (:meth:`hand_over`) is closed once this one runs, or once this create fails or stops;
        when the grant has no room for both it is closed before the create.

        Raises:
            NoRunnerTemplate: No template serves ``gpu_count`` (checked before any slot is taken).
            PoolClosed: The pool was closed or ``stop`` was set while waiting, or the pool closed while the sandbox
                was being created (it is closed again).
            AcquireTimedOut: The deadline passed while the test waited for a slot or for room in the quota.
            SandboxUnavailable, SandboxDenied: As the provider raised them; the slot is given back. A quota that is
                full (``SandboxQuotaExceeded``) is not raised: the create waits for a sandbox to end and is repeated,
                until the test is stopped (``PoolClosed``) or the deadline passes (``AcquireTimedOut``). A sandbox that
                did not run before the deadline is the provider's ``SandboxUnavailable``.
            SandboxGone: Every key the attempt tried was spent (the sandbox of each had ended).
            SandboxCreateFailed: The platform made the sandbox and it failed before it ran; the attempt is over and
                no other key is tried (the slot is given back).
        """
        template = self.template_for(gpu_count)
        lifetime = int(timeout_sec + self._slack)
        if remaining_sec is not None:
            lifetime = min(lifetime, max(1, int(remaining_sec)))
        metadata = self._metadata(tag.request_id, tag.attempt, tag.avoid_nodes, tag.meta or {})
        wait = _Wait(f"test {tag.request_id} attempt {tag.attempt}", _wait_budget(timeout_sec, remaining_sec))
        self._take_slot(stop, wait=wait)
        creating = object()
        parked: _Parked | None = None
        handed = "its create failed or was stopped"
        try:
            try:
                parked = self._claim_parked(int(gpu_count), wait.label, creating)
                sandbox = self._create(template, lifetime, metadata, stop, wait)
            except BaseException:
                self._give_slot(creating)
                raise
            with self._cond:
                closed = self._closed
            if closed:  # the pool closed while the create waited: nobody will release this one
                try:
                    self._close_handle(sandbox)
                finally:
                    self._give_slot(creating)
                raise PoolClosed("the sandbox pool closed while the sandbox was being created")
            log.info("%s: sandbox %s is running after %.0fs", wait.label, sandbox.sandbox_id, wait.elapsed())
            runner = self._adopt(sandbox, _Hold(int(gpu_count), tag.request_id), creating)
            handed = f"sandbox {runner.sandbox_id} of {wait.label} is running"
            return runner
        finally:
            if parked is not None:
                self._end_parked(parked, handed)

    def reattach(
        self, sandbox_id: str, *, stop: threading.Event | None = None, timeout_sec: float | None = None
    ) -> RunnerSandbox:
        """Connect to a sandbox an earlier process created; it takes a slot like any other test.

        ``timeout_sec`` bounds the wait for the slot (``None``: no bound); the wait logs as :meth:`acquire`'s does.

        Raises:
            SandboxGone: The sandbox no longer exists (the slot is given back).
            AcquireTimedOut: No slot came free within ``timeout_sec``.
        """
        budget = float("inf") if timeout_sec is None else max(0.0, float(timeout_sec))
        self._take_slot(stop, reattaching=sandbox_id, wait=_Wait(f"sandbox {sandbox_id} (re-attaching)", budget))
        creating = object()
        with self._cond:
            self._creating[creating] = None  # its GPUs are not known
        try:
            sandbox = self._provider.connect(sandbox_id, role=MEASUREMENT_ROLE)
        except BaseException:
            self._give_slot(creating)
            raise
        return self._adopt(sandbox, _Hold(None, None), creating)

    def release(self, runner: RunnerSandbox) -> None:
        """Close the sandbox and free its slot; calling it again changes nothing."""
        with self._cond:
            held = runner.sandbox_id in self._held
        if not held:
            return
        try:
            if runner.handle is not None:
                runner.handle.kill()
        except SandboxGone:
            pass  # already closed, which is what this asks for
        except SandboxError:
            # A local sandbox has ended all the same (its timer is cancelled) and keeps its cards out of the pool; a hosted
            # one is ended by its own timeout.
            log.exception("sandbox %s could not be closed cleanly", runner.sandbox_id)
        finally:
            self._forget(runner.sandbox_id)

    def hand_over(self, runner: RunnerSandbox) -> None:
        """Keep ``runner``, whose test is done with it, until the next test's sandbox runs; else close it now.

        Call it only once the test no longer uses the sandbox's GPUs: its result has landed and its server has exited.
        It is kept when the pool has a :class:`Handover`, no other sandbox is kept already, a test other than its own
        waits for a sandbox (:meth:`note_upcoming`) and the grant has room for one more sandbox and its GPUs; it gives
        its slot back at once either way. Why it is not kept is logged, and it is then released as :meth:`release` does.
        """
        with self._cond:
            refusal = self._park_refusal(runner)
            if refusal is None:
                parked = self._parked = _Parked(runner, time.monotonic())
                self._claims = max(0, self._claims - 1)
                timeout = self._handover.timeout_sec if self._handover is not None else 0.0
                parked.timer = threading.Timer(timeout, self._expire_parked, args=(parked,))
                parked.timer.name, parked.timer.daemon = HANDOVER_TIMER_NAME, True
                parked.timer.start()
                self._cond.notify_all()
        if refusal is not None:
            if runner.sandbox_id in self._held_ids():
                log.info("sandbox %s is released at once: %s", runner.sandbox_id, refusal)
            self.release(runner)
            return
        log.info(
            "sandbox %s (node %s) is kept until the next test's sandbox runs, at most %.0fs",
            runner.sandbox_id,
            runner.node_name or "unknown",
            timeout,
        )

    def note_upcoming(self, request_ids: Iterable[str]) -> None:
        """Say which tests of the session will ask for a sandbox: those waiting to start and those running.

        A test that holds a sandbox of the pool is not waiting for one. A handed-over sandbox that no test is waiting
        for any more, and that no create has begun to replace, is closed (on a thread of its own: this is called from
        the session's event loop).
        """
        with self._cond:
            self._upcoming = frozenset(request_ids)
            parked = self._parked
            if parked is None or parked.claimed_by is not None or self._waiting_tests(parked.runner):
                return
        threading.Thread(
            target=self._end_parked,
            args=(parked, "no test waits for a sandbox any more"),
            name="sandbox-handover-release",
            daemon=True,
        ).start()

    def discard(self, sandbox_id: str) -> None:
        """Close a ``measurement`` sandbox this pool does not hold, as one an earlier process left behind; a sandbox
        that is already gone is fine."""
        try:
            self._provider.connect(sandbox_id, role=MEASUREMENT_ROLE).kill()
        except SandboxGone:
            pass  # already gone, which is fine
        except SandboxError:
            log.exception("sandbox %s could not be closed", sandbox_id)

    def close(self) -> None:
        """Close every ``measurement`` sandbox still running and refuse new tests."""
        with self._cond:
            self._closed = True
            parked, self._parked = self._parked, None
            self._cond.notify_all()
        if parked is not None and parked.timer is not None:
            parked.timer.cancel()
        for info in self._provider.find(self._scope(), role=MEASUREMENT_ROLE):
            self.discard(info.sandbox_id)
        with self._cond:
            self._held.clear()
            self._claims = 0
            self._cond.notify_all()

    # -- internals ----------------------------------------------------------------------------------------------

    def _scope(self) -> dict[str, str]:
        """The metadata that makes a sandbox this run's: carried by every create, required by every find."""
        return {RUN_ID_KEY: self._run_id}

    def _metadata(
        self, request_id: str, attempt: int, avoid_nodes: tuple[str, ...], meta: Mapping[str, object]
    ) -> dict[str, str]:
        metadata = {
            IDEMPOTENCY_KEY: self.idempotency_key(request_id, attempt),
            PRIORITY_KEY: str(sandbox_priority(_priority_of(meta))),
            REQUEST_ID_KEY: request_id,
            **self._scope(),
        }
        if avoid_nodes:
            metadata[AVOID_NODES_KEY] = ",".join(sorted(set(avoid_nodes)))
        for key, field_name in ((PURPOSE_KEY, "purpose"), (BRANCH_ID_KEY, "branch_id")):
            value = meta.get(field_name)
            if value:
                metadata[key] = str(value)
        return metadata

    def _create(
        self, template: str, lifetime: int, metadata: dict[str, str], stop: threading.Event | None, wait: _Wait
    ) -> Sandbox:
        """Create the sandbox, replacing a spent idempotency key and waiting out a full quota.

        An attempt number comes back when its sandbox ended before the attempt was journalled (a restart, a sheet that
        could not be sent), and the service refuses a create with the key of a sandbox that ended (``SandboxGone``): the
        create is repeated with the attempt's next key, ``<key>:r1``, ``<key>:r2`` and so on, the same sequence every
        time, so a replay still finds a sandbox that is running. A sandbox the platform made and that failed before it
        ran (``SandboxCreateFailed``) ends the attempt: its key is not replayed and no next key is tried, because each
        would only make another sandbox that fails the same way. A full quota (``SandboxQuotaExceeded``) says nothing
        about a node and is no failure of the test: the create waits for a sandbox to end and is repeated, until ``wait``
        runs out (:class:`AcquireTimedOut`). The create's own wait for the sandbox to run is the lesser of the pool's
        ``request_timeout_sec`` and what is left of ``wait``.
        """
        base = metadata[IDEMPOTENCY_KEY]
        spent = 0
        refused: _Ticker | None = None
        said = ""
        while True:
            keyed = {**metadata, IDEMPOTENCY_KEY: f"{base}:r{spent}"} if spent else metadata
            with self._cond:
                ended = self._ended  # taken before the create, so a sandbox that ends during it is not missed
            create_wait = min(self._request_timeout, wait.left())
            if create_wait <= 0:
                if refused is not None:
                    raise AcquireTimedOut(
                        wait.gave_up("room in the grant's quota", f"the service still refuses the create: {said}")
                    )
                raise AcquireTimedOut(
                    wait.gave_up("its sandbox to be created", "the deadline passed before the create")
                )
            log.info(
                "%s: creating its sandbox (template %s, key %s); waits up to %.0fs for it to run",
                wait.label,
                template,
                keyed[IDEMPOTENCY_KEY],
                create_wait,
            )
            try:
                with wait.beating("its sandbox to run", lambda: f"template {template}"):
                    return self._provider.create(
                        template,
                        role=MEASUREMENT_ROLE,
                        timeout_sec=lifetime,
                        metadata=keyed,
                        options=CreateOptions(envs=self._create_envs or None, wait_sec=create_wait),
                    )
            except SandboxGone:
                spent += 1
                if spent > _MAX_SPENT_KEYS:
                    raise
            except SandboxQuotaExceeded as exc:
                said = _one_line(str(exc))
                if refused is None:
                    refused = wait.begin("room in the grant's quota", f"the service refused the create: {said}")
                else:
                    refused.tick(f"the service refused the create: {said}")
                self._await_room(stop, ended, wait)

    def _await_room(self, stop: threading.Event | None, ended: int, wait: _Wait) -> None:
        """Wait until more than ``ended`` sandboxes of this pool have ended, ``_QUOTA_RETRY_SEC`` passes or ``wait`` runs
        out (the caller asks again, and gives up when ``wait`` has).

        Raises:
            PoolClosed: The pool is closed or ``stop`` is set; this wins over a sandbox that ended.
        """
        deadline = time.monotonic() + min(_QUOTA_RETRY_SEC, max(0.0, wait.left()))
        with self._cond:
            while True:
                if self._closed or (stop is not None and stop.is_set()):
                    raise PoolClosed("the sandbox pool is closed or the run is stopping")
                left = deadline - time.monotonic()
                if self._ended != ended or left <= 0:
                    return
                self._cond.wait(min(left, _SLOT_POLL_SEC))

    def _take_slot(self, stop: threading.Event | None, *, wait: _Wait, reattaching: str | None = None) -> None:
        """Wait until fewer than ``max_parallel_tests`` sandboxes are running, then take one of the slots.

        Sandboxes this pool does not hold count too (a measurement a previous process of the run left behind keeps
        its cards busy until it is re-attached or ends); ``reattaching`` is the one sandbox that is about to be held.
        The wait logs when it begins and while it lasts, and ends with :class:`AcquireTimedOut` once ``wait`` runs out.
        """
        waiting: _Ticker | None = None
        while True:
            with self._cond:
                if self._closed or (stop is not None and stop.is_set()):
                    raise PoolClosed("the sandbox pool is closed or the run is stopping")
            others = self._untracked(reattaching)
            with self._cond:
                if self._claims + others < self._max:
                    self._claims += 1
                    if waiting is not None:
                        log.info("%s: took a sandbox slot after %.0fs", wait.label, wait.elapsed())
                    return
                held = f"{self._claims} held by this pool and {others} running that it does not hold, of {self._max}"
                if waiting is None:
                    waiting = wait.begin("a sandbox slot", held)
                else:
                    waiting.tick(held)
                if wait.left() <= 0:
                    raise AcquireTimedOut(wait.gave_up("a sandbox slot", held))
                self._cond.wait(min(_SLOT_POLL_SEC, max(wait.left(), 0.0)))

    def _untracked(self, reattaching: str | None) -> int:
        """How many running ``measurement`` sandboxes of this run this pool does not hold."""
        running = {info.sandbox_id for info in self._provider.find(self._scope(), role=MEASUREMENT_ROLE)}
        with self._cond:
            running.difference_update(self._held)
        running.discard(reattaching or "")
        return len(running)

    def _give_slot(self, creating: object) -> None:
        """Give back the slot of a create or re-attach whose sandbox the pool will not hold."""
        with self._cond:
            self._creating.pop(creating, None)
            self._claims = max(0, self._claims - 1)
            self._cond.notify_all()

    def _adopt(self, sandbox: Sandbox, hold: _Hold, creating: object) -> RunnerSandbox:
        try:
            info = sandbox.info()
        except BaseException:
            try:
                self._close_handle(sandbox)
            finally:
                self._give_slot(creating)
            raise
        metadata = dict(info.metadata)
        with self._cond:
            self._creating.pop(creating, None)
            self._held[sandbox.sandbox_id] = _Hold(hold.gpus, hold.request_id or metadata.get(REQUEST_ID_KEY))
        return RunnerSandbox(
            sandbox_id=sandbox.sandbox_id,
            public_key_pem=metadata.get(SIGNING_PUBLIC_KEY) or None,
            node_name=metadata.get(NODE_NAME_KEY) or None,
            same_node=metadata.get(SAME_NODE_KEY) == "true",
            metadata=metadata,
            handle=sandbox,
        )

    def _forget(self, sandbox_id: str) -> None:
        timer = None
        with self._cond:
            if sandbox_id in self._held:
                del self._held[sandbox_id]
                parked = self._parked
                if parked is not None and parked.runner.sandbox_id == sandbox_id:
                    self._parked, timer = None, parked.timer  # its slot was given back when it was handed over
                else:
                    self._claims = max(0, self._claims - 1)
                self._ended += 1
            self._cond.notify_all()
        if timer is not None:
            timer.cancel()

    def _held_ids(self) -> set[str]:
        with self._cond:
            return set(self._held)

    def _waiting_tests(self, runner: RunnerSandbox) -> frozenset[str]:
        """The tests that will ask for a sandbox and hold none of the pool's: neither ``runner``'s nor any other's.
        Called with the lock held."""
        holding = {hold.request_id for hold in self._held.values()}
        own = self._held.get(runner.sandbox_id)
        return frozenset(self._upcoming - holding - {own.request_id if own else runner.metadata.get(REQUEST_ID_KEY)})

    def _room_for(self, more_gpus: int | None) -> str | None:
        """Why the grant has no room for the sandboxes the pool holds or is creating, one more of ``more_gpus`` GPUs
        and what is reserved; ``None`` when it has. Called with the lock held."""
        handover = self._handover
        if handover is None:
            return "no grant to count the sandboxes in"
        reserved_sandboxes, reserved_gpus = handover.reserved()
        gpus = [hold.gpus for hold in self._held.values()] + list(self._creating.values())
        if more_gpus is None or any(count is None for count in gpus):
            return "a sandbox of the pool has GPUs it does not know"
        sandboxes = len(gpus) + reserved_sandboxes + 1
        in_gpus = sum(count for count in gpus if count is not None) + reserved_gpus + more_gpus
        if sandboxes > handover.max_sandboxes or in_gpus > handover.max_gpus:
            return (
                f"the grant ({handover.max_sandboxes} sandboxes, {handover.max_gpus} GPUs) has no room for {sandboxes} "
                f"sandboxes with {in_gpus} GPUs ({reserved_sandboxes} sandboxes, {reserved_gpus} GPUs reserved)"
            )
        return None

    def _park_refusal(self, runner: RunnerSandbox) -> str | None:
        """Why ``runner`` is not kept for the next test; ``None`` when it is. Called with the lock held."""
        hold = self._held.get(runner.sandbox_id)
        if hold is None:
            return "the pool does not hold it"
        if self._handover is None:
            return "no make-before-break without a grant to count the sandboxes in"
        if self._closed:
            return "the pool is closed"
        if self._parked is not None:
            return f"sandbox {self._parked.runner.sandbox_id} is already kept for the next test"
        if not self._waiting_tests(runner):
            return "no test waits for a sandbox"
        return self._room_for(hold.gpus)

    def _claim_parked(self, gpu_count: int, label: str, creating: object) -> _Parked | None:
        """Count ``label``'s create (``creating``) in the grant, and return the handed-over sandbox it now replaces,
        kept until that create ends; ``None`` when there is none to replace. A handed-over sandbox the grant has no room
        for beside every sandbox the pool holds or creates and the new one is closed before the create, also when
        another test's create replaces it already."""
        with self._cond:
            parked = self._parked
            refusal = None if parked is None else self._room_for(gpu_count)
            self._creating[creating] = gpu_count
            if parked is None:
                return None
            if refusal is None:
                if parked.claimed_by is not None:
                    return None
                parked.claimed_by = label
                kept = time.monotonic() - parked.since
                log.info(
                    "%s: creates its sandbox while sandbox %s is kept (for %.0fs so far)",
                    label,
                    parked.runner.sandbox_id,
                    kept,
                )
                return parked
        self._end_parked(parked, f"{label} creates its sandbox after it is closed: {refusal}")
        return None

    def _end_parked(self, parked: _Parked, why: str) -> None:
        """Close the handed-over sandbox, unless it is closed already; ``why`` is logged."""
        with self._cond:
            if parked.runner.sandbox_id not in self._held:
                return
        log.info(
            "sandbox %s, kept for %.0fs, is closed: %s",
            parked.runner.sandbox_id,
            time.monotonic() - parked.since,
            why,
        )
        self.release(parked.runner)

    def _expire_parked(self, parked: _Parked) -> None:
        with self._cond:
            if parked.runner.sandbox_id not in self._held:
                return
            waiting_for = parked.claimed_by
        log.warning(
            "sandbox %s was kept %.0fs and the next sandbox does not run yet: it is closed anyway; %s",
            parked.runner.sandbox_id,
            time.monotonic() - parked.since,
            f"{waiting_for} keeps waiting for its own" if waiting_for else "no create had begun to replace it",
        )
        self._end_parked(parked, "the make-before-break timeout passed")

    @staticmethod
    def _close_handle(sandbox: Sandbox) -> None:
        try:
            sandbox.kill()
        except SandboxError:
            log.exception("sandbox %s could not be closed", sandbox.sandbox_id)


def _wait_budget(timeout_sec: float, remaining_sec: float | None) -> float:
    """How long an attempt may wait for its sandbox: the test's own timeout, never past what is left of the run."""
    budget = max(0.0, float(timeout_sec))
    return budget if remaining_sec is None else min(budget, max(0.0, float(remaining_sec)))


def _one_line(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


class _Wait:
    """One attempt's deadline for its sandbox, shared by the waits for a slot, for quota and for the sandbox to run."""

    def __init__(self, label: str, budget_sec: float) -> None:
        self.label = label
        self.budget = budget_sec
        self._started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def left(self) -> float:
        return self.budget - self.elapsed()

    def begin(self, what: str, detail: str) -> _Ticker:
        """Log that the attempt now waits for ``what`` and return what logs, while it lasts, that it still does."""
        log.info("%s waits for %s: %s (gives up in %.0fs)", self.label, what, detail, max(self.left(), 0.0))
        return _Ticker(self, what)

    def gave_up(self, what: str, detail: str) -> str:
        """The reason of an attempt that waited for ``what`` until its deadline; logged as well."""
        reason = f"waited {self.elapsed():.0f}s for {what}, the test's timeout is {self.budget:.0f}s: {detail}"
        log.warning("%s gave up waiting: %s", self.label, reason)
        return reason

    @contextmanager
    def beating(self, what: str, detail: Callable[[], str]) -> Iterator[None]:
        """While the block runs (a call that waits inside the provider), log every :data:`WAIT_LOG_EVERY_SEC` that the
        attempt still waits for ``what``."""
        ticker, done = _Ticker(self, what), threading.Event()

        def beat() -> None:
            while not done.wait(WAIT_LOG_EVERY_SEC):
                ticker.tick(detail(), force=True)

        thread = threading.Thread(target=beat, name="sandbox-wait-log", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()


class _Ticker:
    """Says, at most every :data:`WAIT_LOG_EVERY_SEC`, that a wait is still going on."""

    def __init__(self, wait: _Wait, what: str) -> None:
        self._wait, self._what = wait, what
        self._next = time.monotonic() + WAIT_LOG_EVERY_SEC

    def tick(self, detail: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next:
            return
        self._next = now + WAIT_LOG_EVERY_SEC
        wait = self._wait
        log.info(
            "%s still waiting for %s after %.0fs: %s (gives up in %.0fs)",
            wait.label,
            self._what,
            wait.elapsed(),
            detail,
            max(wait.left(), 0.0),
        )


def _priority_of(meta: Mapping[str, object]) -> int | None:
    value = meta.get("priority")
    return None if value is None else int(value)  # type: ignore[call-overload]
