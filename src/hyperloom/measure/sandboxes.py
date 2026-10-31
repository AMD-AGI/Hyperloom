# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool of authoritative sandboxes the hosted measurement backend measures in (design 4.4.6).

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
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
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
    "DEFAULT_MAX_PARALLEL_TESTS",
    "DEFAULT_RUNNER_START_TIMEOUT_SEC",
    "DEFAULT_TIMEOUT_SLACK_SEC",
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
    """The ``agent-sandbox.io/priority`` tier of a test priority: 0 and 1 map to 0, 2 and 3 to 1, 4 to 2 (4.17).

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


@dataclass(frozen=True)
class PoolLimits:
    """How many sandboxes a pool holds and how long it waits for them.

    ``max_parallel_tests``: how many sandboxes may be held at once; ``acquire`` waits for a free slot.
    ``timeout_slack_sec``: added to a test's own timeout to give the sandbox its lifetime.
    ``request_timeout_sec``: how long ``create`` may wait for the sandbox to run.
    """

    max_parallel_tests: int = DEFAULT_MAX_PARALLEL_TESTS
    timeout_slack_sec: int = DEFAULT_TIMEOUT_SLACK_SEC
    request_timeout_sec: float = DEFAULT_RUNNER_START_TIMEOUT_SEC


@dataclass(frozen=True)
class AttemptTag:
    """Which attempt of which request a sandbox is for, the nodes it must avoid and the test's own metadata."""

    request_id: str
    attempt: int
    avoid_nodes: tuple[str, ...] = ()
    meta: Mapping[str, object] | None = None


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
        self._cond = threading.Condition()
        self._held: set[str] = set()
        self._claims = 0
        self._closed = False
        self._ended = 0  # how many held sandboxes have ended: a create waiting for quota wakes on a change

    @property
    def active(self) -> int:
        """How many tests hold a slot right now (a slot is taken before ``create`` returns)."""
        with self._cond:
            return self._claims

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
        ``remaining_sec`` when less) from the call, and each logs when it begins and while it lasts.

        Raises:
            NoRunnerTemplate: No template serves ``gpu_count`` (checked before any slot is taken).
            PoolClosed: The pool was closed or ``stop`` was set while waiting.
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
        try:
            sandbox = self._create(template, lifetime, metadata, stop, wait)
        except BaseException:
            self._give_slot()
            raise
        log.info("%s: sandbox %s is running after %.0fs", wait.label, sandbox.sandbox_id, wait.elapsed())
        return self._adopt(sandbox)

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
        try:
            sandbox = self._provider.connect(sandbox_id, role=MEASUREMENT_ROLE)
        except BaseException:
            self._give_slot()
            raise
        return self._adopt(sandbox)

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
            pass
        except SandboxError:
            log.exception("sandbox %s could not be closed; its own timeout will end it", runner.sandbox_id)
        finally:
            self._forget(runner.sandbox_id)

    def discard(self, sandbox_id: str) -> None:
        """Close a ``measurement`` sandbox this pool does not hold, as one an earlier process left behind; a sandbox
        that is already gone is fine."""
        try:
            self._provider.connect(sandbox_id, role=MEASUREMENT_ROLE).kill()
        except SandboxGone:
            pass
        except SandboxError:
            log.exception("sandbox %s could not be closed", sandbox_id)

    def close(self) -> None:
        """Close every ``measurement`` sandbox still running and refuse new tests."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
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
            running -= self._held
        running.discard(reattaching or "")
        return len(running)

    def _give_slot(self) -> None:
        with self._cond:
            self._claims = max(0, self._claims - 1)
            self._cond.notify_all()

    def _adopt(self, sandbox: Sandbox) -> RunnerSandbox:
        try:
            info = sandbox.info()
        except BaseException:
            self._close_handle(sandbox)
            self._give_slot()
            raise
        with self._cond:
            self._held.add(sandbox.sandbox_id)
        metadata = dict(info.metadata)
        return RunnerSandbox(
            sandbox_id=sandbox.sandbox_id,
            public_key_pem=metadata.get(SIGNING_PUBLIC_KEY) or None,
            node_name=metadata.get(NODE_NAME_KEY) or None,
            same_node=metadata.get(SAME_NODE_KEY) == "true",
            metadata=metadata,
            handle=sandbox,
        )

    def _forget(self, sandbox_id: str) -> None:
        with self._cond:
            if sandbox_id in self._held:
                self._held.discard(sandbox_id)
                self._claims = max(0, self._claims - 1)
                self._ended += 1
            self._cond.notify_all()

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
