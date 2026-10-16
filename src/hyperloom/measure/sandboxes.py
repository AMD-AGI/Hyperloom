# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool of authoritative sandboxes the hosted measurement backend measures in (design 4.4.6).

One test gets one fresh sandbox. :class:`SandboxPool` creates it through the ``sandbox`` socket with the ``measurement``
key, never lets more tests run at once than ``max_parallel_tests``, hands back what the platform said about the sandbox
(its signing public key, the node it landed on, whether the node could not be avoided), and closes everything on
shutdown. Which runner template serves a card count, what the sandbox's lifetime is and what metadata it carries are
decided here, so the backend never builds a ``create`` call itself.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
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
)

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_PARALLEL_TESTS",
    "DEFAULT_RUNNER_START_TIMEOUT_SEC",
    "DEFAULT_TIMEOUT_SLACK_SEC",
    "MEASUREMENT_ROLE",
    "NoRunnerTemplate",
    "PoolClosed",
    "RunnerSandbox",
    "SandboxPool",
    "sandbox_priority",
]

MEASUREMENT_ROLE: Role = "measurement"
DEFAULT_MAX_PARALLEL_TESTS = 8
DEFAULT_RUNNER_START_TIMEOUT_SEC = 3600
DEFAULT_TIMEOUT_SLACK_SEC = 900
REQUEST_ID_KEY = "hyperloom.request-id"
PURPOSE_KEY = "hyperloom.purpose"
BRANCH_ID_KEY = "hyperloom.branch-id"
_SLOT_POLL_SEC = 0.5
_PRIORITY_TIERS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
_DEFAULT_PRIORITY = 2


class NoRunnerTemplate(LookupError):
    """No runner template serves the card count a test needs."""


class PoolClosed(RuntimeError):
    """The pool was closed, or the caller's stop signal fired, while a test waited for a slot."""


def sandbox_priority(priority: int | None) -> int:
    """The ``claw.io/priority`` tier of a test priority: 0 and 1 map to 0, 2 and 3 to 1, 4 to 2 (4.17).

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
        run_id: The run's id, the first part of every idempotency key.
        limits: The slot cap and the two waits, see :class:`PoolLimits`.
    """

    def __init__(
        self, provider: SandboxProvider, templates: Mapping[int, str], *, run_id: str, limits: PoolLimits = PoolLimits()
    ) -> None:
        max_parallel_tests = limits.max_parallel_tests
        if max_parallel_tests < 1:
            raise ValueError("max_parallel_tests must be at least 1")
        if not run_id:
            raise ValueError("the pool needs the run id for its idempotency keys")
        self._provider = provider
        self._templates = {int(count): name for count, name in templates.items()}
        self._run_id = run_id
        self._max = max_parallel_tests
        self._slack = limits.timeout_slack_sec
        self._request_timeout = limits.request_timeout_sec
        self._cond = threading.Condition()
        self._held: set[str] = set()
        self._claims = 0
        self._closed = False

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
        ``remaining_sec`` (what is left of the run). ``stop`` ends the wait for a slot with :class:`PoolClosed`.

        Raises:
            NoRunnerTemplate: No template serves ``gpu_count`` (checked before any slot is taken).
            PoolClosed: The pool was closed or ``stop`` was set while waiting.
            SandboxUnavailable, SandboxDenied: As the provider raised them; the slot is given back.
        """
        template = self.template_for(gpu_count)
        lifetime = int(timeout_sec + self._slack)
        if remaining_sec is not None:
            lifetime = min(lifetime, max(1, int(remaining_sec)))
        metadata = self._metadata(tag.request_id, tag.attempt, tag.avoid_nodes, tag.meta or {})
        self._take_slot(stop)
        try:
            sandbox = self._provider.create(
                template,
                role=MEASUREMENT_ROLE,
                timeout_sec=lifetime,
                metadata=metadata,
                options=CreateOptions(wait_sec=self._request_timeout),
            )
        except BaseException:
            self._give_slot()
            raise
        return self._adopt(sandbox)

    def reattach(self, sandbox_id: str, *, stop: threading.Event | None = None) -> RunnerSandbox:
        """Connect to a sandbox an earlier process created; it takes a slot like any other test.

        Raises:
            SandboxGone: The sandbox no longer exists (the slot is given back).
        """
        self._take_slot(stop, reattaching=sandbox_id)
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

    def close(self) -> None:
        """Close every ``measurement`` sandbox still running and refuse new tests."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for info in self._provider.find({}, role=MEASUREMENT_ROLE):
            self._close_one(info.sandbox_id)
        with self._cond:
            self._held.clear()
            self._claims = 0
            self._cond.notify_all()

    # -- internals ----------------------------------------------------------------------------------------------

    def _metadata(
        self, request_id: str, attempt: int, avoid_nodes: tuple[str, ...], meta: Mapping[str, object]
    ) -> dict[str, str]:
        metadata = {
            IDEMPOTENCY_KEY: self.idempotency_key(request_id, attempt),
            PRIORITY_KEY: str(sandbox_priority(_priority_of(meta))),
            REQUEST_ID_KEY: request_id,
        }
        if avoid_nodes:
            metadata[AVOID_NODES_KEY] = ",".join(sorted(set(avoid_nodes)))
        for key, field_name in ((PURPOSE_KEY, "purpose"), (BRANCH_ID_KEY, "branch_id")):
            value = meta.get(field_name)
            if value:
                metadata[key] = str(value)
        return metadata

    def _take_slot(self, stop: threading.Event | None, reattaching: str | None = None) -> None:
        """Wait until fewer than ``max_parallel_tests`` sandboxes are running, then take one of the slots.

        Sandboxes this pool does not hold count too (a measurement a previous process of the run left behind keeps
        its cards busy until it is re-attached or ends); ``reattaching`` is the one sandbox that is about to be held.
        """
        while True:
            with self._cond:
                if self._closed or (stop is not None and stop.is_set()):
                    raise PoolClosed("the sandbox pool is closed or the run is stopping")
            others = self._untracked(reattaching)
            with self._cond:
                if self._claims + others < self._max:
                    self._claims += 1
                    return
                self._cond.wait(_SLOT_POLL_SEC)

    def _untracked(self, reattaching: str | None) -> int:
        """How many running ``measurement`` sandboxes this pool does not hold."""
        running = {info.sandbox_id for info in self._provider.find({}, role=MEASUREMENT_ROLE)}
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
            self._cond.notify_all()

    def _close_one(self, sandbox_id: str) -> None:
        try:
            self._provider.connect(sandbox_id, role=MEASUREMENT_ROLE).kill()
        except SandboxGone:
            pass
        except SandboxError:
            log.exception("sandbox %s could not be closed on shutdown", sandbox_id)

    @staticmethod
    def _close_handle(sandbox: Sandbox) -> None:
        try:
            sandbox.kill()
        except SandboxError:
            log.exception("sandbox %s could not be closed", sandbox.sandbox_id)


def _priority_of(meta: Mapping[str, object]) -> int | None:
    value = meta.get("priority")
    return None if value is None else int(value)  # type: ignore[call-overload]
