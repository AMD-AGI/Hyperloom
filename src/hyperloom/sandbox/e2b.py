# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The E2B sandbox provider: the sandbox socket over the official ``e2b`` SDK.

It talks to any E2B-compatible service (the Claw sandbox service, or the stand-in the contract tests start) through
``E2B_API_URL`` and ``E2B_SANDBOX_URL``. Each role has its own API key; a key is read once into the provider, is passed
to the SDK per call and never appears in a log line, an error message or ``repr``.

Control-plane calls (everything except commands and files) are retried with exponential backoff while the failure is
transient: a network error, a 5xx, a rate limit or a request timeout. A create is retried only when it carries an
idempotency key, because only then does a replay return the same sandbox instead of a second one. Every SDK failure
is classified into the socket exceptions; one the socket has no word for (a bad argument, say) propagates as it is.

Two refusals of a create carry a meaning of their own. A 410 means the idempotency key's sandbox has ended, so the key is
spent (``SandboxGone``). A 429 whose error message names ``grant_quota_exceeded`` means the session's quota is full
(``SandboxQuotaExceeded``, a ``SandboxUnavailable``). The SDK keeps only the ``message`` of a 429 body, so the service
must put that reason code in ``message``, as it does for the 503 reasons ``pending_timeout`` and ``no_capacity``; a 503
stays a plain ``SandboxUnavailable``.
"""

from __future__ import annotations

import contextlib
import enum
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypeVar

import e2b
import httpx

from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    CommandResult,
    CreateOptions,
    OutputSinks,
    Role,
    RunSettings,
    Sandbox,
    SandboxDenied,
    SandboxGone,
    SandboxInfo,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)

__all__ = ["E2BSandboxProvider", "RetryPolicy"]

T = TypeVar("T")

_KEY_ENV: dict[Role, str] = {
    "measurement": "HYPERLOOM_SANDBOX_KEY_MEASUREMENT",
    "subrun": "HYPERLOOM_SANDBOX_KEY_SUBRUN",
    "dev": "HYPERLOOM_SANDBOX_KEY_DEV",
}
_API_URL_ENV = "E2B_API_URL"
_SANDBOX_URL_ENV = "E2B_SANDBOX_URL"
_LIST_PAGE = 100

#: Everything the SDK raises for a failed call. ``ServiceBusyException`` (a 503) is not a ``SandboxException``, and a
#: gateway's HTML error page reaches the caller as the ``JSONDecodeError`` of the SDK's parse of an error body.
_SDK_ERRORS = (
    e2b.SandboxException,
    e2b.AuthenticationException,
    e2b.ServiceBusyException,
    httpx.TransportError,
    json.JSONDecodeError,
)

#: How the data plane words a refusal when it has no status code of its own (a Connect-RPC ``permission_denied``).
_REFUSAL_PREFIXES = ("Code.PERMISSION_DENIED", "permission_denied")

#: The reason a 429 carries when a create is over the session grant's quota (and not a plain rate limit).
_QUOTA_REASON = "grant_quota_exceeded"

#: Statuses that mean the request was not processed and may succeed if repeated; a 503 means the service has no room
#: and is not retried on create. A 501 or 505 will not change on a retry.
_SERVER_ERROR = 500
_SERVER_RETRY_STATUSES = frozenset({408, _SERVER_ERROR, 502, 504})
_PERMANENT_SERVER_STATUSES = frozenset({501, 505})


@dataclass(frozen=True)
class RetryPolicy:
    """How long and how fast a control-plane call is retried while its failure is transient.

    Attributes:
        budget_sec: Give up (raise ``SandboxUnavailable``) once this much time has passed.
        first_delay_sec: The wait before the first retry; it doubles each time.
        max_delay_sec: The longest single wait.
    """

    budget_sec: float = 300.0
    first_delay_sec: float = 0.5
    max_delay_sec: float = 10.0

    def __post_init__(self) -> None:
        if self.budget_sec < 0 or self.first_delay_sec <= 0 or self.max_delay_sec < self.first_delay_sec:
            raise ValueError("a retry policy needs budget >= 0 and 0 < first_delay <= max_delay")


class _Failure(enum.Enum):
    GONE = "gone"
    UNAVAILABLE = "unavailable"
    OVER_QUOTA = "over quota"
    DENIED = "denied"
    UNREACHABLE = "unreachable"
    NO_FILE = "no file"


class _Retry(enum.Enum):
    """Which failures a control-plane call is repeated after."""

    TRANSIENT = "any transient failure, a rate limit and a 503 included"
    SERVER_ERROR = "a network error or a server error that did not process the request, not a rate limit or a 503"
    NEVER = "none"


_UNAVAILABLE = (
    e2b.TimeoutException,
    e2b.RateLimitException,
    e2b.ServiceBusyException,
    httpx.TransportError,
    json.JSONDecodeError,
)

_ERROR_OF = {
    _Failure.GONE: SandboxGone,
    _Failure.UNAVAILABLE: SandboxUnavailable,
    _Failure.OVER_QUOTA: SandboxQuotaExceeded,
    _Failure.DENIED: SandboxDenied,
}


def _status(exc: Exception) -> int | None:
    """The HTTP status an SDK exception stands for: its attribute, or the ``NNN:`` the data plane puts first."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status
    head = str(exc).split(":", 1)[0]
    return int(head) if head.isdigit() else None


def _classify(exc: Exception) -> _Failure | None:
    """Which socket failure an SDK exception is, or None when the socket has no word for it."""
    if isinstance(exc, e2b.FileNotFoundException):
        return _Failure.NO_FILE
    if isinstance(exc, (e2b.SandboxNotFoundException, e2b.SandboxNotRunningException)):
        return _Failure.GONE
    if isinstance(exc, e2b.SandboxUnreachableException):
        return _Failure.UNREACHABLE
    if isinstance(exc, e2b.RateLimitException) and _QUOTA_REASON in str(exc):
        return _Failure.OVER_QUOTA
    if isinstance(exc, _UNAVAILABLE):
        return _Failure.UNAVAILABLE
    if isinstance(exc, e2b.AuthenticationException) or str(exc).startswith(_REFUSAL_PREFIXES):
        return _Failure.DENIED
    status = _status(exc)
    if status is None:
        return None
    if status in (404, 410):  # 410: a create with an idempotency key whose sandbox has ended
        return _Failure.GONE
    if status in (408, 429) or status >= _SERVER_ERROR:
        return _Failure.UNAVAILABLE
    return _Failure.DENIED if status >= 400 else None


def _is_transient(exc: Exception, *, capacity: bool) -> bool:
    """Whether the same call may succeed if repeated; ``capacity`` also counts a rate limit or a 503."""
    if isinstance(exc, (e2b.SandboxNotRunningException, e2b.SandboxUnreachableException)):
        return False
    if isinstance(exc, (httpx.TransportError, json.JSONDecodeError, e2b.TimeoutException)):
        return True
    if isinstance(exc, (e2b.RateLimitException, e2b.ServiceBusyException)):
        return capacity
    status = _status(exc)
    if status is None:
        return False
    return status in _SERVER_RETRY_STATUSES or (
        capacity and status >= _SERVER_ERROR and status not in _PERMANENT_SERVER_STATUSES
    )


class _LineSink:
    """Turns the chunks the SDK delivers into whole lines (each keeps its newline) for a caller's callback."""

    def __init__(self, callback: Callable[[str], None] | None) -> None:
        self._callback = callback
        self._pending = ""

    def feed(self, chunk: str) -> None:
        if self._callback is None:
            return
        self._pending += chunk
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._callback(line + "\n")

    def flush(self) -> None:
        if self._callback is not None and self._pending:
            self._callback(self._pending)
        self._pending = ""


class _Control:
    """One role's access to the service: its SDK parameters, retries and failure classification."""

    def __init__(self, params: Mapping[str, object], retry: RetryPolicy) -> None:
        self.params = dict(params)
        self._retry = retry

    def call(self, fn: Callable[[], T], *, retry: _Retry = _Retry.TRANSIENT, until: float | None = None) -> T:
        """Run one control-plane call, repeating it while its failure is one ``retry`` allows and time remains.

        Args:
            fn: The call.
            retry: Which failures to repeat after.
            until: A ``time.monotonic()`` deadline; the retry budget's end by default.

        Raises:
            SandboxUnavailable, SandboxGone, SandboxDenied: By :func:`_classify`.
        """
        deadline = time.monotonic() + self._retry.budget_sec if until is None else until
        delay = self._retry.first_delay_sec
        while True:
            try:
                return fn()
            except _SDK_ERRORS as exc:
                again = retry is not _Retry.NEVER and _is_transient(exc, capacity=retry is _Retry.TRANSIENT)
                if not again or time.monotonic() + delay >= deadline:
                    self.fail(exc)
            time.sleep(delay)
            delay = min(delay * 2, self._retry.max_delay_sec)

    def fail(self, exc: Exception, sandbox_id: str | None = None) -> NoReturn:
        """Raise the socket exception for ``exc``; one with no socket meaning is raised unchanged.

        ``sandbox_id`` lets a failure that cannot say whether the sandbox is gone be settled by asking the service.
        """
        failure = _classify(exc)
        if failure is _Failure.UNREACHABLE:
            gone = sandbox_id is not None and not self._exists(sandbox_id)
            failure = _Failure.GONE if gone else _Failure.UNAVAILABLE
        if failure is None:
            raise exc
        if failure is _Failure.NO_FILE:
            raise FileNotFoundError(str(exc)) from exc
        raise _ERROR_OF[failure](str(exc)) from exc

    def _exists(self, sandbox_id: str) -> bool:
        try:
            e2b.Sandbox.get_info(sandbox_id, **self.params)
        except _SDK_ERRORS as exc:
            return _classify(exc) is not _Failure.GONE
        return True


class E2BSandboxProvider:
    """Creates, reconnects to and finds sandboxes on an E2B-compatible service.

    Args:
        keys: The API key of each role this provider serves; a role without one is refused with ``SandboxDenied``.
        api_url: The control plane, ``E2B_API_URL`` by default.
        sandbox_url: The data plane that routes by the ``E2b-Sandbox-Id`` header, ``E2B_SANDBOX_URL`` by default.
        retry: How control-plane calls are retried, see :class:`RetryPolicy`.

    Needs ``e2b>=2.53.1``: an older SDK's ``connect`` sends no ``E2b-Sandbox-Id`` header, so a service that routes by
    it cannot serve a reconnected handle.
    """

    name = "e2b"

    def __init__(
        self,
        keys: Mapping[Role, str],
        *,
        api_url: str | None = None,
        sandbox_url: str | None = None,
        retry: RetryPolicy = RetryPolicy(),
    ) -> None:
        self._controls = {role: _Control(self._params(key, api_url, sandbox_url), retry) for role, key in keys.items()}

    @staticmethod
    def _params(key: str, api_url: str | None, sandbox_url: str | None) -> dict[str, object]:
        params: dict[str, object] = {"api_key": key, "retries": 0}
        if api_url is not None:
            params["api_url"] = api_url
        if sandbox_url is not None:
            params["sandbox_url"] = sandbox_url
        return params

    @classmethod
    def from_environ(cls, environ: Mapping[str, str], *, retry: RetryPolicy = RetryPolicy()) -> E2BSandboxProvider:
        """Build a provider from ``HYPERLOOM_SANDBOX_KEY_<ROLE>``, ``E2B_API_URL`` and ``E2B_SANDBOX_URL``.

        The mapping is only read; the caller decides whether to scrub the keys from the process environment.
        """
        keys = {role: environ[name] for role, name in _KEY_ENV.items() if environ.get(name)}
        return cls(keys, api_url=environ.get(_API_URL_ENV), sandbox_url=environ.get(_SANDBOX_URL_ENV), retry=retry)

    def __repr__(self) -> str:
        return f"E2BSandboxProvider(roles={sorted(self._controls)})"

    def _control(self, role: Role) -> _Control:
        control = self._controls.get(role)
        if control is None:
            raise SandboxDenied(f"no API key is configured for role {role!r}")
        return control

    # -- socket ------------------------------------------------------------

    def create(
        self,
        template: str,
        *,
        role: Role,
        timeout_sec: int,
        metadata: Mapping[str, str],
        options: CreateOptions = CreateOptions(),
    ) -> Sandbox:
        """Create a sandbox and wait until the service reports it running.

        The wait is bounded by ``options.wait_sec``. A metadata ``claw.io/idempotency-key`` makes a repeated create
        (also this method's own retry after a server error) return the sandbox the service already made.

        Raises:
            ValueError: When ``timeout_sec`` is not positive.
            SandboxDenied: No key for ``role``, or the service refused the key, template, environment or metadata.
            SandboxGone: The idempotency key's sandbox has ended (410); the key is spent, a new key makes a new one.
            SandboxQuotaExceeded: The session's quota is full (429 ``grant_quota_exceeded``); it clears when a running
                sandbox ends.
            SandboxUnavailable: No capacity (503), or the sandbox was not running within ``wait_sec``.
        """
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        control = self._control(role)
        deadline = time.monotonic() + options.wait_sec
        envs = dict(options.envs) if options.envs else None

        def attempt() -> e2b.Sandbox:
            return e2b.Sandbox.create(
                template=template,
                timeout=timeout_sec,
                metadata=dict(metadata),
                envs=envs,
                **{**control.params, "request_timeout": max(deadline - time.monotonic(), 0.001)},
            )

        retry = _Retry.SERVER_ERROR if IDEMPOTENCY_KEY in metadata else _Retry.NEVER
        return _E2BSandbox(control.call(attempt, retry=retry, until=deadline), control)

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox:
        """Return a handle on the live sandbox ``role`` created.

        Raises:
            SandboxGone: When it does not exist, ended, or another key made it.
        """
        control = self._control(role)
        return _E2BSandbox(control.call(lambda: e2b.Sandbox.connect(sandbox_id, **control.params)), control)

    def find(self, metadata: Mapping[str, str], *, role: Role) -> list[SandboxInfo]:
        """List the running sandboxes of ``role`` whose metadata contains every given pair."""
        control = self._control(role)
        query = e2b.SandboxQuery(metadata=dict(metadata) or None, state=[e2b.SandboxState.RUNNING])

        def list_all() -> list[e2b.SandboxInfo]:
            paginator = e2b.Sandbox.list(query=query, limit=_LIST_PAGE, **control.params)
            found: list[e2b.SandboxInfo] = []
            while paginator.has_next:
                found.extend(paginator.next_items())
            return found

        return sorted((_info(item) for item in control.call(list_all)), key=lambda i: i.sandbox_id)


def _info(item: e2b.SandboxInfo) -> SandboxInfo:
    return SandboxInfo(item.sandbox_id, item.name or item.template_id, dict(item.metadata or {}))


class _E2BSandbox:
    """A handle on one sandbox: control-plane calls go to the API, files and commands to the data plane."""

    def __init__(self, handle: e2b.Sandbox, control: _Control) -> None:
        self.sandbox_id = handle.sandbox_id
        self._handle = handle
        self._control = control

    @contextlib.contextmanager
    def _data_plane(self) -> Iterator[None]:
        try:
            yield
        except _SDK_ERRORS as exc:
            self._control.fail(exc, self.sandbox_id)

    def info(self) -> SandboxInfo:
        """Return the sandbox's id, template and metadata.

        Raises:
            SandboxGone: When the sandbox has ended or is not visible to this key.
        """
        return _info(self._control.call(lambda: e2b.Sandbox.get_info(self.sandbox_id, **self._control.params)))

    def write_file(self, path: str, data: bytes) -> None:
        """Write a file, creating its parent directories."""
        with self._data_plane():
            self._handle.files.write(path, data)

    def read_file(self, path: str) -> bytes:
        """Read a file.

        Raises:
            FileNotFoundError: When there is no such file.
        """
        with self._data_plane():
            return bytes(self._handle.files.read(path, format="bytes"))

    def copy_out(self, path: str, dest: Path) -> int:
        """Stream a file to ``dest`` in chunks and return the byte count.

        Raises:
            FileNotFoundError: When there is no such file.
        """
        total = 0
        with self._data_plane(), self._handle.files.read(path, format="stream") as stream, dest.open("wb") as out:
            for chunk in stream:
                out.write(chunk)
                total += len(chunk)
        return total

    def list_dir(self, path: str) -> list[str]:
        """List entry names of a directory, sorted.

        Raises:
            FileNotFoundError: When there is no such directory.
        """
        with self._data_plane():
            return sorted(entry.name for entry in self._handle.files.list(path))

    def run(
        self,
        cmd: str,
        *,
        background: bool = False,
        settings: RunSettings = RunSettings(),
        sinks: OutputSinks = OutputSinks(),
    ) -> CommandResult:
        """Run a shell command inside the sandbox.

        A background command returns at once with exit code 0 and keeps running until it ends or the sandbox does.

        Raises:
            SandboxDenied: When the service refuses commands (a sealed sandbox).
            SandboxGone: When the sandbox has ended (also if it ends during the command).
            SandboxUnavailable: When ``timeout_sec`` expires; the command is killed.
        """
        envs = dict(settings.envs) if settings.envs else None
        timeout = 0 if background or settings.timeout_sec is None else settings.timeout_sec
        with self._data_plane():
            handle = self._handle.commands.run(cmd, background=True, envs=envs, cwd=settings.cwd, timeout=timeout)
            if background:
                handle.disconnect()
                return CommandResult(0, "", "")
            return self._wait(handle, sinks)

    def _wait(self, handle: e2b.CommandHandle, sinks: OutputSinks) -> CommandResult:
        out, err = _LineSink(sinks.on_stdout), _LineSink(sinks.on_stderr)
        try:
            done = handle.wait(on_stdout=out.feed, on_stderr=err.feed)
        except e2b.CommandExitException as exited:
            done = exited
        except e2b.TimeoutException:
            with contextlib.suppress(*_SDK_ERRORS):
                handle.kill()
            raise
        finally:
            out.flush()
            err.flush()
        return CommandResult(done.exit_code, done.stdout, done.stderr)

    def set_timeout(self, timeout_sec: int) -> None:
        """Restart the lifetime clock: the sandbox ends ``timeout_sec`` from now.

        Raises:
            SandboxGone: When the sandbox has ended.
        """
        self._control.call(lambda: e2b.Sandbox.set_timeout(self.sandbox_id, timeout_sec, **self._control.params))

    def kill(self) -> None:
        """End the sandbox and everything running in it; safe to call again."""
        self._control.call(lambda: e2b.Sandbox.kill(self.sandbox_id, **self._control.params))
