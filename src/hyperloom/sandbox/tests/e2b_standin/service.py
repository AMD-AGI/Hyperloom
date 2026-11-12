# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stand-in control plane's rules: which key may do what, and the lifecycle of the sandboxes it made.

It answers what the E2B SDK asks of the control plane, with the behaviour the design gives the Claw sandbox service
that a stand-in can honestly reproduce: a key is one role and sees only the sandboxes it made; a role may use only
its own templates; a create may pass only the environment names its template accepts (``accepted_envs``, none for a
sealed template that lists none: 403 ``env_not_allowed``); a ``agent-sandbox.io/idempotency-key`` makes a repeated create
return the live sandbox, and refuses it with 410 once that sandbox has ended; the number of live sandboxes of a role is
capped (429 ``grant_quota_exceeded``); a sandbox ends at its timeout. It does not enforce the sealed path policy, sign
anything or write platform metadata: those belong to the real service.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from hyperloom.sandbox.socket import IDEMPOTENCY_KEY, Role
from hyperloom.sandbox.tests.e2b_standin.envd_host import EnvdHost, RunningSandbox

__all__ = ["Box", "ControlPlane", "ServiceError", "StandInTemplate"]

_REAP_INTERVAL_SEC = 0.05


class ServiceError(Exception):
    """A refusal with the HTTP status the service answers it with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class StandInTemplate:
    """One template the stand-in knows.

    Attributes:
        name: The template name callers pass (it is also the template ID).
        role: The only role whose key may create it.
        sealed: A sealed template accepts no caller environment unless ``accepted_envs`` lists it. The sealed path policy
            itself is not enforced.
        start: Command started inside the sandbox when it is created (empty: none).
        binds: Host directories the sandbox sees read-only, for what ``start`` needs (an interpreter, a source tree).
        envs: Fixed environment of the start command.
        accepted_envs: The environment names a create may pass (the design's ``accepted_envs``). ``None`` is the default
            of the kind of template: none for a sealed one, any for an open one; a collection, even an empty one, is
            exactly the names allowed.
    """

    name: str
    role: Role
    sealed: bool = False
    start: tuple[str, ...] = ()
    binds: tuple[str, ...] = ()
    envs: Mapping[str, str] = field(default_factory=dict)
    accepted_envs: Collection[str] | None = None

    def refused_envs(self, names: Iterable[str]) -> list[str]:
        """The names of ``names`` that a create of this template may not pass, sorted."""
        if self.accepted_envs is None and not self.sealed:
            return []
        allowed = self.accepted_envs or ()
        return sorted(name for name in names if name not in allowed)


@dataclass
class Box:
    """One sandbox: its metadata, lifetime and the process that is its envd."""

    sandbox_id: str
    template: StandInTemplate
    metadata: dict[str, str]
    timeout_sec: int
    token: str
    started: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ready: threading.Event = field(default_factory=threading.Event)
    running: RunningSandbox | None = None
    error: ServiceError | None = None

    @property
    def port(self) -> int:
        assert self.running is not None
        return self.running.plan.port


class ControlPlane:
    """The registry and rules behind the HTTP surface.

    Args:
        host: Starts and stops the sandboxes' envd processes.
        templates: The templates that exist.
        keys: The API key of each role.
        max_sandboxes: Per role, how many may be live at once; a create over it is refused with 429.
        node_name: Reported to sandboxes as ``NODE_NAME``.

    ``create_delay_sec`` holds every create for that long before it starts the sandbox (a queue that takes time) and
    :meth:`fail_next` makes the next calls of an operation fail with a given status.
    """

    def __init__(
        self,
        host: EnvdHost,
        templates: list[StandInTemplate],
        keys: Mapping[Role, str],
        *,
        max_sandboxes: Mapping[Role, int] | None = None,
        node_name: str = "standin-node",
    ) -> None:
        self._host = host
        self._templates = {t.name: t for t in templates}
        self._roles = {key: role for role, key in keys.items()}
        self._max = dict(max_sandboxes or {})
        self._node = node_name
        self._lock = threading.RLock()
        self._boxes: dict[Role, dict[str, Box]] = {role: {} for role in keys}
        self._by_key: dict[tuple[Role, str], Box] = {}
        self._spent: dict[tuple[Role, str], str] = {}  # the idempotency keys whose sandbox ended -> its id
        self._failures: list[tuple[str, int, str]] = []
        self._closed = False
        self.create_delay_sec = 0.0
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap, name="standin-reaper", daemon=True)
        self._reaper.start()

    @property
    def envd_version(self) -> str:
        """The version the sandboxes' envd reports."""
        return self._host.version

    def starting(self) -> int:
        """How many sandboxes have been accepted but are not running yet."""
        with self._lock:
            return sum(1 for per_role in self._boxes.values() for b in per_role.values() if not b.ready.is_set())

    def live_sandboxes(self) -> list[RunningSandbox]:
        """The processes of the sandboxes that are running now."""
        with self._lock:
            boxes = [b for per_role in self._boxes.values() for b in per_role.values()]
        return [b.running for b in boxes if b.running is not None]

    def lookup(self, sandbox_id: str | None) -> Box | None:
        """The live sandbox with this ID whichever role made it (the data plane authenticates by access token)."""
        with self._lock:
            for boxes in self._boxes.values():
                box = boxes.get(sandbox_id or "")
                if box is not None and box.ready.is_set() and box.error is None:
                    return box
        return None

    # -- rules -------------------------------------------------------------

    def role_of(self, api_key: str | None) -> Role:
        """The role a key stands for.

        Raises:
            ServiceError: 401 for a missing or unknown key.
        """
        role = self._roles.get(api_key or "")
        if role is None:
            raise ServiceError(401, "Invalid API key")
        return role

    def fail_next(self, operation: str, status: int, *, times: int = 1, message: str = "injected failure") -> None:
        """Make the next ``times`` calls of ``operation`` (create, connect, get, list, kill, timeout) fail."""
        with self._lock:
            self._failures += [(operation, status, message)] * times

    def _maybe_fail(self, operation: str) -> None:
        with self._lock:
            for index, (name, status, message) in enumerate(self._failures):
                if name == operation:
                    del self._failures[index]
                    raise ServiceError(status, message)

    # -- operations --------------------------------------------------------

    def create(self, role: Role, body: Mapping[str, object]) -> Box:
        """Create a sandbox, or return the live one this role's idempotency key already made.

        Raises:
            ServiceError: 403 for a template the role may not use or environment names it does not accept, 410 for an
                idempotency key whose sandbox has ended, 429 over the role's cap, 500 when envd does not start, or the
                injected failure.
        """
        self._maybe_fail("create")
        template = self._templates.get(str(body.get("templateID")))
        if template is None or template.role != role:
            raise ServiceError(403, f"template {body.get('templateID')!r} is not allowed for this key")
        envs = _strings(body.get("envVars"))
        refused = template.refused_envs(envs)
        if refused:
            raise ServiceError(403, f"env_not_allowed: template {template.name!r} does not accept {', '.join(refused)}")
        metadata = _strings(body.get("metadata"))
        key = (role, metadata[IDEMPOTENCY_KEY]) if IDEMPOTENCY_KEY in metadata else None
        box, earlier = self._reserve(role, template, metadata, _seconds(body.get("timeout")), key)
        if earlier:
            box.ready.wait()
            if box.error is not None:
                raise box.error
            return box
        self._start(box, envs)
        return box

    def _reserve(
        self,
        role: Role,
        template: StandInTemplate,
        metadata: dict[str, str],
        timeout: int,
        key: tuple[Role, str] | None,
    ) -> tuple[Box, bool]:
        """Register the sandbox about to be made, or find the one this idempotency key made (the flag is True)."""
        with self._lock:
            if self._closed:
                raise ServiceError(503, "the stand-in is shutting down")
            earlier = self._by_key.get(key) if key else None
            if earlier is not None:
                return earlier, True
            if key is not None and key in self._spent:
                raise ServiceError(410, f"idempotency_key_spent: its sandbox {self._spent[key]} has ended")
            cap = self._max.get(role)
            if cap is not None and len(self._boxes[role]) >= cap:
                raise ServiceError(429, f"grant_quota_exceeded: role {role!r} already has {cap} sandboxes")
            box = Box(f"sbx{secrets.token_hex(6)}", template, metadata, timeout, secrets.token_hex(16))
            self._boxes[role][box.sandbox_id] = box
            if key:
                self._by_key[key] = box
            return box, False

    def _start(self, box: Box, envs: Mapping[str, str]) -> None:
        defaults = {
            "E2B_SANDBOX_ID": box.sandbox_id,
            "E2B_TEMPLATE_ID": box.template.name,
            "NODE_NAME": self._node,
        }
        everything = {**defaults, **box.template.envs, **envs}
        time.sleep(self.create_delay_sec)
        try:
            box.running = self._host.start(
                box.sandbox_id,
                binds=box.template.binds,
                start=box.template.start,
                start_env=everything,
                init={"envVars": everything, "accessToken": box.token},
            )
        except (RuntimeError, OSError) as exc:
            box.error = ServiceError(500, f"could not start the sandbox: {exc}")
            self._end(box)
            raise box.error from exc
        else:
            box.started = datetime.now(timezone.utc)
            box.end = box.started + timedelta(seconds=box.timeout_sec)
        finally:
            box.ready.set()

    def get(self, role: Role, sandbox_id: str) -> Box:
        """Return the live sandbox ``role`` made.

        Raises:
            ServiceError: 404 when there is none.
        """
        self._maybe_fail("get")
        return self._find(role, sandbox_id)

    def _find(self, role: Role, sandbox_id: str) -> Box:
        with self._lock:
            box = self._boxes[role].get(sandbox_id)
        if box is not None:
            box.ready.wait()
        if box is None or box.error is not None:
            raise ServiceError(404, f"sandbox {sandbox_id} not found")
        return box

    def connect(self, role: Role, sandbox_id: str, timeout: int | None) -> Box:
        """Like :meth:`get`; a ``timeout`` only ever extends the sandbox's life."""
        self._maybe_fail("connect")
        box = self._find(role, sandbox_id)
        if timeout:
            box.end = max(box.end, datetime.now(timezone.utc) + timedelta(seconds=timeout))
        return box

    def list(self, role: Role, metadata: Mapping[str, str]) -> list[Box]:
        """The live sandboxes of ``role`` whose metadata holds every pair, oldest first."""
        self._maybe_fail("list")
        with self._lock:
            boxes = [b for b in self._boxes[role].values() if b.ready.is_set() and b.error is None]
        matching = [b for b in boxes if all(b.metadata.get(k) == v for k, v in metadata.items())]
        return sorted(matching, key=lambda b: (b.started, b.sandbox_id))

    def set_timeout(self, role: Role, sandbox_id: str, timeout: int) -> None:
        """End the sandbox ``timeout`` seconds from now."""
        self._maybe_fail("timeout")
        self._find(role, sandbox_id).end = datetime.now(timezone.utc) + timedelta(seconds=timeout)

    def kill(self, role: Role, sandbox_id: str) -> None:
        """Kill the sandbox and everything in it.

        Raises:
            ServiceError: 404 when there is none (a repeated kill).
        """
        self._maybe_fail("kill")
        self._end(self._find(role, sandbox_id))

    # -- teardown ----------------------------------------------------------

    def _end(self, box: Box) -> None:
        """Remove the sandbox from the registry and kill its envd; a second call for the same box does nothing."""
        with self._lock:
            owned = self._boxes[box.template.role].pop(box.sandbox_id, None) is box
            for key in [k for k, b in self._by_key.items() if b is box]:
                del self._by_key[key]
                if box.error is None:  # a sandbox that never started leaves its key free
                    self._spent[key] = box.sandbox_id
        if owned and box.running is not None:
            self._host.stop(box.running)

    def _reap(self) -> None:
        while not self._stop.wait(_REAP_INTERVAL_SEC):
            now = datetime.now(timezone.utc)
            with self._lock:
                due = [b for boxes in self._boxes.values() for b in boxes.values() if b.ready.is_set() and b.end <= now]
            for box in due:
                self._end(box)

    def close(self) -> None:
        """Refuse new sandboxes, wait for those still starting and kill every sandbox."""
        with self._lock:
            self._closed = True
            live = [b for boxes in self._boxes.values() for b in boxes.values()]
        self._stop.set()
        self._reaper.join(timeout=5)
        for box in live:
            box.ready.wait()
            self._end(box)


def _strings(value: object) -> dict[str, str]:
    return {str(k): str(v) for k, v in dict(value or {}).items()} if isinstance(value, Mapping) else {}


def _seconds(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and value > 0 else 300
