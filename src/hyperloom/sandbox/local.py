# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local sandbox provider: every sandbox is a temporary directory on this machine.

Used for development, CI, the consistency check and multi-branch MB1. It speaks
the same socket as a hosted provider (:mod:`hyperloom.sandbox.socket`), with
the same sealed policy, the same ``claw.io/`` metadata and one process group
per runner that is reaped as a tree.
"""

from __future__ import annotations

import logging
import shutil
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.outside_facts import ROCR_VISIBLE_DEVICES_ENV
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom.sandbox.identity import generate_signing_identity
from hyperloom.sandbox.node import node_name
from hyperloom.sandbox.local_sandbox import WORK_DIR, LocalSandbox, SandboxPlan, SandboxRuntime
from hyperloom.sandbox.socket import (
    AVOID_NODES_KEY,
    IDEMPOTENCY_KEY,
    NODE_NAME_KEY,
    PLATFORM_METADATA_KEYS,
    SAME_NODE_KEY,
    SIGNING_PUBLIC_KEY,
    CreateOptions,
    Role,
    Sandbox,
    SandboxDenied,
    SandboxError,
    SandboxGone,
    SandboxInfo,
    SandboxUnavailable,
)
from hyperloom.sandbox.templates import RESERVED_ENV_NAMES, LocalTemplate

log = logging.getLogger(__name__)

__all__ = ["LocalSandboxProvider", "LocalTemplate", "ProcessUsers"]


@dataclass(frozen=True)
class ProcessUsers:
    """The python that runs each process's launcher and the accounts the processes run as.

    ``launcher_python`` must be executable by ``sealed_user`` / ``unsealed_user`` when those are set. ``sealed_user``
    is the optional ``(uid, gid)`` that sealed sandboxes' processes run as, ``unsealed_user`` the one for dev and
    sub-run sandboxes.
    """

    launcher_python: str = sys.executable
    sealed_user: tuple[int, int] | None = None
    unsealed_user: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Order:
    """What a create asks of one sandbox once its template and idempotency key are accepted."""

    role: Role
    timeout_sec: int
    metadata: Mapping[str, str]
    envs: Mapping[str, str]


class LocalSandboxProvider:
    """Creates, reconnects to and finds sandboxes on this machine.

    Args:
        templates: The templates callers may create; names must be unique.
        gpus: The card pool, for example ``range(8)``; leased per template.
        base_dir: Where sandbox directories live; a temporary directory that
            :meth:`close` removes when omitted.
        hostname: The node name to report; ``NODE_NAME`` or this machine's hostname by default (:func:`~hyperloom.sandbox.node.node_name`).
        users: The launcher's python and the accounts the processes run as, see :class:`ProcessUsers`.

    Without a ``sealed_user`` and an ``unsealed_user`` every sandbox process runs as the
    calling account, so one sandbox's processes can read another's files,
    including its signing key: the sandboxes are separated by directory and by
    process tree, not by account. Giving the two kinds different accounts (this
    needs root) puts the signing key out of reach of every other process.
    """

    name = "local"

    def __init__(
        self,
        templates: Iterable[LocalTemplate],
        *,
        gpus: Iterable[int] = (),
        base_dir: Path | None = None,
        hostname: str | None = None,
        users: ProcessUsers = ProcessUsers(),
    ) -> None:
        sealed_user, unsealed_user = users.sealed_user, users.unsealed_user
        self._templates: dict[str, LocalTemplate] = {}
        for template in templates:
            if template.name in self._templates:
                raise ValueError(f"duplicate template {template.name!r}")
            self._templates[template.name] = template
        self._users = {True: sealed_user, False: unsealed_user}
        if (sealed_user or unsealed_user) and os.geteuid() != 0:
            raise ValueError("running sandbox processes as another account needs root")
        self._launcher_python = users.launcher_python
        self.pool = GpuPool(gpus)
        self.hostname = hostname or node_name()
        self._owns_base = base_dir is None
        self._base = Path(tempfile.mkdtemp(prefix="hl-sandbox-")) if base_dir is None else base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        if sealed_user or unsealed_user:
            self._base.chmod(0o711)  # other accounts must traverse it to reach their own directory
        self._cond = threading.Condition()
        self._live: dict[str, LocalSandbox] = {}
        self._by_key: dict[tuple[Role, str], LocalSandbox | SandboxError] = {}
        self._creating: set[tuple[Role, str]] = set()
        self._cards: dict[str, tuple[int, ...]] = {}
        self.quarantined: set[int] = set()

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
        """Create a fresh sandbox, or return the one an idempotency key already made.

        Args:
            template: A configured template name.
            role: The caller's role; it must match the template's role.
            timeout_sec: Lifetime of the sandbox; its processes are reaped then.
            metadata: Caller metadata; platform ``claw.io/`` keys are refused.
            options: Extra environment (``envs``; refused for a sealed template and for platform variable names) and
                how long to wait for GPU cards (``wait_sec``).

        Returns:
            Sandbox: The new (or previously created) sandbox.

        Raises:
            SandboxDenied: Unknown template, wrong role, platform names in
                metadata or envs, or envs on a sealed template.
            SandboxUnavailable: No GPU cards within ``wait_sec``, or the original
                failure of a repeated idempotency key.
        """
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        envs, wait_sec = options.envs, options.wait_sec
        tpl = self._check_create(template, role, metadata, envs)
        key = (role, metadata[IDEMPOTENCY_KEY]) if IDEMPOTENCY_KEY in metadata else None
        if key is not None:
            earlier = self._claim_key(key, wait_sec)
            if earlier is not None:
                if isinstance(earlier, SandboxError):
                    raise earlier
                return earlier
        try:
            cards = self.pool.acquire(tpl.gpus, wait_sec)
        except BaseException:
            # Capacity is transient: a retry with the same key may succeed.
            self._finish_key(key, None)
            raise
        try:
            sandbox = self._build(tpl, _Order(role, timeout_sec, metadata, envs or {}), cards)
        except SandboxError as exc:
            self._finish_key(key, exc)
            raise
        except BaseException:
            self._finish_key(key, None)
            raise
        self._finish_key(key, sandbox)
        return sandbox

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox:
        """Return the live sandbox ``role`` created.

        Raises:
            SandboxGone: When it does not exist, ended, or another role made it.
        """
        with self._cond:
            sandbox = self._live.get(sandbox_id)
        if sandbox is None or sandbox.role != role or sandbox.ended:
            raise SandboxGone(f"no sandbox {sandbox_id!r} for role {role!r}")
        return sandbox

    def find(self, metadata: Mapping[str, str], *, role: Role) -> list[SandboxInfo]:
        """List live sandboxes of ``role`` whose metadata contains every given pair."""
        with self._cond:
            candidates = [s for s in self._live.values() if s.role == role and not s.ended]
        found: list[SandboxInfo] = []
        for sandbox in candidates:
            try:
                info = sandbox.info()
            except SandboxGone:
                continue
            if all(info.metadata.get(k) == v for k, v in metadata.items()):
                found.append(info)
        return sorted(found, key=lambda i: i.sandbox_id)

    def close(self) -> None:
        """Kill every live sandbox and remove the provider's directory if it made one."""
        with self._cond:
            live = list(self._live.values())
        for sandbox in live:
            try:
                sandbox.kill()
            except SandboxUnavailable:
                log.exception("closing: sandbox %s could not be reaped", sandbox.sandbox_id)
        if self._owns_base:
            shutil.rmtree(self._base, ignore_errors=True)

    def __enter__(self) -> LocalSandboxProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- creation ----------------------------------------------------------

    def _check_create(
        self, template: str, role: Role, metadata: Mapping[str, str], envs: Mapping[str, str] | None
    ) -> LocalTemplate:
        tpl = self._templates.get(template)
        if tpl is None:
            raise SandboxDenied(f"template {template!r} is not allowed")
        if tpl.role != role:
            raise SandboxDenied(f"role {role!r} may not use template {template!r}")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise SandboxDenied("metadata keys and values must be strings")
        owned = PLATFORM_METADATA_KEYS & set(metadata)
        if owned:
            raise SandboxDenied(f"metadata keys are written by the platform: {sorted(owned)}")
        if envs and tpl.sealed:
            raise SandboxDenied("a sealed template does not accept caller environment variables")
        reserved = RESERVED_ENV_NAMES & set(envs or {})
        if reserved:
            raise SandboxDenied(f"envs name platform variables {sorted(reserved)}")
        return tpl

    def _claim_key(self, key: tuple[Role, str], wait_sec: float) -> LocalSandbox | SandboxError | None:
        """Return what an earlier create with this key produced, or claim the key (None)."""
        with self._cond:
            if not self._cond.wait_for(lambda: key not in self._creating, timeout=max(0.0, wait_sec)):
                raise SandboxUnavailable("a create with the same idempotency key is still in progress")
            if key in self._by_key:
                return self._by_key[key]
            self._creating.add(key)
            return None

    def _finish_key(self, key: tuple[Role, str] | None, outcome: LocalSandbox | SandboxError | None) -> None:
        if key is None:
            return
        with self._cond:
            self._creating.discard(key)
            if outcome is not None:
                self._by_key[key] = outcome
            self._cond.notify_all()

    def _build(self, tpl: LocalTemplate, order: _Order, cards: tuple[int, ...]) -> LocalSandbox:
        role, metadata = order.role, order.metadata
        sandbox_id = f"lsb-{uuid.uuid4().hex}"
        directory = self._base / sandbox_id
        sandbox: LocalSandbox | None = None
        with self._cond:
            self._cards[sandbox_id] = cards
        try:
            for sub in ("fs/in", "fs/out", f"fs/{WORK_DIR}"):
                (directory / sub).mkdir(parents=True)
            (directory / "private").mkdir(mode=0o700)
            self._hand_over(directory, tpl)
            meta, platform_envs = self._platform_view(tpl, sandbox_id, directory, cards, metadata)
            plan = SandboxPlan(sandbox_id, tpl, role, directory, meta, order.envs, platform_envs, order.timeout_sec)
            runtime = SandboxRuntime(self._on_end, self._launcher_python, self._users[tpl.sealed])
            sandbox = LocalSandbox(plan, runtime)
            with self._cond:
                self._live[sandbox_id] = sandbox
            sandbox.start_template_command()
        except BaseException as exc:
            self._abort_build(sandbox, directory, cards, sandbox_id)
            if isinstance(exc, (OSError, RuntimeError)):
                raise SandboxUnavailable(f"could not create sandbox from {tpl.name!r}: {exc}") from exc
            raise
        return sandbox

    def _hand_over(self, directory: Path, tpl: LocalTemplate) -> None:
        """Give the sandbox's files to the account its processes run as (when one is configured).

        ``private/`` stays with the provider; only a signing key is handed to the sealed account.
        """
        user = self._users[tpl.sealed]
        if user is None:
            return
        directory.chmod(0o711)
        for path in (directory / "fs", *(directory / "fs").iterdir()):
            os.chown(path, *user)

    def _abort_build(
        self, sandbox: LocalSandbox | None, directory: Path, cards: tuple[int, ...], sandbox_id: str
    ) -> None:
        if sandbox is not None:
            try:
                sandbox.kill()  # releases the cards through _on_end
            except SandboxUnavailable:
                log.exception("create failed and sandbox %s could not be reaped", sandbox_id)
            return
        shutil.rmtree(directory, ignore_errors=True)
        with self._cond:
            self._cards.pop(sandbox_id, None)
        self.pool.release(cards)

    def _platform_view(
        self,
        tpl: LocalTemplate,
        sandbox_id: str,
        directory: Path,
        cards: tuple[int, ...],
        caller_metadata: Mapping[str, str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """The metadata callers see and the environment only the platform sets."""
        meta = dict(caller_metadata)
        meta[NODE_NAME_KEY] = self.hostname
        avoid = [n.strip() for n in caller_metadata.get(AVOID_NODES_KEY, "").split(",") if n.strip()]
        if self.hostname in avoid:
            # One node: it cannot be avoided, so create anyway and say so, as Claw does.
            meta[SAME_NODE_KEY] = "true"
        envs = {
            "NODE_NAME": self.hostname,
            "E2B_SANDBOX_ID": sandbox_id,
            "HYPERLOOM_SANDBOX_ROOT": str(directory / "fs"),
        }
        if tpl.gpus:
            envs[ROCR_VISIBLE_DEVICES_ENV] = ",".join(str(c) for c in cards)
        if tpl.signing:
            identity = generate_signing_identity(directory / "private", self._users[True])
            meta[SIGNING_PUBLIC_KEY] = identity.public_key_pem
            envs["HYPERLOOM_SIGNING_KEY_FILE"] = str(identity.private_key_file)
        return meta, envs

    def _on_end(self, sandbox: LocalSandbox, reaped: bool) -> None:
        """Take a finished sandbox out of the registry; return its cards only if it is provably empty."""
        with self._cond:
            self._live.pop(sandbox.sandbox_id, None)
            cards = self._cards.pop(sandbox.sandbox_id, ())
        if reaped:
            self.pool.release(cards)
        else:
            self.quarantined.update(cards)
