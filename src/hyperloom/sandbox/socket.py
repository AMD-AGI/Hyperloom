# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sandbox socket: the protocol every sandbox implementation must satisfy.

The socket is implementation-neutral. The local implementation
(:mod:`hyperloom.sandbox.local`) and a hosted one both speak these types, and
callers classify failures by the three exception classes below without knowing
which implementation is underneath.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

__all__ = [
    "AVOID_NODES_KEY",
    "IDEMPOTENCY_KEY",
    "NODE_NAME_KEY",
    "PLATFORM_METADATA_KEYS",
    "PRIORITY_KEY",
    "SAME_NODE_KEY",
    "SIGNING_PUBLIC_KEY",
    "CommandResult",
    "Role",
    "Sandbox",
    "SandboxDenied",
    "SandboxError",
    "SandboxGone",
    "SandboxInfo",
    "SandboxProvider",
    "SandboxUnavailable",
]

#: Which key a caller uses: a measurement runner, a sub-run, or a dev sandbox.
Role = Literal["measurement", "subrun", "dev"]

#: Metadata the caller writes.
IDEMPOTENCY_KEY = "claw.io/idempotency-key"
AVOID_NODES_KEY = "claw.io/avoid-nodes"  # comma-separated node names
PRIORITY_KEY = "claw.io/priority"

#: Metadata only the platform writes; a caller that passes one is refused.
NODE_NAME_KEY = "claw.io/node-name"
SIGNING_PUBLIC_KEY = "claw.io/signing-public-key"
SAME_NODE_KEY = "claw.io/same-node"
PLATFORM_METADATA_KEYS = frozenset({NODE_NAME_KEY, SIGNING_PUBLIC_KEY, SAME_NODE_KEY})


class SandboxError(Exception):
    """Base of the three sandbox failure classes."""


class SandboxUnavailable(SandboxError):
    """Creation timed out, capacity or quota is short, or a command timed out."""


class SandboxGone(SandboxError):
    """The sandbox does not exist, is not visible to this role, or has ended."""


class SandboxDenied(SandboxError):
    """Authentication failed, the template is not allowed, or sealed policy refused the call."""


@dataclass(frozen=True)
class SandboxInfo:
    """What a caller may learn about a sandbox.

    Attributes:
        sandbox_id: The sandbox's identifier.
        template: The template it was created from.
        metadata: Keys the caller wrote plus the platform's ``claw.io/`` keys.
    """

    sandbox_id: str
    template: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one command."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RunSettings:
    """Where and how a command runs: extra environment, working directory and a time limit."""

    envs: Mapping[str, str] | None = None
    cwd: str | None = None
    timeout_sec: float | None = None


@dataclass(frozen=True)
class OutputSinks:
    """Callbacks that receive a foreground command's output line by line while it runs."""

    on_stdout: Callable[[str], None] | None = None
    on_stderr: Callable[[str], None] | None = None


@dataclass(frozen=True)
class CreateOptions:
    """What a ``create`` may add: extra environment (refused for a sealed template) and the wait for capacity."""

    envs: Mapping[str, str] | None = None
    wait_sec: float = 60


class Sandbox(Protocol):
    """One isolated execution environment, bound to the role that created it."""

    sandbox_id: str

    def info(self) -> SandboxInfo: ...

    def write_file(self, path: str, data: bytes) -> None: ...

    def read_file(self, path: str) -> bytes: ...

    def copy_out(self, path: str, dest: Path) -> int:
        """Stream a (possibly large) file to ``dest``; return the byte count."""
        ...

    def list_dir(self, path: str) -> list[str]: ...

    def run(
        self,
        cmd: str,
        *,
        background: bool = False,
        settings: RunSettings = RunSettings(),
        sinks: OutputSinks = OutputSinks(),
    ) -> CommandResult: ...

    def set_timeout(self, timeout_sec: int) -> None: ...

    def kill(self) -> None:
        """End the sandbox and everything running in it; safe to call again."""
        ...


class SandboxProvider(Protocol):
    """Creates, reconnects to and finds sandboxes."""

    name: str

    def create(
        self,
        template: str,
        *,
        role: Role,
        timeout_sec: int,
        metadata: Mapping[str, str],
        options: CreateOptions = CreateOptions(),
    ) -> Sandbox: ...

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox: ...

    def find(self, metadata: Mapping[str, str], *, role: Role) -> list[SandboxInfo]: ...
