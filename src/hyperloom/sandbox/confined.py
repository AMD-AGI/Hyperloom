# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local test sandbox's processes in bubblewrap: what they write, what they read, and what they never see.

A test's server runs a branch's patches. Where a provider is given a :class:`RunnerPlan` and names no account for a
sandbox's processes (:class:`~hyperloom.sandbox.local.ProcessUsers`, the account tier), they run in bubblewrap where
the chooser gives it (:func:`~hyperloom.common.agent_isolation.choose_test_isolation`), and as before where it does
not, which is said once at WARNING; a hosted run refuses that instead. Each sandbox starts from an empty root: its
``fs/`` (the file API's ``/in``, ``/out`` and ``/work``) and a private ``/tmp`` and full-size ``/dev/shm`` are
writable, and so is a sealed sandbox's own ``private/``, where its runner reads its signing key and then removes it
(``measure.runner.run``) so that nothing it starts can sign; the toolchain of its launcher and command and the plan's
trees (the model, the Hugging Face cache, the runner cache, the compiled-code store, the framework snapshot) are
read-only; the app's credential locations those show are masked; a template with cards gets ``/dev/kfd`` and
``/dev/dri``; ``/sys`` is read-only; the network is shared, so its server listens on loopback for the app's benchmark
client. Nothing else of the host is there: not the app's state, another sandbox's directory or key, the ``private/`` of
a sandbox that is not sealed (the app's logs of its commands, which the app opens by name as they start), nor any
process of the app's.

The launcher runs inside, so its subreaper keeps what the command leaves in the command's tree, and the sandbox's id
(:data:`~hyperloom.sandbox.card_occupancy.SANDBOX_TAG_ENV`) is set inside only: bubblewrap's own processes carry no
mark, a reap's SIGTERM reaches the sandbox's processes with their grace, and the tag sweep and the card guard find
them by it as before. A sealed sandbox runs one command, in a pid namespace of its own; any other runs its commands in
one holder's (:class:`~hyperloom_common.bwrap.Holder`), so a later command sees and stops what an earlier one started,
as it would on the host.

A sandbox whose paths bubblewrap cannot bind safely runs as before, said at WARNING, and a hosted run refuses it: a
plan's path that is itself one of the app's credential locations (a home, a login file) is one, since a mask hides
what lies below a bind, never the bound path itself, and so is a plan whose maker refused a tree it found
(:attr:`RunnerPlan.refused`). A sealed sandbox, whose one launch finds its scratch as it was made, falls back to a
fresh ``/tmp`` of its own where a bubblewrap older than :data:`~hyperloom_common.bwrap.NESTED_SAFE` refuses binds below
a bound one.

Tests unconfined only because bubblewrap's probe gave no answer in time are so for that launch alone: the next
sandbox is chosen for again, until a choice is final.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import select
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom_common.bwrap import NESTED_SAFE, BwrapPolicy, BwrapPolicyError, Holder, Scratch

from hyperloom.common.agent_isolation import (
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
    choose_test_isolation,
)
from hyperloom.common.confinement import Confinement, credential_paths, toolchain_paths
from hyperloom.sandbox.card_occupancy import SANDBOX_TAG_ENV
from hyperloom.sandbox.socket import SandboxDenied

__all__ = [
    "GPU_DEVICES",
    "MARKS",
    "UNCONFINED_TESTS",
    "RunnerPlan",
    "SandboxConfinement",
    "SandboxShape",
    "SandboxConfiner",
    "sandbox_confiner",
]

log = logging.getLogger(__name__)

#: The cards' device nodes, bound into the sandbox of a template with cards (after the root binds, so not ``nodev``).
GPU_DEVICES = (Path("/dev/kfd"), Path("/dev/dri"))
#: Set inside only (``--setenv``): the sandbox's id, by which the reap, the tag sweep and the card guard find its
#: processes.
MARKS = (SANDBOX_TAG_ENV,)
#: The variable naming the app's state directory, which no bind may show.
STATE_DIR_ENV = "HYPERLOOM_STATE_DIR"
#: The shell a sandbox's commands run under.
SHELL = "/bin/sh"
#: What is said once when the tests of this host run unconfined.
UNCONFINED_TESTS = "tests run unconfined on this host, as before: they can read this account's files and credentials"


_said: set[str] = set()
_said_lock = threading.Lock()


def _warn_once(message: str) -> None:
    with _said_lock:
        if message in _said:
            return
        _said.add(message)
    log.warning("%s", message)


@dataclass(frozen=True)
class RunnerPlan:
    """What a confined test sandbox's processes read besides their toolchain, their own ``fs/`` and their own key:
    ``readable`` trees (the model, the Hugging Face cache, the runner cache, the compiled-code store, the framework
    snapshot); ``app_only`` paths no bind may show, nor anything above them (the app's state directory and the
    provider's directory of sandboxes, which holds every sandbox's key, are added); ``devices``, bound for a template
    with cards. ``refused`` says why a tree its maker found it must read cannot be bound safely (one in a credential
    location, whose credentials no mask would hide); a sandbox of a plan that says any runs as before, said at
    WARNING, and a hosted run refuses it."""

    readable: tuple[Path, ...] = ()
    app_only: tuple[Path, ...] = ()
    devices: tuple[Path, ...] = GPU_DEVICES
    refused: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxShape:
    """What a sandbox's policy is made of: its ``directory``, the environment ``env`` its processes run with, its
    template's startup ``command`` (if any), whether it has cards (``gpus``), and whether it is ``sealed`` (it runs that
    one command; any other runs several, which share one holder)."""

    directory: Path
    env: Mapping[str, str]
    command: Sequence[str] = ()
    gpus: bool = False
    sealed: bool = True


@dataclass
class SandboxConfinement:
    """One sandbox's bubblewrap, policy and private ``/tmp`` and ``/dev/shm``; ``shared`` when its commands join one
    holder, started for the first and again after one that ended."""

    isolation: BwrapIsolation
    policy: BwrapPolicy
    scratch: Scratch
    shared: bool
    holder: Holder | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @contextlib.contextmanager
    def launched(
        self, argv: Sequence[str], env: Mapping[str, str], cwd: Path
    ) -> Iterator[tuple[list[str], dict[str, str], tuple[int, ...]]]:
        """``argv`` run with ``env`` in ``cwd`` inside the sandbox: bubblewrap's command line, the environment to start
        it with (the marks moved inside) and the descriptors to pass it, open until the block ends.

        Raises:
            IsolationRefused: The policy is one this bubblewrap must not launch.
            BwrapSetupError: The holder could not be set up.
            TimeoutError: The holder neither started nor failed in time.
        """
        policy = dataclasses.replace(self.policy, chdir=str(cwd))
        if not self.shared:
            command, child_env = self.isolation.launch(argv, policy, env, marks=MARKS)
            yield command, child_env, ()
            return
        with self._holder().joined(policy) as joined:
            command, child_env = self.isolation.launch(argv, joined, env, marks=MARKS)
            yield command, child_env, joined.join or ()

    def _holder(self) -> Holder:
        """The holder the commands join; a new one when none runs or the last has ended, which ended every process
        in it."""
        with self._lock:
            holder = self.holder
            if holder is not None and holder.pidfd >= 0 and not select.select([holder.pidfd], [], [], 0)[0]:
                return holder
            if holder is not None:
                holder.stop()
            self.holder = Holder.start(self.isolation.probe, self.policy)
            return self.holder

    def release(self) -> None:
        """End the holder, and every process still in it, and delete the scratch."""
        with self._lock:
            holder, self.holder = self.holder, None
        if holder is not None:
            holder.stop()
        try:
            self.scratch.remove()
        except OSError as exc:
            log.warning("the scratch of a test sandbox could not be removed: %s", exc)


def _within(path: Path, trees: Sequence[Path]) -> bool:
    """Whether ``path``, as given, is one of ``trees`` or below one."""
    given = Path(os.path.normpath(path))
    return any(given == tree or tree in given.parents for tree in (Path(os.path.normpath(t)) for t in trees))


def _credential_locations(paths: Sequence[Path], env: Mapping[str, str], trees: Sequence[Path]) -> tuple[str, ...]:
    """Why each of ``paths`` that is, resolved, one of the credential locations of the app or of the sandbox's
    environment ``env`` (a home, a login file or directory; :func:`~hyperloom.common.confinement.credential_paths`
    with nothing allowed names each, whole) must not be bound: a mask hides what lies below a bound path, never the
    path itself, so its bind would show every credential in it."""
    locations = set(credential_paths(env, writable=trees))
    return tuple(
        f"{path} is a credential location, which its bind would show unmasked"
        for path in paths
        if Path(os.path.realpath(path)) in locations
    )


class SandboxConfiner:
    """How a provider confines the sandboxes it names no account for: with ``isolation`` (bubblewrap, or unconfined
    as before), by ``plan``; ``base`` is the provider's directory of sandboxes, ``python`` its launcher's
    interpreter."""

    def __init__(self, isolation: BwrapIsolation | NoIsolation, plan: RunnerPlan, base: Path, python: str) -> None:
        self.isolation = isolation
        self.plan = plan
        self.base = base
        self.python = python
        self._lock = threading.Lock()

    def _chosen(self) -> BwrapIsolation | NoIsolation:
        """The isolation of the sandbox made now: chosen again while the last choice was unconfined only because
        bubblewrap's probe gave no answer in time (which the probe does not keep), as the first was otherwise.

        Raises:
            IsolationRefused: In a hosted run, where bubblewrap cannot sandbox a test.
        """
        with self._lock:
            if isinstance(self.isolation, NoIsolation) and self.isolation.transient:
                self.isolation = choose_test_isolation()
                if isinstance(self.isolation, NoIsolation):
                    _warn_once(f"{UNCONFINED_TESTS} ({self.isolation.why})")
            return self.isolation

    def confine(self, sandbox: SandboxShape) -> SandboxConfinement | None:
        """The confinement of ``sandbox``, whose processes run its command and commands under :data:`SHELL`. ``None``
        where it runs unconfined, as before: no bubblewrap here, or one that cannot bind its paths safely, which is
        said at WARNING.

        Raises:
            SandboxDenied: In a hosted run, where the sandbox's paths cannot be bound safely.
            IsolationRefused: In a hosted run, where bubblewrap cannot sandbox a test.
        """
        isolation = self._chosen()
        if not isinstance(isolation, BwrapIsolation):
            return None
        scratch = Scratch.create()
        try:
            found = self._policy(isolation, sandbox, scratch)
        except BaseException:
            scratch.remove()
            raise
        if isinstance(found, BwrapPolicy):
            return SandboxConfinement(isolation, found, scratch, shared=not sandbox.sealed)
        scratch.remove()
        failed = dataclasses.replace(isolation.probe, problems=found)
        try:
            unconfined = choose_test_isolation(known=failed)
        except IsolationRefused as refused:
            raise SandboxDenied(f"a hosted run's test sandbox must run confined: {refused}") from None
        why = unconfined.why if isinstance(unconfined, NoIsolation) else "; ".join(found)
        log.warning("the test sandbox at %s runs unconfined, as before: %s", sandbox.directory, why)
        return None

    def _policy(
        self, isolation: BwrapIsolation, sandbox: SandboxShape, scratch: Scratch
    ) -> BwrapPolicy | tuple[str, ...]:
        """The policy of every launch of the sandbox, or why bubblewrap must not launch it: ``fs/`` writable, and a
        sealed sandbox's ``private/``; the toolchain and the plan's trees read-only, but none in a tree it writes,
        which shows it already, none of the plan's that is a credential location, and none of a plan that refused a
        tree (:attr:`RunnerPlan.refused`); the credential locations they show masked; the devices; the app's state and
        the provider's directory never bound. No launch dies with the thread that started it: the sandbox's kill and
        timeout end it, with their grace."""
        directory, env = sandbox.directory, sandbox.env
        writable = (directory / "fs", *((directory / "private",) if sandbox.sealed else ()))
        trees = (*writable, Path(scratch.tmp), Path(scratch.shm))
        launcher_path = os.pathsep.join(filter(None, [str(Path(self.python).parent), env.get("PATH")]))
        toolchain = list(toolchain_paths({**env, "PATH": launcher_path}, self.python, writable=trees))
        for program in dict.fromkeys((*sandbox.command[:1], SHELL)):
            toolchain.extend(toolchain_paths(env, program, writable=trees))
        named = (*toolchain, *self.plan.readable)
        readable = tuple(path for path in dict.fromkeys(named) if not _within(path, writable))
        exposed = (*self.plan.refused, *_credential_locations(self.plan.readable, env, trees))
        if exposed:
            return exposed
        masked = credential_paths(env, allowed=(*readable, *writable), writable=trees)
        state = os.environ.get(STATE_DIR_ENV, "")
        confinement = Confinement(
            writable=writable,
            readable=readable,
            masked=masked,
            devices=self.plan.devices if sandbox.gpus else (),
            app_only=(*self.plan.app_only, self.base, *((Path(state),) if os.path.isabs(state) else ())),
        )
        version = isolation.probe.version or (0, 0, 0)
        try:
            policy = dataclasses.replace(confinement.policy(scratch=scratch), die_with_parent=False)
        except BwrapPolicyError as error:
            return error.problems
        problems = tuple(policy.problems(version))
        if problems and sandbox.sealed and version < NESTED_SAFE:
            fresh = dataclasses.replace(policy, tmp=None)
            if not fresh.problems(version):
                return fresh
        return problems or policy


def sandbox_confiner(plan: RunnerPlan, *, base: Path, python: str) -> SandboxConfiner:
    """How a provider with ``plan`` confines the sandboxes it names no account for
    (:func:`~hyperloom.common.agent_isolation.choose_test_isolation`); tests that will run unconfined are said once, at
    WARNING, with the reasons.

    Raises:
        IsolationRefused: In a hosted run, where bubblewrap cannot sandbox a test.
    """
    isolation = choose_test_isolation()
    if isinstance(isolation, NoIsolation):
        _warn_once(f"{UNCONFINED_TESTS} ({isolation.why})")
    return SandboxConfiner(isolation, plan, base, python)
