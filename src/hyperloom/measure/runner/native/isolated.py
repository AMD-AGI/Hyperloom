# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CPU build runner of a native build: the compile runs out of the process that holds the signing key.

A compile runs code the candidate controls (its sources, and the build's own scripts over them), so it must not share a
process, a user or a network with the build service, which signs what the store holds. :class:`IsolatedBuilds` lays the
tree in this process (nothing of the candidate runs then), hands it to a child process (:mod:`.child`) that runs under
:class:`Isolation`, with none of this process's variables, and reads back only what the child staged: its record, as
untrusted data, and its output files, which the store copies by hash. The service then signs (:func:`~hyperloom.measure.
runner.native.service.ensure_build`).

:func:`local_isolation` is the minimal local isolation, the first of two tiers that works on this host:

1. The unprivileged user 65534, to whom the scratch directory is given, so the compile can read neither the signing key
   nor any credential of this user, in a network namespace of its own (no egress) where the host lets this process make
   one (``unshare --net`` needs a capability many pods do not have) (:data:`ISOLATION_UID_NETNS`, else
   :data:`ISOLATION_UID`). A host whose user namespace does not map 65534 (one that maps only its root) cannot take
   this tier: ``setpriv`` cannot become 65534 there, and under a fakeroot preload it would seem to while the compile ran
   as root (:func:`account_problem`).
2. Else a bubblewrap sandbox (:func:`compile_policy`, :data:`ISOLATION_BWRAP_NETNS`) as this process's own user, which
   starts from an empty root: only the work directory is writable, the toolchain, the image's packages and the tree the
   child imports from are read-only, the signing key and this user's credential locations do not exist inside (or are
   masked where a bound tree holds one), and it has a network of its own (loopback only).

Which one a build got is the ``isolation`` entry of its record's toolchain, written by this process, never by the
child. A host where neither works builds nothing: the build fails as the machine's (``BuildInfraError``), naming both
reasons (:func:`isolation_problem`). A remote build pool is another
:class:`~hyperloom.measure.runner.native.service.Build` behind the same ``runner(inputs)`` seam.

The child imports the builder from the tree this process runs from (``source``), not from whatever ``hyperloom`` the
interpreter's environment has installed, so an environment installed from an older tree cannot compile with old code; a
child that resolves the builder elsewhere refuses to run.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from hyperloom_common.bwrap import BwrapPolicy, BwrapPolicyError

import hyperloom
from hyperloom.common.agent_isolation import BwrapIsolation, NoIsolation, bwrap_isolation
from hyperloom.common.confinement import Confinement, credential_paths, toolchain_paths
from hyperloom.common.id_map import unmapped_ids
from hyperloom.measure import fsafe
from hyperloom.measure.compile import BlobStore
from hyperloom.measure.docs import DocumentError, Reader, parse_json_object, string_map
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import COMPILE_ENV_PREFIX, FAILED, READY, OutputFile
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.jit.store import SnapshotInvalid, read_bounded
from hyperloom.measure.runner.native.builder import LaidTree
from hyperloom.measure.runner.native.planning import NativePlanner, PlannedBuild
from hyperloom.measure.runner.native.process_group import run_in_group
from hyperloom.measure.runner.native.service import Build, Staged

__all__ = [
    "ChildInterpreter",
    "ISOLATION_BWRAP_NETNS",
    "ISOLATION_KEY",
    "ISOLATION_UID",
    "ISOLATION_UID_NETNS",
    "JOB_FILE",
    "SOURCE_ROOT",
    "STAGED_FILE",
    "IsolatedBuilds",
    "Isolation",
    "account_problem",
    "compile_policy",
    "isolation_problem",
    "job_document",
    "local_isolation",
    "staged_record",
    "unisolated",
]

JOB_FILE = "job.json"
STAGED_FILE = "staged.json"
NOBODY = 65534
#: The toolchain entry of a build's record that says how its compile was isolated, and the levels it names.
ISOLATION_KEY = "isolation"
ISOLATION_UID_NETNS = "uid+netns"
ISOLATION_UID = "uid"
ISOLATION_BWRAP_NETNS = "bwrap+netns"
#: The level of a stand-in that runs the compile as this process's own user (a test's; never a deployment's).
_NOT_ISOLATED = "none"
#: The directory this process imports ``hyperloom`` from: the child imports the builder from the same tree.
SOURCE_ROOT = Path(hyperloom.__file__).resolve().parent.parent
_PROBE_TIMEOUT_SEC = 10
#: What the child runs under ``python -I``: the builder of ``argv[1]``'s tree, refused if it resolves anywhere else.
_CHILD_BOOTSTRAP = """\
import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
from hyperloom.measure.runner.native import child
if not Path(child.__file__).resolve().is_relative_to(source):
    sys.exit(f"the build child resolved the builder to {child.__file__}, not to the tree under test {source}")
sys.exit(child.main(sys.argv[2:]))
"""
_WORK = "work"
_MARGIN_SEC = 300
_STDERR_TAIL = 500
_INFRA = "infra"
#: The job is read by the compile alone, which owns it once the work directory is given to it (:func:`_give`).
_JOB_MODE = 0o600
#: The permission another user needs to reach a directory below one: search, not read.
_PASS_THROUGH = stat.S_IXOTH
#: All of ``/sys`` a compile reads under bubblewrap: the CPU tree (how many CPUs it may use), nothing of the GPUs.
_COMPILE_SYS = (Path("/sys/devices/system/cpu"),)


@dataclass(frozen=True)
class Isolation:
    """How a compile is cut off from this process: ``prefix`` starts it, as ``uid`` (``None``: this process's own, which
    a test's stand-in may use and a deployment must not, unless ``bwrap`` sandboxes it); ``level`` is what its record
    says of it (:data:`ISOLATION_KEY`). With ``bwrap`` the compile runs in that bubblewrap's sandbox
    (:func:`compile_policy`) instead of under ``prefix``."""

    prefix: tuple[str, ...]
    uid: int | None
    level: str = _NOT_ISOLATED
    bwrap: BwrapIsolation | None = None


def private_network(unshare: str) -> bool:
    """Whether this process can start a process in a network namespace of its own: ``unshare --net true`` is run once."""
    try:
        done = subprocess.run(
            [unshare, "--net", "--", "true"], capture_output=True, timeout=_PROBE_TIMEOUT_SEC, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def account_problem() -> str | None:
    """Why this host cannot run a compile as user 65534: this process is not root, has no ``setpriv``, or its user
    namespace does not map 65534 (``setpriv`` would fail there, or under a fakeroot preload seem to succeed and leave the
    compile running as root); ``None`` when it can."""
    euid = os.geteuid()
    if euid != 0:
        return f"it runs as uid {euid}, not root, and only root can compile as user {NOBODY}"
    if shutil.which("setpriv") is None:
        return f"setpriv (util-linux) is not on PATH, and the compile runs as user {NOBODY} through it"
    unmapped = unmapped_ids((NOBODY,), (NOBODY,))
    if unmapped is not None:
        return (
            f"{unmapped}: setpriv cannot become user {NOBODY} there, and under a fakeroot preload it would seem to "
            "while the compile ran as root"
        )
    return None


def unisolated(account: str, bwrap: NoIsolation) -> str:
    """The reasons a compile can be isolated neither as user 65534 (``account``) nor under bubblewrap (``bwrap``)."""
    return f"it can compile neither as user {NOBODY} ({account}) nor under bubblewrap ({bwrap.why})"


def isolation_problem() -> str | None:
    """Why this host can isolate a compile in neither tier (see the module), naming both reasons; ``None`` when one
    works."""
    account = account_problem()
    if account is None:
        return None
    found = bwrap_isolation()
    return None if isinstance(found, BwrapIsolation) else unisolated(account, found)


def local_isolation(network_probe: Callable[[str], bool] = private_network) -> Isolation | None:
    """User 65534 (``setpriv``), in a network namespace of its own (``unshare --net``) where ``network_probe`` finds the
    host lets this process make one; else, where this host cannot change the user (:func:`account_problem`), a
    bubblewrap sandbox with a network of its own where bubblewrap's probe passes (its ``notes`` say why not as 65534);
    ``None`` when neither works (:func:`isolation_problem` says why)."""
    setpriv = shutil.which("setpriv")
    account = account_problem()
    if setpriv is None or account is not None:
        found = bwrap_isolation()
        if isinstance(found, NoIsolation):
            return None
        noted = replace(found, notes=(f"compiling as user {NOBODY}: {account}",))
        return Isolation((), None, ISOLATION_BWRAP_NETNS, noted)
    drop = (setpriv, f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--no-new-privs", "--")
    unshare = shutil.which("unshare")
    if unshare is not None and network_probe(unshare):
        return Isolation((unshare, "--net", "--", *drop), NOBODY, ISOLATION_UID_NETNS)
    return Isolation(drop, NOBODY, ISOLATION_UID)


@dataclass(frozen=True)
class ChildInterpreter:
    """What runs the build child: ``python`` under ``-I``, importing the builder from ``source`` (by default this process's
    own interpreter and the tree it runs from)."""

    python: str = sys.executable
    source: Path = SOURCE_ROOT

    def command(self, work: Path) -> list[str]:
        """The child's command line for the laid tree in ``work``."""
        return [self.python, "-I", "-c", _CHILD_BOOTSTRAP, str(self.source), str(work)]


def compile_policy(work: Path, child: ChildInterpreter, laid: LaidTree, environ: Mapping[str, str]) -> BwrapPolicy:
    """The bubblewrap sandbox of the compile of ``laid`` in ``work`` by ``child``, whose environment is ``environ``.

    Writable: ``work`` alone. Read-only: what running the child's interpreter and the image's (which runs the build)
    needs under ``environ`` (:func:`~hyperloom.common.confinement.toolchain_paths`: the system roots, this process's
    interpreter's prefixes (the child's by default), the import path of the first ``python3`` on ``PATH``, every
    ``PATH`` entry, and each interpreter's own directory), the tree the child imports the builder from, the image's
    packages and the upstream build directory. ``/sys`` is only the CPU tree, ``/dev`` only the base nodes
    (``/dev/null`` and the like, no GPU), and the network is its own (loopback only). Every credential location of this
    process and the files it registers (the signing key among them) is absent, or masked where a bound tree holds it
    (:func:`~hyperloom.common.confinement.credential_paths`).

    ``--die-with-parent`` is kept on whatever thread starts the compile: that thread waits for it, so it cannot end
    first, and only that flag lets a timeout's kill of bubblewrap end the sandbox, which runs in a session of its own.

    Raises:
        BwrapPolicyError: A path the compile reads is reached through ``work``, which the compile writes.
    """
    writable = (work,)
    readable = (
        *toolchain_paths(environ, child.python, writable=writable),
        *toolchain_paths(environ, laid.machine.python, writable=writable),
        child.source,
        laid.image_root,
        laid.tracked,
    )
    masked = credential_paths(environ, allowed=(*readable, *writable), writable=writable)
    confinement = Confinement(
        writable=writable, readable=readable, masked=masked, sys_paths=_COMPILE_SYS, own_network=True
    )
    return replace(confinement.policy(chdir=work), die_with_parent=True)


def job_document(laid: LaidTree, factory: str) -> dict[str, Any]:
    """What the child needs to compile ``laid``: the recipe by the factory that makes it, and the laid tree's facts."""
    machine = laid.machine
    return {
        "factory": factory,
        "machine": {
            "image_digest": machine.image_digest,
            "builder_image_digest": machine.builder_image_digest,
            "gfx_arch": machine.gfx_arch,
            "python": machine.python,
        },
        "tracked": str(laid.tracked),
        "image_root": str(laid.image_root),
        "laid": sorted(laid.laid),
        "environment": dict(laid.environment),
    }


class IsolatedBuilds:
    """``runner(inputs) -> Build`` for :class:`~hyperloom_multibranch.toolbox.store_builds.StoreBuildDriver` (see the
    module). ``factories`` name each component's recipe factory as ``module:function``; ``child`` says what runs the
    child."""

    def __init__(
        self,
        planner: NativePlanner,
        blobs: BlobStore,
        isolation: Isolation | None,
        factories: Mapping[str, str],
        child: ChildInterpreter = ChildInterpreter(),
    ) -> None:
        self._planner, self._blobs, self._isolation = planner, blobs, isolation
        self._factories, self._child = factories, child

    def __call__(self, inputs: BuildInputs) -> Build:
        return partial(self._build, self._planner.planned(inputs))

    def _build(self, planned: PlannedBuild, scratch: Path) -> Staged:
        isolation = self._isolation
        if isolation is None:
            why = isolation_problem() or "the build runner was given no isolation"
            raise BuildInfraError(f"this host cannot isolate a native build: {why}; nothing was compiled")
        work = Path(scratch) / _WORK
        work.mkdir()
        laid = planned.lay(self._blobs, work)
        _write_job(work, job_document(laid, self._factories[laid.recipe.component]))
        command, env = self._command(isolation, work, laid)
        started, cpu_before = time.monotonic(), _children_cpu_seconds()
        try:
            done = run_in_group(
                command, cwd=work, env=env, capture_output=True, text=True, timeout=laid.recipe.budget_sec + _MARGIN_SEC
            )
        except subprocess.TimeoutExpired:
            raise BuildInfraError(f"the build runner of {laid.recipe.component} did not end in time") from None
        timing = (_children_cpu_seconds() - cpu_before, time.monotonic() - started)
        if done.returncode != 0:
            tail = done.stderr.strip()[-_STDERR_TAIL:]
            raise BuildInfraError(f"the build runner exited {done.returncode}: {tail}")
        staged = staged_record(work, timing)
        return replace(staged, toolchain=_service_toolchain(staged.toolchain, laid, isolation.level))

    def _command(self, isolation: Isolation, work: Path, laid: LaidTree) -> tuple[list[str], dict[str, str]]:
        """The child's command line under ``isolation`` and its environment; the work directory is given to the
        compile's user first, when it has one of its own."""
        env, command = _child_environment(work), self._child.command(work)
        if isolation.bwrap is None:
            if isolation.uid is not None:
                _give(work, isolation.uid)
            return [*isolation.prefix, *command], env
        try:
            return compile_policy(work, self._child, laid, env).argv(isolation.bwrap.probe, command), env
        except BwrapPolicyError as refused:
            raise BuildInfraError(f"bubblewrap cannot sandbox this compile: {refused}; nothing was compiled") from None


def _service_toolchain(child: Mapping[str, str], laid: LaidTree, level: str) -> dict[str, str]:
    """The record's toolchain: the child's facts, less any it wrote under the names this process writes (the variables of
    the compile and its isolation), which a compile running candidate code must not be able to claim."""
    own = {f"{COMPILE_ENV_PREFIX}{name}": value for name, value in laid.environment.items()}
    kept = {name: value for name, value in child.items() if not name.startswith(COMPILE_ENV_PREFIX)}
    return {**kept, **own, ISOLATION_KEY: level}


def _write_job(work: Path, job: Mapping[str, Any]) -> None:
    descriptor = os.open(work / JOB_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _JOB_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as out:
        json.dump(job, out)


def _give(work: Path, uid: int) -> None:
    """Make ``work`` and everything below it the compile user's, links included (never followed), and let that user pass
    through the scratch directory above it (which the service made private to itself) without listing or reading it."""
    scratch = work.parent
    os.chmod(scratch, stat.S_IMODE(os.stat(scratch).st_mode) | _PASS_THROUGH)
    os.lchown(work, uid, uid)
    for directory, names, files in os.walk(work):
        for name in (*names, *files):
            os.lchown(os.path.join(directory, name), uid, uid)


def _child_environment(work: Path) -> dict[str, str]:
    """The child's whole environment: no variable of this process (none of its credentials, proxies or key paths)."""
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(work), "TMPDIR": str(work), "LC_ALL": "C.UTF-8"}


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def staged_record(work: Path, timing: tuple[float, float]) -> Staged:
    """What the child staged, read as untrusted data: a record in the schema, never a link; the time is this process's
    own measurement of the child, not the child's word."""
    root_fd = fsafe.open_root(work)
    try:
        document = parse_json_object(read_bounded(root_fd, STAGED_FILE), what="staged build")
    except (SnapshotInvalid, DocumentError) as exc:
        raise BuildInfraError(f"the build runner staged nothing readable: {exc}") from None
    finally:
        os.close(root_fd)
    reader = Reader(document, "staged")
    try:
        infra = reader.optional(_INFRA)
        if infra is not None:
            raise BuildInfraError(f"the build runner failed: {str(infra)[:_STDERR_TAIL]}")
        status = str(reader.choice("status", (READY, FAILED)))
        staged = Staged(
            status=status,
            outputs=reader.sequence("outputs", OutputFile.from_document),
            toolchain=dict(reader.section("toolchain", string_map)),
            cpu_seconds=timing[0],
            wall_seconds=timing[1],
            log=reader.string("log"),
            staging=work,
            failure=reader.string("failure"),
        )
        reader.done()
    except DocumentError as exc:
        raise BuildInfraError(f"the build runner staged a record outside the schema: {exc}") from None
    return staged
