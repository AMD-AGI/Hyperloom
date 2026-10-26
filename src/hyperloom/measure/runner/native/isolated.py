# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CPU build runner of a native build: the compile runs out of the process that holds the signing key.

A compile runs code the candidate controls (its sources, and the build's own scripts over them), so it must not share a
process, a user or a network with the build service, which signs what the store holds. :class:`IsolatedBuilds` lays the
tree in this process (nothing of the candidate runs then), hands it to a child process (:mod:`.child`) that runs under
:class:`Isolation`, with none of this process's variables, and reads back only what the child staged: its record, as
untrusted data, and its output files, which the store copies by hash. The service then signs (:func:`~hyperloom.measure.
runner.native.service.ensure_build`).

:func:`local_isolation` is the minimal local isolation: a new network namespace (no egress) and the unprivileged user
65534, to whom the scratch directory is given, so the compile can read neither the signing key nor any credential of
this user. A host that cannot provide it builds nothing: the build fails as the machine's (``BuildInfraError``). A remote
build pool is another :class:`~hyperloom.measure.runner.native.service.Build` behind the same ``runner(inputs)`` seam.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from hyperloom.measure import fsafe
from hyperloom.measure.compile import BlobStore
from hyperloom.measure.docs import DocumentError, Reader, parse_json_object, string_map
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import FAILED, READY, OutputFile
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.jit.store import SnapshotInvalid, read_bounded
from hyperloom.measure.runner.native.builder import LaidTree, lay_tree
from hyperloom.measure.runner.native.planning import NativePlanner, Planned
from hyperloom.measure.runner.native.service import Build, Staged

__all__ = [
    "CHILD_MODULE",
    "JOB_FILE",
    "STAGED_FILE",
    "SGL_KERNEL_FACTORY",
    "IsolatedBuilds",
    "Isolation",
    "job_document",
    "local_isolation",
    "staged_record",
]

CHILD_MODULE = "hyperloom.measure.runner.native.child"
JOB_FILE = "job.json"
STAGED_FILE = "staged.json"
#: The recipe factory of sgl-kernel, as the child imports it.
SGL_KERNEL_FACTORY = "hyperloom.measure.runner.native.recipes:sgl_kernel_recipe"
NOBODY = 65534
_WORK = "work"
_MARGIN_SEC = 300
_STDERR_TAIL = 500
_INFRA = "infra"
_JOB_MODE = 0o644


@dataclass(frozen=True)
class Isolation:
    """How a compile is cut off from this process: ``prefix`` starts it, as ``uid`` (``None``: this process's own, which
    a test's stand-in may use and a deployment must not)."""

    prefix: tuple[str, ...]
    uid: int | None


def local_isolation() -> Isolation | None:
    """A new network namespace and user 65534 (``unshare --net``, ``setpriv``); ``None`` when this host cannot give it
    (not root, or either tool missing)."""
    unshare, setpriv = shutil.which("unshare"), shutil.which("setpriv")
    if os.geteuid() != 0 or unshare is None or setpriv is None:
        return None
    drop = (setpriv, f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--no-new-privs", "--")
    return Isolation((unshare, "--net", "--", *drop), NOBODY)


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
    }


class IsolatedBuilds:
    """``runner(inputs) -> Build`` for :class:`~hyperloom_multibranch.toolbox.store_builds.StoreBuildDriver` (see the
    module). ``factories`` name each component's recipe factory as ``module:function``; ``python`` runs the child and must
    import ``hyperloom``."""

    def __init__(
        self,
        planner: NativePlanner,
        blobs: BlobStore,
        isolation: Isolation | None,
        factories: Mapping[str, str],
        python: str = sys.executable,
    ) -> None:
        self._planner, self._blobs, self._isolation = planner, blobs, isolation
        self._factories, self._python = factories, python

    def __call__(self, inputs: BuildInputs) -> Build:
        return partial(self._build, self._planner.planned(inputs))

    def _build(self, planned: Planned, scratch: Path) -> Staged:
        isolation = self._isolation
        if isolation is None:
            raise BuildInfraError(
                "this host cannot isolate a native build (it needs root, unshare and setpriv to compile as an "
                "unprivileged user with no network); nothing was compiled"
            )
        work = Path(scratch) / _WORK
        work.mkdir()
        laid = lay_tree(planned.request, planned.root, self._blobs, work)
        _write_job(work, job_document(laid, self._factories[laid.recipe.component]))
        if isolation.uid is not None:
            _give(work, isolation.uid)
        started, cpu_before = time.monotonic(), _children_cpu_seconds()
        try:
            done = subprocess.run(
                [*isolation.prefix, self._python, "-I", "-m", CHILD_MODULE, str(work)],
                cwd=work,
                env=_child_environment(work),
                capture_output=True,
                text=True,
                timeout=laid.recipe.budget_sec + _MARGIN_SEC,
            )
        except subprocess.TimeoutExpired:
            raise BuildInfraError(f"the build runner of {laid.recipe.component} did not end in time") from None
        timing = (_children_cpu_seconds() - cpu_before, time.monotonic() - started)
        if done.returncode != 0:
            tail = done.stderr.strip()[-_STDERR_TAIL:]
            raise BuildInfraError(f"the build runner exited {done.returncode}: {tail}")
        return staged_record(work, timing)


def _write_job(work: Path, job: Mapping[str, Any]) -> None:
    descriptor = os.open(work / JOB_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _JOB_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as out:
        json.dump(job, out)


def _give(work: Path, uid: int) -> None:
    """Make ``work`` and everything below it the compile user's, links included (never followed)."""
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
