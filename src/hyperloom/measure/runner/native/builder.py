# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build a component's patched native tree into the store.

Front: the stack is applied with :func:`~hyperloom.measure.compile.apply_patch_stack` over the build directory of the
pinned upstream checkout (never in place, never into the image), which yields the changed files and so the key. Middle:
under the key's lock, a copy of that directory with the changed files laid over it is compiled by the recipe, in an
environment of its own, within the recipe's budget. Back: the declared outputs are harvested and the store publishes them.

A compiler that exits non-zero, and outputs that are not what the recipe declares, are the candidate's ``compile_error``,
stored signed so the next test of the key gets the same answer. A failed preflight, a timeout and a compiler killed by a
signal are machine failures (:class:`~hyperloom.measure.jit.native.store.BuildInfraError`): nothing is stored.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches
from hyperloom.measure.compile import BlobStore, EnvRoot, PackageBase, apply_patch_stack
from hyperloom.measure.runner.native.harvest import HarvestError, harvest
from hyperloom.measure.jit.native.key import BuildInputs, native_digest
from hyperloom.measure.jit.native.record import EXECUTABLE, FAILED, READY
from hyperloom.measure.runner.native.recipes import NativeRecipe
from hyperloom.measure.runner.native.service import Ensured, Staged, ensure_build
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import Signer
from hyperloom.orchestrator.enablement.runtime.build_utils import run_argv

__all__ = ["BuildMachine", "BuildRequest", "build_native", "compile_tree", "plan", "tree_dir"]

COMPILE_ERROR = "compile_error"
_TREE = "tree"
_SOURCE_MODES = {EXECUTABLE: 0o700}
_SOURCE_MODE = 0o600
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True)
class BuildMachine:
    """Where the outputs run and where they are built."""

    image_digest: str
    builder_image_digest: str
    gfx_arch: str
    python: str = sys.executable


@dataclass(frozen=True)
class BuildRequest:
    """One stack of one component: ``upstream`` is the checkout of ``stack.upstream_ref``; ``image_root`` is the image's
    directory that holds the installed package (``site-packages``), which ``recipe.package_dir`` stands for in the tree:
    outputs equal to its files are not stored."""

    recipe: NativeRecipe
    stack: ComponentPatches
    upstream: Path
    image_root: Path
    machine: BuildMachine


def plan(request: BuildRequest, blobs: BlobStore) -> tuple[BuildInputs, EnvRoot]:
    """The key's inputs and the changed native files, from the stack alone (nothing is compiled)."""
    recipe, machine = request.recipe, request.machine
    base_dir = Path(request.upstream) / recipe.source_dir
    root = apply_patch_stack(
        request.stack, recipe.layout(), PackageBase(recipe.import_name, base_dir, None, None), blobs
    )
    inputs = BuildInputs(
        machine.image_digest,
        machine.builder_image_digest,
        recipe.component,
        machine.gfx_arch,
        native_digest(root.entries, base_dir),
        recipe.version,
    )
    return inputs, root


def build_native(request: BuildRequest, blobs: BlobStore, store: BuildStore, signer: Signer) -> Ensured:
    """The store's entry for ``request``, compiled when no request built it before."""
    inputs, root = plan(request, blobs)
    return ensure_build(store, inputs, lambda scratch: compile_tree(request, root, blobs, scratch), signer)


def tree_dir(scratch: Path) -> Path:
    """Where :func:`compile_tree` builds below its scratch directory."""
    return Path(scratch) / _TREE


def compile_tree(
    request: BuildRequest, root: EnvRoot, blobs: BlobStore, scratch: Path, *, against_image: bool = True
) -> Staged:
    """Compile the upstream build directory with ``root`` laid over it, and harvest its outputs.

    With ``against_image`` false every output is kept, not only those that differ from the image's.

    Raises:
        BuildInfraError: When the preflight fails, the compile times out or is killed by a signal.
    """
    recipe = request.recipe
    tree = tree_dir(scratch)
    shutil.copytree(Path(request.upstream) / recipe.source_dir, tree, symlinks=True)
    _lay(tree, root, blobs)
    checked = recipe.preflight(Path(scratch), request.machine.gfx_arch)
    if checked.problem is not None:
        raise BuildInfraError(f"the builder failed its preflight: {checked.problem}")
    toolchain = {**checked.toolchain, "gfx": request.machine.gfx_arch, "max_jobs": str(recipe.max_jobs)}
    started, cpu_before = time.monotonic(), _children_cpu_seconds()
    result = run_argv(
        recipe.argv(request.machine.python),
        cwd=tree,
        env=recipe.environment(request.machine.gfx_arch, Path(scratch)),
        timeout_sec=recipe.budget_sec,
    )
    timing = (_children_cpu_seconds() - cpu_before, time.monotonic() - started)
    log = f"{result.stdout_tail}\n{result.stderr_tail}"
    if result.timed_out or result.returncode < 0:
        raise BuildInfraError(f"the compile of {recipe.component} timed out or was killed (rc={result.returncode})")
    if result.returncode != 0:
        return _failed(f"{COMPILE_ERROR}: the compile exited {result.returncode}", toolchain, timing, log, scratch)
    image = Path(request.image_root) if against_image else None
    try:
        outputs = harvest(tree / recipe.package_dir, recipe.outputs, image, recipe.max_output_bytes, Path(scratch))
    except HarvestError as exc:
        return _failed(f"{COMPILE_ERROR}: {exc}", toolchain, timing, log, scratch)
    return Staged(READY, outputs, toolchain, timing[0], timing[1], log, Path(scratch))


def _failed(failure: str, toolchain: dict[str, str], timing: tuple[float, float], log: str, scratch: Path) -> Staged:
    return Staged(FAILED, (), toolchain, timing[0], timing[1], log, Path(scratch), failure=failure)


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _lay(tree: Path, root: EnvRoot, blobs: BlobStore) -> None:
    """Lay the stack's changed files over the copied tree: each written new (a symlink there is replaced, never
    followed), or removed."""
    for entry in root.entries:
        target = tree / entry.path
        if os.path.lexists(target):
            target.unlink()
        if entry.op != "upsert":
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, _WRITE_FLAGS, _SOURCE_MODES.get(entry.mode or "", _SOURCE_MODE))
        with os.fdopen(descriptor, "wb") as out:
            out.write(blobs.get(entry.blob or ""))
