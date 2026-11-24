# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's native builds: the one assembly the multi-branch toolbox builds through.

:func:`open_session` puts together what every build of a session shares: the store and the key its entries are signed
with, the planner that keys each stack (:class:`~hyperloom.measure.runner.native.planning.NativePlanner`), the plan of
aiter's touched modules, and the build runner that compiles out of this process
(:class:`~hyperloom.measure.runner.native.isolated.IsolatedBuilds`). A host that cannot isolate a compile fails each
build as the machine's, and is asked again at each build: bubblewrap may answer later.

:meth:`NativeBuildSession.base` builds a component's pinned upstream commit as it is, :meth:`NativeBuildSession.build`
a stack of patches over it, each through :func:`~hyperloom.measure.runner.native.service.ensure_build` (one build per
key, the key's lock held, the entry signed here). The answer is a :class:`BuildOutcome`: built, the candidate's compile
error (stored, so the next request of the key gets it too), or the machine's failure (nothing stored; never the
candidate's). :meth:`NativeBuildSession.environment` compiles a stack into the environment a test runs in
(:meth:`~hyperloom.measure.compile.EnvironmentCompiler.compile`), its builds named once they are ready;
:meth:`NativeBuildSession.base_environment` is the one of the pinned commit as it is, which runs its build when the
build makes executables.

A component whose recipe builds a checkout the operator pinned (:attr:`~hyperloom.measure.runner.native.recipes.
CommitSource.PINNED_CHECKOUT`) is built at the ``HEAD`` of the framework checkout the session names, when that checkout
is the component's (:func:`~hyperloom.measure.compile.locate_source`): :func:`pinned_checkouts`. A checkout whose
tracked files differ from its ``HEAD`` is refused there, since its builds would not be of what the operator sees; a
session that already pinned its commit finds the checkout again with :func:`component_checkout`, whatever it changed in
it since.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, Recipe, RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    BlobStore,
    CompiledEnvironment,
    CompileError,
    EnvironmentCompiler,
    UnsupportedComponentError,
    locate_checkout,
)
from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.laying import manifest_sha256
from hyperloom.measure.jit.native.needs import BuildNeed, BuildNotReady, BuildRef
from hyperloom.measure.jit.native.record import FAILED, READY, BuildRecord
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused, Signer
from hyperloom.measure.runner.native.aiter_modules import AITER_COMPONENT, AITER_RECIPE_FACTORY, AiterModules
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_UID_NETNS,
    ChildInterpreter,
    IsolatedBuilds,
    Isolation,
    Unisolated,
    local_isolation,
)
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.measure.runner.native.recipes import RECIPE_FACTORIES, CommitSource, NativeRecipe
from hyperloom.measure.runner.native.service import ensure_build

__all__ = [
    "BuildOutcome",
    "BuildStatus",
    "NativeBuildSession",
    "PinnedCheckout",
    "component_checkout",
    "open_session",
    "pinned_checkouts",
]

logger = logging.getLogger(__name__)
_GIT_TIMEOUT_SEC = 60


class BuildStatus(Enum):
    """What a build came to: its outputs are stored, the candidate's compile failed, or the machine failed it."""

    BUILT = "built"
    COMPILE_ERROR = "compile_error"
    INFRA = "infra"


@dataclass(frozen=True)
class BuildOutcome:
    """One build: the inputs that key it, the stored entry (none when the machine failed it), the compile's failure or
    the machine's reason (``detail``), and whether another request built the entry first (``reused``)."""

    status: BuildStatus
    inputs: BuildInputs
    record: BuildRecord | None = None
    detail: str = ""
    reused: bool = False


@dataclass(frozen=True)
class NativeBuildSession:
    """A session's native builds (see the module)."""

    store: BuildStore
    signer: Signer
    planner: NativePlanner
    modules: AiterModules
    runner: IsolatedBuilds
    compiler: EnvironmentCompiler

    def base(self, component: str) -> BuildOutcome:
        """The build of ``component``'s upstream commit as the site pins it.

        Raises:
            UnsupportedComponentError: The commit the site pins has no checkout.
        """
        return self._ensure(self.planner.base(component).inputs)

    def build(self, stack: ComponentPatches) -> BuildOutcome:
        """The build of ``stack``, which must change a native file of its component.

        Raises:
            CompileError: The stack changes nothing a native build takes, or something no build may take (its build
                definition, a build product, a file outside its layout), or its commit has no checkout.
        """
        planned = self.planner(stack)
        if planned is None:
            raise CompileError(f"the stack of {stack.component} changes no native file; it needs no native build")
        return self._ensure(planned.inputs)

    def environment(self, stack: ComponentPatches) -> CompiledEnvironment:
        """The environment a test of ``stack`` runs in, each build it needs named by its stored entry.

        Raises:
            CompileError: The stack cannot be compiled.
            BuildNotReady: A build it needs is not ready in the store (:meth:`build` it first).
        """
        recipe = Recipe(base_image_digest=self.compiler.active_image_digest, components=(stack,))
        compiled = self.compiler.compile(recipe, RecipeDelta())
        return compiled.with_builds(tuple(self._ready(need) for need in compiled.builds))

    def base_environment(self, component: str) -> CompiledEnvironment:
        """The environment a test of ``component``'s pinned commit as it is runs in: the build :meth:`base` made, when it
        makes executables (the server runs those, never whatever ``PATH`` finds); the image as it is otherwise.

        Raises:
            UnsupportedComponentError: As :meth:`base`.
            BuildNotReady: The build is not ready in the store (:meth:`base` it first).
        """
        planned = self.planner.base(component)
        recipe = Recipe(base_image_digest=self.compiler.active_image_digest)
        compiled = self.compiler.compile(recipe, RecipeDelta())
        need = BuildNeed(component, planned.inputs)
        root = self.compiler.native_root(self.planner.base_stack(component), planned)
        return replace(compiled, roots=(root,), builds=(need,), build_refs=(self._ready(need),))

    def _ready(self, need: BuildNeed) -> BuildRef:
        record = self.store.read(need.inputs)
        if record is None or record.status != READY:
            raise BuildNotReady(f"the {need.inputs.component} build {need.inputs.key} is not ready in the store")
        return BuildRef(need.root_id, need.inputs.key, manifest_sha256(record))

    def _ensure(self, inputs: BuildInputs) -> BuildOutcome:
        try:
            ensured = ensure_build(self.store, inputs, self.runner(inputs), self.signer)
        except (BuildInfraError, BuildRefused) as exc:
            logger.warning("the %s build %s failed as the machine's: %s", inputs.component, inputs.key, exc)
            return BuildOutcome(BuildStatus.INFRA, inputs, detail=str(exc))
        record = ensured.record
        if record.status == FAILED:
            return BuildOutcome(BuildStatus.COMPILE_ERROR, inputs, record, record.failure, ensured.reused)
        return BuildOutcome(BuildStatus.BUILT, inputs, record, reused=ensured.reused)


def open_session(
    store: BuildStore,
    signer: Signer,
    site: NativeSite,
    blobs: BlobStore,
    isolation: Callable[[], Isolation | Unisolated] = local_isolation,
) -> NativeBuildSession:
    """The native builds of a session on ``site``, signed with ``signer`` into ``store``, isolated as ``isolation``
    finds at the start (and at each build while it finds nothing)."""
    planner = NativePlanner(site, blobs)
    modules = AiterModules(planner, history=partial(store.ready_records, AITER_COMPONENT, site.machine.image_digest))
    found = isolation()
    chosen = found if isinstance(found, Isolation) else isolation
    factories = {**RECIPE_FACTORIES, AITER_COMPONENT: AITER_RECIPE_FACTORY}
    runner = IsolatedBuilds(planner, blobs, chosen, factories, ChildInterpreter(cpus=site.build_cpus))
    _log_isolation(found)
    compiler = EnvironmentCompiler(blobs, site.machine.image_digest, native=planner, modules=modules)
    return NativeBuildSession(store, signer, planner, modules, runner, compiler)


def _log_isolation(isolated: Isolation | Unisolated) -> None:
    """Say once, at the start of the session, how this host isolates a compile (the level each build's record names)."""
    if isinstance(isolated, Unisolated):
        logger.warning(
            "native builds are configured and this host cannot isolate a compile now: each build asks again, and "
            "fails as the machine's while it cannot: %s",
            isolated.why,
        )
    elif isolated.bwrap is not None:
        logger.info(
            "native builds compile under bubblewrap %s as this process's user, in a network namespace of their own "
            "(%s): %s",
            isolated.bwrap.shown,
            isolated.level,
            "; ".join(isolated.bwrap.notes),
        )
    elif isolated.level == ISOLATION_UID_NETNS:
        logger.info(
            "native builds compile as user %s in a network namespace of their own (%s)", isolated.uid, isolated.level
        )
    else:
        logger.warning(
            "native builds compile as user %s with this host's network: it cannot make a network namespace (%s)",
            isolated.uid,
            isolated.level,
        )


@dataclass(frozen=True)
class PinnedCheckout:
    """The checkout ``checkout`` the operator pinned ``component`` to, at its ``HEAD`` ``commit``."""

    component: str
    commit: str
    checkout: Path


def pinned_checkouts(
    recipes: Mapping[str, NativeRecipe], environ: Mapping[str, str] = os.environ
) -> list[PinnedCheckout]:
    """The pinned checkout of each of ``recipes`` built from one, where the framework checkout ``environ`` names is it.

    Raises:
        BuildStoreConfigError: That checkout's ``HEAD`` cannot be read, or its tracked files differ from it.
    """
    found = []
    for component, recipe in recipes.items():
        if recipe.commit_source is not CommitSource.PINNED_CHECKOUT:
            continue
        checkout = component_checkout(component, environ)
        if checkout is not None:
            found.append(PinnedCheckout(component, _clean_head(checkout), checkout))
    return found


def component_checkout(component: str, environ: Mapping[str, str] = os.environ) -> Path | None:
    """The framework checkout ``environ`` names, when it is a checkout of ``component``."""
    try:
        return locate_checkout(COMPONENT_LAYOUTS[component], environ).path
    except UnsupportedComponentError as exc:
        logger.debug("%s is not built from the framework checkout: %s", component, exc)
        return None


def _clean_head(checkout: Path) -> str:
    head = _git(checkout, "rev-parse", "--verify", "HEAD")
    if len(head) != 40:
        raise BuildStoreConfigError(f"the framework checkout {checkout} has no commit at HEAD to build: {head!r}")
    changed = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise BuildStoreConfigError(
            f"the framework checkout {checkout} has changes to tracked files ({changed.splitlines()[0]} ...); its "
            "builds would be of its HEAD, not of what is there: commit or stash them"
        )
    return head


def _git(checkout: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(checkout), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildStoreConfigError(f"git cannot read the framework checkout {checkout}: {exc}") from None
    if done.returncode != 0:
        raise BuildStoreConfigError(f"git cannot read the framework checkout {checkout}: {done.stderr.strip()[-300:]}")
    return done.stdout.strip()
