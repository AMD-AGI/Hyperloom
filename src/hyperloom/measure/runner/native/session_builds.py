# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A single-branch session's native builds: the server of a framework built from its source, never a binary ``PATH``
happens to find.

A framework whose server is built from a checkout (:func:`~hyperloom.inference_optimizer.framework_registry.
native_component`) is built from ``FRAMEWORK_REPO_PATH`` at the ``HEAD`` it has when the session starts, its base, which
``native/base.json`` keeps for every later leg. A checkout whose tracked files differ from its ``HEAD`` is refused when
the session starts; a later leg takes the checkout as the session left it (a candidate kept, applied only, or left
pending by a crash) and runs the build the session accepted last. The builds go through the
:class:`~hyperloom.measure.runner.native.session.NativeBuildSession` the multi-branch toolbox builds through, into a
store in the session directory (``native/store``). Each process signs with a key of its own that never leaves it;
``native/trust.json`` keeps the public half of every leg's key, so a later leg reads the builds an earlier one made.

A build is of the tree as it stands: the checkout's ``HEAD`` with the given paths as the working tree has them, as one
patch over the base commit (:meth:`SessionNativeBuilds.build_tree`). A built tree's executables are laid once per build
under ``native/farms/<key>``. The session's accepted build (:meth:`SessionNativeBuilds.accept`) is what
``native/current`` links to, and that directory is first on this process's ``PATH`` from :func:`start_native_builds`
on, so every round of the session, the base's included, runs it; a round of a candidate puts its own build's directory
ahead of it.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path

from hyperloom.common.provenance import detect_gfx_arch
from hyperloom.contracts.recipe import PATCH_ROOT, ComponentPatches, PatchRef
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.framework_paths import GENERIC_FRAMEWORK_ROOT_ENV
from hyperloom.inference_optimizer.native_builds import NativeBuildsRefused
from hyperloom.measure import signing
from hyperloom.measure.compile import CompileError, CompiledEnvironment, LocalBlobStore
from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.pins import ImagePinError, active_image
from hyperloom.measure.jit.native.trust import Signer, Trust, TrustedKey
from hyperloom.measure.runner.native.builder import BUILD_PYTHON_ENV, BuildMachine
from hyperloom.measure.runner.native.isolated import build_cpus
from hyperloom.measure.runner.native.planning import NativeSite
from hyperloom.measure.runner.native.recipes import native_recipes
from hyperloom.measure.farm import FarmError, build_farm, built_roots
from hyperloom.measure.jit.native.laying import NamedBuild, NativeLayError, sheet_builds
from hyperloom.measure.runner.native.session import (
    NativeBuildSession,
    component_checkout,
    open_session,
    pinned_checkouts,
)
from hyperloom.measure.spec import EnvRoot as SpecEnvRoot
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild, TreeBuildStatus

__all__ = [
    "SessionNativeBuilds",
    "check_native_tree",
    "lay_executables",
    "open_native_builds",
    "start_native_builds",
    "tune_native_builds",
]

logger = logging.getLogger(__name__)

_NATIVE_DIR = "native"
_BASE_FILE = "base.json"
_TRUST_FILE = "trust.json"
_CURRENT = "current"
_SOURCES = "sources"
_CHECKS = "checks"
_TUNES = "tunes"
_KEY_EPOCH = 1
_GIT_TIMEOUT_SEC = 120
_UTF8 = "utf-8"


class SessionNativeBuilds:
    """The native builds of one single-branch session (see the module); integration's
    :class:`~hyperloom.orchestrator.actions.executors.candidate_builds.CandidateBuilds`."""

    def __init__(self, root: Path, component: str, checkout: Path, session: NativeBuildSession) -> None:
        self.root, self.component, self.checkout, self.session = root, component, checkout, session

    @property
    def base_commit(self) -> str:
        return self.session.planner.site.commits[self.component]

    @property
    def current(self) -> Path:
        """The directory the accepted build's executables are in, through the link :meth:`accept` moves."""
        return self.root / _CURRENT

    def build_tree(self, paths: Sequence[str] = ()) -> TreeBuild:
        """The build of the checkout's ``HEAD`` with its changed tracked files and ``paths`` as its working tree has
        them, over the base commit, its executables laid. A tree git cannot read, and a build whose executables cannot
        be laid, are the machine's failure."""
        return self._laid(*self._compiled(paths))

    def build_patch(self, patch: bytes) -> TreeBuild:
        """The build of ``patch``, a git patch over the base commit (empty for the base as it is), its executables laid
        as :meth:`build_tree` lays them."""
        stack = self._stack(patch)
        return self._laid(self._build(stack), stack)

    def read_upstream(self, path: str) -> str:
        """The text of ``path`` (a path in the upstream tree) at the base commit."""
        sources = self.session.planner.site.sources
        return (sources / self.component / self.base_commit / path).read_text(encoding=_UTF8)

    def _laid(self, found: TreeBuild, stack: ComponentPatches) -> TreeBuild:
        if found.status is not TreeBuildStatus.BUILT:
            return found
        environment = (
            self.session.environment(stack) if stack.patches else self.session.base_environment(self.component)
        )
        try:
            return replace(found, bin=self._lay(found.key, environment))
        except (FarmError, NativeLayError, OSError) as unlaid:
            logger.warning("the %s build %s cannot be laid: %s", self.component, found.key, unlaid)
            return replace(found, status=TreeBuildStatus.INFRA, detail=f"the build cannot be laid: {unlaid}")

    def check_tree(self, paths: Sequence[str] = ()) -> TreeBuild:
        """Whether the tree builds: :meth:`build_tree` without laying its executables, for a check that runs nothing."""
        return self._compiled(paths)[0]

    def _compiled(self, paths: Sequence[str]) -> tuple[TreeBuild, ComponentPatches]:
        """The build of the tree and its stack."""
        stack = self.session.planner.base_stack(self.component)
        try:
            stack = self._stack(_tree_patch(self.checkout, self.base_commit, paths))
        except NativeBuildsRefused as unreadable:
            return TreeBuild(TreeBuildStatus.INFRA, detail=str(unreadable)), stack
        return self._build(stack), stack

    def accept(self, build: TreeBuild) -> None:
        """Make ``build`` the one every later round runs."""
        staged = self.root / f".{_CURRENT}.{os.getpid()}"
        with contextlib.suppress(FileNotFoundError):
            staged.unlink()
        staged.symlink_to(build.bin, target_is_directory=True)
        os.replace(staged, self.current)

    def _stack(self, patch: bytes) -> ComponentPatches:
        base = self.session.planner.base_stack(self.component)
        if not patch:
            return base
        sha256 = self.session.compiler.blobs.put(patch)
        return replace(base, patches=(PatchRef(f"{PATCH_ROOT}/{self.component}/{sha256}.patch", sha256),))

    def _build(self, stack: ComponentPatches) -> TreeBuild:
        try:
            outcome = self.session.build(stack) if stack.patches else self.session.base(self.component)
        except CompileError as refused:
            return TreeBuild(TreeBuildStatus.COMPILE_ERROR, detail=str(refused))
        return TreeBuild(TreeBuildStatus(outcome.status.value), outcome.inputs.key, outcome.reused, outcome.detail)

    def _lay(self, key: str, environment: CompiledEnvironment) -> Path:
        return lay_executables(self.session, self.component, key, environment, self.root / "farms")


def lay_executables(
    session: NativeBuildSession, component: str, key: str, environment: CompiledEnvironment, farms: Path
) -> Path:
    """The directory of the executables of ``component``'s ready build ``key``, compiled to run as ``environment``, laid
    once below ``farms/<key>`` from ``session``'s store.

    Raises:
        FarmError: The build laid no executables of its own.
        NativeLayError: Its outputs in the store are not the ones its environment names.
    """
    farm_dir = farms / key
    laid = farm_dir / component / "bin"
    if laid.is_dir():
        return laid
    named = tuple(NamedBuild(ref.root_id, ref.key, ref.manifest_sha256) for ref in environment.build_refs)
    builds = sheet_builds(named, session.store, session.planner.site.machine.image_digest)
    roots = [SpecEnvRoot.parse(root.to_document(), "environment.roots") for root in environment.roots]
    staged = Path(tempfile.mkdtemp(prefix=f".{key[:16]}.", dir=_directory(farms)))
    farm = build_farm(built_roots(roots, builds), staged, python=sys.executable, read_blob=_no_blob)
    if farm is None or farm.bin != (staged / component / "bin",):
        raise FarmError(f"the {component} build {key} laid no executables of its own")
    staged.rename(farm_dir)
    return laid


def _no_blob(address: str) -> bytes:
    raise FileNotFoundError(f"an executable build lays no recipe file, and {address} was asked for")


def open_native_builds(session_dir: Path, framework: str, environ: Mapping[str, str]) -> SessionNativeBuilds | None:
    """The native builds of the session at ``session_dir`` when ``framework`` builds its server from its source;
    ``None`` when it does not.

    Raises:
        NativeBuildsRefused: No checkout of the component is named, a session that starts finds changes to its tracked
            files, or the build site cannot be read from ``environ`` (the image by digest, the GPU architecture, the
            build CPUs).
    """
    component = framework_registry.native_component(framework)
    if component is None:
        return None
    if not environ.get(GENERIC_FRAMEWORK_ROOT_ENV, "").strip():
        raise NativeBuildsRefused(
            f"--framework {framework} builds its server from its source: pass --framework-path with a {component} "
            "checkout"
        )
    try:
        return _open(Path(session_dir) / _NATIVE_DIR, component, environ)
    except BuildStoreConfigError as exc:
        raise NativeBuildsRefused(str(exc)) from None


def start_native_builds(
    session_dir: Path, framework: str, environ: MutableMapping[str, str]
) -> SessionNativeBuilds | None:
    """Open the session's native builds (:func:`open_native_builds`), build and accept the tree its checkout holds
    unless an earlier leg accepted a build, and put the accepted build first on ``environ``'s ``PATH``; ``None`` when
    ``framework`` builds none. The earlier leg's build is kept rather than the tree rebuilt: the tree may hold a
    candidate a crash left pending, which the resume takes out again.

    Raises:
        NativeBuildsRefused: As :func:`open_native_builds`, or the tree did not build.
    """
    native = open_native_builds(session_dir, framework, environ)
    if native is None:
        return None
    if not native.current.is_dir():
        built = native.build_tree()
        if built.status is not TreeBuildStatus.BUILT:
            raise NativeBuildsRefused(
                f"the {native.component} checkout {native.checkout} did not build ({built.status.value}): {built.detail}"
            )
        native.accept(built)
    path = [part for part in environ.get("PATH", os.defpath).split(os.pathsep) if part and part != str(native.current)]
    environ["PATH"] = os.pathsep.join([str(native.current), *path])
    return native


def _open(root: Path, component: str, environ: Mapping[str, str]) -> SessionNativeBuilds:
    """The builds over the commit ``base.json`` keeps, or, for a session that keeps none, over the ``HEAD`` of a
    checkout with no changes to its tracked files."""
    commit = _recorded_commit(root)
    if commit is None:
        pinned = next(
            (found for found in pinned_checkouts(native_recipes(), environ) if found.component == component), None
        )
        checkout, commit = (pinned.checkout, pinned.commit) if pinned is not None else (None, "")
    else:
        checkout = component_checkout(component, environ)
    if checkout is None:
        raise NativeBuildsRefused(f"{environ[GENERIC_FRAMEWORK_ROOT_ENV]} is not a checkout of {component}")
    machine, cpus = _environ_site(environ)
    _record_base(_directory(root), component, commit, machine, cpus)
    upstream = root / _SOURCES / component / commit
    if not upstream.is_dir():
        _directory(upstream.parent)
        _git(checkout, "worktree", "add", "--quiet", "--detach", str(upstream), commit)
    session = _site_session(root, root, {component: commit}, machine, cpus)
    return SessionNativeBuilds(root, component, checkout, session)


def check_native_tree(session_dir: Path, tree: Path, paths: Sequence[str] = ()) -> TreeBuild:
    """Whether ``tree``, a checkout of the framework the session at ``session_dir`` builds, builds over the session's
    base with its changed tracked files and ``paths`` (:meth:`SessionNativeBuilds.check_tree`), on the build site the
    session recorded, into a store of the session's checks (``native/checks``), apart from the one its integration
    reads.

    Raises:
        NativeBuildsRefused: The session builds no framework server from source.
    """
    return _recorded(Path(session_dir), _CHECKS, Path(tree)).check_tree(paths)


def tune_native_builds(session_dir: Path) -> SessionNativeBuilds:
    """The builds of patches over the base of the session at ``session_dir`` (:meth:`SessionNativeBuilds.build_patch`)
    that a sweep of a kernel knob times, on the build site the session recorded, into a store of the session's sweeps
    (``native/tunes``), apart from the one its integration reads.

    Raises:
        NativeBuildsRefused: The session builds no framework server from source.
    """
    return _recorded(Path(session_dir), _TUNES, None)


def _recorded(session_dir: Path, own: str, tree: Path | None) -> SessionNativeBuilds:
    """The builds of the session's recorded base and build site into the store below ``native/<own>``, of ``tree`` (the
    pinned upstream tree when ``None``)."""
    root = session_dir / _NATIVE_DIR
    path = root / _BASE_FILE
    if not path.is_file():
        raise NativeBuildsRefused(f"the session at {session_dir} builds no framework server from source: no {path}")
    base = json.loads(path.read_text(encoding=_UTF8))
    component, commit = str(base["component"]), str(base["commit"])
    machine = BuildMachine(base["image_digest"], base["image_digest"], base["gfx_arch"], base["python"])
    stored = _directory(root / own)
    session = _site_session(root, stored, {component: commit}, machine, str(base["build_cpus"]))
    return SessionNativeBuilds(stored, component, tree or root / _SOURCES / component / commit, session)


def _site_session(
    root: Path, own: Path, commits: Mapping[str, str], machine: BuildMachine, cpus: str
) -> NativeBuildSession:
    """The build session over the session's upstream checkouts (below ``root``) on ``machine``, compiling on ``cpus``,
    storing into ``own`` signed by a key of this process."""
    signer, trust = _session_key(own)
    site = NativeSite(machine, native_recipes(), root / _SOURCES, _directory(root / "image"), commits, cpus)
    return open_session(BuildStore(own / "store", trust), signer, site, LocalBlobStore(own / "patches"))


def _environ_site(environ: Mapping[str, str]) -> tuple[BuildMachine, str]:
    """The build machine and the build CPUs ``environ`` names."""
    image = _image_digest(environ)
    return BuildMachine(image, image, _gfx_arch(environ), _python(environ)), build_cpus(environ)


def _recorded_commit(root: Path) -> str | None:
    """The commit ``base.json`` keeps, the one the session builds over; ``None`` for a session that keeps none."""
    path = root / _BASE_FILE
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding=_UTF8))
    return str(document["commit"])


def _record_base(root: Path, component: str, commit: str, machine: BuildMachine, cpus: str) -> None:
    """Record ``commit`` in ``base.json`` with the build site this leg found, which a check of a specialist's tree
    builds on: its process is not given the session's environment."""
    path = root / _BASE_FILE
    site = {"image_digest": machine.image_digest, "gfx_arch": machine.gfx_arch, "python": machine.python}
    record = {"component": component, "commit": commit, "build_cpus": cpus, **site}
    staged = path.with_name(f".{_BASE_FILE}.{os.getpid()}")
    staged.write_text(json.dumps(record, sort_keys=True), encoding=_UTF8)
    os.replace(staged, path)


def _session_key(root: Path) -> tuple[Signer, Trust]:
    """A key of this process, and the trust of every key a leg of the session signed with (its own the last)."""
    key = signing.generate_private_key()
    public = signing.public_key_pem(key)
    path = root / _TRUST_FILE
    with _locked(root / f".{_TRUST_FILE}.lock"):
        keys = json.loads(path.read_text(encoding=_UTF8))["keys"] if path.is_file() else []
        keys.append({"public_key_pem": public, "epoch": _KEY_EPOCH})
        staged = path.with_name(f".{_TRUST_FILE}.{os.getpid()}")
        staged.write_text(json.dumps({"min_epoch": _KEY_EPOCH, "keys": keys}), encoding=_UTF8)
        os.replace(staged, path)
    trust = Trust(tuple(TrustedKey(item["public_key_pem"], int(item["epoch"])) for item in keys), _KEY_EPOCH)
    return Signer(key), trust


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _image_digest(environ: Mapping[str, str]) -> str:
    try:
        return active_image(environ).digest
    except ImagePinError as exc:
        raise NativeBuildsRefused(f"a native build is keyed by the image it runs in, and {exc}") from None


def _gfx_arch(environ: Mapping[str, str]) -> str:
    gfx = detect_gfx_arch(environ)
    if not gfx:
        raise NativeBuildsRefused(
            "a native build needs the GPU architecture: pass --gpu-type or set HYPERLOOM_GFX_ARCH"
        )
    return gfx


def _python(environ: Mapping[str, str]) -> str:
    return environ.get(BUILD_PYTHON_ENV, "").strip() or sys.executable


def _tree_patch(checkout: Path, base_commit: str, paths: Sequence[str]) -> bytes:
    """The checkout's ``HEAD`` with its changed tracked files and ``paths`` as its working tree has them, as one git
    patch over ``base_commit``; empty when they are the same tree. Read through an index of its own, so the checkout's
    is never touched."""
    with tempfile.TemporaryDirectory(prefix="hl-native-index-") as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        _git(checkout, "read-tree", "HEAD", env=env)
        stage = functools.partial(_git, checkout, "add", env=env)
        stage("--update")
        if paths:
            stage("--all", "--", *paths)
        return _git(checkout, "diff", "--cached", "--binary", "--no-renames", "--no-color", base_commit, env=env)


def _git(checkout: Path, *args: str, env: Mapping[str, str] | None = None) -> bytes:
    command = ["git", "-c", "safe.directory=*", "-c", "core.quotePath=false", "-C", str(checkout), *args]
    try:
        done = subprocess.run(command, capture_output=True, timeout=_GIT_TIMEOUT_SEC, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeBuildsRefused(f"git {args[0]} cannot read {checkout}: {exc}") from None
    if done.returncode != 0:
        stderr = done.stderr.decode(_UTF8, errors="replace").strip()[-500:]
        raise NativeBuildsRefused(f"git {' '.join(args[:2])} in {checkout} exited {done.returncode}: {stderr}")
    return done.stdout


def _directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
