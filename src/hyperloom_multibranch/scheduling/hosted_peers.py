# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session measures on: runner sandboxes of the host's sandbox service, which install Hyperloom from the
runner cache of the commit the run executes.

A hosted session's peers are the ones :func:`~hyperloom_multibranch.scheduling.local_peers.local_peers` builds -- the
repository, the specialists and the MCP service run in the application's container -- on one other
:class:`~hyperloom_multibranch.scheduling.local_peers.ToolboxBackend`, :class:`HostedSandboxes`: every test runs in a
runner sandbox of the service (:func:`~hyperloom_multibranch.toolbox.hosted_toolbox.build_hosted_toolbox`), and the
toolbox keeps its journal, archive and landed output under ``MB-toolbox``. A runner sandbox has no network, so
:class:`PinnedToolboxInputs` clones the run's source into ``MB-runner-source`` and has the commit's runner cache entry
built from a private copy of the commit (:func:`private_copy`, :func:`runner_cache_pin`), or finds it built, before
anything is measured, and pins every runner sandbox to that entry's manifest through the toolbox's inputs.

The dev sandboxes a hosted session lends its specialists and its autonomous agents run in ``dev`` sandboxes of the grant
(:func:`~hyperloom_multibranch.scheduling.local_peers.branch_sandboxes`): only their processes run there, and their
files stay in the session directory, which the ``dev`` template mounts at the path it has here. PRELUDE's root tests
and profile of the root run in runner sandboxes too (:mod:`~hyperloom_multibranch.scheduling.prelude`).

The grant is either injected by a host (``hosted run``) or handed to a session started by hand (``run --measurement
hosted``); a grant without a run id of its own names its sandboxes by the session's id.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from hyperloom.measure.runner.cache_build import CacheBuildError, WheelTarget, runner_wheel_target
from hyperloom_multibranch.repo.isolated_git import INIT, git_argv, git_env
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.local_peers import attempt_timeout_sec, measurement_host, peers_on
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeerSetup
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom.hosted.host_info import HostInfo
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom_multibranch.blackboard import LocalBlackboard
    from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolboxInputs
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox

#: The toolbox's inputs for the session's layout and the workload's tensor-parallel size: the hosted entry binds the
#: host's inputs and the runner cache pin into it (:class:`PinnedToolboxInputs`).
ToolboxInputsFor = Callable[[SessionLayout, int], "HostedToolboxInputs"]

#: The umask the runner cache is built under: the sandbox reads the cache as another account.
RUNNER_CACHE_UMASK = 0o022
#: How long fetching the run's source may take before the run fails.
SOURCE_FETCH_TIMEOUT_SEC = 1800
#: ``HYPERLOOM_SOURCE_REPO`` as ``OWNER/NAME``, a repository on GitHub; any other value is a git URL or path.
_GITHUB_NAME = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
_QUIET = "--quiet"
_GIT = "git"
_FETCH = "fetch"
#: The clone's top directory once whole: the next leg's account reads it, and no account but this one writes it.
_SOURCE_MODE = 0o755


@dataclass(frozen=True)
class HostedSandboxes:
    """Tests run in runner sandboxes of the host's sandbox service, on the inputs ``inputs`` gives for the session's
    ``MB-toolbox`` directory and the workload's TP; a finished test's sandbox is kept for the next test as the
    session's ``make_before_break`` says (the operator's ``multibranch.make_before_break``)."""

    inputs: ToolboxInputsFor
    on_this_machine: ClassVar[bool] = False

    def toolbox(
        self, setup: PeerSetup, blackboard: LocalBlackboard, acceptance: AcceptanceSetup, spec: SpecContext
    ) -> RealToolbox:
        from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions, build_hosted_toolbox

        inputs = self.inputs(setup.layout, setup.workload.tp)
        options = HostedOptions(
            timeout_sec=attempt_timeout_sec(setup.workload), make_before_break=setup.make_before_break
        )
        return build_hosted_toolbox(inputs, blackboard, acceptance, spec, options=options)


def load_hosted_peers(workload: Workload, inputs: ToolboxInputsFor) -> PeerFactory:
    """The factory of the real peers of a hosted session of ``workload``, whose tests run in the sandboxes ``inputs``
    describe; raises :class:`~hyperloom_multibranch.scheduling.peers.PeersUnavailable` as
    :func:`~hyperloom_multibranch.scheduling.local_peers.load_peers` does, for the same reasons. Creates nothing."""

    return peers_on(measurement_host(os.environ, workload, HostedSandboxes(inputs)), workload)


def runner_cache_pin(source: Path, sha: str, cache_root: Path, target: WheelTarget) -> str:
    """The SHA-256 of the manifest of commit ``sha``'s entry in the runner cache at ``cache_root``, built from the
    clone ``source`` when it is not there yet, under umask 022 so the sandbox's account can read what was built.

    The wheels are for ``target``, the runner's interpreter (the application's own is not it), and come from the index
    pip is configured with.

    Raises:
        CacheBuildError: When the entry cannot be had. A runner sandbox has no network, so no test could start: the
            run fails before it measures anything.
    """

    from hyperloom.measure.runner.cache_build import BuildOptions, ensure_runner_cache

    previous = os.umask(RUNNER_CACHE_UMASK)
    try:
        return ensure_runner_cache(source, sha, cache_root, BuildOptions(target=target)).manifest_sha256
    finally:
        os.umask(previous)


def runner_source(destination: Path, repo: str, sha: str) -> Path:
    """A clone of ``repo`` (``OWNER/NAME`` on GitHub, or a git URL or path) that holds commit ``sha``, at
    ``destination``: fetched there once, and found there by a restart, as it is moved there only once it holds the
    commit.

    The session directory is the group's to write, the run's sandboxes included, so the clone is made in a new
    directory only this account can enter, whose name no one can guess, so no one can put a config or a hook in it
    first; it is then made readable, never writable, by the next leg's account, which only reads it, and renamed into
    place. A clone found there is used whoever made it, and only through a copy of the commit in a repository of this
    account's (:func:`private_copy`): what is built from it is the commit's own, whose ids git computed.

    Raises:
        CacheBuildError: When the commit cannot be fetched; nothing is left at ``destination``.
    """

    if destination.is_dir():
        return destination
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent))
    try:
        url = f"https://github.com/{repo}.git" if _GITHUB_NAME.fullmatch(repo) else repo
        _git([_GIT, "init", _QUIET, str(staging)], f"creating {staging}")
        try:
            _git([_GIT, "-C", str(staging), _FETCH, _QUIET, "--depth", "1", url, sha], f"fetching {sha} from {repo}")
        except CacheBuildError:
            _git([_GIT, "-C", str(staging), _FETCH, _QUIET, url], f"fetching {repo}")
        _git([_GIT, "-C", str(staging), "cat-file", "-e", f"{sha}^{{commit}}"], f"finding {sha} in {repo}")
        staging.chmod(_SOURCE_MODE)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _git(argv: list[str], doing: str, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True, timeout=SOURCE_FETCH_TIMEOUT_SEC, env=env)
    except subprocess.CalledProcessError as exc:
        raise CacheBuildError(f"{doing} failed: {(exc.stderr or '').strip()[-500:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CacheBuildError(f"{doing} took more than {SOURCE_FETCH_TIMEOUT_SEC} s") from exc


@dataclass(frozen=True)
class PinnedToolboxInputs:
    """The toolbox's inputs of a hosted session: what ``host`` gave, every runner sandbox pinned to the runner cache
    entry below ``cache_root`` of the commit the run executes, built from a private copy of the session's clone of it."""

    host: HostInfo
    cache_root: Path

    def __call__(self, layout: SessionLayout, tp: int) -> HostedToolboxInputs:
        """Raises :class:`~hyperloom.hosted.toolbox_inputs.ToolboxInputsError` naming what the host did not give, and
        :class:`CacheBuildError` when the commit or its runner cache entry cannot be had, the clone does not hold the
        commit whole, or when
        ``$HYPERLOOM_RUNNER_WHEEL_TARGET`` does not name the runner's interpreter."""

        from hyperloom.hosted.toolbox_inputs import hosted_toolbox_inputs
        from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolboxInputs

        given = hosted_toolbox_inputs(self.host, session_dir=layout.toolbox, tp=tp)
        given["run_id"] = self.host.run_id or layout.session_id
        given["session_id"] = layout.session_id
        sha = str(given["source_sha"])
        source = runner_source(layout.runner_source, str(self.host.source_repo), sha)
        with tempfile.TemporaryDirectory(prefix="hyperloom-runner-source-") as private:
            copy = private_copy(source, Path(private), sha)
            pin = runner_cache_pin(copy, sha, self.cache_root, runner_wheel_target(os.environ))
        return HostedToolboxInputs(**given, runner_cache_manifest_sha256=pin)  # type: ignore[arg-type]


def private_copy(source: Path, private: Path, sha: str) -> Path:
    """A new bare repository in ``private``, a directory of this account's own, holding commit ``sha`` fetched from the
    clone ``source``; what the runner cache is built from.

    The clone lies in the session directory, which the group's other accounts can write: they could put in its place
    a repository whose config git would obey as the application, or plant an object under the id of one of the
    commit's own, which git does not check as it reads one. The fetch sends the commit's objects through a pack whose
    ids this side computes from their content, so a planted object fails it; it runs with no system config and starts
    no program a config could name (:mod:`~hyperloom_multibranch.repo.isolated_git`), and the clone's config is read
    only by git's serving side, which starts none.

    The clone is another account's when an earlier leg of the run made it. git's serving side refuses a repository
    another account owns unless ``safe.directory`` names it in the system or global config (git 2.45.1 to 2.47, and
    distribution builds that carry the CVE-2024-32004 fix, such as Ubuntu 22.04's and 24.04's), and a ``-c`` does not
    reach it: git drops it from the environment of the local transport's ``git-upload-pack``. So the fetch's global
    config is a file in ``private`` that holds only ``safe.directory`` for the clone, and the fetch names the clone by
    the same resolved path.

    Raises:
        CacheBuildError: When the clone does not hold the commit whole as its ids name it.
    """

    copy = private / "source.git"
    clone = os.path.realpath(source)
    trusted = private / "trusted.gitconfig"
    object_format = ["--object-format=sha256"] if len(sha) == 64 else []
    env = git_env(GIT_ALLOW_PROTOCOL="file", GIT_CONFIG_GLOBAL=str(trusted))
    _git(git_argv(*INIT, _QUIET, "--bare", *object_format, str(copy)), f"creating {copy}", env)
    for directory in (clone, os.path.join(clone, ".git")):
        trust = ["--git-dir", str(copy), "config", "--file", str(trusted), "--add", "safe.directory", directory]
        _git(git_argv(*trust), f"trusting {clone}", env)
    fetch = ["--git-dir", str(copy), "-c", "protocol.version=2", _FETCH, _QUIET, "--no-tags", "--depth", "1"]
    _git(git_argv(*fetch, clone, sha), f"copying {sha} from {source}", env)
    return copy


__all__ = [
    "RUNNER_CACHE_UMASK",
    "SOURCE_FETCH_TIMEOUT_SEC",
    "HostedSandboxes",
    "PinnedToolboxInputs",
    "ToolboxInputsFor",
    "load_hosted_peers",
    "private_copy",
    "runner_cache_pin",
    "runner_source",
]
