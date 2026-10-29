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
built from it (:func:`runner_cache_pin`), or finds it built, before anything is measured, and pins every runner sandbox
to that entry's manifest through the toolbox's inputs.

A hosted session lends its specialists no dev sandbox: a dev lease's commands run in a sandbox on this machine's cards
(:func:`~hyperloom_multibranch.specialist.dev_sandbox.local_dev_sandboxes`), and a hosted lease's cards are in another
machine. Nor does PRELUDE start a child on this machine's cards: it roots the session on root tests and a profile of the
root in runner sandboxes (:mod:`~hyperloom_multibranch.scheduling.prelude`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from hyperloom.measure.runner.cache_build import CacheBuildError, WheelTarget, runner_wheel_target
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


@dataclass(frozen=True)
class HostedSandboxes:
    """Tests run in runner sandboxes of the host's sandbox service, on the inputs ``inputs`` gives for the session's
    ``MB-toolbox`` directory and the workload's TP."""

    inputs: ToolboxInputsFor
    on_this_machine: ClassVar[bool] = False

    def toolbox(
        self, setup: PeerSetup, blackboard: LocalBlackboard, acceptance: AcceptanceSetup, spec: SpecContext
    ) -> RealToolbox:
        from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions, build_hosted_toolbox

        inputs = self.inputs(setup.layout, setup.workload.tp)
        options = HostedOptions(timeout_sec=attempt_timeout_sec(setup.workload))
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

    Raises:
        CacheBuildError: When the commit cannot be fetched; nothing is left at ``destination``.
    """

    if destination.is_dir():
        return destination
    staging = destination.with_name(f"{destination.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    url = f"https://github.com/{repo}.git" if _GITHUB_NAME.fullmatch(repo) else repo
    _git(["init", _QUIET, str(staging)], f"creating {staging}")
    try:
        _git(["-C", str(staging), "fetch", _QUIET, "--depth", "1", url, sha], f"fetching {sha} from {repo}")
    except CacheBuildError:
        _git(["-C", str(staging), "fetch", _QUIET, url], f"fetching {repo}")
    _git(["-C", str(staging), "cat-file", "-e", f"{sha}^{{commit}}"], f"finding {sha} in {repo}")
    staging.rename(destination)
    return destination


def _git(args: list[str], doing: str) -> None:
    try:
        subprocess.run(["git", *args], check=True, capture_output=True, text=True, timeout=SOURCE_FETCH_TIMEOUT_SEC)
    except subprocess.CalledProcessError as exc:
        raise CacheBuildError(f"{doing} failed: {(exc.stderr or '').strip()[-500:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CacheBuildError(f"{doing} took more than {SOURCE_FETCH_TIMEOUT_SEC} s") from exc


@dataclass(frozen=True)
class PinnedToolboxInputs:
    """The toolbox's inputs of a hosted session: what ``host`` gave, every runner sandbox pinned to the runner cache
    entry below ``cache_root`` of the commit the run executes, built from the session's clone of it."""

    host: HostInfo
    cache_root: Path

    def __call__(self, layout: SessionLayout, tp: int) -> HostedToolboxInputs:
        """Raises :class:`~hyperloom.hosted.toolbox_inputs.ToolboxInputsError` naming what the host did not give, and
        :class:`CacheBuildError` when the commit or its runner cache entry cannot be had, or when
        ``$HYPERLOOM_RUNNER_WHEEL_TARGET`` does not name the runner's interpreter."""

        from hyperloom.hosted.toolbox_inputs import hosted_toolbox_inputs
        from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolboxInputs

        given = hosted_toolbox_inputs(self.host, session_dir=layout.toolbox, tp=tp)
        sha = str(given["source_sha"])
        source = runner_source(layout.runner_source, str(given["source_repo"]), sha)
        pin = runner_cache_pin(source, sha, self.cache_root, runner_wheel_target(os.environ))
        return HostedToolboxInputs(**given, runner_cache_manifest_sha256=pin)  # type: ignore[arg-type]


__all__ = [
    "RUNNER_CACHE_UMASK",
    "SOURCE_FETCH_TIMEOUT_SEC",
    "HostedSandboxes",
    "PinnedToolboxInputs",
    "ToolboxInputsFor",
    "load_hosted_peers",
    "runner_cache_pin",
    "runner_source",
]
