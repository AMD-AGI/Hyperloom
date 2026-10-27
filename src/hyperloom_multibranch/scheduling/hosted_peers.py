# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session measures on: runner sandboxes of the host's sandbox service, which install Hyperloom from the
runner cache of the commit the run executes.

A hosted session's peers are the ones :func:`~hyperloom_multibranch.scheduling.local_peers.local_peers` builds -- the
repository, the specialists and the MCP service run in the application's container -- on one other
:class:`~hyperloom_multibranch.scheduling.local_peers.ToolboxBackend`, :class:`HostedSandboxes`: every test runs in a
runner sandbox of the service (:func:`~hyperloom_multibranch.toolbox.hosted_toolbox.build_hosted_toolbox`), and the
toolbox keeps its journal, archive and landed output under ``MB-toolbox``. A runner sandbox has no network, so
:func:`runner_cache_pin` has the commit's runner cache entry built, or finds it built, before anything is measured, and
the hosted entry pins every runner sandbox to that entry's manifest through the toolbox's inputs.

A hosted session lends its specialists no dev sandbox: a dev lease's commands run in a sandbox on this machine's cards
(:func:`~hyperloom_multibranch.specialist.dev_sandbox.local_dev_sandboxes`), and a hosted lease's cards are in another
machine.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom_multibranch.scheduling.local_peers import attempt_timeout_sec, measurement_host, peers_on
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeerSetup
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom_multibranch.notebook import LocalNotebook
    from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolboxInputs
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox

#: The toolbox's inputs for the session's toolbox directory and the workload's tensor-parallel size: the hosted entry
#: binds the host's inputs and the runner cache pin into it.
ToolboxInputsFor = Callable[[Path, int], "HostedToolboxInputs"]

#: The umask the runner cache is built under: the sandbox reads the cache as another account.
RUNNER_CACHE_UMASK = 0o022


@dataclass(frozen=True)
class HostedSandboxes:
    """Tests run in runner sandboxes of the host's sandbox service, on the inputs ``inputs`` gives for the session's
    ``MB-toolbox`` directory and the workload's TP."""

    inputs: ToolboxInputsFor

    def toolbox(
        self, setup: PeerSetup, notebook: LocalNotebook, acceptance: AcceptanceSetup, spec: SpecContext
    ) -> RealToolbox:
        from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions, build_hosted_toolbox

        inputs = self.inputs(setup.layout.toolbox, setup.workload.tp)
        options = HostedOptions(timeout_sec=attempt_timeout_sec(setup.workload))
        return build_hosted_toolbox(inputs, notebook, acceptance, spec, options=options)


def load_hosted_peers(workload: Workload, inputs: ToolboxInputsFor) -> PeerFactory:
    """The factory of the real peers of a hosted session of ``workload``, whose tests run in the sandboxes ``inputs``
    describe; raises :class:`~hyperloom_multibranch.scheduling.peers.PeersUnavailable` as
    :func:`~hyperloom_multibranch.scheduling.local_peers.load_peers` does, for the same reasons. Creates nothing."""

    return peers_on(measurement_host(os.environ, workload, HostedSandboxes(inputs)), workload)


def runner_cache_pin(source: Path, sha: str, cache_root: Path) -> str:
    """The SHA-256 of the manifest of commit ``sha``'s entry in the runner cache at ``cache_root``, built from the
    clone ``source`` when it is not there yet, under umask 022 so the sandbox's account can read what was built.

    The wheels are for this interpreter and come from the index pip is configured with: the application runs in the
    framework image the runner sandboxes run.

    Raises:
        CacheBuildError: When the entry cannot be had. A runner sandbox has no network, so no test could start: the
            run fails before it measures anything.
    """

    from hyperloom.measure.runner.cache_build import ensure_runner_cache

    previous = os.umask(RUNNER_CACHE_UMASK)
    try:
        return ensure_runner_cache(source, sha, cache_root).manifest_sha256
    finally:
        os.umask(previous)


__all__ = [
    "RUNNER_CACHE_UMASK",
    "HostedSandboxes",
    "ToolboxInputsFor",
    "load_hosted_peers",
    "runner_cache_pin",
]
