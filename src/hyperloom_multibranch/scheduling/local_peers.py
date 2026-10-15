# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session on one machine: the git experiment repository and the local hosted toolbox.

:func:`local_peers` builds them inside the session directory: the repository at ``MB-repo``, the toolbox's
sandboxes, archive, submit log and measurement points under ``MB-toolbox``, and the patch and kernel files its
environment compiler reads from the session's blob store at ``MB-blobs``. The specialist runtime's lifecycle comes from
a :data:`LifecycleFactory`; that runtime has no implementation yet, so :func:`load_peers` refuses and the command
line fails before it creates anything.

The toolbox and its measurement points load the main flow's renderer, so they are imported inside the functions that
use them: importing this module, the command line or the session loads none of the main-flow runtime.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom_multibranch.contracts.lifecycle import SpecialistLifecycle
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.peers import (
    PeerFactory,
    Peers,
    PeerSetup,
    PeersUnavailable,
    git_root_writer,
)
from hyperloom_multibranch.scheduling.settings import Workload

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

TOOLBOX_WRITER = "toolbox"
CANONICAL_DIR = "canonical"
POINTS_DIR = "points"

#: The modules a real session needs and that hold no implementation yet; emptied as each module lands, and deleted
#: with :func:`mb_fake_specialist_lifecycles` once the last one has.
MB_FAKE_MISSING_PEER_MODULES: tuple[str, ...] = ("hyperloom_multibranch.specialist",)
#: Stands in for the image the local sandboxes run in: they run on this machine, in no image.
MB_FAKE_IMAGE_REF = "local"
#: Stands in for the Magpie and InferenceX commits the runner checks out; the instruction sheet refuses no commit
#: of this shape, but a runner cannot fetch it. Replaced when enablement or the command line name the real commits.
MB_FAKE_TOOL_COMMIT = "0" * 40

#: Builds the specialist runtime's lifecycle on the peer setup, the repository and the toolbox the branches use.
LifecycleFactory = Callable[[PeerSetup, ExperimentRepo, "RealToolbox"], SpecialistLifecycle]


@dataclass(frozen=True)
class MeasurementHost:
    """What the toolbox measures on and the session's workload does not name: the cards of the machine, the image
    the tests run in (``image_digest`` is also the base image every recipe must carry) and the Magpie and InferenceX
    commits the runner checks out."""

    cards: tuple[int, ...]
    image_ref: str
    image_digest: str
    magpie_commit: str
    inferencex_commit: str


def workload_environment(workload: Workload) -> Mapping[str, str]:
    """The workload as the process environment carries it to the main flow's config renderer, which the toolbox
    renders the canonical measurement and every test's points with; ``hyperloom optimize`` exports the same, the
    model resolved to the directory a server loads it from."""

    return {
        "FRAMEWORK": workload.framework,
        "MODEL_PATH": resolve_serving_model_path(workload.model),
        "TP": str(workload.tp),
        "ISL": str(workload.isl),
        "OSL": str(workload.osl),
        "PRECISION": workload.precision,
        "GPU_TYPE": workload.gpu_type,
    }


def local_peers(host: MeasurementHost, lifecycles: LifecycleFactory, setup: PeerSetup) -> Peers:
    """The session's peers on this machine; exports :func:`workload_environment` into this process.

    The toolbox is built, not started: the session starts it, and its close releases the backend and the sandboxes.
    """

    from hyperloom.measure.compile import EnvironmentCompiler
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom.measure.points import default_canonical
    from hyperloom_multibranch.toolbox.local_hosted import build_local_toolbox
    from hyperloom_multibranch.toolbox.planner import HostedPlanner

    layout, workload = setup.layout, setup.workload
    os.environ.update(workload_environment(workload))
    repo = GitExperimentRepo(layout.repo)
    canonical = default_canonical(layout.toolbox / CANONICAL_DIR, primary_concurrency=workload.conc)
    compiler = EnvironmentCompiler(setup.blobs, host.image_digest)
    spec = SpecContext(
        image_ref=host.image_ref,
        image_digest=host.image_digest,
        gpu_type=workload.gpu_type,
        magpie_commit=host.magpie_commit,
        inferencex_commit=host.inferencex_commit,
        blobs=setup.blobs,
        blob_root=str(setup.blobs.root),
    )
    toolbox = build_local_toolbox(
        session_dir=layout.toolbox,
        run_id=layout.session_id,
        cards=host.cards,
        notebook=LocalNotebook(layout.notebook, Writer(Role.TOOLBOX, TOOLBOX_WRITER)),
        repo=repo,
        planner=HostedPlanner(compiler, canonical, layout.toolbox / POINTS_DIR),
        spec=spec,
        active_image_digest=host.image_digest,
        tp=workload.tp,
    )
    return Peers(lifecycles(setup, repo, toolbox), toolbox, repo, git_root_writer(layout.repo))


def mb_fake_measurement_host() -> MeasurementHost:
    """The cards of one node as the toolbox counts them, the root recipe's base image and :data:`MB_FAKE_TOOL_COMMIT`
    for both tools.

    Stands in for the measurement host enablement or the command line will name; replaced once one of them does.
    """

    from hyperloom_multibranch.toolbox.local_hosted import MAX_CARDS_PER_NODE

    cards = tuple(range(MAX_CARDS_PER_NODE))
    return MeasurementHost(
        cards, MB_FAKE_IMAGE_REF, MB_FAKE_BASE_IMAGE_DIGEST, MB_FAKE_TOOL_COMMIT, MB_FAKE_TOOL_COMMIT
    )


def machine_parallel_tests(tp: int) -> int:
    """How many tests of ``tp`` cards the local toolbox runs at once on this machine's cards; raises ``ValueError``
    when a test of ``tp`` cards does not fit on them."""

    from hyperloom_multibranch.toolbox.local_hosted import max_parallel_tests

    return max_parallel_tests(len(mb_fake_measurement_host().cards), tp)


def mb_fake_specialist_lifecycles() -> LifecycleFactory:
    """Always raises :class:`PeersUnavailable` naming :data:`MB_FAKE_MISSING_PEER_MODULES`.

    Stands in for the specialist runtime's lifecycle factory; replaced by it once ``hyperloom_multibranch.specialist``
    ships one.
    """

    raise PeersUnavailable(MB_FAKE_MISSING_PEER_MODULES)


def load_peers() -> PeerFactory:
    """The factory of a session's real peers; raises :class:`PeersUnavailable` while one has no implementation.

    It takes no argument and creates nothing, so the command line calls it before the session directory exists.
    """

    return partial(local_peers, mb_fake_measurement_host(), mb_fake_specialist_lifecycles())


__all__ = [
    "MB_FAKE_MISSING_PEER_MODULES",
    "LifecycleFactory",
    "MeasurementHost",
    "load_peers",
    "local_peers",
    "machine_parallel_tests",
    "mb_fake_measurement_host",
    "mb_fake_specialist_lifecycles",
    "workload_environment",
]
