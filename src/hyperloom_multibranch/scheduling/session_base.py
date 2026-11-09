# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The base launch every test of a session runs on, frozen once in the toolbox's session directory.

Every inference_suite round refuses a session that froze no base launch
(:func:`hyperloom.orchestrator.state.base_launch.frozen_base`). A fresh session resolves its base as
``hyperloom optimize`` does (:func:`~hyperloom.inference_optimizer.cli.base_launch.resolve_server_base`) for its
workload, on the cards one test runs on, InferenceX read at the operator's commit when one is given; a session that
froze one, a resumed one, keeps it. The operator's ``--reference-script`` is the base as it is for ``hyperloom
optimize``: a base launch record is frozen without reading any recipe source. The Arbor agents of such a session are
served the frozen base (:func:`served_launch`), the one the gate measures on. The workload fixes the topology every test is sized by, so a base on another one is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.settings import Workload
from inference_suite.contracts import BaseLaunchError
from inference_suite.contracts import Workload as LaunchWorkload

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.cli.base_launch import SessionLaunch
    from hyperloom.orchestrator.state.base_launch import FrozenBase


def freeze_session_base(
    toolbox: Path, workload: Workload, reference_script: str, inferencex_commit: str = ""
) -> FrozenBase:
    """Freeze the base launch of a session of ``workload`` in ``toolbox`` unless it froze one, resolved from the
    operator's ``reference_script`` when one is given (``""`` for none), else from InferenceX at ``inferencex_commit``
    (``""`` for the head of its ``main``), and return the frozen one; raises :class:`PeersUnavailable` when the frozen
    one is not what the session recorded, or none can be resolved."""

    from hyperloom.inference_optimizer.cli.base_launch import BaseLaunchUnresolved, resolve_server_base
    from hyperloom.orchestrator.state.base_launch import BaseLaunchMismatch, freeze_base, frozen_base

    try:
        frozen = frozen_base(toolbox)
        if frozen is not None:
            return frozen
    except (BaseLaunchMismatch, BaseLaunchError) as exc:
        raise PeersUnavailable(f"the session's frozen base launch cannot be read: {exc}") from exc
    try:
        base = resolve_server_base(_session_launch(workload, reference_script, inferencex_commit))
    except BaseLaunchUnresolved as exc:
        raise PeersUnavailable(str(exc)) from exc
    return freeze_base(toolbox, base)


def served_launch(frozen: FrozenBase, reference_script: str, agent_launch: str) -> str:
    """The launch file the session's Arbor agents are served: the session's ``frozen`` base when the operator gave a
    ``reference_script``, which the base was frozen from or, on a resume, stands in for; else the Arbor
    ``agent_launch`` as it is."""

    return str(frozen.path) if reference_script else agent_launch


def _session_launch(workload: Workload, reference_script: str, inferencex_commit: str) -> SessionLaunch:
    from hyperloom.inference_optimizer.cli.base_launch import SessionLaunch

    shape = workload.inferencex
    served = resolve_serving_model_path(workload.model)
    return SessionLaunch(
        model=served,
        framework=workload.framework,
        board=workload.gpu_type,
        workload=LaunchWorkload.AGENTX if shape is None else LaunchWorkload.FIXED,
        concurrencies=workload.final_concs,
        gpus=workload.tp,
        tp=workload.tp,
        ep=workload.ep,
        isl=None if shape is None else shape.isl,
        osl=None if shape is None else shape.osl,
        model_display_name="" if served == workload.model else workload.model,
        reference=reference_script,
        inferencex_commit=inferencex_commit,
    )


__all__ = ["freeze_session_base", "served_launch"]
