# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening the builds of a session whose framework server is built from its source, for the entry point that hands them
to integration.

The builds are the hosted native-build engine's (:mod:`hyperloom.measure.runner.native.session_builds`), which builds on
this layer and which only its owners may import (the plug architecture guard), so this layer asks the ``native_builds``
socket of the process's plug host for them (:mod:`hyperloom.plugs.native_builds`); a framework whose server is the
image's never asks it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from hyperloom.inference_optimizer import framework_registry
from hyperloom.plugs import active
from hyperloom.plugs.native_builds import NATIVE_BUILDS_SOCKET, NativeBuilds

if TYPE_CHECKING:
    from hyperloom.orchestrator.actions.executors.candidate_builds import CandidateBuilds

__all__ = ["NativeBuildsRefused", "start_candidate_builds"]


class NativeBuildsRefused(RuntimeError):
    """The session's framework cannot be built from its source here; the reason says what to change."""


def _engine() -> NativeBuilds:
    """What fills the ``native_builds`` socket of the process's current plug host."""
    return cast(NativeBuilds, active().get(NATIVE_BUILDS_SOCKET))


def start_candidate_builds(session_dir: Path, framework: str) -> CandidateBuilds | None:
    """The session's builds of ``framework``'s server, its accepted build first on this process's ``PATH``; ``None``
    when the server is the image's.

    Raises:
        NativeBuildsRefused: The framework builds its server from source and that cannot be done here.
    """
    if framework_registry.native_component(framework) is None:
        return None
    return cast("CandidateBuilds | None", _engine().open_builds(session_dir, framework))
