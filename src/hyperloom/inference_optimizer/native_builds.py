# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening the builds of a session whose framework server is built from its source, for the entry point that hands them
to integration, and the option list of the server so built, for the session's specialists.

The builds are the hosted native-build engine's (:mod:`hyperloom.measure.runner.native.session_builds`), which builds on
this layer and which only its owners may import (the hosted-import guard), so this layer imports the engine's entry
(:mod:`hyperloom.measure.runner.native.session_engine`) by its path, on the first call that needs it; a framework whose
server is the image's never loads it. The option list is the same engine's
(:mod:`hyperloom.measure.runner.native.session_options`), asked of the same entry.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from hyperloom.inference_optimizer import framework_registry

if TYPE_CHECKING:
    from hyperloom.orchestrator.actions.executors.candidate_builds import CandidateBuilds

__all__ = [
    "NativeBuildsRefused",
    "OptionCoverage",
    "OptionVerdict",
    "ServerOptions",
    "record_server_options",
    "server_options",
    "start_candidate_builds",
]


class NativeBuildsRefused(RuntimeError):
    """The session's framework cannot be built from its source here; the reason says what to change."""


class OptionVerdict(str, Enum):
    """What a specialist concludes of one option the base launch leaves unset. ``CANDIDATE``: worth a measured
    test. ``NO_EFFECT_ON_STEP``: it cannot change the time of a decode step of the workload.
    ``DEFAULT_ALREADY``: the launch already gets this behaviour by default. ``CHANGES_WORKLOAD_OR_WEIGHTS``: it
    would change what is served or measured. ``NOT_USABLE_HERE``: this build, card or launch cannot run it."""

    CANDIDATE = "candidate"
    NO_EFFECT_ON_STEP = "no_effect_on_step"
    DEFAULT_ALREADY = "default_already"
    CHANGES_WORKLOAD_OR_WEIGHTS = "changes_workload_or_weights"
    NOT_USABLE_HERE = "not_usable_here"


@dataclass(frozen=True)
class OptionCoverage:
    """How far a review covers the options a session's base launch leaves unset: how many there are, the ones the
    review gives no verdict on, and the spellings it names that the server lists no option for."""

    unset: int
    missing: tuple[str, ...]
    unknown: tuple[str, ...]


class ServerOptions(Protocol):
    """The option list a session keeps of the server it built, in ``file``."""

    file: Path

    def coverage(self, reviewed: Iterable[str]) -> OptionCoverage:
        """How far ``reviewed``, the spellings a review gives a verdict on, covers the options the base leaves unset."""


class NativeBuilds(Protocol):
    """Builds the server of a session's framework from its checkout.

    What a caller asks of the builds and of the option list is declared by the layer that uses them, which this module
    must not import, so they come back as ``object``.
    """

    def open_builds(self, session_dir: Path, framework: str) -> object | None:
        """The builds of ``framework``'s server in the session at ``session_dir``, the accepted one first on this
        process's ``PATH``; ``None`` when ``framework`` builds none."""

    def record_options(self, session_dir: Path, framework: str) -> Path | None:
        """Keep the options the server first on this process's ``PATH`` takes in the session at ``session_dir``,
        unless it keeps them already; the file, or ``None`` when ``framework`` builds no server."""

    def options(self, session_dir: Path) -> object | None:
        """The option list the session at ``session_dir`` keeps; ``None`` when it keeps none."""


#: The engine's entry, by import path (see the module docstring).
_ENGINE_MODULE = "hyperloom.measure.runner.native.session_engine"


def _engine() -> NativeBuilds:
    """The native-build engine, imported on first use."""
    return cast(NativeBuilds, importlib.import_module(_ENGINE_MODULE).SessionNativeEngine())


def start_candidate_builds(session_dir: Path, framework: str) -> CandidateBuilds | None:
    """The session's builds of ``framework``'s server, its accepted build first on this process's ``PATH``; ``None``
    when the server is the image's.

    Raises:
        NativeBuildsRefused: The framework builds its server from source and that cannot be done here.
    """
    if framework_registry.native_component(framework) is None:
        return None
    return cast("CandidateBuilds | None", _engine().open_builds(session_dir, framework))


def record_server_options(session_dir: Path, framework: str) -> Path | None:
    """Keep the options the server first on this process's ``PATH`` takes in the session at ``session_dir``, read
    against its frozen base launch, unless an earlier leg kept them; the file, or ``None`` when the server is the
    image's. Called once :func:`start_candidate_builds` has put the session's build there.

    Raises:
        FrameworkProbeError: The server is not on ``PATH``, or cannot print its ``--help``.
    """
    if framework_registry.native_component(framework) is None:
        return None
    return _engine().record_options(session_dir, framework)


def server_options(session_dir: Path | None, framework: str) -> ServerOptions | None:
    """The option list the session at ``session_dir`` keeps of ``framework``'s server; ``None`` without a session,
    when the server is the image's, or when the session kept none."""
    if session_dir is None or framework_registry.native_component(framework) is None:
        return None
    return cast("ServerOptions | None", _engine().options(session_dir))
