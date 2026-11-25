# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's option list: what the server it built takes, kept once for its specialists.

When a session starts, the build first on ``PATH`` is its base
(:func:`~hyperloom.measure.runner.native.session_builds.start_native_builds`), so that is when the server's ``--help``
is read against the frozen base launch and kept in the directory the base is frozen in: the session directory, or a
multi-branch session's toolbox directory (:mod:`~hyperloom.measure.runner.native.server_options`). A later leg keeps
the file it finds: by then the build first on ``PATH`` may be a candidate's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.inference_optimizer.native_builds import OptionCoverage
from hyperloom.measure.runner.native.server_options import (
    OptionInventory,
    inventory_path,
    read_inventory,
    server_inventory,
    write_inventory,
)
from hyperloom.orchestrator.state.base_launch import ServerBase, frozen_base, server_base
from inference_suite import runner_owned

__all__ = ["SessionServerOptions", "record_session_options", "session_options"]


@dataclass(frozen=True)
class SessionServerOptions:
    """The option list a session keeps, in ``file``; the entry point's
    :class:`~hyperloom.inference_optimizer.native_builds.ServerOptions`."""

    file: Path
    inventory: OptionInventory

    def coverage(self, reviewed: Iterable[str]) -> OptionCoverage:
        """How far ``reviewed``, the spellings a review gives a verdict on, covers the options the base leaves unset."""
        named = {spelling: self.inventory.option(spelling) for spelling in dict.fromkeys(reviewed)}
        covered = [option for option in named.values() if option is not None]
        unset = self.inventory.unset
        return OptionCoverage(
            unset=len(unset),
            missing=tuple(option.name for option in unset if option not in covered),
            unknown=tuple(spelling for spelling, option in named.items() if option is None),
        )


def record_session_options(session_dir: Path, framework: str, environ: Mapping[str, str]) -> Path | None:
    """Keep the options ``framework``'s server takes, as ``environ``'s ``PATH`` resolves it, against the frozen base of
    the session at ``session_dir`` (its arguments, and the flags the runner sets on every launch of it), unless the
    session keeps them already; the file, or ``None`` for a framework whose server is no native program.

    The server is asked under ``environ`` with the frozen launch's environment over it, as a test of the session
    runs it.

    Raises:
        FrameworkProbeError: The program is not on ``environ``'s ``PATH``, or cannot print its ``--help``.
        BaseLaunchMismatch: As :func:`~hyperloom.orchestrator.state.base_launch.frozen_base`.
    """
    kept = inventory_path(session_dir)
    if kept is not None:
        return kept
    frozen = frozen_base(session_dir)
    launch = ServerBase() if frozen is None else server_base(frozen.launch)
    owned = () if frozen is None else runner_owned(frozen.launch).args
    env = {**environ, **launch.env}
    inventory = server_inventory(framework, env, launch.args, cwd=session_dir, runner_owned=owned)
    return None if inventory is None else write_inventory(inventory, session_dir)


def session_options(session_dir: Path) -> SessionServerOptions | None:
    """The option list the session at ``session_dir`` keeps; ``None`` when it keeps none."""
    path = inventory_path(session_dir)
    return None if path is None else SessionServerOptions(path, read_inventory(path))
