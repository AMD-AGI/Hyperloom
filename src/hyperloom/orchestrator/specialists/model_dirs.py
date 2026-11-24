# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist of a server built from its checkout reads of the model files a launch names.

Such a specialist is told to list the directory of every model file its launch names, following symlinks, for other
builds of that file. In a sandbox it sees its model's directory and nothing a link in it leads to, and nothing of a
file the launch names elsewhere. So what it reads is derived here from the launch, once for a single-branch and a
multi-branch specialist alike: the directory behind the served model file where that file is a link, and the directory
of each file the base launch names with a flag the rule pack lists as naming an auxiliary model
(``NATIVE_AUX_MODEL_FLAGS``), with the one behind it. A directory that is more than a model's gives the file in it
alone (:func:`_shown`). A framework the rule pack lists no such flag of gains nothing.

A session that keeps the option list of the server it built keeps it in its directory, which no sandbox shows, so
:func:`session_reads` names that one file with the models.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.confinement import credential_paths
from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS
from hyperloom.inference_optimizer.framework_registry import native_component
from hyperloom.inference_optimizer.native_builds import server_options
from hyperloom.orchestrator.state.base_launch import frozen_base, server_base
from inference_suite.argv import canonical_flag_name, flag_occurrences
from inference_suite.frameworks import Framework, ModelFileError, framework_facts

__all__ = ["launch_model_reads", "session_reads"]

log = logging.getLogger(__name__)


def launch_model_reads(framework: str, model_path: str, launch_args: str, session: Path | None) -> tuple[Path, ...]:
    """What a specialist of ``framework`` in the session at ``session`` reads besides the model at ``model_path``,
    for a server launched with ``launch_args``: the directory of the file the served model file leads to when it is a
    link, and of each auxiliary model file the arguments name by an absolute path and of the file that one leads to;
    the file alone where its directory is more than a model's (:func:`_shown`). None for a framework built from no
    checkout."""
    component = native_component(framework)
    flags = frozenset(map(canonical_flag_name, NATIVE_AUX_MODEL_FLAGS.get(component or "", ())))
    if not flags:
        return ()
    named = [Path(value) for name, value in flag_occurrences(launch_args) if name in flags and os.path.isabs(value)]
    files = [*_led_to(_served_file(framework, model_path)), *(f for file in named for f in (file, *_led_to(file)))]
    return tuple(dict.fromkeys(_shown(file, session) for file in files))


def session_reads(session: Path | None, params: Mapping[str, Any] | None) -> tuple[Path, ...]:
    """What a specialist whose task has ``params`` reads besides its workspace in the session at ``session``: the
    model the task names, what the session's frozen base launch names for its framework (:func:`launch_model_reads`;
    nothing without a session or a frozen base), and the option list the session keeps of the server it built.

    Raises:
        BaseLaunchMismatch: As :func:`~hyperloom.orchestrator.state.base_launch.frozen_base`.
    """
    task = params or {}
    framework, model_path = str(task.get("framework") or ""), str(task.get("model_path") or "")
    kept = server_options(session, framework)
    frozen = None if session is None else frozen_base(session)
    args = "" if frozen is None else server_base(frozen.launch).args
    models = launch_model_reads(framework, model_path, args, session)
    return (Path(model_path), *models, *(() if kept is None else (kept.file,)))


def _served_file(framework: str, model_path: str) -> Path | None:
    """The file ``framework``'s server loads for the model at ``model_path`` when that is a file here; ``None`` for a
    model that is not (a hub id), and for a directory the launch itself refuses (no one model in it)."""
    try:
        served = framework_facts(Framework(framework)).served_model(model_path)
    except ModelFileError as refused:
        log.info("no model file is read at %s for a specialist's reads: %s", model_path, refused)
        return None
    return Path(served) if os.path.isabs(served) and os.path.isfile(served) else None


def _led_to(file: Path | None) -> tuple[Path, ...]:
    """The file ``file`` leads to when it is a link out of its own directory."""
    if file is None:
        return ()
    real = Path(os.path.realpath(file))
    return () if real.parent == Path(os.path.realpath(file.parent)) else (real,)


def _shown(file: Path, session: Path | None) -> Path:
    """``file``'s directory, for the other builds beside it, or ``file`` alone, said at WARNING, where that directory
    is more than a model's: one that holds the session at ``session``, whose rounds, branches and keys no sandbox
    shows, or one that is a credential location (a home), which a bind of it would show unmasked. One above a
    credential location is shown: the masks hide the location in it."""
    directory = Path(os.path.realpath(file.parent))
    held = None if session is None else Path(os.path.realpath(session))
    if held is not None and (directory == held or directory in held.parents):
        why = "holds the session"
    elif directory in credential_paths(os.environ):
        why = "is a credential location"
    else:
        return file.parent
    log.warning(
        "%s %s, so it is not shown to a specialist for other builds of %s: only that file is", directory, why, file
    )
    return file
