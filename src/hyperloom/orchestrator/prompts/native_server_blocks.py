# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a single-branch specialist is told of a framework whose server the session builds from its checkout: how the
gains of such a server are found."""

from __future__ import annotations

from hyperloom.inference_optimizer.framework_registry import native_component
from hyperloom.inference_optimizer.session.paths import session_dir
from hyperloom.inference_optimizer.session.session_paths import runs_root

from ..specialists.prompt_inputs import SpecialistPromptInputs
from .native_discovery import native_discovery

__all__ = ["carries_timing_lines", "native_server_blocks"]

_CARRIED = (
    "the runtime findings of the current best round, given with your task once a round is measured, end with them."
)
_LOGS = "under {runs} (find them by name)"


def native_server_blocks(inp: SpecialistPromptInputs) -> list[str]:
    """The block of the identity section for a framework whose server the session builds from source, empty for any
    other framework: how the gains of the server are found, the section a multi-branch specialist reads, with what
    carries this session's timing lines and where its rounds log."""
    component = native_component(inp.framework)
    if component is None:
        return []
    section = native_discovery(component, carried=_CARRIED, logs=_LOGS.format(runs=runs_root(session_dir())))
    return ["", "### Finding gains in a server built from its checkout", "", section] if section else []


def carries_timing_lines(inp: SpecialistPromptInputs) -> bool:
    """Whether the identity section tells the specialist that the runtime findings of its task end with its timing
    lines (:data:`_CARRIED`): a prompt that says so holds those findings."""
    return bool(native_server_blocks(inp))
