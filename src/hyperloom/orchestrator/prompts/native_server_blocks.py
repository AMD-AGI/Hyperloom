# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a single-branch specialist is told of a framework whose server the session builds from its checkout: how the
gains of such a server are found, and the helper that compiles its worktree as the integrate gate does."""

from __future__ import annotations

from hyperloom.inference_optimizer.framework_registry import native_component
from hyperloom.inference_optimizer.session.paths import session_dir
from hyperloom.inference_optimizer.session.session_paths import runs_root

from ..specialists.profile import MODE_PATCH
from ..specialists.prompt_inputs import SpecialistPromptInputs
from .native_discovery import native_discovery

__all__ = ["carries_timing_lines", "native_server_blocks"]

_CARRIED = (
    "the runtime findings of the current best round, given with your task once a round is measured, end with them."
)
_LOGS = "under {runs} (find them by name)"


def native_server_blocks(inp: SpecialistPromptInputs) -> list[str]:
    """The blocks of the identity section for a framework whose server the session builds from source; empty for any
    other framework."""
    return [*_discovery_block(inp), *_build_check_block(inp)]


def carries_timing_lines(inp: SpecialistPromptInputs) -> bool:
    """Whether the identity section tells the specialist that the runtime findings of its task end with its timing
    lines (:data:`_CARRIED`): a prompt that says so holds those findings."""
    return bool(_discovery_block(inp))


def _discovery_block(inp: SpecialistPromptInputs) -> list[str]:
    """How the gains of the server are found, for every specialist of a framework whose server the session builds from
    source: the section a multi-branch specialist reads, with what carries this session's timing lines and where
    its rounds log."""
    component = native_component(inp.framework)
    if component is None:
        return []
    section = native_discovery(component, carried=_CARRIED, logs=_LOGS.format(runs=runs_root(session_dir())))
    return ["", "### Finding gains in a server built from its checkout", "", section] if section else []


def _build_check_block(inp: SpecialistPromptInputs) -> list[str]:
    """Optional ``build_check`` helper for a patch specialist with a worktree of a framework whose server the session
    builds from source."""
    if inp.mode != MODE_PATCH or not inp.worktree_base or native_component(inp.framework) is None:
        return []
    return [
        "",
        "Optional helper: ``build_check`` compiles the server from your worktree exactly as the integrate gate",
        "builds it (your changed tracked files over the session's base commit), and measures nothing:",
        "    python -m hyperloom.measure.runner.native.build_check \\",
        f"        --tree {inp.workspace_path} --session {session_dir()} [--add <new untracked file> ...]",
        "  It prints a JSON result: ``status`` built / compile_error / infra, and ``detail`` with the compiler's",
        "  error. A full build takes minutes. A candidate that does not build is reverted without a bench.",
    ]
