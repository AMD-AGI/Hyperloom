# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a single-branch specialist is told of a framework whose server the session builds from its checkout: how the
gains of such a server are found, and the helper that sweeps a kernel knob of the server."""

from __future__ import annotations

import os

from hyperloom.common.agent_isolation import NONE, agent_isolation_report
from hyperloom.common.outside_facts import kernel_knobs
from hyperloom.common.provenance import detect_gfx_arch
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
    return [*_discovery_block(inp), *_knob_tune_block(inp)]


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


def _knob_tune_block(inp: SpecialistPromptInputs) -> list[str]:
    """Optional ``knob_tune`` helper for a patch specialist with cards of its own and a worktree of a framework whose
    server the session builds from source, when the rule pack states a kernel knob of that framework in a branch
    compiled for the architecture the session builds for. Only for a specialist launched unconfined: in a sandbox or
    as another account it is not shown the session's recorded base and cannot write the session's sweeps."""
    component = native_component(inp.framework)
    gfx = detect_gfx_arch(os.environ, gpu_type=inp.gpu_type or None, probe=False)
    declared = any(knob.component == component and gfx in knob.archs for knob in kernel_knobs().values())
    if not declared or not inp.allocated_gpu_ids or inp.mode != MODE_PATCH or not inp.worktree_base:
        return []
    if agent_isolation_report().specialists != NONE:
        return []
    cards = ",".join(str(g) for g in inp.allocated_gpu_ids)
    return [
        "",
        "Optional helper: ``knob_tune`` sweeps a compile-time kernel knob of the server: one build per value over the",
        "session's base, each kernel instance timed with the server's own operator benchmark against the base build at",
        "the weight shapes you give (your model's), a value recommended only where it beats the default beyond the",
        "measured noise:",
        f"    ROCR_VISIBLE_DEVICES={cards} python -m hyperloom.measure.runner.native.knob_tune \\",
        f"        --session {session_dir()} --knob <knob> --axis <axis>=<v1,v2> --shape <MxK> [--values <v1,v2>]",
        "  ``<knob>`` is one the rule pack declares: a run that names another is refused with the knobs it takes, each with",
        "  its axes and values, before anything is built. It prints a JSON report and writes ``winning.patch`` (a git",
        "  patch over the base); apply it to your worktree. Builds take minutes each; the times are advice, the",
        "  integrate gate decides.",
    ]
