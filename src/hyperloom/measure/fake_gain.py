# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public names for main's refusals of a fake gain, so the multi-branch toolbox refuses the same requests.

Main refuses three gains that a measurement can show without the candidate being better: a context-copying
speculative draft on a synthetic benchmark, a change of a draft whose acceptance length is simulated, and an AgentX
baseline served with the prefix cache off. Its flow does that in the explore and baseline executors; a test launched
through ``hyperloom.measure`` never reaches them. This module re-exports the checks without copying their logic.
"""

from __future__ import annotations

from typing import Any

from hyperloom.contracts.recipe import LaunchConfig
from hyperloom.orchestrator.actions.executors import (
    CONTEXT_COPY_REFUSAL,
    SPECULATIVE_DECODING_PINNED,
    SpeculativeDecodingPinnedError,
    context_copy_refusal,
    refuse_unpinned_speculative_change,
)

__all__ = [
    "CONTEXT_COPY_REFUSAL",
    "SPECULATIVE_DECODING_PINNED",
    "launch_context_copy_refusal",
    "pinned_speculative_refusal",
]


def _argv(launch: LaunchConfig) -> tuple[str, dict[str, Any]]:
    return launch.server_args, dict(launch.envs)


def launch_context_copy_refusal(base: LaunchConfig, candidate: LaunchConfig) -> str:
    """Why ``candidate`` may not run on a synthetic (InferenceX) benchmark, or ``""``; main's #1904 check."""
    return context_copy_refusal(*_argv(base), *_argv(candidate))


def pinned_speculative_refusal(base: LaunchConfig, candidate: LaunchConfig) -> str:
    """Why ``candidate`` may not change the draft or simulated acceptance its base pins, or ``""``; main's #1885."""
    try:
        refuse_unpinned_speculative_change(*_argv(base), *_argv(candidate))
    except SpeculativeDecodingPinnedError as error:
        return str(error)
    return ""
