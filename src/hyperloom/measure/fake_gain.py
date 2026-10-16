# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public names for main's refusals of a fake gain, so the multi-branch toolbox refuses the same requests.

Main refuses three gains that a measurement can show without the candidate being better: a context-copying
speculative draft on a synthetic benchmark, a change of a draft whose acceptance length is simulated, and an AgentX
baseline served with the prefix cache off. Its flow does that in the explore and baseline executors; a test launched
through ``hyperloom.measure`` never reaches them. This module re-exports the checks without copying their logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.contracts.recipe import LaunchConfig
from hyperloom.orchestrator.actions.executors import (
    PREFIX_CACHE_OFF_ERROR_CLASS,
    CONTEXT_COPY_REFUSAL,
    SPECULATIVE_DECODING_PINNED,
    SpeculativeDecodingPinnedError,
    build_launch_evidence,
    context_copy_refusal,
    refuse_cacheless_agentx_baseline,
    refuse_unpinned_speculative_change,
)

__all__ = [
    "CONTEXT_COPY_REFUSAL",
    "PREFIX_CACHE_OFF_ERROR_CLASS",
    "SPECULATIVE_DECODING_PINNED",
    "baseline_prefix_cache_refusal",
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


def _server_log(round_dir: Path) -> str | None:
    """The server log of a round: the one in its directory, else the newest below it (a warm-up round's launch)."""
    direct = round_dir / "server.log"
    if direct.is_file():
        return str(direct)
    found = sorted(round_dir.rglob("server.log"), key=lambda path: path.stat().st_mtime)
    return str(found[-1]) if found else None


def baseline_prefix_cache_refusal(point: Path, round_dir: Path) -> str:
    """Why an AgentX baseline round served with the prefix cache off may not end ``succeeded``, or ``""``.

    ``point`` is the measurement point the round ran and ``round_dir`` the directory it landed in; the launch evidence
    is built from both like the main flow does, and main's #1884 check reads it.
    """
    evidence = build_launch_evidence(
        config_path=point, actual_server_log=_server_log(round_dir), framework="", slot=round_dir
    )
    measured: dict[str, Any] = {"status": "succeeded", "launch_evidence": evidence}
    refuse_cacheless_agentx_baseline(measured)
    return str(measured.get("error") or "") if measured["status"] != "succeeded" else ""
