# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Refuse an AgentX baseline whose server ran with the prefix cache off."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from inference_suite.argv import argv_flag_occurrences

from ._round_folder import is_round_folder, read_round_launch

log = logging.getLogger(__name__)

#: The ``error_class`` of a baseline refused for running its replay with the prefix cache off. Writeback owns what
#: happens next: it stops the run on the first occurrence and keeps the refusal out of the enablement lane.
PREFIX_CACHE_OFF_ERROR_CLASS = "agentx_baseline_prefix_cache_off"

#: ATOM's prefix-cache switch (an argparse ``BooleanOptionalAction``, on by default) under both its spellings. ATOM
#: prints no launch record to read the resolved setting back from, so the argv decides, its last spelling winning.
_ATOM_CACHE_ON, _ATOM_CACHE_OFF = "enable-prefix-caching", "no-enable-prefix-caching"


def _atom_evidence(workspace: str | None) -> str:
    """Read from the argv the round's ``launch.json`` records: the base the session froze and every layer above it."""
    run_dir = Path(workspace) if workspace else None
    launch = read_round_launch(run_dir) if run_dir is not None and is_round_folder(run_dir) else None
    if launch is None:
        return ""
    names = (name for name, _value in reversed(argv_flag_occurrences(launch.argv)))
    switch = next((name for name in names if name in (_ATOM_CACHE_ON, _ATOM_CACHE_OFF)), _ATOM_CACHE_ON)
    return "the ATOM server was launched with --no-enable_prefix_caching" if switch == _ATOM_CACHE_OFF else ""


def prefix_cache_off_evidence(result: Mapping[str, Any]) -> str:
    """Why this AgentX measurement ran with the prefix cache off, or ``""``.

    Each engine is read from its own account of the launch: SGLang's resolved ``disable_radix_cache``, vLLM's launch
    record (which lists ``enable_prefix_caching`` only when the launch turned it off), and, since ATOM prints no
    record, the argv the round's ``launch.json`` in ``result["workspace"]`` records. A run none of these can be read
    from is left alone.
    """
    evidence = result.get("launch_evidence")
    if not isinstance(evidence, Mapping):
        return ""
    server_config = evidence.get("observed_server_config") or {}
    framework = str(evidence.get("framework") or "")
    if framework == "sglang" and server_config.get("disable_radix_cache") is True:
        return "the SGLang server ran with disable_radix_cache=True"
    if framework == "vllm" and server_config.get("enable_prefix_caching") is False:
        return "the vLLM server ran with enable_prefix_caching=False"
    if framework == "atom":
        return _atom_evidence(result.get("workspace"))
    return ""


def refuse_cacheless_agentx_baseline(result: dict[str, Any]) -> None:
    """Mark a succeeded AgentX baseline measured with the prefix cache off as failed.

    The caller runs it only for a genuine AgentX baseline, the measurement every later candidate is graded against; a
    candidate validated through a ``baseline``-kind task (``quality_ref_exempt``) is measured as it ran.
    """
    if result.get("status") != "succeeded":
        return
    evidence = prefix_cache_off_evidence(result)
    if not evidence:
        return
    result["status"] = "failed"
    result["error_class"] = PREFIX_CACHE_OFF_ERROR_CLASS
    result["error"] = (
        f"AgentX baseline refused: {evidence}. An AgentX replay is mostly shared prefix and every later candidate is "
        "graded against this baseline, so any change that turns the cache back on would be credited with the cache "
        "itself as a gain. Keep the prefix cache on for agentic replays (SGLang: no --disable-radix-cache; vLLM: no "
        "--no-enable-prefix-caching; ATOM: no --no-enable_prefix_caching), as upstream's agentic recipes do."
    )
    log.error("baseline_executor: %s", result["error"])
