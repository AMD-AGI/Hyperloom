# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Refuse an AgentX baseline whose server ran with the prefix cache off."""

from __future__ import annotations

import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)

#: The ``error_class`` of a baseline refused for running its replay with the prefix cache off. Writeback owns what
#: happens next: it stops the run on the first occurrence and keeps the refusal out of the enablement lane.
PREFIX_CACHE_OFF_ERROR_CLASS = "agentx_baseline_prefix_cache_off"

#: ATOM's prefix cache is off unless this flag is passed, and ATOM prints no launch record to read it back from.
_ATOM_PREFIX_CACHE_FLAG = "enable_prefix_caching"


def _atom_evidence(requested_server_args: str) -> str:
    flags = {token.lstrip("-").replace("-", "_") for token in requested_server_args.split()}
    return "" if _ATOM_PREFIX_CACHE_FLAG in flags else "the ATOM server was launched without --enable_prefix_caching"


def prefix_cache_off_evidence(result: Mapping[str, Any]) -> str:
    """Why this AgentX measurement ran with the prefix cache off, or ``""``.

    Each engine is read from its own account of the launch: SGLang's resolved ``disable_radix_cache``, vLLM's launch
    record (which lists ``enable_prefix_caching`` only when the launch turned it off), and, since ATOM prints no record
    and keeps the cache off by default, the server args it was launched with. A run none of these can be read from is
    left alone.
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
        return _atom_evidence(str(evidence.get("requested_server_args") or ""))
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
        "--no-enable-prefix-caching; ATOM: pass --enable_prefix_caching), as upstream's agentic recipes do."
    )
    log.error("baseline_executor: %s", result["error"])
