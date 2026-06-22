# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared LLM-transport stability env for spawned claude-CLI / SDK children.

Orchestrator-side twin of ``kernel-agent/tools/backends/_llm_stability_env.py``
(the two packages are independent, so the helper is duplicated rather than
cross-imported). See that module for the full RCA: a streaming request to the
SaFE/LiteLLM gateway can return a partial response (``stop_reason=None``) and
then stop pushing chunks while the socket stays open; without a client-side
request timeout the spawned ``claude`` CLI hangs forever on ``read()`` and
freezes the specialist -> coordinator -> optimizer chain.

``API_TIMEOUT_MS`` bounds the claude-code client's own HTTP request so a stalled
stream raises a normal error instead. ``setdefault`` keeps operator overrides
authoritative.
"""

from __future__ import annotations

from typing import MutableMapping

# Match forge's original mitigation (5 min) so all paths behave identically.
DEFAULT_API_TIMEOUT_MS = "300000"

__all__ = ["DEFAULT_API_TIMEOUT_MS", "apply_llm_stability_env"]


def apply_llm_stability_env(
    env: MutableMapping[str, str],
    *,
    api_timeout_ms: str = DEFAULT_API_TIMEOUT_MS,
) -> None:
    """Inject client-side LLM-transport timeout/stability knobs into ``env``.

    Mutates ``env`` in place via ``setdefault`` (operator overrides win).

    Args:
        env: The environment mapping to harden (mutated in place).
        api_timeout_ms: Per-request claude-code timeout, in milliseconds.
    """
    env.setdefault("API_TIMEOUT_MS", str(api_timeout_ms))
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
