# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared LLM-transport stability env for spawned claude-CLI / SDK children.

Orchestrator-side twin of ``kernel-agent/tools/backends/_llm_stability_env.py``
(the two packages are independent, so the helper is duplicated rather than
cross-imported). See that module for the full RCA: a streaming request to the
SaFE/LiteLLM gateway can return a partial response (``stop_reason=None``) and
then stop pushing chunks while the socket stays open.

Only local, Hyperloom-controlled idle monitors should decide whether a
long-running stream is stalled. ``API_TIMEOUT_MS`` is therefore opt-in: some
external clients may interpret it as a total request timeout, which could kill a
legitimate long streaming response. The default helper only cuts non-essential /
auto-update traffic that can block in headless containers. ``setdefault`` keeps
operator overrides authoritative.
"""

from __future__ import annotations

from typing import MutableMapping

# Match forge's original mitigation (5 min) so all paths behave identically.
DEFAULT_API_TIMEOUT_MS = "300000"

__all__ = ["DEFAULT_API_TIMEOUT_MS", "apply_llm_stability_env"]


def apply_llm_stability_env(
    env: MutableMapping[str, str],
    *,
    api_timeout_ms: str | None = None,
) -> None:
    """Inject client-side LLM-transport timeout/stability knobs into ``env``.

    Mutates ``env`` in place via ``setdefault`` (operator overrides win).

    Args:
        env: The environment mapping to harden (mutated in place).
        api_timeout_ms: Optional per-request claude-code timeout, in
            milliseconds. ``None`` leaves ``API_TIMEOUT_MS`` untouched.
    """
    if api_timeout_ms is not None:
        env.setdefault("API_TIMEOUT_MS", str(api_timeout_ms))
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
