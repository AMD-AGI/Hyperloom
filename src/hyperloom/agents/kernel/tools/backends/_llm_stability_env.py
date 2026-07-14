# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared LLM-transport stability env for claude-CLI / claude-agent-sdk / forge children.

Hyperloom RCA (Sandbox hang): a streaming request to the SaFE/LiteLLM gateway
can return a partial response (``stop_reason=None``) and then stop pushing
chunks while the TCP connection stays alive. Hyperloom-owned idle monitors
(process-log/heartbeat stale checks, or per-message SDK ``wait_for``) should
decide whether that stream is stalled.

``API_TIMEOUT_MS`` is opt-in because external clients may interpret it as a
total request timeout, which could kill a legitimate long streaming response.
The default helper only cuts non-essential / auto-update traffic that can also
block in headless containers.

This is the single source of truth for those knobs; ``forge_submit`` already
applied them inline (its original RCA fix) and now delegates here. ``setdefault``
semantics keep any operator-provided override authoritative.
"""

from __future__ import annotations

from typing import MutableMapping

# Match forge's original mitigation (Hyperloom forge RCA root cause 4): 5 min.
DEFAULT_API_TIMEOUT_MS = "300000"

__all__ = ["DEFAULT_API_TIMEOUT_MS", "apply_llm_stability_env"]


def apply_llm_stability_env(
    env: MutableMapping[str, str],
    *,
    api_timeout_ms: str | None = None,
) -> None:
    """Inject client-side LLM-transport timeout/stability knobs into ``env``.

    Mutates ``env`` in place via ``setdefault`` (operator overrides win). Safe to
    call on a child-process env dict or on ``os.environ`` directly.

    Args:
        env: The environment mapping to harden (mutated in place).
        api_timeout_ms: Optional per-request claude-code timeout, in
            milliseconds. ``None`` leaves ``API_TIMEOUT_MS`` untouched.
    """
    if api_timeout_ms is not None:
        env.setdefault("API_TIMEOUT_MS", str(api_timeout_ms))
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
