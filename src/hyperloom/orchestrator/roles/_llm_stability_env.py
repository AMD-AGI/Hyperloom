# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared LLM-transport stability env for spawned claude-CLI / SDK children.

``API_TIMEOUT_MS`` is opt-in (some clients treat it as a total request timeout);
the default helper only cuts non-essential / auto-update traffic that can block
in headless containers. ``setdefault`` keeps operator overrides authoritative.
"""

from __future__ import annotations

from typing import MutableMapping

# Default per-request API timeout: 5 minutes (in ms).
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
