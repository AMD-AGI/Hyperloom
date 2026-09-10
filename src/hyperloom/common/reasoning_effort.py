# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one reasoning-effort vocabulary Hyperloom and KernelForge both speak.

Five names, because five is what the deeper surface can express. The Claude
CLI's ``--effort`` takes ``low | medium | high | xhigh | max``; the
OpenAI-compatible gateway's ``reasoning_effort`` takes
``none | minimal | low | medium | high | xhigh`` and returns 400 on ``max``.
``low``..``xhigh`` are common to both and are levels as written. ``max`` is a
real Claude level, so it is one here too, and :func:`gateway_reasoning_effort`
projects it onto ``xhigh`` -- the gateway's deepest -- for anything speaking
the OpenAI protocol. ``minimal`` and ``none`` go the other way: the gateway
takes them but the Claude CLI does not know them, and there is no Claude level
below ``low`` to project them onto, so they are not levels at all.

Before this module each component carried its own table, and the tables
disagreed -- ``HYPERLOOM_REASONING_EFFORT=minimal`` was accepted by Hyperloom's
own calls and then crashed the Forge Codex backend, which is exactly the
failure a shared vocabulary exists to prevent. The projection lives here for
the same reason: ``max`` used to be folded inside the Codex backend only, so
the identical value reaching Hyperloom's own chat.completions was a 400.

The ordering is meaningful: an index is the level's rank, so a call site that
caps a session can compare what it allows against what the deployment asked
for. Nothing here talks to a provider -- this module is a name list, so both
components can import it without importing a transport.
"""

from __future__ import annotations


#: Accepted levels, cheapest first. Index is rank.
REASONING_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

#: Rank per level, for comparing a requested effort against a ceiling.
REASONING_EFFORT_RANK: dict[str, int] = {name: rank for rank, name in enumerate(REASONING_EFFORT_LEVELS)}

#: What a session runs at when neither the operator nor the call site names one.
DEFAULT_REASONING_EFFORT = "high"


def normalize_reasoning_effort(value: str | None) -> str:
    """Return ``value`` as a canonical level name, or ``""`` when it is not one.

    Empty string is the single "not a level" answer, so callers choose their own
    response to an unrecognized name -- ignoring it, or refusing the run --
    rather than inheriting one from here.
    """
    name = (value or "").strip().lower()
    return name if name in REASONING_EFFORT_RANK else ""


#: Levels an OpenAI-compatible gateway cannot be told by name, and the level it
#: is told instead. ``max`` is Claude's deepest; ``xhigh`` is the gateway's, so
#: the projection preserves "as deep as this surface goes" rather than silently
#: dropping the operator a step.
_GATEWAY_PROJECTION: dict[str, str] = {"max": "xhigh"}


def gateway_reasoning_effort(value: str | None) -> str:
    """Return the level to send over the OpenAI protocol, or ``""``.

    Every caller that puts an effort in a ``reasoning_effort`` field or a Codex
    config goes through here rather than through
    :func:`normalize_reasoning_effort` directly, because the name the operator
    chose and the name that surface accepts are not always the same one.
    """
    name = normalize_reasoning_effort(value)
    return _GATEWAY_PROJECTION.get(name, name)
