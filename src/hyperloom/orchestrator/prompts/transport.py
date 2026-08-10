# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a backend carries an intent, as prompt-rendering vocabulary.

The Claude backend mounts ``emit_intent`` and the read-only context tools as
real tools; the Codex backend has no tool surface and is bound by a
provider-enforced output schema instead. A prompt block that documents one is
an instruction the other cannot follow, so the prompt builder scopes those
blocks by transport and each backend declares its own.

This is a leaf module on purpose: backends state a fact about themselves
without taking a dependency on the prompt builder that reads it.
"""

from __future__ import annotations

TRANSPORT_TOOLS = "tools"
TRANSPORT_STRUCTURED_OUTPUT = "structured_output"
TRANSPORTS: frozenset[str] = frozenset({TRANSPORT_TOOLS, TRANSPORT_STRUCTURED_OUTPUT})


__all__ = [
    "TRANSPORTS",
    "TRANSPORT_STRUCTURED_OUTPUT",
    "TRANSPORT_TOOLS",
]
