# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a backend carries an intent, as prompt-rendering vocabulary."""

from __future__ import annotations

TRANSPORT_TOOLS = "tools"
TRANSPORT_STRUCTURED_OUTPUT = "structured_output"
TRANSPORTS: frozenset[str] = frozenset({TRANSPORT_TOOLS, TRANSPORT_STRUCTURED_OUTPUT})


__all__ = [
    "TRANSPORTS",
    "TRANSPORT_STRUCTURED_OUTPUT",
    "TRANSPORT_TOOLS",
]
