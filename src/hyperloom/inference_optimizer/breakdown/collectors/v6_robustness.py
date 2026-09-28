# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read historical robustness fragments without synthesizing new turns.

The fixed V6 wire key remains empty for sessions without recorded turns.
Historical fragments pass through unchanged; workdirs and finding logs are
not sources for rebuilding the retired agent's account.
"""

from __future__ import annotations

from typing import Any

from ._common import _dict_rows, _mapping

__all__ = ["collect_v6_robustness"]


def collect_v6_robustness(recorded: Any = None) -> dict[str, Any]:
    """Put the recorded robustness turns on the wire.

    Args:
        recorded (Any): The assembled ``robustness`` view, when present.

    Returns:
        dict[str, Any]: ``{"turns": [...]}``. Always a full object: an empty
        ``turns`` says the agent never completed a turn, which is itself the
        answer a reader is after.
    """
    return {"turns": _dict_rows(_mapping(recorded).get("turns"))}
