# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The critic agent's own run, for SBD V6.

``critic`` is a top-level V6 key rather than a timeline event: the agent
reviews the session as a whole and its iterations are not a stage competing
for wall-clock with the others. The per-proposal verdicts stay with the
proposals they judge; what this key carries is the agent's own run.

The iterations are recorded when each review comes back (see
:mod:`..recorder.critic_out`), so this collector only puts them on the wire.
There is no projection behind it: an iteration's topic and verdict exist only
in the exchange that produced them, and rebuilding those from whatever
workdirs survived pruning is what the section it replaces did.
"""

from __future__ import annotations

from typing import Any

from ._common import _dict_rows, _mapping

__all__ = ["collect_v6_critic"]


def collect_v6_critic(recorded: Any = None) -> dict[str, Any]:
    """Put the recorded critic iterations on the wire.

    Args:
        recorded (Any): The assembled ``critic`` view, when present.

    Returns:
        dict[str, Any]: ``{"iterations": [...]}``. An empty ``iterations`` says
        the critic never completed a review, which is itself an answer.
    """
    return {"iterations": _dict_rows(_mapping(recorded).get("iterations"))}