# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A propose_action intent waiting for Critic Review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PendingProposal:
    """A propose_action intent waiting for Critic Review."""

    proposal_msg_id: str
    from_agent: str
    action_name: str
    predicted_gain_pct: float
    payload: dict[str, Any]
    #: The task this proposal materialized into, once the Critic approved it.
    task_id: str | None = None


__all__ = ["PendingProposal"]
