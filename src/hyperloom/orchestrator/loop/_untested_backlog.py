# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A new specialist waits while the proposals waiting for a bench fill two automatic explore grids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.common.export_gate import export_gate_is_shut

from ..phases.machine_state import PHASE_FRAMEWORK_AGENT
from ..phases.proposal_bench import UNTESTED_BACKLOG_CAP, waiting_proposal_rows
from ..policy.denial import PolicyDenied
from ..policy.gate import SPECIALIST_ACTION_NAME
from ..specialists.domains import is_custom_specialist_dispatch

__all__ = ["SPECIALIST_UNTESTED_BACKLOG_FULL", "untested_backlog_denial"]

SPECIALIST_UNTESTED_BACKLOG_FULL = "specialist_untested_backlog_full"


def untested_backlog_denial(
    action_name: str,
    params: Mapping[str, Any] | None,
    shared_state: Any,
    tasks: Any,
) -> PolicyDenied | None:
    """Refuse a new FRAMEWORK_AGENT specialist while ``UNTESTED_BACKLOG_CAP`` proposals wait for a bench.

    The cap holds back dispatch only: a specialist already queued or running still records every proposal it
    returns. The operator-defined specialist, which the Coordinator dispatches once per macro cycle, is exempt.
    """
    if action_name != SPECIALIST_ACTION_NAME or is_custom_specialist_dispatch(dict(params or {})):
        return None
    if str(shared_state.phase or "").strip().upper() != PHASE_FRAMEWORK_AGENT:
        return None
    if export_gate_is_shut(shared_state):
        # Nothing can drain the backlog: the automatic bench is held for the same reason,
        # and a patch specialist is the only lane that can land the export that reopens
        # both. Holding one back here would stall the run on a queue it cannot empty.
        return None
    waiting = len(waiting_proposal_rows(shared_state, tasks))
    if waiting < UNTESTED_BACKLOG_CAP:
        return None
    return PolicyDenied(
        f"delegate{{action='specialist'}}: {waiting} untested proposals are waiting for a bench "
        f"(cap {UNTESTED_BACKLOG_CAP})",
        rule=SPECIALIST_UNTESTED_BACKLOG_FULL,
        hint=(
            "Let explore grids bench the waiting proposals; a new specialist is admitted once fewer than "
            f"{UNTESTED_BACKLOG_CAP} wait."
        ),
    )
