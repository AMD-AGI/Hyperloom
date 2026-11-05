# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experiment blackboard's contracts: what the other modules code against.

* :mod:`.records`: what the blackboard keeps and answers with: test records, claims, lineage, progress, startup briefs,
  the session profile and snapshot;
* :mod:`.vocab`: its vocabulary: writer roles, claim and dedup states, conclusions, lineage kinds;
* :mod:`.interfaces`: one protocol per interface the other modules use, and the errors they raise;
* :mod:`.peers`: the objects other modules define, re-exported: the toolbox's test requests and results, and the
  intents, judgements and branch-log entries.
* :mod:`.exhausted`: the branch-log record a specialist writes when its direction is done on a head.

``hyperloom_multibranch.blackboard.LocalBlackboard`` implements every protocol and is the fake the skeleton runs on.
"""

from hyperloom_multibranch.contracts.blackboard.interfaces import (
    AlreadyTested,
    CeilingTooLow,
    ClaimHeld,
    ClaimLost,
    ConflictingWrite,
    Blackboard,
    BlackboardError,
    NotRecorded,
    UnknownRequest,
    WriterNotAllowed,
)
from hyperloom_multibranch.contracts.blackboard.records import ClaimKey, Writer
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, Role

__all__ = [
    "INTEGRATION_BRANCH",
    "AlreadyTested",
    "CeilingTooLow",
    "ClaimHeld",
    "ClaimKey",
    "ClaimLost",
    "ConflictingWrite",
    "NotRecorded",
    "Blackboard",
    "BlackboardError",
    "Role",
    "UnknownRequest",
    "Writer",
    "WriterNotAllowed",
]
