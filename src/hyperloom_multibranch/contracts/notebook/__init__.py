# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experiment notebook's contracts: what the other modules code against.

* :mod:`.records`: what the notebook keeps and answers with: test records, claims, lineage, progress, startup briefs,
  the session profile and snapshot;
* :mod:`.vocab`: its vocabulary: writer roles, claim and dedup states, conclusions, lineage kinds;
* :mod:`.interfaces`: one protocol per interface of multi-branch design section 5.1, and the errors they raise;
* :mod:`.peers`: the objects other modules define, re-exported: the toolbox's test requests and results, and the
  intents, judgements and branch-log entries.

``hyperloom_multibranch.notebook.LocalNotebook`` implements every protocol and is the fake the skeleton runs on.
"""

from hyperloom_multibranch.contracts.notebook.interfaces import (
    AlreadyTested,
    CeilingTooLow,
    ClaimHeld,
    ClaimLost,
    ConflictingWrite,
    Notebook,
    NotebookError,
    NotRecorded,
    UnknownRequest,
    WriterNotAllowed,
)
from hyperloom_multibranch.contracts.notebook.records import ClaimKey, Writer
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Role

__all__ = [
    "INTEGRATION_BRANCH",
    "AlreadyTested",
    "CeilingTooLow",
    "ClaimHeld",
    "ClaimKey",
    "ClaimLost",
    "ConflictingWrite",
    "NotRecorded",
    "Notebook",
    "NotebookError",
    "Role",
    "UnknownRequest",
    "Writer",
    "WriterNotAllowed",
]
