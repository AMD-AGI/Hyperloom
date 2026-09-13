# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delivery layer: what a round declared, and what its apply can undo."""

from __future__ import annotations

from hyperloom.orchestrator.delivery.archive import (
    ROLE_LAUNCH_CONFIG,
    ROLE_PATCH,
    ROLE_PATCH_EVIDENCE,
    ROLE_PROMPT,
    ROLE_SERVER_LOG,
    ROLE_SPECIALIST_RESULT,
    ArchivedFile,
    RoundArchive,
)
from hyperloom.orchestrator.delivery.deliverable import (
    Artifact,
    Deliverable,
    DeliverableRefused,
    parse_deliverable,
)
from hyperloom.orchestrator.delivery.ledger import file_digest, load_records

__all__ = [
    "ROLE_LAUNCH_CONFIG",
    "ROLE_PATCH",
    "ROLE_PATCH_EVIDENCE",
    "ROLE_PROMPT",
    "ROLE_SERVER_LOG",
    "ROLE_SPECIALIST_RESULT",
    "ArchivedFile",
    "Artifact",
    "Deliverable",
    "DeliverableRefused",
    "RoundArchive",
    "file_digest",
    "load_records",
    "parse_deliverable",
]
