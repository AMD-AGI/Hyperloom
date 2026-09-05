# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delivery layer: what a round declared, what it was validated against, what shipped."""

from __future__ import annotations

from hyperloom.orchestrator.delivery.archive import (
    ROLE_LAUNCH_CONFIG,
    ROLE_PATCH,
    ROLE_PROMPT,
    ROLE_SERVER_LOG,
    ROLE_SPECIALIST_RESULT,
    ArchivedFile,
    RoundArchive,
)
from hyperloom.orchestrator.delivery.deliverable import (
    NO_PRE_IMAGE,
    Artifact,
    Deliverable,
    DeliverableRefused,
    freeze_digests,
    mismatched_recorded_artifacts,
    parse_deliverable,
)
from hyperloom.orchestrator.delivery.ledger import load_records
from hyperloom.orchestrator.delivery.manifest import (
    TreeBaseline,
    baseline_path,
    capture_baseline,
    drifted_paths,
    file_digest,
    post_images_from_diff,
    write_baseline,
    write_post_images,
)

__all__ = [
    "NO_PRE_IMAGE",
    "ROLE_LAUNCH_CONFIG",
    "ROLE_PATCH",
    "ROLE_PROMPT",
    "ROLE_SERVER_LOG",
    "ROLE_SPECIALIST_RESULT",
    "ArchivedFile",
    "Artifact",
    "Deliverable",
    "DeliverableRefused",
    "RoundArchive",
    "TreeBaseline",
    "baseline_path",
    "capture_baseline",
    "drifted_paths",
    "file_digest",
    "freeze_digests",
    "load_records",
    "mismatched_recorded_artifacts",
    "parse_deliverable",
    "post_images_from_diff",
    "write_baseline",
    "write_post_images",
]
