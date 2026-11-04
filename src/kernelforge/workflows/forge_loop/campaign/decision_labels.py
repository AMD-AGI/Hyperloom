# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The decision labels the forge-loop records for each finished iteration, in its events and its candidate archive.

Hyperloom-KB's ``kernelforge-run-v1`` declaration accepts exactly these values for ``outcome.decision``: collect skips
a candidate whose label it does not list.
"""

from __future__ import annotations

from enum import Enum


class DecisionLabel(str, Enum):
    """How the forge-loop decided one finished iteration: KEEP, or why nothing was kept."""

    KEEP = "KEEP"
    REVERT_PERF = "REVERT_PERF"
    REVERT_VALIDATION = "REVERT_VALIDATION"
    REVERT_VALIDATION_TIMEOUT = "REVERT_VALIDATION_TIMEOUT"
    REVERT_VALIDATION_ERROR = "REVERT_VALIDATION_ERROR"
    REVERT_INTEGRITY = "REVERT_INTEGRITY"
    REVERT_CONTENDED = "REVERT_CONTENDED"
    CRASH = "CRASH"
    BUILD_FAILED = "BUILD_FAILED"
    NO_CHANGES = "NO_CHANGES"
    API_ERROR = "API_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    ORCHESTRATION_ERROR = "ORCHESTRATION_ERROR"


DECISION_LABELS = frozenset(label.value for label in DecisionLabel)
