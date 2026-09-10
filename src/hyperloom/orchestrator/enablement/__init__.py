# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement: make a ``(model, backend)`` combination runnable at all.

Public surface — data types and pure functions from the modules that have
cross-package consumers.  The four ``CoordinatorCollaborator`` subclasses
(``EnablementLane``, ``EnablementParams``, ``EnablementBuild``,
``EnablementRevalidation``) are *not* exported here: they are resolved by
``Coordinator._COLLAB_MODULES`` via dotted strings, and exposing them would
invite callers to instantiate them outside the coordinator.

This follows the shape of ``bringup/__init__.py``.
"""

from __future__ import annotations

from .mandate import build_mandate, build_search_plan, score_enablement_title

__all__ = [
    "build_mandate",
    "build_search_plan",
    "score_enablement_title",
]
