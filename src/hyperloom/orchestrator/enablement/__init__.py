# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement: make a ``(model, backend)`` combination runnable at all.

Exports the pure functions with consumers outside the package. The four
collaborator classes are withheld; they are instantiated inside the Coordinator
and accessed through its collaborator properties.
"""

from __future__ import annotations

from .mandate import build_mandate, build_search_plan, score_enablement_title

__all__ = [
    "build_mandate",
    "build_search_plan",
    "score_enablement_title",
]
