# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement: make a ``(model, backend)`` combination runnable at all.

Exports the pure functions with consumers outside the package. The four
collaborator classes are withheld: they are bases of ``Coordinator``, not objects
to instantiate on their own.
"""

from __future__ import annotations

from .mandate import build_mandate, build_search_plan, score_enablement_title

__all__ = [
    "build_mandate",
    "build_search_plan",
    "score_enablement_title",
]
