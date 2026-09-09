# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for per-phase handler collaborators."""

from __future__ import annotations

from ..collaborator import CoordinatorCollaborator


class PhaseHandler(CoordinatorCollaborator):
    """A coordinator collaborator that owns one phase's methods."""
