# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for Coordinator collaborators."""

from __future__ import annotations

from typing import Any


class CoordinatorCollaborator:
    """An object that borrows the Coordinator's attributes for its own methods."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_coord"), name)
