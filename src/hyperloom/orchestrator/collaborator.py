# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for Coordinator collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class CoordinatorCollaborator:
    """Base of every class ``Coordinator`` inherits.

    A collaborator's methods run with the Coordinator as ``self`` and use state and methods that
    ``Coordinator.__init__`` and the other collaborators define. A type checker cannot see those
    from one collaborator, so it is told that any other attribute exists.
    """

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...
