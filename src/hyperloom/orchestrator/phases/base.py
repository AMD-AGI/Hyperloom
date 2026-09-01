# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for per-phase handler collaborators.

Each phase's coordinator methods live in a ``PhaseHandler`` subclass. The
attribute delegation it relies on lives in
:class:`~hyperloom.orchestrator.collaborator.CoordinatorCollaborator`; this
subclass exists to say that the collaborator is bound to one phase of the
state machine, which the ``_on_enter_*`` dispatch in ``phases/machine.py``
keys on.
"""

from __future__ import annotations

from ..collaborator import CoordinatorCollaborator


class PhaseHandler(CoordinatorCollaborator):
    """A coordinator collaborator that owns one phase's methods."""
