# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why a forge-loop run was refused, in words meant for the operator who started it."""

from __future__ import annotations


class ForgeLoopError(Exception):
    """The run cannot start or cannot go on; ``exit_code`` is the status a command line leaves with."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ForgeLoopUsageError(ForgeLoopError):
    """The options the caller passed cannot be used together or as given."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=2)


class InvalidOptionValue(ForgeLoopUsageError):
    """One option holds a value the run cannot use; ``param_hint`` names it when the message does not."""

    def __init__(self, message: str, *, param_hint: str = "") -> None:
        super().__init__(message)
        self.param_hint = param_hint


class TaskPreparationFailed(ForgeLoopError):
    """Task preparation could not make the driver conform; the run's published result already says why."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=2)


__all__ = ["ForgeLoopError", "ForgeLoopUsageError", "InvalidOptionValue", "TaskPreparationFailed"]
