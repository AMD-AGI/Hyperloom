# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR Monitor client stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRMonitorClient:
    """Minimal PR Monitor stub."""

    enabled: bool = True

    @classmethod
    def from_args(cls, *, enabled: bool = True) -> "PRMonitorClient":
        """Build a client from the enablement flag."""
        return cls(enabled=bool(enabled))


__all__ = [
    "PRMonitorClient",
]
