# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data sources used by the reactor.

The local probe is the only collector; :class:`DegradeRouter` keeps it
behind a silent fallback so a failing probe degrades to "no data" instead
of failing the tick. The contract is narrow so each source is a drop-in
replacement.
"""

from .base import (
    DegradeRouter,
    HealthState,
    Source,
    SourceData,
    SourceUnavailable,
)
from .local_probe import LocalProbeConfig, LocalProbeSource

__all__ = [
    "DegradeRouter",
    "HealthState",
    "LocalProbeConfig",
    "LocalProbeSource",
    "Source",
    "SourceData",
    "SourceUnavailable",
]
