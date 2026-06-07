# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Data sources used by the reactor.

Layered as ``robustness-server`` (preferred) -> ``local probe`` (fallback)
through :class:`DegradeRouter`. Higher milestones add a cluster proxy
client; the contract here is intentionally narrow so each source is a
drop-in replacement.
"""

from .base import (
    DegradeRouter,
    HealthState,
    Source,
    SourceData,
    SourceUnavailable,
)
from .local_probe import LocalProbeConfig, LocalProbeSource
from .server_client import RobustnessServerClient, RobustnessServerSource

__all__ = [
    "DegradeRouter",
    "HealthState",
    "LocalProbeConfig",
    "LocalProbeSource",
    "RobustnessServerClient",
    "RobustnessServerSource",
    "Source",
    "SourceData",
    "SourceUnavailable",
]
