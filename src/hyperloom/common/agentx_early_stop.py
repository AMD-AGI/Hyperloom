# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The modes of the AgentX early stop, read from one variable by single-lineage and multi-branch alike.

Each flow passes its own default: single-lineage enforces, multi-branch only records the ruling.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from hyperloom_common.env import EnvValueError

EARLY_STOP_ENV = "HYPERLOOM_AGENTX_EARLY_STOP"


class EarlyStopMode(str, Enum):
    """``enforce`` records the ruling and stops a trailing round, ``shadow`` only records it, ``off`` skips it."""

    ENFORCE = "enforce"
    SHADOW = "shadow"
    OFF = "off"


def early_stop_mode(environ: Mapping[str, str], default: EarlyStopMode) -> EarlyStopMode:
    """The mode ``environ`` names, ``default`` when it names none; any other value raises ``EnvValueError``."""
    raw = (environ.get(EARLY_STOP_ENV) or "").strip().lower()
    if not raw:
        return default
    try:
        return EarlyStopMode(raw)
    except ValueError:
        modes = ", ".join(mode.value for mode in EarlyStopMode)
        raise EnvValueError(f"{EARLY_STOP_ENV}={raw!r} is not one of {modes}") from None


__all__ = ["EARLY_STOP_ENV", "EarlyStopMode", "early_stop_mode"]
