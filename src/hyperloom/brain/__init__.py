# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain: what the run decides. It may import ``hyperloom.brain_api``, ``hyperloom.common`` and ``hyperloom_common``, nothing else."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain.v0 import BrainV0


def create_brain(config: Mapping[str, Any]) -> BrainV0:
    """The factory ``hyperloom.brain_api.loader`` calls."""
    return BrainV0(config)


__all__ = ["BrainV0", "create_brain"]
