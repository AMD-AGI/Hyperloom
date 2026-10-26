# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which launch controls ``current_best`` carries after a KEEP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LAUNCH_CONTROL_KEYS = ("remove_args", "unset_envs", "args_mode")


def kept_launch_controls(winner: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """The launch controls the new ``current_best`` takes: the winner's when it declares any, else the previous best's.

    A winner that declares none (an append-mode source patch on top of a replace-mode config) leaves the launch
    config as it was, so the flags the stack already removed stay removed.
    """
    source = winner if any(winner.get(key) for key in LAUNCH_CONTROL_KEYS) else previous
    return {key: source[key] for key in LAUNCH_CONTROL_KEYS if source.get(key)}
