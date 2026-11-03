# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which launch controls ``current_best`` carries after a KEEP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.inference_optimizer.grid_server_args import removals_absent_from_args
from hyperloom_common.coerce import to_str_list

LAUNCH_CONTROL_KEYS = ("remove_args", "unset_envs", "args_mode")


def kept_launch_controls(
    winner: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    extra_server_args: str | None,
    extra_envs: Mapping[str, Any],
) -> dict[str, Any]:
    """The launch controls the new ``current_best`` takes: the winner's when it declares any, else the previous best's.

    A winner that declares none (an append-mode source patch on top of a replace-mode config) leaves the launch
    config as it was, so the flags the stack already removed stay removed.

    ``extra_server_args`` and ``extra_envs`` are the composed launch state the same ``current_best`` stores, and the
    two deltas are pruned against it: a consumer lays that state down and applies the deltas after it, so a removal
    or an unset that still names something the snapshot asserts would delete what the winner was measured with. What
    is left names nothing here, so it keeps suppressing -- in whatever recipe or environment a consumer applies it to
    -- only what that base carries.
    """
    source = winner if any(winner.get(key) for key in LAUNCH_CONTROL_KEYS) else previous
    controls = {key: source[key] for key in LAUNCH_CONTROL_KEYS if source.get(key)}
    if "remove_args" in controls:
        controls["remove_args"] = removals_absent_from_args(extra_server_args, controls["remove_args"])
    if "unset_envs" in controls:
        controls["unset_envs"] = [name for name in to_str_list(controls["unset_envs"]) if name not in extra_envs]
    return {key: value for key, value in controls.items() if value}
