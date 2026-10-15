# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a warm recipe changes anything about the launch the baseline was measured with."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    SKIP_BEST_CONFIG_EMPTY,
    SKIP_RECIPE_ALREADY_IN_BASELINE,
)

from ..actions.executors.launch_change import launch_change, stack_launch_evidence


def replay_skip_code(
    state: Any,
    args: str,
    envs: Mapping[str, str],
    *,
    patches: Sequence[Any],
    kernel_pending: Sequence[Any],
) -> str:
    """The skip code for a recipe that changes nothing about the baseline launch; "" when it changes something.

    A recipe carrying patches or kernel work changes more than the launch, so it always replays.
    """
    if patches or kernel_pending:
        return ""
    if not args and not envs:
        return SKIP_BEST_CONFIG_EMPTY
    change = launch_change(
        stack_launch_evidence(state),
        str(getattr(state, "framework", "") or ""),
        extra_args=args,
        extra_envs=envs,
    )
    return "" if change else SKIP_RECIPE_ALREADY_IN_BASELINE
