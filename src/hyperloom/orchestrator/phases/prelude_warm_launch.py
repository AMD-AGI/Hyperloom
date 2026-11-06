# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recipe config a PRELUDE warm replay launches with, and the task it opens."""

from __future__ import annotations

from typing import Any

from ..state.task_registry import TaskSpec


def suppress_recipe_config(config: dict[str, Any]) -> None:
    """Blank a low-confidence recipe's own config so only its kernel section is replayed."""
    config["extra_server_args"] = ""
    config["extra_envs"] = {}
    config["config_source"] = ""
    config["config_tier"] = "suppressed_low_confidence"
    config["donor_expected_gain"] = 0.0


def warm_replay_task_spec(params: dict[str, Any], *, lanes: list[str] | None, ttl: int) -> TaskSpec:
    """The single PRELUDE warm-replay task ``params`` opens."""
    return TaskSpec(
        kind="replay_warm_recipe",
        params=params,
        idempotency_key="warm-replay-prelude",
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
        dispatch_class="coordinator",
    )
