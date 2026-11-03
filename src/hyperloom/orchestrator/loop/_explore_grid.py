# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore reaches dispatch with variants to measure: its own grid, or a seed source that fills one."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..actions.executors.explore import runnable_grid_variants
from ..policy.denial import PolicyDenied

__all__ = ["EXPLORE_ACTION_NAME", "EXPLORE_GRID_MISSING", "explore_grid_denial"]

# Grid measurement action name; the gate is called for every action, so it needs its own name to
# answer only for itself.
EXPLORE_ACTION_NAME: str = "explore"

EXPLORE_GRID_MISSING = "explore_grid_missing"


def explore_grid_denial(
    action_name: str,
    params: Mapping[str, Any] | None,
    shared_state: Any,
) -> PolicyDenied | None:
    """Refuse an explore with no grid that no seed source would fill, naming what to send instead.

    The round answers the same question from the materialized config minutes later and fails the
    task with ``error_class="empty_grid"``; answering it at admission spends no dispatch on a round
    that cannot start, and hands the caller a denial it can act on.

    It lives in ``loop`` rather than beside the other gates in ``policy``: the question is the
    round's own, so it is asked of ``actions.executors.explore``, and ``policy`` sits under
    ``actions`` in the architecture contracts.
    """
    if action_name != EXPLORE_ACTION_NAME:
        return None
    # The round's own reckoning of what it would measure, so the two cannot answer differently.
    # Variants the round then drops with the materialized config in hand (config-pinned envs,
    # baseline no-ops) stay its to report, not admission's to predict.
    if runnable_grid_variants(params, shared_state):
        return None
    rows = (params or {}).get("grid")
    sent = rows if isinstance(rows, list) else []
    # The framework the round will materialize and seed from, as ``runnable_grid_variants`` reads it.
    framework = str(getattr(shared_state, "framework", "") or "") or "?"
    detail = (
        f"params.grid holds {len(sent)} row(s) but no variant dict with a name, and a grid that was sent "
        "is never replaced by a default one"
        if sent
        else "params.grid is empty and this session has no default grid, framework-rewrite lever or "
        "declared lever to seed one from"
    )
    return PolicyDenied(
        f"{EXPLORE_ACTION_NAME}{{framework='{framework}'}}: {detail}",
        rule=EXPLORE_GRID_MISSING,
        hint=(
            "Send params.grid as a non-empty list of variant dicts "
            "({'name': ..., 'extra_server_args': ..., 'extra_envs': {...}}) built from the GRID INPUT "
            "and specialist proposals; an explore with nothing to measure has no result to report."
        ),
    )
