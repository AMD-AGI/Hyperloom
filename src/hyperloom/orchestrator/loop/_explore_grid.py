# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore reaches dispatch with variants to measure, none of them a file change a ``--no-source-changes`` session refuses."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import launch_settings_naming_path_under

from ..actions.executors.explore_prefilter.grid_payload import runnable_grid_variants
from ..policy.denial import PolicyDenied

__all__ = [
    "EXPLORE_ACTION_NAME",
    "EXPLORE_GRID_MISSING",
    "EXPLORE_NAMES_SESSION_FILE",
    "explore_grid_denial",
    "explore_session_file_denial",
]

# Grid measurement action name; the gate is called for every action, so it needs its own name to
# answer only for itself.
EXPLORE_ACTION_NAME: str = "explore"

EXPLORE_GRID_MISSING = "explore_grid_missing"
EXPLORE_NAMES_SESSION_FILE = "explore_names_session_file"


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


def explore_session_file_denial(
    action_name: str,
    params: Mapping[str, Any] | None,
    shared_state: Any,
    session_dir: Path,
) -> PolicyDenied | None:
    """Refuse, under ``--no-source-changes``, an explore with a variant whose setting names a file of this session.

    Everything a specialist or a stage writes lands under the session directory, so a setting that points the server
    there carries that file into the kept launch by name: a tuned table, a template, a script. The integrate gate
    refuses the same file when it arrives as a deliverable; a grid is the other way a setting reaches a kept step.
    """
    if action_name != EXPLORE_ACTION_NAME or not getattr(shared_state, "no_source_changes", False):
        return None
    naming = {
        variant.name: settings
        for variant in runnable_grid_variants(params, shared_state)
        if (settings := launch_settings_naming_path_under(session_dir, variant.extra_server_args, variant.extra_envs))
    }
    if not naming:
        return None
    detail = "; ".join(f"{name}: {', '.join(settings)}" for name, settings in naming.items())
    return PolicyDenied(
        f"{EXPLORE_ACTION_NAME}: --no-source-changes keeps server arguments and environment variables only, and "
        f"these variants point the server at a file under the session directory ({detail})",
        rule=EXPLORE_NAMES_SESSION_FILE,
        hint="Resend the grid without those variants, or with values that name no file this session wrote.",
    )
