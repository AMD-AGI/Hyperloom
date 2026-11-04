# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A group-readable view of ``state.json`` for the retest runner.

``state.json`` is the program's own persistence and stays owner-only and unmasked. The retest runner (the
``retest_flow.py`` tool of the shared retest area) reads only the fields below, so ``SharedState.save`` writes just
those, with credentials scrubbed once, to ``retest_view.json``. No ``SharedState`` reader loads it back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values
from hyperloom_common.io import WriteOptions, atomic_write_json

RETEST_VIEW = "retest_view.json"

#: The top-level fields retest_flow.py reads from state.json.
VIEW_FIELDS = ("framework", "framework_version", "baseline_config_path", "baseline_tput", "claw_session_id")
#: The optimization_stack entry fields it reads (entry_config, entry_overlays and the flow's target_entry record).
STACK_ENTRY_FIELDS = (
    "action",
    "variant_name",
    "task_id",
    "workspace",
    "source_snapshot",
    "patch_path",
    "target_file",
    "framework_root",
    "tput",
    "gain_pct",
    "accuracy",
    "lever_kind",
    "scope",
)


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secret_values(value)
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def write_retest_view(session_dir: Path, state: Any) -> None:
    """Write ``retest_view.json`` (group-readable inside a setgid shared tree) from ``state``'s retest fields."""
    view = {name: getattr(state, name, None) for name in VIEW_FIELDS}
    view["optimization_stack"] = [
        {name: entry[name] for name in STACK_ENTRY_FIELDS if name in entry} for entry in state.optimization_stack
    ]
    atomic_write_json(
        Path(session_dir) / RETEST_VIEW, _scrub(view), options=WriteOptions(make_parents=True, group_read=True)
    )
