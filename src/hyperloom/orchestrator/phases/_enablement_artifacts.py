# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy enablement round deliverables into ``reports/enablement/<task_id>/``.

The archive collector drops ``runs/`` wholesale and retains ``reports/``, so a
patch or launch config left in the specialist workspace never reaches the
archive and the fix cannot be replayed by a later session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.inference_optimizer.session.session_paths import (
    enablement_round_dir,
    runs_dir,
)

# A patch is a few KB; anything this large is a stray build output and would
# eat into the archive's per-session budget.
_FILE_SIZE_LIMIT = 2 * 1024 * 1024

_LAUNCH_LOG_EXCERPT_CHARS = 1200


def _copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` when it exists and is under the size limit."""
    if not src.is_file() or src.stat().st_size > _FILE_SIZE_LIMIT:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def snapshot_round(session_dir: str | Path, res: dict[str, Any]) -> None:
    """Archive one enablement round's patches, specialist result and launch config.

    Rounds the phase synthesises carry no task id and no deliverables, and are
    skipped rather than colliding on a shared directory.

    Args:
        session_dir: The session root directory.
        res: The ``integrate_patch`` result for an enablement round.
    """
    task_id = str(res.get("specialist_task_id") or "").strip()
    if not task_id:
        return
    round_dir = enablement_round_dir(Path(session_dir), task_id)
    round_dir.mkdir(parents=True, exist_ok=True)

    launch_log = str(res.get("enablement_launch_log") or "")
    atomic_write_json(
        round_dir / "round.json",
        {
            "status": res.get("status"),
            "specialist_task_id": task_id,
            "patches_applied": res.get("patches_applied") or [],
            "config_changes_applied": res.get("config_changes_applied") or {},
            "extra_envs_applied": res.get("extra_envs_applied") or {},
            "extra_server_args_applied": res.get("extra_server_args_applied") or "",
            "setup_commands_applied": res.get("setup_commands_applied") or [],
            "framework_switch_problems": res.get("framework_switch_problems") or [],
            "after_signature": res.get("after_signature") or {},
            "enablement_accepted_config_path": res.get("enablement_accepted_config_path") or "",
            "enablement_effective_config": res.get("enablement_effective_config") or {},
            "launch_log_excerpt": launch_log[:_LAUNCH_LOG_EXCERPT_CHARS],
        },
    )

    patches_dir = round_dir / "patches"
    copied: set[str] = set()
    for applied in res.get("patches_applied") or []:
        src = Path(str(applied))
        _copy(src, patches_dir / src.name)
        copied.add(src.name)

    workspace = runs_dir(Path(session_dir), "specialist", task_id)
    _copy(workspace / "specialist_done.json", round_dir / "specialist_done.json")
    _copy(workspace / "prompt.md", round_dir / "prompt.md")

    # Patches the round did not apply still explain what was attempted.
    for base in (workspace, workspace / "worktree"):
        for pattern in ("*.patch", "*.diff"):
            for src in sorted((base / "patches").glob(pattern)):
                if src.name not in copied:
                    _copy(src, patches_dir / src.name)
                    copied.add(src.name)

    accepted_config = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_config:
        _copy(Path(accepted_config), round_dir / "launch_config.yaml")


__all__ = ["snapshot_round"]
