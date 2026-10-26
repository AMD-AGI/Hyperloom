# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a campaign's device sentinel lives: the one file every run that touches its GPU flocks."""

from __future__ import annotations

from pathlib import Path

DEVICE_LOCK_SENTINEL = ".forge-device-bench.lock"


def campaign_device_lock_path(workspace: str | Path) -> Path:
    """The sentinel beside the campaign workspace, outside every checkout a lane or probe measures."""
    workspace = Path(workspace).expanduser().resolve()
    return workspace.parent / f"{workspace.name}{DEVICE_LOCK_SENTINEL}"


__all__ = ["DEVICE_LOCK_SENTINEL", "campaign_device_lock_path"]
