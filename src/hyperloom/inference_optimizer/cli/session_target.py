# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``optimize --session-dir``: where a fresh session is created."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..session.paths import SESSION_FILES, is_path_within, workspace_root


def session_dir_target(value: str) -> Path:
    """Parse ``--session-dir``: an absolute path below ``$USER_DATA_PATH``, the same rule ``--resume-from`` applies."""
    target = Path(value).expanduser()
    root = workspace_root()
    if not target.is_absolute():
        raise argparse.ArgumentTypeError(f"{value!r} is not an absolute path")
    if target.resolve() == root.resolve() or not is_path_within(target, root):
        raise argparse.ArgumentTypeError(f"{value!r} is not a directory below $USER_DATA_PATH={root}")
    held = [name for name in SESSION_FILES if (target / name).exists()]
    if held:
        raise argparse.ArgumentTypeError(
            f"{value!r} already holds a session ({', '.join(held)}); resume it with --resume-from"
        )
    return target


def add_session_dir_argument(opt: argparse.ArgumentParser) -> None:
    """Register ``--session-dir`` on ``optimize``; absent, the session directory is generated as always."""
    opt.add_argument(
        "--session-dir",
        dest="session_dir",
        type=session_dir_target,
        default=None,
        metavar="DIR",
        help="Create the new session in exactly this directory (absolute, below $USER_DATA_PATH) instead of "
        "<workspace>/<model>/<timestamp>-<random>. Used by a launcher that must find the session again; "
        "ignored by --resume-from, which names the directory it resumes.",
    )
