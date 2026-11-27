# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom inspect``: resolve the session directory and start the Inspector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hyperloom.inspector.session_layout import LEDGER_MARK, SESSION_FILES, SHARED_DB

_SEARCH_DEPTH = 3
# A session directory holds one of these once it has started: a session file, the database every account opens, or
# the mark that the database is kept outside it.
_SESSION_MARKS = (*SESSION_FILES, SHARED_DB, LEDGER_MARK)
#: The argparse action of a switch, an option that takes no value and is on when given.
SWITCH = "store_true"


def _is_session(path: Path) -> bool:
    return any((path / mark).is_file() for mark in _SESSION_MARKS)


def _started_at(path: Path) -> float:
    """When the session at *path* last wrote one of its session files; 0 when it has none."""
    times = [(path / name).stat().st_mtime for name in SESSION_FILES if (path / name).is_file()]
    return max(times, default=0.0)


def resolve_session_dir(path: Path) -> Path:
    """Return *path* if it is a session, else the newest session up to three levels below it.

    This lets ``--session-dir`` take ``$USER_DATA_PATH/<model>`` (or the workspace root) as well as a session.
    Raises ``FileNotFoundError`` when there is none.
    """
    path = path.expanduser().absolute()
    if not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    if _is_session(path):
        return path
    found: list[Path] = []
    level = [path]
    for _ in range(_SEARCH_DEPTH):
        level = [c for p in level for c in _subdirs(p)]
        found.extend(c for c in level if _is_session(c))
    if not found:
        raise FileNotFoundError(f"no session ({', '.join(_SESSION_MARKS)}) under {path}")
    return max(found, key=lambda p: (_started_at(p), p.name))


def _subdirs(path: Path) -> list[Path]:
    try:
        return [c for c in path.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except OSError:
        return []


def add_inspect_arguments(insp: argparse.ArgumentParser) -> None:
    """The ``inspect`` options, shared by ``hyperloom inspect`` and ``python -m hyperloom.inspector``."""
    insp.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory, or a directory above sessions (e.g. $USER_DATA_PATH/<model>): the newest "
        "session below it is used.",
    )
    insp.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Default: 127.0.0.1; reach it from a laptop with kubectl port-forward.",
    )
    insp.add_argument("--port", type=int, default=8765, help="Port. Default: 8765.")
    insp.add_argument(
        "--replay",
        action=SWITCH,
        help="Replay the session on a virtual clock from its first record, with a scrubber in the page.",
    )
    insp.add_argument("--speed", type=float, default=1.0, help="Replay speed factor. Default: 1.0.")


def run_inspect(args: argparse.Namespace) -> int:
    """Entry point for the ``inspect`` subcommand."""
    from hyperloom.inspector.server import ServeOptions, serve
    from hyperloom.inspector.watcher import WatchOptions

    try:
        session_dir = resolve_session_dir(Path(args.session_dir))
    except FileNotFoundError as exc:
        print(f"inspect: {exc}", file=sys.stderr)
        return 2
    try:
        options = ServeOptions(
            host=args.host,
            port=args.port,
            watch=WatchOptions(replay=args.replay, speed=args.speed),
        )
        serve(session_dir, options)
    except OSError as exc:
        print(f"inspect: cannot serve on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    return 0
