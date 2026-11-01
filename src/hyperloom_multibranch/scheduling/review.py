# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A person's review of the session's agent map, before the agent branches open (ArborDevelopment #30).

With the review on (the default; ``run --no-review-gate`` turns it off), a session with Arbor agents pauses once their
map is made: the dossier pass's, one generated from the root's shares, or the one ``--agent-directions`` gives. It
pauses before it splits the map and before any branch opens, holding no lease, and writes two files in its
``MB-review`` directory (:func:`request`): ``review.json``, the review's state (``pending``), and ``REVIEW.md``, what
the person reads: the map's directions, the dossier, and the commands. The person may edit the map file and the
dossier where they lie, then decides with ``hyperloom-multibranch review --session DIR approve|reject``
(:func:`decide`). That needs no running session: the file on the session's storage is the whole protocol, so a person
on a login node can decide for a session on a GPU node.

The session polls the file (:func:`wait`). Approved, it reads the map and the dossier again from disk, so the edits
are what its branches get, and ``review.json`` names every file the review changed; rejected, the session ends with
``review_rejected`` and opens nothing. The pause does not count against ``--max-hours``: EXPLORE ends that much later.

A session that stops after its map (``--stop-after-map``) does not pause: its stop is the review point, and its map is
reviewed offline across the points (:mod:`~hyperloom_multibranch.points`).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom_multibranch.scheduling.layout import REVIEW_DIR

REVIEW_FILE = "review.json"
REVIEW_PAGE = "REVIEW.md"
PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"
#: How often the session reads the review's state while it waits.
POLL_SECONDS = 15.0
#: The ``PRELUDE`` decisions of the review, as the lifecycle records them.
REVIEW_REQUESTED = "review requested"
REVIEW_APPROVED = "review approved"
REVIEW_REJECTED = "review rejected"
REVIEW_SKIPPED = "review skipped (--no-review-gate)"
#: What made the map the review shows.
MADE_BY_USER = "given by --agent-directions"
#: The command a person decides with, its actions, and its exit code when there is nothing to show or to decide.
COMMAND = "hyperloom-multibranch review"
SHOW, APPROVE, REJECT = "show", "approve", "reject"
DONE = {APPROVE: APPROVED, REJECT: REJECTED}
EXIT_REFUSED = 2
ENCODING = "utf-8"


class ReviewRejected(Exception):
    """A person rejected the session's agent map; the message is who and why."""


def _iso(now: float) -> str:
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digests(map_file: Path | None, dossier: Path | None) -> dict[str, str]:
    """The sha256 of the map file and of every file of the dossier, by path; a file that cannot be read is
    ``unreadable``."""

    files = [] if map_file is None else [map_file]
    if dossier is not None and dossier.is_dir():
        files += sorted(path for path in dossier.rglob("*") if path.is_file())
    found = {}
    for path in files:
        try:
            found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            found[str(path)] = "unreadable"
    return found


def directions(map_file: Path | None) -> list[Mapping[str, Any]]:
    """The directions of an Arbor map file (``{"directions": [...]}`` or a list), ``[]`` when it cannot be read."""

    if map_file is None:
        return []
    try:
        document = json.loads(map_file.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return []
    listed = document.get("directions", []) if isinstance(document, Mapping) else document
    return [entry for entry in listed if isinstance(entry, Mapping)] if isinstance(listed, list) else []


def request(
    directory: Path, session: Path, map_file: Path | None, dossier: Path | None, made_by: str, now: float
) -> dict[str, Any]:
    """Ask for a review of ``map_file`` and ``dossier`` (either may be ``None``), which ``made_by`` made, for the
    session in ``session``: write ``REVIEW.md`` and a pending ``review.json`` in ``directory``; the state written."""

    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / REVIEW_PAGE, _page(session, map_file, dossier, made_by))
    state = {
        "status": PENDING,
        "requested_at_iso": _iso(now),
        "decided_at_iso": None,
        "by": None,
        "note": "",
        "map": None if map_file is None else str(map_file),
        "dossier": None if dossier is None else str(dossier),
        "made_by": made_by,
        "requested_digests": digests(map_file, dossier),
        "changed": [],
    }
    atomic_write_json(directory / REVIEW_FILE, state)
    return state


def load(directory: Path) -> dict[str, Any] | None:
    """The review's state in ``directory``; ``None`` when there is none or it cannot be read."""

    try:
        state = json.loads((directory / REVIEW_FILE).read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


async def wait(
    directory: Path, sleep: Callable[[float], Awaitable[None]], poll_seconds: float = POLL_SECONDS
) -> dict[str, Any]:
    """Wait, reading ``directory``'s state every ``poll_seconds``, until a person decided; the decided state."""

    while True:
        state = load(directory)
        if state is not None and state.get("status") in (APPROVED, REJECTED):
            return state
        await sleep(poll_seconds)


def decide(directory: Path, approve: bool, by: str | None, note: str, now: float) -> list[str]:
    """Record a person's approval (``approve``) or rejection of the pending review in ``directory``, by ``by`` (the
    login name when ``None``), with ``note``, and which files the review changed; ``[]`` when recorded, else why it
    was not."""

    state = load(directory)
    if state is None:
        return [f"no review is pending in {directory}"]
    if state.get("status") != PENDING:
        return [f"the review is already {state.get('status')}"]
    map_file = None if state.get("map") is None else Path(state["map"])
    dossier = None if state.get("dossier") is None else Path(state["dossier"])
    before, after = state.get("requested_digests") or {}, digests(map_file, dossier)
    state.update(
        status=APPROVED if approve else REJECTED,
        decided_at_iso=_iso(now),
        by=by or _login(),
        note=note,
        changed=sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path)),
    )
    atomic_write_json(directory / REVIEW_FILE, state)
    return []


def show(directory: Path) -> str | None:
    """``REVIEW.md`` and the review's state, as the ``show`` command prints them; ``None`` when there is no review."""

    state = load(directory)
    if state is None:
        return None
    try:
        page = (directory / REVIEW_PAGE).read_text(encoding=ENCODING)
    except OSError:
        page = ""
    decided = f" by {state.get('by')} at {state.get('decided_at_iso')}" if state.get("by") else ""
    return f"{page}\nStatus: {state.get('status')}{decided}\n"


def _login() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return os.environ.get("USER", "")


def _page(session: Path, map_file: Path | None, dossier: Path | None, made_by: str) -> str:
    listed = directions(map_file)
    rows = [
        f"| `{entry.get('id', '')}` | {entry.get('status', '')} | {str(entry.get('title') or '').replace('|', '/')} |"
        for entry in listed
    ]
    parts = sorted(path.name for path in dossier.iterdir()) if dossier is not None and dossier.is_dir() else []
    mapped = f"`{map_file}` ({made_by})" if map_file is not None else "none could be made: each Arbor maps its own"
    lines = [
        f"# Review before the agent branches open: {session.name}",
        "",
        "The session paused once its Arbor agents' map was made, before it splits the map and before any agent branch",
        "opens; it holds no cards. Read the map and the dossier below, edit either where it lies if you wish, then:",
        "",
        f'    {COMMAND} --session {session} approve --note "..."',
        f'    {COMMAND} --session {session} reject --note "..."',
        "",
        "Approved, the session reads the map and the dossier again and opens its branches on them; rejected, it ends",
        "(`review_rejected`) and opens none. The pause does not count against --max-hours.",
        "",
        f"- **Map:** {mapped}",
        f"- **Dossier:** `{dossier}`: {', '.join(parts)}" if parts else "- **Dossier:** none",
        "",
        f"## The map's directions ({len(listed)})",
        "",
        "| id | status | title |",
        "| --- | --- | --- |",
        *rows,
        "",
    ]
    return "\n".join(lines)


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``review`` command: show a session's review, or approve or reject its agent map."""

    parser = commands.add_parser("review", help="Show a session's review of its agent map, or approve or reject it.")
    parser.add_argument("action", choices=(SHOW, APPROVE, REJECT))
    parser.add_argument("--session", type=Path, required=True, help="The session's directory, as this machine sees it.")
    parser.add_argument("--note", default="", help="Why, for the record.")
    parser.add_argument("--by", default=None, help="Who decides (default: the login name).")


def main(args: argparse.Namespace) -> int:
    """Carry out the ``review`` action ``args`` name: 0 when done, :data:`EXIT_REFUSED` when there is no review to
    show or none pending to decide on."""

    directory = args.session / REVIEW_DIR
    if args.action == SHOW:
        shown = show(directory)
        if shown is None:
            print(f"no review in {directory}", file=sys.stderr)
            return EXIT_REFUSED
        print(shown, end="")
        return 0
    refused = decide(directory, args.action == APPROVE, args.by, args.note, time.time())
    if refused:
        print(f"not {DONE[args.action]}: {'; '.join(refused)}", file=sys.stderr)
        return EXIT_REFUSED
    state = load(directory) or {}
    print(f"{DONE[args.action]} by {state.get('by')}; files changed during the review: {len(state.get('changed', []))}")
    return 0


__all__ = [
    "APPROVE",
    "APPROVED",
    "MADE_BY_USER",
    "PENDING",
    "POLL_SECONDS",
    "REJECTED",
    "REVIEW_APPROVED",
    "REVIEW_FILE",
    "REVIEW_PAGE",
    "REVIEW_REJECTED",
    "REVIEW_REQUESTED",
    "REVIEW_SKIPPED",
    "ReviewRejected",
    "add_parser",
    "decide",
    "digests",
    "directions",
    "load",
    "main",
    "request",
    "show",
    "wait",
]
