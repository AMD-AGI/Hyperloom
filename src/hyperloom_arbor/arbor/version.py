"""Arbor's version, and the versions of the interfaces a caller depends on.

``python3 -m arbor --version`` prints one line. ``python3 -m arbor version --json`` prints what a caller (Hyperloom, a
launcher) checks before it relies on a command, a pack format or a record, so it can keep working with an older
checkout and refuse one it cannot drive:

- ``version``: the release line (``VERSION``); ``git``: the checkout's ``git describe``, when it is one.
- ``commands``: what ``python3 -m arbor`` accepts. ``master`` and ``worker`` from ``cli`` 2.
- ``interfaces``: each interface's version, raised whenever it changes in a way an old caller would break on:
  - ``cli``: the commands and their flags. 2 adds ``master``, ``worker`` and ``run --split``; 3 adds ``run --split
    workers`` (a config's ``workers`` block, ``arbor/dealer.py``) and ``master --deal``;
  - ``pack``: ``pack.yaml``'s skills. 1 is ``v0.9.x`` (``skills:``, ``plugins:``, per-chapter ``lifecycle:``); 2 is
    ``lifecycle.load``, each skill placed by its own metadata;
  - ``component``: the config's component block, its keys in ``component_keys``, and the exit record's reasons in
    ``exit_reasons`` (``arbor/component.py``). 2 adds the reasons ``ordered`` and ``reported``; 3 adds the reason
    ``retraction_conflict`` and a KEEP snapshot's ``retracts`` (``arbor/retraction.py``);
  - ``roles``: the master's work order, the worker's report and a command's ``role_exit.json``, their schemas in
    ``records`` (``arbor/roles.py``). 2 gives the order an ``id``, which the report names, and adds ``role_exit``; 3
    lets a worker of its own session end ``spent``, its spent directions closed ``killed`` with ``spent`` set.
- ``chapters`` and ``events``: the lifecycle's (``arbor/lifecycle.py``); ``roles``: which side runs each chapter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: The release line. ``v0.9.2`` was tagged at ``cfbb7b9``; the pack format changed after it (interface ``pack`` 2).
VERSION = "0.10.0.dev0"
#: Each interface's version. Raise one when it changes in a way a caller written for the old one would break on.
INTERFACES: dict[str, int] = {"cli": 3, "pack": 2, "component": 3, "roles": 3}
#: What ``python3 -m arbor`` accepts.
COMMANDS: tuple[str, ...] = ("run", "master", "worker", "review", "conflicts", "iterate", "version")
ROOT = Path(__file__).resolve().parent.parent


def git_describe(root: Path = ROOT) -> str | None:
    """The checkout's ``git describe --tags --always --dirty``; None outside a git checkout."""

    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def info(root: Path = ROOT) -> dict:
    """What ``version --json`` prints."""

    from arbor import component, lifecycle, roles

    return {
        "name": "arbor",
        "version": VERSION,
        "git": git_describe(root),
        "commands": list(COMMANDS),
        "interfaces": dict(INTERFACES),
        "component_keys": list(component.BLOCK_KEYS),
        "exit_reasons": list(component.REASONS),
        "records": {"work_order": roles.ORDER_SCHEMA, "worker_report": roles.REPORT_SCHEMA,
                    "role_exit": roles.EXIT_SCHEMA},
        "chapters": list(lifecycle.CHAPTER_NAMES),
        "events": list(lifecycle.EVENT_NAMES),
        "roles": {side: list(chapters) for side, chapters in lifecycle.ROLES.items()},
    }


def line(root: Path = ROOT) -> str:
    """What ``--version`` prints: ``arbor <version> (<git describe>)``."""

    described = git_describe(root)
    return f"arbor {VERSION}" + (f" ({described})" if described else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m arbor version",
                                     description="Arbor's version, and its interfaces' versions.")
    parser.add_argument("--json", action="store_true", help="everything a caller checks, as JSON")
    args = parser.parse_args(argv)
    print(json.dumps(info(), indent=2) if args.json else line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
