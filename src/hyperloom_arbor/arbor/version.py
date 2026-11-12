"""Arbor's version, and the versions of the interfaces a caller depends on.

``python3 -m arbor --version`` prints one line. ``python3 -m arbor version --json`` prints what a caller (Hyperloom, a
launcher) checks before it relies on a command, a pack format or a record, so it can keep working with an older
checkout and refuse one it cannot drive:

- ``version``: the release line (``VERSION``); ``git``: the checkout's ``git describe``, when it is one.
- ``python``: the oldest Python the checkout runs on (``PYTHON_MIN``); a host checks it before it starts Arbor.
- ``interface``: the host contract's version (``arbor.host/2.0``), which ``python3 -m arbor interface --json`` prints
  in full (``arbor/interface.py``); ``exit_codes`` and ``timing``: its exit codes by name and the numbers a host
  plans by.
- ``commands``: what ``python3 -m arbor`` accepts. ``master`` and ``worker`` from ``cli`` 2.
- ``interfaces``: each interface's version, raised whenever it changes in a way an old caller would break on:
  - ``cli``: the commands and their flags. 2 adds ``master``, ``worker`` and ``run --split``; 3 adds ``run --split
    workers`` (a config's ``workers`` block, ``arbor/dealer.py``) and ``master --deal``; 4 adds ``--opt-objective``
    and the config's ``opt_objective``, described in ``objective`` (``arbor/objective.py``); 5 adds ``--payload`` on
    every command and ``python3 -m arbor.directions``, the ``schema`` and ``directions`` commands, and
    ``<session>/invocation.json``, described in ``payload`` (``arbor/cli/payload.py``); 6 adds ``restore``, ``pause``
    (exit 76 for a paused run) and ``checkpoint list|show|verify|create|fork`` (``arbor/checkpoints.py``); 7 adds
    ``interface`` (``arbor/interface.py``); 8 adds ``run --exit-at-handover`` (a host deals each worker report);
  - ``pack``: ``pack.yaml``'s skills. 1 is ``v0.9.x`` (``skills:``, ``plugins:``, per-chapter ``lifecycle:``); 2 is
    ``lifecycle.load``, each skill placed by its own metadata;
  - ``component``: the config's component block, its keys in ``component_keys``, and the exit record's reasons in
    ``exit_reasons`` (``arbor/component.py``). 2 adds the reasons ``ordered`` and ``reported``; 3 adds the reason
    ``retraction_conflict`` and a KEEP snapshot's ``retracts`` (``arbor/retraction.py``); 4 adds ``bench_read``, the
    metrics read by path from the host client's report; 5 adds the reason ``paused`` (``arbor pause``);
  - ``roles``: the master's work order, the worker's report and a command's ``role_exit.json``, their schemas in
    ``records`` (``arbor/roles.py``). 2 gives the order an ``id``, which the report names, and adds ``role_exit``; 3
    lets a worker of its own session end ``spent``, its spent directions closed ``killed`` with ``spent`` set;
  - ``session``: the session's own format, what a checkpoint holds (``arbor/checkpoints.py``). 0 is a session with no
    checkpoint; 1 adds ``checkpoints/`` and ``restore``; 2 makes a checkpoint immutable and verified (its ``id``), and
    adds the workspace's exact history, the prompts, the dossier, the fingerprint, the bars, named checkpoints,
    ``pause`` and ``fork`` (``fork.json``).
- ``continues``: the range of each format this Arbor continues, ``{"session": [low, high]}``. ``run``, ``master`` and
  ``worker`` refuse a session outside it, and ``restore`` a checkpoint outside it. A new format raises ``high``; ``low``
  moves only when an Arbor can no longer read the oldest format, which its release notes say.
- ``chapters`` and ``events``: the lifecycle's (``arbor/lifecycle.py``); ``roles``: which side runs each chapter.
- ``objective``: how a caller names the metric to optimize in place of the pack's, checked before it writes it: the
  config key, the flag, the variable, the metric keys and names Arbor knows, and the record a run writes.
- ``payload``: a command's parameters as one JSON or YAML document (``--payload FILE``): the format's version, the
  formats, the commands it drives, the schema's id (``python3 -m arbor schema`` prints it) and the record a run writes.
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
INTERFACES: dict[str, int] = {"cli": 8, "pack": 2, "component": 5, "roles": 3, "session": 2}
#: The formats this Arbor continues, each a range ``(low, high)``: a session written in a format outside it is refused.
CONTINUES: dict[str, tuple[int, int]] = {"session": (0, 2)}
#: What ``python3 -m arbor`` accepts.
COMMANDS: tuple[str, ...] = ("run", "master", "worker", "review", "conflicts", "restore", "pause", "checkpoint",
                             "iterate", "version", "schema", "directions", "interface")
#: The oldest Python this checkout runs on, for a host to check before it starts Arbor (``import arbor.version``
#: needs nothing else). 3.10 is what serving images ship; a checkout without this name needs 3.11.
PYTHON_MIN: tuple[int, int] = (3, 10)
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


_PROVENANCE: dict | None = None


def provenance() -> dict:
    """The Arbor that writes a record, as each host record carries it (``arbor``): ``{"version", "commit"}``, the
    commit None when Arbor is not its own git checkout (a vendored copy inside another repository).

    Read once per process, and early (``component.Run.start``): a record written in a SIGTERM handler must not wait on
    git."""
    global _PROVENANCE
    if _PROVENANCE is None:
        commit = None
        try:
            top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            lines = top.stdout.split()
            if top.returncode == 0 and len(lines) == 2 and Path(lines[0]).resolve() == ROOT.resolve():
                commit = lines[1]
        except (OSError, subprocess.SubprocessError):
            pass
        _PROVENANCE = {"version": VERSION, "commit": commit}
    return dict(_PROVENANCE)


def info(root: Path = ROOT) -> dict:
    """What ``version --json`` prints."""

    from arbor import component, interface, lifecycle, objective, roles
    from arbor.cli import payload

    return {
        "name": "arbor",
        "version": VERSION,
        "git": git_describe(root),
        "python": ".".join(map(str, PYTHON_MIN)),
        "interface": interface.ID,
        "exit_codes": interface.exit_codes(),
        "timing": interface.timing(),
        "commands": list(COMMANDS),
        "interfaces": dict(INTERFACES),
        "continues": {name: list(span) for name, span in CONTINUES.items()},
        "component_keys": list(component.BLOCK_KEYS),
        "exit_reasons": list(component.REASONS),
        "records": {"work_order": roles.ORDER_SCHEMA, "worker_report": roles.REPORT_SCHEMA,
                    "role_exit": roles.EXIT_SCHEMA},
        "chapters": list(lifecycle.CHAPTER_NAMES),
        "events": list(lifecycle.EVENT_NAMES),
        "roles": {side: list(chapters) for side, chapters in lifecycle.ROLES.items()},
        "objective": {"config_key": objective.CONFIG_KEY, "flag": "--opt-objective", "env": objective.ENV,
                      "record": objective.FILE, "keys": list(objective.KNOWN), "names": dict(objective.NAMES),
                      "forms": list(objective.FORMS)},
        "payload": {"version": payload.VERSION, "formats": list(payload.FORMATS), "flag": payload.FLAG,
                    "commands": list(payload.COMMANDS), "schema": payload.SCHEMA_ID, "record": payload.RECORD},
    }


def line(root: Path = ROOT) -> str:
    """What ``--version`` prints: ``arbor <version> (<git describe>)``."""

    described = git_describe(root)
    return f"arbor {VERSION}" + (f" ({described})" if described else "")


def parser() -> argparse.ArgumentParser:
    """The command's parser, built without parsing (arbor/cli/payload.py makes its argv and schema)."""
    parser = argparse.ArgumentParser(prog="python3 -m arbor version",
                                     description="Arbor's version, and its interfaces' versions.")
    parser.add_argument("--json", action="store_true", help="everything a caller checks, as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(info(), indent=2) if args.json else line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
