# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sessions at several operating points, in two stages with a person's review between them.

A point is a concurrency, and each point is one session. On a machine it shares with other points, a point's session
runs on its own share of the machine's cards (its visible-device mask), with its own ``PORT``, its own Arbor port
blocks (``--agent-port``) and its own ``USER_DATA_PATH``, so the sessions share nothing they would collide on. A
points directory holds every point's directory, whichever machine ran it: one machine can run every point, or each
point can run on a machine of its own, into one directory on shared storage. It keeps the run arguments the points
share, so the second stage can be started with none.

- **map**: each point's session runs PRELUDE and its map, and ends (``run --stop-after-map``); the review page
  (:data:`REVIEW_PAGE`) then lists every point's map and dossier, and how to approve or reject it;
- **approve**, **reject**: a person's decision on one point, once they have read its map and edited it if they wish;
- **run**: each approved point's session, rooted on its first stage's PRELUDE (``--reuse-prelude``) and given its
  reviewed map and dossier (``--agent-directions``, ``--agent-dossier``). A rejected point, or one nobody reviewed, does
  not run, and the others go on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH
from hyperloom_multibranch.scheduling.agent_plans import KEPT_SUMMARY
from hyperloom_multibranch.scheduling.layout import AGENT_MAP_DIR, SESSION_PREFIX
from hyperloom_multibranch.scheduling.settings import AGENT_PORT

POINTS_FILE = "points.json"
POINT_FILE = "point.json"
DECISION_FILE = "review.json"
REVIEW_PAGE = "REVIEW.md"
USER_DATA_DIR = "userdata"
#: The ``PORT`` the first point's session serves its tests on, and how far apart two points' ports and two points'
#: Arbor port blocks are: wider than all the blocks one session takes.
FIRST_PORT = 8888
PORT_STEP = 100
#: The flags each point's session is given by its stage, which the run arguments the points share may not set.
POINT_FLAGS = ("--conc", "--agent-port", "--stop-after-map", "--reuse-prelude", "--agent-directions", "--agent-dossier")
#: The device mask each point's session is given; the other masks are dropped so that none names other cards.
MASK = "ROCR_VISIBLE_DEVICES"
OTHER_MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
#: How a point's session is started: this interpreter running the ``hyperloom-multibranch`` command.
COMMAND = (sys.executable, "-m", "hyperloom_multibranch")
ENCODING = "utf-8"


class Stage(str, Enum):
    MAP = "map"
    RUN = "run"


class Decision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


#: The ``points`` actions that record a decision, and the one that writes the review page again.
DECISIONS = {"approve": Decision.APPROVED, "reject": Decision.REJECTED}
REVIEW = "review"


@dataclass(frozen=True)
class Point:
    """One operating point: its ``conc``, the ``cards`` its session runs on, the ``port`` it serves its tests on and
    the first port of its Arbor port blocks, ``agent_port``; ``root`` is its directory."""

    conc: int
    cards: tuple[int, ...]
    port: int
    agent_port: int
    root: Path

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        """``base`` with this point's cards, port and user data path in place of the machine's."""

        env = {name: value for name, value in base.items() if name not in OTHER_MASKS}
        env.update({MASK: ",".join(map(str, self.cards)), "PORT": str(self.port)})
        env[ENV_USER_DATA_PATH] = str(self.root / USER_DATA_DIR)
        return env

    def sessions(self) -> set[Path]:
        """Every session directory this point's sessions have made."""

        return set((self.root / USER_DATA_DIR).glob(f"*/{SESSION_PREFIX}*"))


@dataclass(frozen=True)
class Reviewed:
    """What the review of one point reads: its first stage's ``session`` (``None`` when it made none), how that stage
    ``exited``, the ``summary`` of the map it kept (empty without one) and the person's ``decision`` (``None`` before
    anyone made one)."""

    conc: int
    session: Path | None
    exited: int | None
    summary: Mapping[str, object]
    decision: Mapping[str, object] | None


def place_points(concs: Sequence[int], cards: Sequence[int], root: Path, per_point: int | None) -> tuple[Point, ...]:
    """The points of ``concs`` in ``root``: each an equal share of ``cards`` in order, ``per_point`` each when given,
    a port and Arbor port blocks of its own; raise ValueError when there are no points or the cards do not go round."""

    if not concs or len(set(concs)) != len(concs):
        raise ValueError("give each --conc once, and at least one")
    share = per_point or len(cards) // len(concs)
    if share < 1 or share * len(concs) > len(cards):
        raise ValueError(f"{len(cards)} cards do not give {len(concs)} points {share or 1} each")
    return tuple(
        Point(conc, tuple(cards[at * share : (at + 1) * share]), *_ports(at), root / f"c{conc}")
        for at, conc in enumerate(concs)
    )


def _ports(at: int) -> tuple[int, int]:
    return FIRST_PORT + at * PORT_STEP, AGENT_PORT + at * PORT_STEP


def shared_args(args: Sequence[str]) -> list[str]:
    """The run arguments the points share; raise ValueError when one is a flag each point's stage sets itself."""

    set_by_stage = sorted({arg.split("=", 1)[0] for arg in args} & set(POINT_FLAGS))
    if set_by_stage:
        raise ValueError(f"each point's stage sets {', '.join(set_by_stage)} itself; leave it out of the run arguments")
    return list(args)


def stage_argv(point: Point, stage: Stage, run_args: Sequence[str], reviewed: Reviewed | None = None) -> list[str]:
    """The ``run`` arguments of ``point``'s session in ``stage``: the shared ``run_args``, its conc and Arbor ports, and
    either the flag that stops it after its map or, from ``reviewed``, the PRELUDE it roots on and its reviewed map and
    dossier."""

    argv = ["run", *run_args, "--conc", str(point.conc), "--agent-port", str(point.agent_port)]
    if stage is Stage.MAP:
        return [*argv, "--stop-after-map"]
    assert reviewed is not None and reviewed.session is not None
    argv += ["--reuse-prelude", str(reviewed.session), "--agent-directions", str(reviewed.summary["map"])]
    dossier = str(reviewed.summary.get("dossier") or "")
    return [*argv, "--agent-dossier", dossier] if dossier else argv


async def run_point(point: Point, stage: Stage, argv: Sequence[str], command: Sequence[str]) -> tuple[Path | None, int]:
    """Run ``point``'s session of ``stage`` with ``argv``, its output in the point's log of that stage; the session
    directory it made (``None`` when it made none) and its exit code."""

    point.root.mkdir(parents=True, exist_ok=True)
    before = point.sessions()
    with (point.root / f"{stage.value}.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command, *argv, env=point.env(os.environ), stdout=log, stderr=asyncio.subprocess.STDOUT
        )
        exited = await process.wait()
    made = sorted(point.sessions() - before)
    session = made[-1] if made else None
    _record(point, stage, session, exited)
    return session, exited


def _record(point: Point, stage: Stage, session: Path | None, exited: int) -> None:
    path = point.root / POINT_FILE
    document = _read(path) or {}
    document.update(conc=point.conc, cards=list(point.cards), port=point.port, agent_port=point.agent_port)
    document[stage.value] = {"session": str(session or ""), "exit_code": exited, "at": _now()}
    atomic_write_json(path, document)


def reviewed_points(root: Path) -> list[Reviewed]:
    """Every point in ``root`` whose map stage ran, by conc."""

    found = []
    for path in sorted(root.glob(f"c*/{POINT_FILE}")):
        document = _read(path) or {}
        mapped = document.get(Stage.MAP.value)
        if not isinstance(mapped, dict) or "conc" not in document:
            continue
        session = Path(mapped["session"]) if mapped.get("session") else None
        summary = (_read(session / AGENT_MAP_DIR / KEPT_SUMMARY) if session else None) or {}
        found.append(
            Reviewed(document["conc"], session, mapped.get("exit_code"), summary, _read(path.parent / DECISION_FILE))
        )
    return sorted(found, key=lambda point: point.conc)


def decide(root: Path, conc: int, decision: Decision, why: str) -> Path:
    """Record a person's ``decision`` on the point of ``conc``, and write the review page again; refuse a point whose
    map stage kept no map."""

    point = next((point for point in reviewed_points(root) if point.conc == conc), None)
    if point is None or "map" not in point.summary:
        raise ValueError(f"the point C{conc} in {root} has no map to decide on")
    atomic_write_json(root / f"c{conc}" / DECISION_FILE, {"decision": decision.value, "why": why, "at": _now()})
    return write_review(root)


def write_review(root: Path) -> Path:
    """Write the review page of every point in ``root``; its path."""

    lines = [f"# Review: {len(reviewed_points(root))} points in {root}", ""]
    for point in reviewed_points(root):
        lines += _point_lines(root, point)
    page = root / REVIEW_PAGE
    atomic_write_text(page, "\n".join(lines) + "\n")
    return page


def _point_lines(root: Path, point: Reviewed) -> list[str]:
    state = "awaiting review" if point.decision is None else str(point.decision.get("decision"))
    lines = [f"## C{point.conc}: {state}", "", f"- Session: `{point.session}`, map stage exited {point.exited}"]
    summary = point.summary
    if "map" not in summary:
        return [*lines, "- No map was kept: read the session's MB-reports/MB-report.md.", ""]
    lines += [
        f"- Map: `{summary['map']}`, {summary.get('made_by')}; the dossier pass: {summary.get('pass')}",
        f"- Dossier: `{summary.get('dossier') or 'none'}`",
        "",
        "| rank | id | kind | direction | ceiling |",
        "|---|---|---|---|---|",
        *_direction_rows(Path(str(summary["map"]))),
        "",
        f"Edit the map if you wish, then: `hyperloom-multibranch points approve --dir {root} --conc {point.conc}`",
        f" or `hyperloom-multibranch points reject --dir {root} --conc {point.conc} --why ...`",
        "",
    ]
    return lines


def _direction_rows(map_file: Path) -> list[str]:
    document = _read(map_file)
    directions = document.get("directions") if isinstance(document, dict) else document
    if not isinstance(directions, list):
        return [f"| | | | the map {map_file} holds no directions | |"]
    cells = ("rank", "id", "kind", "title", "ceiling")
    return [
        "| " + " | ".join(str(entry.get(cell, "")) for cell in cells) + " |"
        for entry in directions
        if isinstance(entry, dict)
    ]


async def map_points(points: Sequence[Point], run_args: Sequence[str], command: Sequence[str]) -> list[int]:
    """Run every point's map stage at once, and write the review page; each point's exit code."""

    runs = (run_point(point, Stage.MAP, stage_argv(point, Stage.MAP, run_args), command) for point in points)
    exited = [code for _, code in await asyncio.gather(*runs)]
    write_review(points[0].root.parent)
    return exited


async def run_approved(
    root: Path, cards: Sequence[int], per_point: int | None, run_args: Sequence[str], command: Sequence[str]
) -> dict[int, int | str]:
    """Run, at once on ``cards``, the second stage of every point in ``root`` a person approved; by conc, each run
    point's exit code, and for every other point why it did not run."""

    reviewed = reviewed_points(root)
    approved = [point for point in reviewed if _approved(point)]
    outcome: dict[int, int | str] = {point.conc: _held_back(point) for point in reviewed if point not in approved}
    if not approved:
        return outcome
    placed = place_points([point.conc for point in approved], cards, root, per_point)
    runs = (
        run_point(point, Stage.RUN, stage_argv(point, Stage.RUN, run_args, review), command)
        for point, review in zip(placed, approved)
    )
    for point, (_, code) in zip(placed, await asyncio.gather(*runs)):
        outcome[point.conc] = code
    return outcome


def _approved(point: Reviewed) -> bool:
    decided = point.decision or {}
    return decided.get("decision") == Decision.APPROVED.value and point.session is not None and "map" in point.summary


def _held_back(point: Reviewed) -> str:
    if point.decision is None:
        return "not reviewed"
    return f"{point.decision.get('decision')}: {point.decision.get('why') or 'no reason given'}"


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``points`` command and its actions."""

    points = commands.add_parser("points", help="Sessions at several concs: map each, review, then run the approved.")
    actions = points.add_subparsers(dest="points_command", required=True)

    def action(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = actions.add_parser(name, help=help_text)
        parser.add_argument("--dir", type=Path, required=True, help="The points directory, on shared storage.")
        return parser

    mapping = action(Stage.MAP.value, "Run each --conc's session to its map, then write the review page.")
    mapping.add_argument("--conc", type=int, action="append", required=True, help="An operating point; repeatable.")
    running = action(Stage.RUN.value, "Run the approved points, each on its reviewed map and dossier.")
    for stage in (mapping, running):
        stage.add_argument("--cards-per-point", type=int, default=None, help="Default: the machine's cards shared out.")
        stage.add_argument("run_args", nargs=argparse.REMAINDER, help="-- and the run arguments the points share.")
    for verb, decision in DECISIONS.items():
        deciding = action(verb, f"Record one point as {decision.value}.")
        deciding.add_argument("--conc", type=int, required=True)
        deciding.add_argument("--why", default="", required=decision is Decision.REJECTED)
    action(REVIEW, "Write the review page again.")


def main(args: argparse.Namespace) -> int:
    """Carry out the ``points`` action ``args`` name; 0 when every session it ran ended as asked."""

    root: Path = args.dir.absolute()
    action = args.points_command
    if action == REVIEW:
        print(write_review(root))
        return 0
    if action in DECISIONS:
        print(decide(root, args.conc, DECISIONS[action], args.why))
        return 0
    from hyperloom_multibranch.scheduling.local_peers import machine_cards

    if action == Stage.MAP.value:
        run_args = shared_args(_after_dashes(args.run_args))
        atomic_write_json(root / POINTS_FILE, {"run_args": run_args})
        points = place_points(args.conc, machine_cards(), root, args.cards_per_point)
        exited = asyncio.run(map_points(points, run_args, COMMAND))
        print(root / REVIEW_PAGE)
        return 0 if not any(exited) else 1
    given = _after_dashes(args.run_args)
    run_args = shared_args(given or (_read(root / POINTS_FILE) or {}).get("run_args", []))
    outcome = asyncio.run(run_approved(root, machine_cards(), args.cards_per_point, run_args, COMMAND))
    for conc, ended in sorted(outcome.items()):
        print(f"C{conc}: {ended}")
    return 0 if all(isinstance(ended, str) or ended == 0 for ended in outcome.values()) else 1


def _after_dashes(args: Sequence[str]) -> list[str]:
    return list(args[1:]) if args and args[0] == "--" else list(args)


def _read(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, (dict, list)) else None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "COMMAND",
    "REVIEW_PAGE",
    "Decision",
    "Point",
    "Reviewed",
    "Stage",
    "decide",
    "place_points",
    "reviewed_points",
    "run_point",
    "shared_args",
    "stage_argv",
    "write_review",
]
