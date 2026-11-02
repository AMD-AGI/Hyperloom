# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start several multi-branch sessions on one machine, one per operating point, and wait for them all.

An MB session has one operating point, its ``--conc``. ``hyperloom-multibranch-launch`` starts one ordinary
``hyperloom-multibranch run`` per ``--point``, each on its own share of the machine's cards, with its own server port
and Arbor port base, and keeps each one's log under ``--out``. It holds no session state: each session reviews its own
agent map, makes its own FINAL choice and ends with its own recipe. Nothing is merged across the points.

Each point runs in a container of its own (``--image``): Magpie ends a benchmark by killing every serving process it
can see, so points that share a PID namespace kill each other's servers. The containers are prepared one at a time
(``--prepare``, Hyperloom's ``install.sh`` say, whose installs race on a shared virtualenv), and then the sessions start
together. Each session is given its cards as ``ROCR_VISIBLE_DEVICES``, set only for it, since Ray refuses that mask at
import. No Ray head is started or used: one per container would collide on the machine's network, and one shared head
has a single serving slot for the machine. Without ``--image``, the sessions are processes of the launcher, for one
point, or for several with ``--same-host``.

While they run, ``STATUS.md`` in ``--out`` gives each point's state and session directory, and the commands for a
session that waits for a person's review of its agent map. ``launch.json`` holds the plan and how each point ended.
The launcher exits 0 when every session did, 1 when one did not, and 2 when it refuses the launch before starting
anything. A SIGTERM or SIGINT stops every session with SIGTERM, which MB answers with its CLOSE; ``--stop-grace``
later, what is left is killed and every container removed.

The module needs only the standard library, so that it runs on a machine's own Python, outside the image.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, NamedTuple

PROG = "hyperloom-multibranch-launch"
#: The command each point runs, as ``<program> run <run arguments>``.
MB_PROGRAM = "hyperloom-multibranch"
#: The ``run`` options the launcher sets for each point, refused among the run arguments.
OWNED = ("--conc", "--agent-port")
TP_OPTION = "--tp"
#: The first point's server port and Arbor port base; each further point's are one step on.
PORT_BASE = 8888
AGENT_PORT_BASE = 31100
PORT_STEP = 100
#: What a point's session is told: its cards, its server's port, and to execute without Ray.
CARDS_ENV = "ROCR_VISIBLE_DEVICES"
PORT_ENV = "PORT"
SESSION_ENV = {"INFERENCE_OPTIMIZER_RAY_EXEC": "0"}
#: What a container's preparation is told: to start no Ray head.
PREPARE_ENV = {"SKIP_RAY_START": "1"}
#: The variables a container takes from the launcher's environment when they are set, by name only, so that no value
#: is on a command line: the gateways' credentials and headers, the CA bundle and Hyperloom's storage.
FORWARDED = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "HF_TOKEN",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "USER_DATA_PATH",
    "HYPERLOOM_RUNNER_CACHE",
    "HYPERLOOM_RSI_ARCHIVE",
)
#: How every container starts: the GPUs, the host's network and IPC, room for NCCL's shared memory, and a process that
#: keeps it up until the launcher removes it.
CONTAINER_FLAGS = (
    "--init",
    "--network=host",
    "--ipc=host",
    "--shm-size=64g",
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add=video",
    "--security-opt=seccomp=unconfined",
    "--ulimit=nofile=1048576:1048576",
)
KEEPALIVE = ("--entrypoint", "tail")
KEEPALIVE_ARGS = ("-f", "/dev/null")
#: Where a session's process id is kept in its container, for the stop.
PID_FILE = "/tmp/hyperloom-multibranch-launch.pid"
#: The line ``run`` prints first, its session directory (``scheduling/layout.py``'s session names).
SESSION_LINE = re.compile(r"^(/\S*/MB-\d{8}T\d{6}Z-[0-9a-f]+)\s*$", re.MULTILINE)
#: What the launcher reads in a session directory: its review's state and its report (``scheduling/layout.py``).
REVIEW_FILE = "MB-review/review.json"
REVIEW_PAGE = "MB-review/REVIEW.md"
REPORT_FILE = "MB-reports/MB-report.json"
STATUS_FILE = "STATUS.md"
LAUNCH_FILE = "launch.json"
POLL_SECONDS = 30.0
#: How long the sessions have to end after a stop: MB's CLOSE grace is at most 120 s.
STOP_GRACE_SECONDS = 180.0
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
SESSION_STEP = "the session"
EXIT_FAILED = 1
EXIT_REFUSED = 2
#: A point's states, in ``STATUS.md`` and ``launch.json``.
WAITING, PREPARING, RUNNING, REVIEW, DONE, FAILED = "waiting", "preparing", "running", "review", "done", "failed"
ENCODING = "utf-8"
BASH = "bash"
DOCKER = "docker"
APPEND = "append"
ISO = "%Y-%m-%dT%H:%M:%SZ"


class LaunchRefused(ValueError):
    """The launch cannot start as given; the message names every reason."""


@dataclass(frozen=True)
class Point:
    """One operating point: its conc, its ``--max-hours`` (``None`` for the run arguments' own), its cards, and its
    server port and Arbor port base."""

    conc: int
    hours: float | None
    cards: tuple[int, ...]
    port: int
    agent_port: int

    @property
    def name(self) -> str:
        return f"c{self.conc}"


@dataclass
class PointState:
    """How a point stands: its state, its session directory once ``run`` printed it, the review it waits for, and how
    it ended."""

    point: Point
    state: str = WAITING
    session: str = ""
    review: str = ""
    exit_code: int | None = None
    stop_reason: str = ""
    final: str = ""
    error: str = ""


def plan(
    concs: Sequence[int], hours: Mapping[int, float], cards: Sequence[int], per_point: int, tp: int
) -> tuple[Point, ...]:
    """The points, in the order given, each on the next ``per_point`` of ``cards`` (an equal share when 0), with its
    ports one step on from the last; raise :class:`LaunchRefused` with every reason the plan cannot hold."""

    reasons = []
    repeated = sorted({conc for conc in concs if list(concs).count(conc) > 1})
    if repeated:
        reasons.append(f"each --point once: {', '.join(map(str, repeated))} given twice")
    stray = sorted(set(hours) - set(concs))
    if stray:
        reasons.append(f"--point-hours names {', '.join(map(str, stray))}, which no --point gives")
    per = per_point or (len(cards) // len(concs) if concs else 0)
    if per < tp:
        reasons.append(f"{per} cards a point is fewer than {TP_OPTION} {tp}")
    elif per * len(concs) > len(cards):
        reasons.append(f"{len(concs)} points of {per} cards need {per * len(concs)}; {len(cards)} are given")
    if reasons:
        raise LaunchRefused("; ".join(reasons))
    return tuple(
        Point(
            conc,
            hours.get(conc),
            tuple(cards[index * per : (index + 1) * per]),
            PORT_BASE + index * PORT_STEP,
            AGENT_PORT_BASE + index * PORT_STEP,
        )
        for index, conc in enumerate(concs)
    )


def run_arg_refusals(run_args: Sequence[str]) -> list[str]:
    """Why the run arguments cannot be given to every point: an option the launcher sets per point, or no TP."""

    owned = sorted({arg.split("=", 1)[0] for arg in run_args if arg.split("=", 1)[0] in OWNED})
    reasons = [f"{option} is set by the launcher, one per point" for option in owned]
    if tp_of(run_args) is None:
        reasons.append(f"the run arguments give no {TP_OPTION}")
    return reasons


def tp_of(run_args: Sequence[str]) -> int | None:
    """The run arguments' ``--tp``, the last one given; ``None`` when there is none or it is not a number."""

    found = None
    for index, arg in enumerate(run_args):
        if arg == TP_OPTION and index + 1 < len(run_args):
            found = run_args[index + 1]
        elif arg.startswith(f"{TP_OPTION}="):
            found = arg.split("=", 1)[1]
    return int(found) if found is not None and found.isdigit() else None


def session_argv(point: Point, run_args: Sequence[str], program: str = MB_PROGRAM) -> list[str]:
    """The command of ``point``'s session: ``run`` with the run arguments, then the point's conc, its Arbor port base
    and, when the point has its own, ``--max-hours``, which wins over the run arguments' as the last given."""

    argv = [program, "run", *run_args, "--conc", str(point.conc), "--agent-port", str(point.agent_port)]
    return argv if point.hours is None else [*argv, "--max-hours", f"{point.hours:g}"]


def session_env(point: Point) -> dict[str, str]:
    """The environment ``point``'s session is given over the one it starts in."""

    return {CARDS_ENV: ",".join(map(str, point.cards)), PORT_ENV: str(point.port), **SESSION_ENV}


def shell(setup: str, argv: Sequence[str], pid_file: str = "") -> list[str]:
    """``argv`` run by bash after ``setup``, in its place (``exec``), its process id written to ``pid_file`` first."""

    lines = [*([setup] if setup else []), *([f"echo $$ > {pid_file}"] if pid_file else []), 'exec "$@"']
    return [BASH, "-c", "\n".join(lines), BASH, *argv]


@dataclass(frozen=True)
class Containers:
    """How each point's container is made: the image, the paths mounted at the same path in it (``SRC:DST`` for
    another), the variables forwarded by name, further ``docker run`` arguments, the working directory, and what is
    run in it before its session: as root, then as the container's user."""

    image: str
    tag: str
    mounts: tuple[str, ...] = ()
    forwarded: tuple[str, ...] = FORWARDED
    extra: tuple[str, ...] = ()
    workdir: str = ""
    root_prepare: str = ""
    prepare: str = ""

    def name(self, point: Point) -> str:
        return f"mbl-{self.tag}-{point.name}"

    def start(self, point: Point, env: Mapping[str, str]) -> list[str]:
        mounts = [f"--volume={mount if ':' in mount else f'{mount}:{mount}'}" for mount in self.mounts]
        forwarded = [f"--env={name}" for name in self.forwarded if name in env]
        head = [DOCKER, "run", "--detach", f"--name={self.name(point)}", *CONTAINER_FLAGS, *mounts, *forwarded]
        return [*head, *self.extra, *KEEPALIVE, self.image, *KEEPALIVE_ARGS]

    def exec(
        self, point: Point, command: Sequence[str], env: Mapping[str, str] | None = None, root: bool = False
    ) -> list[str]:
        where = [f"--workdir={self.workdir}"] if self.workdir and not root else []
        given = [f"--env={key}={value}" for key, value in (env or {}).items()]
        return [DOCKER, "exec", *(["--user=0"] if root else []), *where, *given, self.name(point), *command]

    def stop(self, point: Point) -> list[str]:
        return self.exec(point, [BASH, "-c", f'kill -TERM "$(cat {PID_FILE})"'])

    def remove(self, point: Point) -> list[str]:
        return [DOCKER, "rm", "--force", self.name(point)]


def _iso(now: float) -> str:
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime(ISO)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def session_of(log: Path) -> str:
    """The session directory ``run`` printed in ``log``, ``""`` until it has."""

    try:
        with log.open(encoding=ENCODING, errors="replace") as handle:
            head = handle.read(1 << 16)
    except OSError:
        return ""
    found = SESSION_LINE.search(head)
    return found.group(1) if found else ""


def review_of(session: str) -> str:
    """What a person must do for ``session``'s pending review, ``""`` when none is pending."""

    state = _read_json(Path(session) / REVIEW_FILE)
    if state is None or state.get("status") != "pending":
        return ""
    by = f"; decide by about {state['open_by_iso']}" if state.get("open_by_iso") else ""
    return f"read {Path(session) / REVIEW_PAGE}, then `{MB_PROGRAM} review --session {session} approve`{by}"


class Ending(NamedTuple):
    """How a session ended, from its report: its stop reason, its FINAL choice and its error."""

    stop_reason: str
    final: str
    error: str


def ending_of(session: str) -> Ending:
    """How ``session`` ended, from its report; empty before it has one."""

    report = _read_json(Path(session) / REPORT_FILE) or {}
    final = report.get("final") if isinstance(report.get("final"), dict) else {}
    chosen = final.get("chosen") if isinstance(final.get("chosen"), dict) else {}
    pick = f"{chosen.get('branch', '')}: {final.get('reason', '')}" if chosen else str(final.get("reason") or "")
    return Ending(str(report.get("stop_reason") or ""), pick, str(report.get("error") or ""))


def _atomic(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text, encoding=ENCODING)
    os.replace(partial, path)


def _cell(text: str) -> str:
    return text.replace("|", "/") or "-"


def status_page(states: Sequence[PointState], out: Path, now: float) -> str:
    """``STATUS.md``: each point's cards, ports, state, session and ending, then what waits for a person."""

    rows = [
        f"| C{s.point.conc} | {','.join(map(str, s.point.cards))} | {s.point.port}, {s.point.agent_port} | {s.state}"
        f" | {_cell(s.session)} | {_cell(s.stop_reason)} | {_cell(s.final)} |"
        for s in states
    ]
    reviews = [f"- **C{s.point.conc}:** {s.review}" for s in states if s.review]
    errors = [f"- **C{s.point.conc}:** {s.error}" for s in states if s.error]
    lines = [
        f"# Operating points: {out.name}",
        "",
        f"Updated {_iso(now)}. Each point is one `{MB_PROGRAM} run`; its log is `c<conc>.log` beside this page.",
        "",
        "| Point | Cards | Ports (server, Arbor) | State | Session | Stop reason | FINAL |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
    ]
    lines += ["", "## Waiting for a review", "", *reviews] if reviews else []
    lines += ["", "## Errors", "", *errors] if errors else []
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Sessions:
    """What every point's session shares: the run arguments, the launch's directory, the shell lines run before each
    session, the multi-branch command, and the containers (``None`` to run the sessions as the launcher's processes)."""

    run_args: tuple[str, ...]
    out: Path
    setup: str = ""
    program: str = MB_PROGRAM
    containers: Containers | None = None


@dataclass(frozen=True)
class Host:
    """What a launch runs with: the environment the containers and sessions start from, ``runner`` to run a
    container's step to its end and ``spawn`` to start a session (as :func:`subprocess.run` and
    :class:`subprocess.Popen` do), and the clock."""

    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    runner: Callable[..., Any] = subprocess.run
    spawn: Callable[..., Any] = subprocess.Popen
    clock: Callable[[], float] = time.time


class Launch:
    """The points' sessions, as ``sessions`` says, on ``host``: started (:meth:`start`), followed (:meth:`poll`),
    stopped (:meth:`stop`) and cleaned up (:meth:`close`)."""

    def __init__(self, points: Sequence[Point], sessions: Sessions, host: Host | None = None) -> None:
        host = host or Host()
        self._states = {point.name: PointState(point) for point in points}
        self._run_args, self._out, self._setup = sessions.run_args, sessions.out, sessions.setup
        self._program, self._containers, self._env = sessions.program, sessions.containers, dict(host.env)
        self._runner, self._spawn, self._clock = host.runner, host.spawn, host.clock
        self._procs: dict[str, Any] = {}
        self._logs: dict[str, IO[str]] = {}
        self._started_at = host.clock()

    @property
    def states(self) -> tuple[PointState, ...]:
        return tuple(self._states.values())

    def commands(self, point: Point) -> list[tuple[str, list[str]]]:
        """What runs for ``point``, in order, each with what it is: its container's steps, then its session."""

        argv = session_argv(point, self._run_args, self._program)
        containers = self._containers
        if containers is None:
            return [(SESSION_STEP, shell(self._setup, argv))]
        steps = [("starting the container", containers.start(point, self._env))]
        if containers.root_prepare:
            root = containers.exec(point, [BASH, "-c", containers.root_prepare], root=True)
            steps.append(("the root preparation", root))
        if containers.prepare:
            steps.append(("the preparation", containers.exec(point, [BASH, "-c", containers.prepare], PREPARE_ENV)))
        session = containers.exec(point, shell(self._setup, argv, PID_FILE), session_env(point))
        return [*steps, (SESSION_STEP, session)]

    def start(self, stopping: Callable[[], bool] = lambda: False) -> None:
        """Prepare each point's container, one at a time, then start every session whose container is ready;
        ``stopping`` is asked before each point."""

        self._out.mkdir(parents=True, exist_ok=True)
        for state in self.states:
            if stopping():
                return
            if self._containers is not None:
                self._prepare(state)
        for state in self.states:
            if stopping():
                return
            if state.state != FAILED:
                self._start_session(state)
        self.write()

    def _log_path(self, point: Point) -> Path:
        return self._out / f"{point.name}.log"

    def _prepare(self, state: PointState) -> None:
        state.state = PREPARING
        self.write()
        with self._log_path(state.point).open("a", encoding=ENCODING) as log:
            for what, step in self.commands(state.point)[:-1]:
                done = self._runner(step, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self._env)
                if done.returncode != 0:
                    state.state, state.error = FAILED, f"{what} exited {done.returncode}; see {log.name}"
                    self._say(state.point, state.error)
                    return

    def _start_session(self, state: PointState) -> None:
        point = state.point
        env = self._env if self._containers is not None else {**self._env, **session_env(point)}
        log = self._logs[point.name] = self._log_path(point).open("a", encoding=ENCODING)
        self._procs[point.name] = self._spawn(
            self.commands(point)[-1][1],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        state.state = RUNNING
        self._say(
            point, f"started on cards {','.join(map(str, point.cards))}, ports {point.port} and {point.agent_port}"
        )

    def poll(self) -> bool:
        """Follow every session once and write ``STATUS.md`` and ``launch.json``; whether any still runs."""

        for state in self.states:
            self._follow(state)
        self.write()
        return any(state.state in (RUNNING, REVIEW) for state in self.states)

    def _follow(self, state: PointState) -> None:
        proc = self._procs.get(state.point.name)
        if proc is None or state.exit_code is not None:
            return
        state.session = state.session or session_of(self._log_path(state.point))
        code = proc.poll()
        if code is None:
            review = review_of(state.session) if state.session else ""
            if review != state.review:
                self._say(state.point, f"waits for a review: {review}" if review else "its review is decided")
            state.review, state.state = review, REVIEW if review else RUNNING
            return
        state.exit_code, state.review, state.state = code, "", DONE if code == 0 else FAILED
        if state.session:
            state.stop_reason, state.final, state.error = ending_of(state.session)
        self._logs.pop(state.point.name).close()
        self._say(state.point, f"ended, exit {code}, stop reason {state.stop_reason or 'unknown'}")

    def stop(self, grace: float, sleep: Callable[[float], None] = time.sleep) -> None:
        """Send every running session SIGTERM and wait up to ``grace`` seconds for them to end; then kill what is left
        of a session the launcher runs itself (a container's goes with the container, :meth:`close`)."""

        running = {name: proc for name, proc in self._procs.items() if proc.poll() is None}
        for name, proc in running.items():
            if self._containers is not None:
                self._runner(self._containers.stop(self._states[name].point), capture_output=True, env=self._env)
            else:
                proc.send_signal(signal.SIGTERM)
        ends = self._clock() + grace
        while self._clock() < ends and any(proc.poll() is None for proc in running.values()):
            sleep(1.0)
        for proc in running.values():
            if proc.poll() is None and self._containers is None:
                os.killpg(proc.pid, signal.SIGKILL)

    def close(self) -> None:
        """Close the logs, remove every container the launch made, and write the last ``STATUS.md``."""

        for log in self._logs.values():
            log.close()
        self._logs.clear()
        if self._containers is not None:
            for state in self.states:
                if state.state != WAITING:
                    self._runner(self._containers.remove(state.point), capture_output=True, env=self._env)
        self.write()

    def exit_code(self) -> int:
        return 0 if all(state.state == DONE for state in self.states) else EXIT_FAILED

    def write(self) -> None:
        now = self._clock()
        document = {
            "started_at_iso": _iso(self._started_at),
            "updated_at_iso": _iso(now),
            "image": None if self._containers is None else self._containers.image,
            "run_args": list(self._run_args),
            "points": [{**asdict(state), "name": state.point.name} for state in self.states],
        }
        _atomic(self._out / LAUNCH_FILE, json.dumps(document, indent=2) + "\n")
        _atomic(self._out / STATUS_FILE, status_page(self.states, self._out, now))

    def _say(self, point: Point, what: str) -> None:
        print(f"[{_iso(self._clock())}] C{point.conc}: {what}", flush=True)


def follow(launch: Launch, poll_seconds: float, grace: float, sleep: Callable[[float], None] = time.sleep) -> int:
    """Start ``launch``, follow it every ``poll_seconds`` until every session ended, or until a SIGTERM or SIGINT,
    which stops them all (:meth:`Launch.stop`); clean up and return the launch's exit code."""

    asked: list[int] = []
    previous = {signum: signal.signal(signum, lambda got, _frame: asked.append(got)) for signum in STOP_SIGNALS}
    try:
        launch.start(stopping=lambda: bool(asked))
        while not asked and launch.poll():
            _nap(poll_seconds, asked, sleep)
        if asked:
            launch.stop(grace, sleep)
            launch.poll()
    finally:
        launch.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return launch.exit_code()


def _nap(seconds: float, asked: Sequence[int], sleep: Callable[[float], None]) -> None:
    """Sleep ``seconds``, a second at a time, so that a stop is heard within one."""

    for _ in range(max(1, int(seconds))):
        if asked:
            return
        sleep(min(1.0, seconds))


def parse_cards(text: str) -> tuple[int, ...]:
    """Cards written as ``0-7``, ``0,1,2,3`` or both (``0-3,6,7``); raise ValueError on anything else."""

    cards: list[int] = []
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        cards.extend(range(int(first), int(last or first) + 1))
    return tuple(cards)


def machine_cards() -> tuple[int, ...]:
    """Every card of the machine, one per render node."""

    return tuple(range(len(list(Path("/dev/dri").glob("renderD*")))))


def parse_hours(pairs: Sequence[str]) -> dict[int, float]:
    """``--point-hours`` as ``CONC=HOURS`` pairs; raise ValueError on another form."""

    hours = {}
    for pair in pairs:
        conc, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--point-hours {pair} is not CONC=HOURS")
        hours[int(conc)] = float(value)
    return hours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            f"Start one `{MB_PROGRAM} run` per operating point on this machine and wait for them all. Give the run"
            f" arguments every point shares after `--`, without --conc and --agent-port, which are set per point."
        ),
    )
    parser.add_argument(
        "--point",
        dest="points",
        type=int,
        action=APPEND,
        required=True,
        metavar="CONC",
        help="An operating point's concurrency; repeatable, one session each.",
    )
    parser.add_argument(
        "--point-hours",
        action=APPEND,
        default=[],
        metavar="CONC=HOURS",
        help="That point's --max-hours, over the run arguments'; repeatable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="A new or empty directory for STATUS.md, launch.json and each point's log.",
    )
    parser.add_argument("--cards", default="", help="The cards the points share, as 0-7 or 0,1,2,3 (default: all).")
    parser.add_argument("--cards-per-point", type=int, default=0, help="Default: an equal share of --cards.")
    parser.add_argument("--setup", default="", help="Shell lines each session runs first, to source its environment.")
    parser.add_argument("--program", default=MB_PROGRAM, help=f"The multi-branch command (default {MB_PROGRAM}).")
    parser.add_argument(
        "--same-host", action="store_true", help="Run several points without containers, in one PID namespace."
    )
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--stop-grace", type=float, default=STOP_GRACE_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Print what each point runs, and start nothing.")
    _add_container_options(parser)
    return parser


def _add_container_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default="", help="Run each point in a container of its own, from this image.")
    parser.add_argument(
        "--mount",
        action=APPEND,
        default=[],
        metavar="PATH[:DST]",
        help="A path mounted in each container, at the same path unless DST is given; repeatable.",
    )
    parser.add_argument(
        "--forward",
        action=APPEND,
        default=[],
        metavar="NAME",
        help="A variable each container takes from this environment, by name; repeatable.",
    )
    parser.add_argument(
        "--docker-arg",
        action=APPEND,
        default=[],
        metavar="ARG",
        help="A further `docker run` argument, as --docker-arg=--user=1000:1000; repeatable.",
    )
    parser.add_argument("--workdir", default="", help="Each container's working directory for its steps.")
    parser.add_argument("--root-prepare", default="", help="Shell lines run as root in each container first.")
    parser.add_argument("--prepare", default="", help="Shell lines run in each container, one container at a time.")
    parser.add_argument("--tag", default=str(os.getpid()), help="Names the containers mbl-TAG-c<conc>.")


def split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """The launcher's arguments and the run arguments, which follow ``--``."""

    if "--" not in argv:
        return list(argv), []
    at = list(argv).index("--")
    return list(argv[:at]), list(argv[at + 1 :])


def make_launch(args: argparse.Namespace, run_args: Sequence[str]) -> Launch:
    """The launch the arguments describe; raise :class:`LaunchRefused` with every reason it cannot start."""

    reasons = run_arg_refusals(run_args)
    try:
        hours = parse_hours(args.point_hours)
        cards = parse_cards(args.cards) if args.cards else machine_cards()
    except ValueError as exc:
        reasons.append(f"{exc}")
        hours, cards = {}, ()
    if len(args.points) > 1 and not args.image and not args.same_host:
        reasons.append(
            "several points without --image share one PID namespace, where Magpie's cleanup kills each other's"
            " servers; give --image, or --same-host to accept that"
        )
    if args.out.is_dir() and any(args.out.iterdir()):
        reasons.append(f"--out {args.out} is not empty: each launch writes into a directory of its own")
    if reasons:
        raise LaunchRefused("; ".join(reasons))
    points = plan(args.points, hours, cards, args.cards_per_point, tp_of(run_args) or 0)
    containers = None
    if args.image:
        containers = Containers(
            args.image,
            args.tag,
            tuple(args.mount),
            (*FORWARDED, *args.forward),
            tuple(args.docker_arg),
            args.workdir,
            args.root_prepare,
            args.prepare,
        )
    return Launch(points, Sessions(tuple(run_args), args.out, args.setup, args.program, containers))


def main(argv: Sequence[str] | None = None) -> int:
    own, run_args = split_argv(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(own)
    try:
        launch = make_launch(args, run_args)
    except LaunchRefused as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        for state in launch.states:
            for what, argv_of in launch.commands(state.point):
                print(f"C{state.point.conc}, {what}: {shlex.join(argv_of)}")
        return 0
    return follow(launch, args.poll_seconds, args.stop_grace)


__all__ = [
    "Containers",
    "Ending",
    "Host",
    "Launch",
    "LaunchRefused",
    "Point",
    "PointState",
    "Sessions",
    "build_parser",
    "ending_of",
    "follow",
    "main",
    "make_launch",
    "parse_cards",
    "parse_hours",
    "plan",
    "review_of",
    "run_arg_refusals",
    "session_argv",
    "session_env",
    "session_of",
    "shell",
    "split_argv",
    "status_page",
    "tp_of",
]


if __name__ == "__main__":
    sys.exit(main())
