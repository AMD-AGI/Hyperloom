# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local stand-in for an autonomous branch's sandbox lease: commands run on this host, on the leased cards only.

Every command runs with :data:`~hyperloom_multibranch.specialist.agent_watch.MARKER` in its environment, which every
process it starts inherits, and under a launcher that is a child subreaper (:mod:`hyperloom.sandbox.launcher`), so the
marker and the launchers' trees are the sandbox's whole extent: :meth:`LocalAgentSandbox.stop` ends every process that
carries the marker or descends from a launcher, whichever session or process group it moved to and whatever it did to
its environment (a Claude CLI runs each of its shell commands in a session of its own, and an agent's model starts its
servers through one), and so does the session's watch (:mod:`agent_watch`) once the session's process is gone.
:meth:`LocalProcess.kill` ends one command's alone, selected by
:data:`~hyperloom_multibranch.specialist.agent_watch.PROCESS_MARKER` and its launcher's tree. The pod
forbids a PID namespace, so an agent's ``pkill``, ``pgrep`` and ``killall`` would otherwise reach the whole host; Arbor
stops its server with ``docker exec C pkill -TERM -f sglang.launch_server``, which through the docker shim would stop
MB's measurement servers too. The sandbox puts wrappers for them first on ``PATH`` that match only the processes
carrying the caller's marker and never the caller or its ancestors, as a ``docker exec`` from outside a container would
not: a shell that waits with ``until ! pgrep -f X`` carries ``X`` in its own command line. Nor do they match a launcher,
which would leave what it holds to init. ``killall NAME`` is ``pkill -x`` on the name's first 15 characters, all the
kernel keeps of a name, and takes no option but a signal and ``-q``: its others select by user, age or regex, which an
agent confined to its own processes has no use for.

``ROCR_VISIBLE_DEVICES`` holds the cards and ``HIP_VISIBLE_DEVICES`` numbers them from 0, so a HIP-level mask a command
sets indexes into the cards and reaches no other card of the host; the watch kills a command that sets a ROCr-level
one beyond them.

Output goes to a numbered log file under the sandbox's root, not to a pipe, which an agent running for hours would
fill.

No command may write the image's installed frameworks (:func:`framework_protection`): each runs with them mounted
read-only in a mount namespace of its own, or, on a host that gives a command none, under a
:class:`~hyperloom_multibranch.specialist.framework_guard.FrameworkGuard` that restores them before the command starts,
while it runs, once it and every process it started ended, and when the sandbox stops, and says what it restored in the
logs of the commands that ran; a command whose frameworks neither can protect is refused. A command's end is its
launcher's: a process the command left running in the background, or detached, still writes as the command.

Where bubblewrap can sandbox an agent (:func:`~hyperloom.common.agent_isolation.choose_sandbox_isolation`), every
command runs in one (:class:`~hyperloom_common.bwrap.BwrapPolicy`) that starts from an empty root: only the
toolchain, the paths its sandbox names (:class:`SandboxPaths`) and the installed frameworks are bound, the image's
installed sglang and vllm read-only and aiter writable, as its JIT and tuning files are written beside it; the home,
the app's state and every credential location are not there. The sandbox holds one long-lived holder
(:class:`~hyperloom_common.bwrap.Holder`) whose user and pid namespaces every command joins, so a server one command
starts is seen and stopped by a later one; each command gets its own mount and IPC namespaces, a ``/proc`` of the
sandbox's processes only, and the sandbox's one private ``/tmp`` and full-size ``/dev/shm``, so a file a command
leaves there is the next one's to read. The launcher runs inside, its subreaper in the sandbox's pid namespace, and
the markers are set inside only, so the kills and the watch reach the command's processes, never bubblewrap's. The
environment is no longer the app's: only what a command needs (:func:`confined_env`), with a home, Claude config,
pip install target and Triton cache of the sandbox's own; aiter keeps its JIT modules and FlyDSL cache in its own tree,
where the image's prebuilt ones are. A sandbox whose paths bubblewrap cannot bind safely runs unconfined, as before,
with a WARNING; in a hosted run it is refused. Where bubblewrap cannot sandbox an agent at all, commands run as before,
said once at start-up.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import itertools
import logging
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import BinaryIO

from hyperloom.common.agent_isolation import (
    BwrapIsolation,
    NoIsolation,
    announce_agent_isolation,
    choose_sandbox_isolation,
)
from hyperloom.common.confinement import Confinement, credential_paths, toolchain_paths
from hyperloom.common.outside_facts import ROCR_VISIBLE_DEVICES_ENV
from hyperloom_common.proctree import descendants, running
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.measure.compile import COMPONENT_LAYOUTS, UnsupportedComponentError, locate_installed_package
from hyperloom.measure.local_templates import image_env
from hyperloom.sandbox.launcher import LAUNCHER_SOURCE, StatusPipe, launcher_argv
from hyperloom_common.bwrap import BwrapPolicy, BwrapPolicyError, Holder, Scratch
from hyperloom_common.shared_storage import make_shared_dirs
from hyperloom_multibranch.specialist.agent_onboarding import CommandResult
from hyperloom_multibranch.specialist.agent_session import Processes
from hyperloom_multibranch.specialist.agent_watch import (
    GRACE_SECONDS,
    MARKER,
    POLL_SECONDS,
    PROCESS_MARKER,
    kill_marked,
    revive,
    sandbox_pids,
    watch,
)
from hyperloom_multibranch.specialist.framework_guard import (
    FrameworkGuard,
    FrameworksUnguarded,
    GuardedPackage,
    PATROL_SECONDS,
)

logger = logging.getLogger(__name__)

#: The frameworks whose installed package a sandbox may not write: every test lays its patches over those files, so an
#: edit in place (a tool defaulting to the installed tree) breaks every branch's tests. aiter is left writable, since
#: it builds its JIT modules beside itself unless told otherwise.
READ_ONLY_FRAMEWORKS = ("sglang", "vllm")
#: The frameworks a confined sandbox's commands may write in place: aiter builds its JIT modules and keeps its tuning
#: files beside itself.
WRITABLE_FRAMEWORKS = ("aiter",)
#: The prefix that runs a command in a mount namespace of its own, whose mounts no other process sees; the probe of
#: :func:`mount_namespaces` runs the same one.
_PRIVATE_MOUNTS = ("unshare", "--mount", "--propagation", "private", "--")
#: Mounts each path given before ``--`` read-only in the command's own mount namespace, then runs the rest.
_READ_ONLY_SCRIPT = (
    'while [ "$1" != -- ]; do mount --bind "$1" "$1" && mount -o remount,bind,ro "$1" || exit 97; shift; done; '
    'shift; exec "$@"'
)
BIN_DIR = "bin"
LOG_DIR = "logs"
#: Where in a sandbox's temporary directory its JIT and kernel caches go, and the same name in a confined home's tool
#: directories.
CACHE_DIR = "cache"
#: Where a sandbox's temporary files go, under the host's local temporary directory: HIP unbundles its code objects
#: there and fails on a network file system, where the branch's directory usually is.
TMP_PREFIX = "hyperloom-agent-"
#: How much of a log :meth:`LocalProcess.output` and a run's result keep: the end, where the cause usually is.
TAIL_BYTES = 64 * 1024
#: The exit code a run that timed out reports, as ``timeout(1)`` does.
TIMED_OUT = 124
#: How often a command that ended is asked whether a process it left still runs: each asking walks ``/proc``.
LEFT_POLL_SECONDS = 1.0
CONFINED = ("pkill", "pgrep", "killall")
#: The launchers this process started, by the root of their sandbox, so a kill reaps every one of them.
_launched: dict[str, list[subprocess.Popen[bytes]]] = {}
#: How a sandbox's commands are isolated: in bubblewrap, or unconfined as before.
SandboxIsolation = BwrapIsolation | NoIsolation
#: A sandbox no tier was chosen for runs its commands as before.
UNCONFINED = NoIsolation(())
#: Where a confined command finds its temporary directory: the sandbox's own, bound there for every command.
CONFINED_TMP = Path("/tmp")
#: A confined sandbox's home, below its root unless its factory names another (:attr:`SandboxPaths.home`), and where
#: in it the Claude CLI keeps its state and transcripts, and Python its user base.
HOME_DIR = "home"
CLAUDE_DIR = ".claude"
USER_BASE = ".local"
#: Where in a confined home pip installs (``PIP_TARGET``), with every dependency an install brings: on no import path,
#: so no inference server a command starts imports those ahead of the image's and runs other packages than the
#: baseline it is measured against. Arbor imports from it on its own commands alone (``ARBOR_SITE``,
#: :mod:`~hyperloom_multibranch.specialist.arbor_adapter`).
SITE_DIR = "site-packages"
#: The programs a confined command may run besides those of its toolchain's trees: its shell and the agent CLIs.
AGENT_PROGRAMS = ("bash", "claude", "codex")
#: The cards' device nodes, bound into a confined sandbox where the host has them.
GPU_DEVICES = (Path("/dev/kfd"), Path("/dev/dri"))
#: What a confined command takes from the app's environment by name and by prefix (:func:`confined_env`), besides the
#: framework image's own settings and what its factory forwards: the search paths, the locale and terminal, and the
#: models and benchmark checkout it reads.
BASIC_NAMES = frozenset(
    {"PATH", "PYTHONPATH", "LANG", "LANGUAGE", "TZ", "TERM", "HF_HOME", "HF_HUB_CACHE", "MODEL_PATH", "INFERENCEX_PATH"}
)
BASIC_PREFIXES = ("LC_",)
#: The variable naming the app's state directory, which no bind of a sandbox may show.
STATE_DIR_ENV = "HYPERLOOM_STATE_DIR"
GUARDED_NOTE = (
    "the image's installed frameworks are not mounted read-only on this host: what a command, or a process it left "
    "running, writes into them is restored within seconds, and recorded with the command"
)

CONFINE_SCRIPT = """#!{python} -I
# pkill, pgrep and killall confined to the processes of the caller's sandbox, the caller and its ancestors spared.
import os, re, signal, subprocess, sys

PGREP = {pgrep!r}
LAUNCHER = {launcher!r}.encode()
VALUED = set("dgGOPrstuUF")
SELECTING = set("gGPtuUF")
VALUED_LONG = {{"--delimiter", "--pgroup", "--group", "--parent", "--session", "--terminal", "--euid", "--uid",
               "--ns", "--nslist", "--pidfile", "--signal", "--older", "--runstates"}}
SELECTING_LONG = {{"--pgroup", "--group", "--parent", "--terminal", "--euid", "--uid", "--ns", "--pidfile"}}
tool, args = os.path.basename(sys.argv[0]), sys.argv[1:]
MARK = ("{marker}=" + os.environ["{marker}"]).encode()


def refuse(why):
    sys.exit(f"{{tool}} in this sandbox: {{why}}")


def signum(name):
    name = name.upper().removeprefix("SIG")
    if name.isdigit():
        return int(name)
    number = getattr(signal, "SIG" + name, None)
    return number if isinstance(number, signal.Signals) else None


def ancestors():
    pid, seen = os.getpid(), set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{{pid}}/stat") as stat:
                pid = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return seen


# The sandbox alone would select every process of it, so another criterion is required.
def has_criteria(args):
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg == "--":
            return bool(rest)
        if arg.startswith("--"):
            name = arg.partition("=")[0]
            if name in SELECTING_LONG:
                return True
            if name in VALUED_LONG and "=" not in arg and rest:
                rest.pop(0)
        elif arg.startswith("-") and len(arg) > 1:
            for at, letter in enumerate(arg[1:]):
                if letter in SELECTING:
                    return True
                if letter in VALUED:
                    if at == len(arg) - 2 and rest:
                        rest.pop(0)
                    break
        else:
            return True
    return False


# A launcher holds what its command left behind; killed, it would leave that to init.
def ours(pid):
    try:
        with open(f"/proc/{{pid}}/environ", "rb") as environ, open(f"/proc/{{pid}}/cmdline", "rb") as cmdline:
            return MARK in environ.read().split(b"\\0") and LAUNCHER not in cmdline.read().split(b"\\0")
    except OSError:
        return False


# pgrep's lines for the selection, those of processes outside the sandbox left out.
def found(selection):
    done = subprocess.run([PGREP, *selection], stdout=subprocess.PIPE, text=True)
    if done.returncode > 1:
        sys.exit(done.returncode)
    return [line for line in done.stdout.splitlines() if line.split()[:1] and ours(line.split()[0])]


# pgrep counts, joins and picks the newest or oldest over the host, so the wrapper does it over the sandbox's lines.
def shaping(args):
    selection, shape = [], {{"count": False, "delimiter": "\\n", "pick": None}}
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg == "--":
            selection += [arg, *rest]
            break
        if arg == "--count":
            shape["count"] = True
        elif arg in ("--newest", "--oldest"):
            shape["pick"] = arg[2:]
        elif arg.partition("=")[0] == "--delimiter":
            shape["delimiter"] = arg.partition("=")[2] if "=" in arg else rest.pop(0) if rest else ""
        elif arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            kept = ""
            for at, letter in enumerate(arg[1:], 2):
                if letter == "c":
                    shape["count"] = True
                elif letter in "no":
                    shape["pick"] = "newest" if letter == "n" else "oldest"
                elif letter == "d":
                    shape["delimiter"] = arg[at:] or (rest.pop(0) if rest else "")
                    break
                else:
                    kept += letter
                    if letter in VALUED:
                        kept += arg[at:]
                        if at == len(arg) and rest:
                            selection += ["-" + kept, rest.pop(0)]
                            kept = ""
                        break
            if kept:
                selection.append("-" + kept)
        else:
            selection.append(arg)
    return selection, shape


def started(line):
    try:
        with open(f"/proc/{{line.split()[0]}}/stat") as stat:
            return int(stat.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


def picked(lines, pick):
    if pick is None or not lines:
        return lines
    return [(max if pick == "newest" else min)(lines, key=started)]


def unspared(lines):
    spared = {{str(pid) for pid in ancestors()}}
    return [line for line in lines if line.split()[0] not in spared]


def matching(selection, pick):
    return [int(line.split()[0]) for line in picked(unspared(found(selection)), pick)]


def send(pids, number, echo):
    sent = 0
    for pid in pids:
        try:
            os.kill(pid, number)
        except ProcessLookupError:
            continue
        sent += 1
        if echo:
            print(f"killed (pid {{pid}})")
    return sent


if tool == "pgrep":
    if not has_criteria(args):
        refuse("give a pattern or a selection")
    selection, shape = shaping(args)
    lines = picked(unspared(found(selection)), shape["pick"])
    print(len(lines) if shape["count"] else shape["delimiter"].join(lines), end="\\n" if shape["count"] or lines else "")
    sys.exit(0 if lines else 1)
number, echo, quiet, selection = signal.SIGTERM, False, False, []
rest = list(args)
while rest:
    arg = rest.pop(0)
    if (arg == "--signal" or arg == "-s" and tool == "killall") and rest:
        number = signum(rest.pop(0))
        if number is None:
            refuse(f"unknown signal after {{arg}}")
    elif arg.startswith("--signal="):
        number = signum(arg.partition("=")[2])
        if number is None:
            refuse(f"unknown signal in {{arg}}")
    elif arg.startswith("-") and len(arg) > 1 and signum(arg[1:]) is not None:
        number = signum(arg[1:])
    elif arg in ("-e", "--echo") and tool == "pkill":
        echo = True
    elif arg in ("-q", "--quiet") and tool == "killall":
        quiet = True
    else:
        selection.append(arg)
if tool == "pkill":
    if not has_criteria(selection):
        refuse("give a pattern or a selection")
    selection, shape = shaping(selection)
    sent = send(matching(selection, shape["pick"]), number, echo)
    if shape["count"]:
        print(sent)
    sys.exit(0 if sent else 1)
if not selection or any(name.startswith("-") for name in selection):
    refuse("only `killall [-SIGNAL | -s SIGNAL] [-q] NAME...` is supported")
missed = 0
for name in selection:
    # The kernel keeps 15 characters of a name, and pgrep -x matches no longer one.
    if not send(matching(["-x", re.escape(name[:15])], None), number, False):
        missed += 1
        if not quiet:
            print(f"{{name}}: no process found", file=sys.stderr)
sys.exit(1 if missed else 0)
"""


@dataclass(frozen=True)
class FrameworkProtection:
    """How sandbox commands are kept from writing the image's installed frameworks: ``read_only``, the packages each
    command mounts read-only, or ``guard``, which restores them around each command; neither on a host where none is
    installed."""

    read_only: tuple[Path, ...] = ()
    guard: FrameworkGuard | None = None


@dataclass(frozen=True)
class SandboxPaths:
    """What a confined sandbox's commands may write and read besides its root, its home and its toolchain:
    ``writable`` trees (an agent's directory, a lease's root, a worktree it edits) and ``readable`` ones (a worktree it
    only imports, the paths a worktree links); ``home``, the sandbox's home, its Claude config included (default:
    ``home`` below the sandbox's root)."""

    writable: tuple[Path, ...] = ()
    readable: tuple[Path, ...] = ()
    home: Path | None = None


@dataclass
class _Confined:
    """A confined sandbox's bubblewrap, policy, environment, home and private ``/tmp`` and ``/dev/shm``, and the holder
    its commands join once one started."""

    isolation: BwrapIsolation
    policy: BwrapPolicy
    env: dict[str, str]
    home: Path
    scratch: Scratch
    holder: Holder | None = None

    def release(self) -> None:
        """End the holder, and every process still in it, and delete the scratch."""
        if self.holder is not None:
            self.holder.stop()
            self.holder = None
        self.scratch.remove()


@dataclass(frozen=True)
class _Guarded:
    """A command ``guard`` counts as running by ``ticket``, none without a guard; ``note`` is what its log says
    first."""

    guard: FrameworkGuard | None = None
    ticket: int = 0
    note: str = ""

    def ended(self) -> None:
        if self.guard is not None:
            self.guard.ended(self.ticket)


async def _restore(guard: FrameworkGuard | None, log: Path | None = None) -> None:
    """Restore the frameworks ``guard`` protects, after a command ran: one that cannot be restored is an error of the
    session's log, and of ``log``, and the next command is refused."""

    if guard is None:
        return
    try:
        await asyncio.to_thread(guard.check)
    except FrameworksUnguarded as error:
        logger.exception("the installed frameworks were not restored after a sandbox command")
        if log is not None:
            with log.open("a", encoding="utf-8") as out:
                out.write(f"\n[hyperloom] {error}\n")


@dataclass(frozen=True)
class _Launch:
    """One command's launcher ``argv``, started in ``cwd`` with ``env``, its output to ``out``, holding the status
    descriptor ``status`` and ``fds``."""

    argv: list[str]
    cwd: Path
    env: Mapping[str, str]
    out: BinaryIO
    status: int
    fds: tuple[int, ...] = ()

    def popen(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=dict(self.env),
            stdin=subprocess.DEVNULL,
            stdout=self.out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(self.status, *self.fds),
        )


@dataclass
class LocalProcess:
    """A command started in a :class:`LocalAgentSandbox` whose processes carry ``marker``, its sandbox's, and
    ``process_marker``, its own (:data:`~hyperloom_multibranch.specialist.agent_watch.PROCESS_MARKER`), run by the
    launcher ``popen``, which says on ``status`` how the command ended; its output is in ``log``. ``guard`` restores the
    image's frameworks once the process is killed."""

    popen: subprocess.Popen[bytes]
    log: Path
    marker: str
    process_marker: str
    grace_seconds: float = GRACE_SECONDS
    poll_seconds: float = POLL_SECONDS
    status: StatusPipe | None = None
    guard: FrameworkGuard | None = None
    _code: int | None = field(default=None, init=False, repr=False)

    @property
    def returncode(self) -> int | None:
        """The command's exit code, or the launcher's when it ended without saying; the launcher outlives the command
        while anything it started runs."""

        if self._code is not None or self.status is None:
            return self._code if self._code is not None else self.popen.poll()
        try:
            self._code = self.status.wait_exit(self.popen, 0)
        except TimeoutError:
            return None
        self.status = None
        return self._code if self._code is not None else self.popen.poll()

    def retire(self) -> None:
        """End the launcher of a command that ended and left nothing running, which it would otherwise wait for until
        the sandbox stops. In a confined sandbox :attr:`popen` is bubblewrap's process, and the launcher it runs is
        the one process of its tree left: that launcher is ended, and bubblewrap's process ends with it."""

        if self.returncode is None:
            return
        left = [pid for pid, _ in descendants(self.popen.pid) if running(pid)]
        launchers = [pid for pid in left if _runs_launcher(pid)]
        if len(launchers) < len(left):
            return
        for pid in launchers:
            _kill(pid)
        if self.popen.poll() is None:
            self.popen.send_signal(signal.SIGKILL)
            self.popen.wait()

    def ended(self) -> bool:
        """Whether the command and every process it started ended, its launcher retired then."""

        if self.returncode is None:
            return False
        self.retire()
        return self.popen.poll() is not None

    async def output(self) -> str:
        return await asyncio.to_thread(read_tail, self.log)

    async def kill(self) -> None:
        """SIGTERM every process of the command, SIGKILL what is left after :attr:`grace_seconds`; return once none
        is left. The sandbox's other commands, a server started beside it, run on."""

        reap = partial(_reap, self.marker)
        await kill_marked(self.process_marker, self.grace_seconds, self.poll_seconds, reap, PROCESS_MARKER)
        if self.returncode is None and self.status is not None:
            self.status.close()
            self.status = None
        await _restore(self.guard, self.log)


@dataclass
class LocalAgentSandbox:
    """Commands on this host, confined to ``cards`` (absolute GPU ids) and marked as the sandbox's, importing
    ``python_path`` first; ``root`` is the sandbox's own directory, for its logs and wrappers.
    ``processes`` adopts every command it starts for ``branch``; ``base_env`` is ``os.environ`` when ``None``.
    ``niceness`` lowers every command's CPU priority, for a sandbox whose host work must not slow the tests beside it.
    ``frameworks`` keeps every command from writing the image's installed frameworks (:func:`framework_protection`);
    the default protects none. ``isolation`` runs every command in bubblewrap (see the module doc), with ``paths``
    and ``forwarded`` (what its commands take from the app's environment besides :func:`confined_env`'s), or
    unconfined, the default; a confined sandbox's frameworks are read-only by its binds, so ``frameworks`` is not
    used.

    Raises:
        IsolationRefused: In a hosted run, where bubblewrap cannot bind the sandbox's paths safely.
    """

    cards: tuple[int, ...]
    python_path: tuple[Path, ...]
    root: Path
    processes: Processes | None = None
    branch: str = ""
    base_env: Mapping[str, str] | None = None
    grace_seconds: float = GRACE_SECONDS
    poll_seconds: float = POLL_SECONDS
    niceness: int = 0
    frameworks: FrameworkProtection = field(default_factory=FrameworkProtection)
    isolation: SandboxIsolation = UNCONFINED
    paths: SandboxPaths = field(default_factory=SandboxPaths)
    forwarded: Mapping[str, str] = field(default_factory=dict)
    _numbers: Iterator[int] = field(init=False, repr=False)
    _told: bool = field(default=False, init=False, repr=False)
    _guarding: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _confined: _Confined | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (BIN_DIR, LOG_DIR):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        start = max((int(log.name.split("-", 1)[0]) for log in (self.root / LOG_DIR).glob("*-*.log")), default=0)
        self._numbers = itertools.count(start + 1)
        pgrep = shutil.which("pgrep", path=self._base().get("PATH")) or "/usr/bin/pgrep"
        script = CONFINE_SCRIPT.format(python=sys.executable, pgrep=pgrep, marker=MARKER, launcher=LAUNCHER_SOURCE)
        for tool in CONFINED:
            wrapper = self.root / BIN_DIR / tool
            wrapper.write_text(script, encoding="utf-8")
            wrapper.chmod(0o755)
        self._confinement()

    @property
    def lifetime_s(self) -> float:
        """A local sandbox lives until it is stopped."""
        return math.inf

    @property
    def tmp(self) -> Path:
        """The sandbox's own temporary directory as its commands see it, the same for every command: a confined
        sandbox's private ``/tmp``; else one below this host's, the same for every sandbox at :attr:`root`."""
        if isinstance(self.isolation, BwrapIsolation):
            return CONFINED_TMP
        digest = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"{TMP_PREFIX}{digest}"

    @property
    def cache_root(self) -> Path:
        """Where the sandbox's JIT and kernel caches go: this host's temporary disk, since a CK module's build writes
        thousands of generated sources, which crawl on the shared storage :attr:`root` is on."""
        return self.tmp / CACHE_DIR

    def env(self, env: Mapping[str, str]) -> dict[str, str]:
        """The environment a command runs with, ``env`` added; the cards and the wrappers are not ``env``'s to
        change."""

        confined = self._confinement()
        base = self._base()
        merged = {name: value for name, value in base.items() if name not in GPU_MASK_ENV_NAMES}
        merged["PYTHONPATH"] = os.pathsep.join([*map(str, self.python_path), *filter(None, [base.get("PYTHONPATH")])])
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        if confined is None:
            self.tmp.mkdir(exist_ok=True)
        merged["TMPDIR"] = str(self.tmp)
        merged.update(env)
        merged = {name: value for name, value in merged.items() if name not in GPU_MASK_ENV_NAMES}
        merged[ROCR_VISIBLE_DEVICES_ENV] = ",".join(map(str, self.cards))
        merged["HIP_VISIBLE_DEVICES"] = ",".join(map(str, range(len(self.cards))))
        merged[MARKER] = str(self.root)
        merged["PATH"] = os.pathsep.join(filter(None, [str(self.root / BIN_DIR), merged.get("PATH")]))
        return merged

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        """Run ``command`` to its end or ``timeout``, at which every process of the command is killed.

        Raises :class:`~hyperloom_multibranch.specialist.framework_guard.FrameworksUnguarded` when the image's
        frameworks can be neither mounted read-only nor restored."""

        log = self._log("run")
        guarded = await self._guarded(command, log)
        process = self._spawn_guarded(command, cwd, env, guarded, log)
        try:
            code = await self._waited(process, timeout)
        finally:
            await self._ran(process, guarded)
        if code is None:
            return CommandResult(TIMED_OUT, f"{read_tail(process.log)}\n`{command}` timed out after {timeout:g} s")
        return CommandResult(code, read_tail(process.log))

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> LocalProcess:
        """Start ``command``; raises what :meth:`run` raises."""

        log = self._log("start")
        guarded = await self._guarded(command, log)
        process = self._spawn_guarded(command, cwd, env, guarded, log)
        if guarded.guard is not None:
            self._follow(process, guarded)
        return process

    def alive(self) -> bool:
        """Whether any process of the sandbox is left, its launchers aside."""
        return bool(sandbox_pids(str(self.root))[0])

    def watched(self) -> str:
        """Start the session's watch again when it ended, which left the sandbox unconfined since; what to say of it,
        empty when it runs."""

        ended = revive()
        return "" if ended is None else _restarted(ended)

    async def stop(self) -> None:
        """SIGTERM every process of the sandbox, every command's, SIGKILL what is left after :attr:`grace_seconds`;
        return once none is left. A confined sandbox's holder ends then, and its private ``/tmp`` and ``/dev/shm`` are
        deleted; a command started after this gets new ones."""
        await kill_marked(str(self.root), self.grace_seconds, self.poll_seconds, partial(_reap, str(self.root)))
        await _restore(self.frameworks.guard)
        confined, self._confined = self._confined, None
        if confined is not None:
            await asyncio.to_thread(confined.release)

    async def _waited(self, process: LocalProcess, timeout: float) -> int | None:
        """``process``'s exit code, or ``None`` once it ran ``timeout`` and was killed."""

        until = time.monotonic() + timeout
        try:
            while process.returncode is None and time.monotonic() < until:
                await asyncio.sleep(self.poll_seconds)
        except asyncio.CancelledError:
            await process.kill()
            raise
        if process.returncode is None:
            await process.kill()
            return None
        process.retire()
        return process.returncode

    async def _guarded(self, command: str, log: Path) -> _Guarded:
        """The frameworks restored before ``command``, whose output goes to ``log``, starts, and it counted as
        running."""

        guard = self.frameworks.guard
        if guard is None:
            return _Guarded()
        restored = await asyncio.to_thread(guard.check)
        said = [] if self._told else [GUARDED_NOTE]
        self._told = True
        if restored is not None:
            said.append(f"before this command: {restored.describe()}")
        note = "".join(f"[hyperloom] {line}\n" for line in said)
        return _Guarded(guard, guard.started(self.branch, command, log), note)

    def _spawn_guarded(
        self, command: str, cwd: Path, env: Mapping[str, str], guarded: _Guarded, log: Path
    ) -> LocalProcess:
        try:
            return self._spawn(log, command, cwd, env, guarded.note)
        except BaseException:
            guarded.ended()
            raise

    async def _ran(self, process: LocalProcess, guarded: _Guarded) -> None:
        """Restore the frameworks once a run's command ended; the command counts as running while a process it left
        does."""

        if guarded.guard is None:
            return
        if process.popen.poll() is None:
            self._follow(process, guarded)
        else:
            guarded.ended()
        await _restore(guarded.guard, process.log)

    def _follow(self, process: LocalProcess, guarded: _Guarded) -> None:
        task = asyncio.get_running_loop().create_task(self._guard_until_ended(process, guarded))
        self._guarding.add(task)
        task.add_done_callback(self._guarding.discard)

    async def _guard_until_ended(self, process: LocalProcess, guarded: _Guarded) -> None:
        """Count ``process`` as running until it and every process it started ended, then restore the frameworks.
        Cancelled, it leaves the command running, so the guard's checks go on while its processes may write."""

        while not process.ended():
            await asyncio.sleep(self.poll_seconds if process.returncode is None else LEFT_POLL_SECONDS)
        guarded.ended()
        await _restore(guarded.guard, process.log)

    def _log(self, kind: str) -> Path:
        return self.root / LOG_DIR / f"{next(self._numbers):04d}-{kind}.log"

    def _spawn(self, log: Path, command: str, cwd: Path, env: Mapping[str, str], note: str = "") -> LocalProcess:
        ended = watch(self.root, self.cards)
        if ended is not None:
            logger.warning("%s", _restarted(ended))
        argv = ["bash", "-c", command]
        if self.niceness:
            argv = ["nice", "-n", str(self.niceness), *argv]
        if self.frameworks.read_only:
            argv = read_only_argv(self.frameworks.read_only, argv)
        process_marker = f"{self.root}/{log.name.split('-', 1)[0]}"
        read_fd, write_fd = os.pipe()
        try:
            with log.open("ab") as out:
                out.write(note.encode())
                out.flush()
                launch = _Launch(
                    launcher_argv(sys.executable, write_fd, argv),
                    cwd,
                    {**self.env(env), PROCESS_MARKER: process_marker},
                    out,
                    write_fd,
                )
                popen = launch.popen() if self._confined is None else self._confined_popen(self._confined, launch)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        _launched.setdefault(str(self.root), []).append(popen)
        if self.processes is not None:
            try:
                self.processes.adopt(popen.pid, branch=self.branch, argv=argv)
            except ValueError:
                logger.warning("`%s` of %s ended before the process registry took it over", command, self.branch)
        return LocalProcess(
            popen,
            log,
            str(self.root),
            process_marker,
            self.grace_seconds,
            self.poll_seconds,
            StatusPipe(read_fd),
            self.frameworks.guard,
        )

    def _base(self) -> Mapping[str, str]:
        """What a command's environment starts from: a confined sandbox's own (:func:`confined_env`), else the
        app's (``base_env``)."""
        if self._confined is not None:
            return self._confined.env
        return os.environ if self.base_env is None else self.base_env

    def _confinement(self) -> _Confined | None:
        """The sandbox's confinement, made again after :meth:`stop`; ``None`` for an unconfined sandbox."""
        if self._confined is None and isinstance(self.isolation, BwrapIsolation):
            self._confined = self._confine(self.isolation)
        return self._confined

    def _confine(self, isolation: BwrapIsolation) -> _Confined | None:
        """The confinement of every command, or ``None`` where bubblewrap cannot bind the sandbox's paths safely
        (:meth:`~hyperloom_common.bwrap.BwrapPolicy.problems`): the sandbox then runs unconfined, as before, with
        its frameworks protected as an unconfined sandbox's are, and says so in a WARNING.

        Raises:
            IsolationRefused: In a hosted run, which refuses that.
        """
        home = self.paths.home or self.root / HOME_DIR
        make_shared_dirs(home / CLAUDE_DIR)
        env = confined_env(self._base(), home, self.forwarded)
        scratch = Scratch.create()
        try:
            policy = self._policy(env, home, scratch)
            refused = policy.problems(isolation.probe.version or (0, 0, 0))
        except BwrapPolicyError as error:
            refused = list(error.problems)
        except BaseException:
            scratch.remove()
            raise
        if not refused:
            self.frameworks = FrameworkProtection()
            return _Confined(isolation, policy, env, home, scratch)
        scratch.remove()
        failed = dataclasses.replace(isolation.probe, problems=tuple(refused))
        self.isolation = choose_sandbox_isolation(known=failed)
        logger.warning("the agent sandbox at %s runs unconfined, as before: %s", self.root, self.isolation.why)
        self.frameworks = framework_protection()
        return None

    def _policy(self, env: Mapping[str, str], home: Path, scratch: Scratch) -> BwrapPolicy:
        """Every command's policy: the sandbox's root, its home, the trees it names and the image's writable
        frameworks writable; the toolchain of :data:`AGENT_PROGRAMS`, its import path, the paths it names and the
        image's other installed frameworks read-only, but none in a tree it writes, which shows it already; the
        credential locations they show absent (a read-only tree bound entry by entry without them, so a login the app
        replaces or makes while a command runs never shows, :meth:`Confinement.policy`); the cards' devices; the
        app's state never bound. No command dies with the thread that started it: the session's watch ends what is
        left of a sandbox, with its grace."""
        installed = installed_frameworks()
        writable = (self.root, home, *self.paths.writable, *installed.writable)
        trees = (*writable, Path(scratch.tmp), Path(scratch.shm))
        toolchain = [path for program in AGENT_PROGRAMS for path in toolchain_paths(env, program, writable=trees)]
        named = (*toolchain, *self.python_path, *self.paths.readable, *installed.read_only)
        readable = tuple(path for path in dict.fromkeys(named) if not _within(path, writable))
        masked = credential_paths(env, allowed=(*readable, *writable), writable=trees)
        state = os.environ.get(STATE_DIR_ENV, "")
        confinement = Confinement(
            writable=writable,
            readable=readable,
            masked=masked,
            devices=GPU_DEVICES,
            app_only=(Path(state),) if os.path.isabs(state) else (),
        )
        return dataclasses.replace(confinement.policy(scratch=scratch), die_with_parent=False)

    def _confined_popen(self, confined: _Confined, launch: _Launch) -> subprocess.Popen[bytes]:
        """``launch`` in the sandbox, joined to its holder: the markers are set inside only, so bubblewrap's own
        process carries none and every kill's SIGTERM reaches the command's processes, never bubblewrap's."""
        policy = dataclasses.replace(confined.policy, chdir=str(launch.cwd))
        with self._holder(confined).joined(policy) as joined:
            argv, env = confined.isolation.launch(launch.argv, joined, launch.env, marks=(MARKER, PROCESS_MARKER))
            return dataclasses.replace(launch, argv=argv, env=env, fds=joined.join).popen()

    def _holder(self, confined: _Confined) -> Holder:
        """The holder the sandbox's commands join, started when none runs or the one that ran has ended, which ended
        every process in it."""
        holder = confined.holder
        if holder is not None and holder.pidfd >= 0 and not select.select([holder.pidfd], [], [], 0)[0]:
            return holder
        if holder is not None:
            holder.stop()
        confined.holder = Holder.start(confined.isolation.probe, confined.policy)
        watch(self.root, self.cards, holder=confined.holder.pid)
        return confined.holder


def read_only_argv(paths: tuple[Path, ...], argv: list[str]) -> list[str]:
    """``argv`` run in a private mount namespace where each of ``paths`` is mounted read-only."""
    script = ["sh", "-c", _READ_ONLY_SCRIPT, "read-only", *map(str, paths), "--", *argv]
    return [*_PRIVATE_MOUNTS, *script]


@functools.cache
def framework_protection() -> FrameworkProtection:
    """The protection of :data:`READ_ONLY_FRAMEWORKS` installed in this image: read-only mounts where this host gives a
    command a mount namespace of its own, else the guard, one for every sandbox of this process, which is logged as
    an error since a command can then write the packages until the guard's next check."""

    packages = []
    for name in READ_ONLY_FRAMEWORKS:
        layout = COMPONENT_LAYOUTS[name]
        try:
            packages.append(GuardedPackage(name, locate_installed_package(layout).path, layout.generated))
        except UnsupportedComponentError:
            continue
    if not packages:
        return FrameworkProtection()
    if mount_namespaces():
        return FrameworkProtection(read_only=tuple(package.path for package in packages))
    logger.error(
        "sandbox commands get no mount namespace on this host, so %s cannot be mounted read-only: what a command "
        "writes there is restored within %g s and logged with the command",
        ", ".join(str(package.path) for package in packages),
        PATROL_SECONDS,
    )
    return FrameworkProtection(guard=FrameworkGuard(tuple(packages)))


def mount_namespaces() -> bool:
    """Whether this host gives a command a mount namespace of its own: root, and ``unshare`` permitted."""

    probe = [*_PRIVATE_MOUNTS, "true"]
    try:
        return (
            shutil.which(_PRIVATE_MOUNTS[0]) is not None and subprocess.run(probe, capture_output=True).returncode == 0
        )
    except OSError:
        return False


def sandbox_isolation() -> SandboxIsolation:
    """How this host's agent sandboxes run their commands
    (:func:`~hyperloom.common.agent_isolation.choose_sandbox_isolation`); unconfined is said once, at WARNING, with
    the reasons (:func:`~hyperloom.common.agent_isolation.announce_agent_isolation`).

    Raises:
        IsolationRefused: In a hosted run, where bubblewrap cannot sandbox an agent.
    """
    isolation = choose_sandbox_isolation()
    if isinstance(isolation, NoIsolation):
        announce_agent_isolation()
    return isolation


def confined_env(environ: Mapping[str, str], home: Path, forwarded: Mapping[str, str]) -> dict[str, str]:
    """What a confined sandbox's commands start from, taken from ``environ`` (the app's, a lease's on top): the
    search paths, locale, terminal and the models and benchmark checkout it names (:data:`BASIC_NAMES`), the framework
    image's own settings (:func:`~hyperloom.measure.local_templates.image_env`), and ``forwarded`` (the LLM gateway,
    its credentials and models, proxies and CA bundle); with ``home`` as the home, Claude config, cache, Python's user
    base, pip's install target (:data:`SITE_DIR`) and Triton's cache, which the image's settings may name in the app's
    home. aiter's JIT directory and FlyDSL's cache are left to aiter: its own tree, which a confined command writes,
    where the image's prebuilt modules and FlyDSL kernels are, which a directory of the home would hide. No other
    variable of the app's: not its other credentials, nor its home or tools' configuration. Without a Hugging Face
    location of its own, the hub cache is the app's, which a model id resolves in."""
    kept = {name: value for name, value in environ.items() if name in BASIC_NAMES or name.startswith(BASIC_PREFIXES)}
    kept.update(image_env(environ))
    kept.update(forwarded)
    hub = _hub_cache(environ)
    if hub is not None:
        kept.setdefault("HF_HUB_CACHE", str(hub))
    kept.update(
        {
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(home / CLAUDE_DIR),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "PYTHONUSERBASE": str(home / USER_BASE),
            "PIP_TARGET": str(home / SITE_DIR),
            "TRITON_CACHE_DIR": str(home / ".triton" / CACHE_DIR),
            # The inner user is the app's, root in a container, which the Claude CLI accepts only in a sandbox.
            "IS_SANDBOX": "1",
        }
    )
    return kept


def _hub_cache(environ: Mapping[str, str]) -> Path | None:
    """The app's Hugging Face hub cache where ``environ`` names no Hugging Face location and it exists, else
    ``None``."""
    if environ.get("HF_HOME") or environ.get("HF_HUB_CACHE") or not os.path.isabs(environ.get("HOME", "")):
        return None
    cache = environ.get("XDG_CACHE_HOME") or os.path.join(environ["HOME"], ".cache")
    hub = Path(cache, "huggingface", "hub")
    return hub if os.path.isabs(cache) and hub.is_dir() else None


@dataclass(frozen=True)
class InstalledFrameworks:
    """Where the image's frameworks are installed: ``writable`` (:data:`WRITABLE_FRAMEWORKS`) and ``read_only``
    (every other component, so the links a worktree holds to their compiled modules resolve in a sandbox)."""

    read_only: tuple[Path, ...] = ()
    writable: tuple[Path, ...] = ()


@functools.cache
def installed_frameworks() -> InstalledFrameworks:
    """The image's installed frameworks (:class:`InstalledFrameworks`), found once without importing them."""
    read_only: list[Path] = []
    writable: list[Path] = []
    for name, layout in COMPONENT_LAYOUTS.items():
        try:
            path = locate_installed_package(layout).path
        except UnsupportedComponentError:
            continue
        (writable if name in WRITABLE_FRAMEWORKS else read_only).append(path)
    return InstalledFrameworks(tuple(read_only), tuple(writable))


def _within(path: Path, trees: tuple[Path, ...]) -> bool:
    """Whether ``path``, as given, is one of ``trees`` or below one."""
    given = Path(os.path.normpath(path))
    normalized = [Path(os.path.normpath(tree)) for tree in trees]
    return any(given == tree or tree in given.parents for tree in normalized)


def _runs_launcher(pid: int) -> bool:
    """Whether process ``pid`` runs a sandbox's launcher."""
    try:
        return LAUNCHER_SOURCE.encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _reap(root: str) -> None:
    """Collect the launchers of the sandbox at ``root`` that ended, which would otherwise linger as zombies."""

    _launched[root] = [popen for popen in _launched.get(root, []) if popen.poll() is None]


def _restarted(ended: tuple[int, Path]) -> str:
    status, err = ended
    return f"the sandbox's watch had ended (exit {status}) and was started again; its stderr is in {err}"


def read_tail(log: Path, size: int = TAIL_BYTES) -> str:
    """The last ``size`` bytes of ``log``, decoded."""

    with log.open("rb") as file:
        file.seek(max(0, file.seek(0, os.SEEK_END) - size))
        return file.read().decode(errors="replace")


__all__ = [
    "CONFINE_SCRIPT",
    "CONFINED_TMP",
    "GUARDED_NOTE",
    "READ_ONLY_FRAMEWORKS",
    "SITE_DIR",
    "TAIL_BYTES",
    "TIMED_OUT",
    "UNCONFINED",
    "WRITABLE_FRAMEWORKS",
    "FrameworkProtection",
    "InstalledFrameworks",
    "LocalAgentSandbox",
    "LocalProcess",
    "SandboxIsolation",
    "SandboxPaths",
    "confined_env",
    "framework_protection",
    "installed_frameworks",
    "mount_namespaces",
    "read_only_argv",
    "read_tail",
    "sandbox_isolation",
]
