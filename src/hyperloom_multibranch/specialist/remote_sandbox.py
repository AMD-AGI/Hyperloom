# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's agent sandbox in a remote ``dev`` sandbox of the hosted backend.

The ``dev`` template mounts the shared storage the session directory is on at the path it has on this host, so every
file below the session directory -- the sandbox's root, its logs, the worktree a command imports, the agent's
checkout -- is written and read here directly. Only process control goes through the sandbox's command API: a start
runs a launcher (:data:`LAUNCH_SCRIPT`) detached, which runs the command in a session of its own with its output
appended to a numbered log under the root, and writes its pid and then its exit code into files this host made
beforehand. A file this host made is opened afresh at every
read, which the shared storage revalidates; a file the sandbox made would show here only after its directory's cache
expired.

Every process a command starts carries :data:`~hyperloom_multibranch.specialist.agent_watch.MARKER` set to the root, and
:data:`PROCESS_MARKER` set to the root and the command's number, so a kill (:data:`PROCS_SCRIPT`) ends a command's or
the whole sandbox's processes, whatever session they moved to. The sandbox is a container of its own: nothing else
runs in it, so no wrapper confines ``pkill`` and no watch confines the cards.

A process whose launcher and command are gone without an exit code, or whose sandbox ended, ends with :data:`LOST`
and a line in its log naming the cause; :meth:`RemoteProcess.output` probes for that every
:attr:`RemoteAgentSandbox.probe_seconds`.

A command runs on the sandbox's own environment with only what :func:`forwarded_env` takes from this host's (the LLM
gateway, its credentials and models, the CA bundle the hosting layer puts at the same path in both containers) and the
sandbox's own home, Claude config, caches and a temporary directory on the sandbox's local disk, where HIP unbundles
its code objects. Commands run as root there, as the Claude CLI accepts only in a sandbox it is told of. The platform
sets the device mask; a mask the caller passes is dropped.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import shlex
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.sandbox import RunSettings, Sandbox, SandboxGone, SandboxUnavailable
from hyperloom_multibranch.specialist.agent_onboarding import ENCODING, CommandResult
from hyperloom_multibranch.specialist.agent_sandbox import TIMED_OUT, read_tail
from hyperloom_multibranch.specialist.agent_watch import GRACE_SECONDS, MARKER

logger = logging.getLogger(__name__)

#: Set, besides :data:`MARKER`, to ``<root>/<number>`` on every process of one command.
PROCESS_MARKER = "HYPERLOOM_AGENT_PROCESS"
#: The exit code of a process that ended with no exit code of its own, as a SIGKILL reports it.
LOST = 137
#: How often a running process is probed for a launcher that is gone or a sandbox that ended.
PROBE_SECONDS = 30.0
#: How long one call that starts, probes or lists processes may take.
CONTROL_SECONDS = 60.0
#: The sandbox's temporary directory, on its local disk.
SANDBOX_TMP = "/tmp/hl"
BIN_DIR, LOG_DIR, RUN_DIR, HOME_DIR, CACHE_DIR = "bin", "logs", "run", "home", "cache"
LAUNCHER, PROCS = "hl-launch.sh", "hl-procs.sh"
#: How long a start waits, in tenths of a second, for the launcher to write its pid.
_START_TENTHS = 100

#: What a command takes from this host's environment: the LLM gateway, its credentials and models, and the CA bundle,
#: by prefix and by name.
FORWARDED_PREFIXES = ("ANTHROPIC_", "OPENAI_")
FORWARDED_NAMES = frozenset(
    {
        "CLAUDE_MODEL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CODEX_MODEL",
        "GEAK_CLAUDE_MODEL",
        "HF_ENDPOINT",
        "HF_HUB_OFFLINE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NODE_TLS_REJECT_UNAUTHORIZED",
        "NO_PROXY",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
        "NODE_EXTRA_CA_CERTS",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    }
)

LAUNCH_SCRIPT = f"""#!/bin/bash
# hl-launch.sh RUN N LOG CWD: run $HL_COMMAND in CWD, in a session of its own, its output appended to LOG; write
# "LAUNCHER COMMAND" pids to RUN/N.pid and the exit code to RUN/N.exit. Only the command carries the marks, so a kill
# leaves the launcher to write how the command ended.
run=$1 n=$2 log=$3 cwd=$4
command=$HL_COMMAND
export PATH="${{HL_PATH_PREFIX:+$HL_PATH_PREFIX:}}$PATH"
export PYTHONPATH="$HL_PYTHONPATH${{PYTHONPATH:+:$PYTHONPATH}}"
root=$HL_ROOT
unset HL_COMMAND HL_PATH_PREFIX HL_PYTHONPATH HL_ROOT
mkdir -p "$TMPDIR"
if cd "$cwd" 2>>"$log"; then
  env "{MARKER}=$root" "{PROCESS_MARKER}=$root/$n" setsid bash -c "$command" >>"$log" 2>&1 </dev/null &
  child=$!
  echo "$$ $child" >"$run/$n.pid"
  wait "$child"
  code=$?
else
  echo "$$" >"$run/$n.pid"
  code=1
fi
echo "$code" >"$run/$n.exit"
"""

PROCS_SCRIPT = """#!/bin/bash
# hl-procs.sh list|kill NAME=VALUE [GRACE]: the processes whose environment holds NAME=VALUE; kill sends them SIGTERM
# and, GRACE seconds later, SIGKILL to what is left.
mark=$2
marked() {
  for environ in /proc/[0-9]*/environ; do
    pid=${environ#/proc/}
    pid=${pid%/environ}
    [ "$pid" != "$$" ] && grep -qzxF -- "$mark" "$environ" 2>/dev/null && echo "$pid"
  done
}
pids=$(marked)
if [ "$1" = list ]; then
  echo $pids
  exit 0
fi
[ -z "$pids" ] && exit 0
kill -TERM $pids 2>/dev/null
end=$((SECONDS + $3))
while [ -n "$pids" ] && [ "$SECONDS" -lt "$end" ]; do
  sleep 0.5
  pids=$(marked)
done
[ -n "$pids" ] && kill -KILL $pids 2>/dev/null
exit 0
"""


def forwarded_env(environ: Mapping[str, str]) -> dict[str, str]:
    """What a remote sandbox's commands take from ``environ``, this host's environment: never all of it, whose paths
    and settings are this host's."""

    return {
        name: value for name, value in environ.items() if name in FORWARDED_NAMES or name.startswith(FORWARDED_PREFIXES)
    }


def claude_config_dir(root: Path) -> Path:
    """Where the Claude CLI of the sandbox at ``root`` keeps its state and transcripts."""
    return root / HOME_DIR / ".claude"


@dataclass
class RemoteProcess:
    """Command ``number`` of ``owner``, its output in ``log``."""

    owner: RemoteAgentSandbox
    number: int
    log: Path
    _code: int | None = field(default=None, init=False, repr=False)
    _probed_at: float = field(default=-math.inf, init=False, repr=False)

    @property
    def returncode(self) -> int | None:
        if self._code is None:
            self._code = _code_in(self.exit_file)
        return self._code

    async def output(self) -> str:
        await self.check()
        return await asyncio.to_thread(read_tail, self.log)

    async def check(self) -> None:
        """Probe the process, once :attr:`RemoteAgentSandbox.probe_seconds` passed since the last probe, for a
        launcher gone without an exit code or a sandbox that ended."""

        if self.returncode is None and self.owner.clock() - self._probed_at >= self.owner.probe_seconds:
            await self._probe()

    async def kill(self) -> None:
        """SIGTERM every process of the command, SIGKILL what is left after the grace; return once none is left."""

        try:
            await self.owner.kill_marked(f"{PROCESS_MARKER}={self.owner.root}/{self.number}")
        except SandboxGone as error:
            self._lost(f"the sandbox ended: {error}")
            return
        if self.returncode is None:
            await self._probe()

    async def _probe(self) -> None:
        self._probed_at = self.owner.clock()
        pid, done = shlex.quote(str(self.pid_file)), shlex.quote(str(self.exit_file))
        probe = (
            f"read launcher command < {pid}; if [ -s {done} ]; then cat {done}; "
            'elif kill -0 "$launcher" 2>/dev/null || kill -0 "$command" 2>/dev/null; then echo running; '
            "else echo gone; fi"
        )
        try:
            state = (await self.owner.control(probe)).stdout.strip()
        except SandboxGone as error:
            self._lost(f"the sandbox ended: {error}")
            return
        except SandboxUnavailable:
            logger.warning("process %s of %s could not be probed", self.number, self.owner.root, exc_info=True)
            return
        if state.lstrip("-").isdigit():
            self._code = int(state)
        elif state == "gone":
            self._lost("its launcher and command are gone and left no exit code")

    def _lost(self, cause: str) -> None:
        if self._code is not None:
            return
        self._code = LOST
        with self.log.open("a", encoding=ENCODING) as log:
            log.write(f"\n{cause}; exit {LOST}\n")

    @property
    def pid_file(self) -> Path:
        """Where the launcher writes its pid and the command's."""
        return self.owner.root / RUN_DIR / f"{self.number}.pid"

    @property
    def exit_file(self) -> Path:
        """Where the launcher writes the command's exit code."""
        return self.owner.root / RUN_DIR / f"{self.number}.exit"


@dataclass
class RemoteAgentSandbox:
    """Commands in the remote ``sandbox``, its ``cards`` the platform's, importing ``python_path`` first; ``root``,
    on the shared storage, is the sandbox's own directory, for its logs, scripts, home and caches. ``forwarded`` is
    what its commands take from this host's environment (:func:`forwarded_env`); ``lifetime_s``, how long the sandbox
    lives from its lease."""

    sandbox: Sandbox
    root: Path
    python_path: tuple[Path, ...]
    cards: int
    forwarded: Mapping[str, str] = field(default_factory=dict)
    lifetime_s: float = math.inf
    grace_seconds: float = GRACE_SECONDS
    poll_seconds: float = 1.0
    probe_seconds: float = PROBE_SECONDS
    clock: Callable[[], float] = time.monotonic
    _numbers: Iterator[int] = field(init=False, repr=False)
    _alive: bool = field(default=False, init=False, repr=False)
    _alive_at: float = field(default=-math.inf, init=False, repr=False)
    _refresh: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (BIN_DIR, LOG_DIR, RUN_DIR, HOME_DIR, CACHE_DIR):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        claude_config_dir(self.root).mkdir(exist_ok=True)
        start = max((int(log.name.split("-", 1)[0]) for log in (self.root / LOG_DIR).glob("*-*.log")), default=0)
        self._numbers = itertools.count(start + 1)
        for name, script in ((LAUNCHER, LAUNCH_SCRIPT), (PROCS, PROCS_SCRIPT)):
            path = self.root / BIN_DIR / name
            path.write_text(script, encoding=ENCODING)
            path.chmod(0o755)

    def env(self, env: Mapping[str, str]) -> dict[str, str]:
        """The environment a command runs with on top of the sandbox's own, ``env`` added; a ``PATH`` in ``env`` is
        this host's with directories added, and only those go before the sandbox's own ``PATH``."""

        home = self.root / HOME_DIR
        merged = {
            **self.forwarded,
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(claude_config_dir(self.root)),
            "XDG_CACHE_HOME": str(self.root / CACHE_DIR),
            "TMPDIR": SANDBOX_TMP,
            "PYTHONDONTWRITEBYTECODE": "1",
            # Commands run as root, which the Claude CLI's --dangerously-skip-permissions refuses outside a sandbox.
            "IS_SANDBOX": "1",
            # The shared storage's files belong to another account than the sandbox's, which git refuses to work in.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "*",
        }
        merged.update({name: value for name, value in env.items() if name != "PATH"})
        merged = {name: value for name, value in merged.items() if name not in GPU_MASK_ENV_NAMES}
        # Every card of the sandbox, numbered from 0 as HIP numbers them: the docker shim indexes a mask into it.
        merged["HIP_VISIBLE_DEVICES"] = ",".join(map(str, range(self.cards)))
        merged["HL_PATH_PREFIX"] = os.pathsep.join(_added_path(env.get("PATH", "")))
        merged["HL_PYTHONPATH"] = os.pathsep.join(map(str, self.python_path))
        merged["HL_ROOT"] = str(self.root)
        return merged

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        """Run ``command`` to its end or ``timeout``, at which every process of it is killed."""

        process = await self._spawn("run", command, cwd, env)
        until = self.clock() + timeout
        try:
            while process.returncode is None and self.clock() < until:
                await asyncio.sleep(self.poll_seconds)
                await process.check()
        except asyncio.CancelledError:
            await process.kill()
            raise
        if process.returncode is None:
            await process.kill()
            return CommandResult(TIMED_OUT, f"{read_tail(process.log)}\n`{command}` timed out after {timeout:g} s")
        return CommandResult(process.returncode, read_tail(process.log))

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> RemoteProcess:
        """Start ``command`` in ``cwd``; return once its launcher runs.

        Raises:
            OSError: When the launcher did not start.
            SandboxGone: When the sandbox has ended.
        """

        return await self._spawn("start", command, cwd, env)

    async def _spawn(self, kind: str, command: str, cwd: Path, env: Mapping[str, str]) -> RemoteProcess:
        number = next(self._numbers)
        process = RemoteProcess(self, number, self.root / LOG_DIR / f"{number:04d}-{kind}.log")
        for path in (process.log, process.pid_file, process.exit_file):
            path.write_text("", encoding=ENCODING)
        launch = (self.root / BIN_DIR / LAUNCHER, self.root / RUN_DIR, number, process.log, cwd)
        pid = shlex.quote(str(process.pid_file))
        script = (
            f"setsid nohup {' '.join(shlex.quote(str(arg)) for arg in launch)} >/dev/null 2>&1 </dev/null & "
            f"for _ in $(seq {_START_TENTHS}); do [ -s {pid} ] && exit 0; sleep 0.1; done; exit 1"
        )
        done = await self.control(script, {**self.env(env), "HL_COMMAND": command})
        if done.exit_code != 0:
            raise OSError(f"`{command}` did not start in sandbox {self.sandbox.sandbox_id}: {done.stderr.strip()}")
        self._alive, self._alive_at = True, self.clock()
        return process

    def alive(self) -> bool:
        """Whether any process of the sandbox is left, as last looked; a look older than :attr:`probe_seconds` is
        taken again in the background."""

        if self._refresh is None and self.clock() - self._alive_at >= self.probe_seconds:
            try:
                self._refresh = asyncio.get_running_loop().create_task(self._look())
            except RuntimeError:
                return self._alive
        return self._alive

    def watched(self) -> str:
        """Nothing: the sandbox is a container of its own and confines its processes itself."""
        return ""

    async def stop(self) -> None:
        """Kill every process of the sandbox; one that ended has none left."""

        try:
            await self.kill_marked(f"{MARKER}={self.root}")
        except SandboxGone:
            logger.info("the sandbox of %s had ended", self.root)
        self._alive, self._alive_at = False, self.clock()

    async def kill_marked(self, mark: str) -> None:
        """SIGTERM every process whose environment holds ``mark``, SIGKILL what is left after the grace."""

        grace = math.ceil(self.grace_seconds)
        command = f"{shlex.quote(str(self.root / BIN_DIR / PROCS))} kill {shlex.quote(mark)} {grace}"
        await self.control(command, timeout=grace + CONTROL_SECONDS)

    async def control(
        self, command: str, envs: Mapping[str, str] | None = None, timeout: float = CONTROL_SECONDS
    ) -> CommandResult:
        """Run a control ``command`` in the sandbox on a worker thread.

        Raises:
            SandboxGone: When the sandbox has ended.
            SandboxUnavailable: When the call timed out or the service is unavailable.
        """

        settings = RunSettings(envs=envs, timeout_sec=timeout)
        return await asyncio.to_thread(self.sandbox.run, command, settings=settings)

    async def _look(self) -> None:
        try:
            listed = await self.control(f"{shlex.quote(str(self.root / BIN_DIR / PROCS))} list {MARKER}={self.root}")
            self._alive = bool(listed.stdout.strip())
        except SandboxGone:
            self._alive = False
        except SandboxUnavailable:
            logger.warning("the processes of %s could not be listed", self.root, exc_info=True)
        finally:
            self._alive_at, self._refresh = self.clock(), None


def _code_in(path: Path) -> int | None:
    text = path.read_text(encoding=ENCODING).strip()
    return int(text) if text else None


def _added_path(path: str) -> list[str]:
    host = set(os.environ.get("PATH", "").split(os.pathsep))
    return [entry for entry in path.split(os.pathsep) if entry and entry not in host]


__all__ = [
    "LOST",
    "PROCESS_MARKER",
    "RemoteAgentSandbox",
    "RemoteProcess",
    "claude_config_dir",
    "forwarded_env",
]
