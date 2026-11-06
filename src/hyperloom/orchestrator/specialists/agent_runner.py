# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AI work socket: launch, poll, cancel and collect agent CLI sessions.

The local implementation spawns the argv the existing command builders produced,
and hands the existing reap loop an absolute deadline, so a long-lived session is
not cut by the 4 hour cap that only applies to an unbounded run. Heartbeat rules,
environment allowlist, tree kill and reap-mark sweep are the dispatcher's: each
launch puts its own ``HYPERLOOM_REAP_MARK`` in the CLI's environment, and however
supervision ends (exit, done file, silence, deadline, cancel) the CLI's tree is
killed if its root still runs, and then every process that still carries that
mark, however it detached. Nothing else is found: a process the CLI started that
dropped its environment (one exec'd with an empty one, say) carries no mark, so
unless it is still in the tree of a live root it outlives the session, as it
outlives the dispatcher's. Whether a session that ended or was lost is restarted,
and from what, is the caller's decision; this module only launches, polls, cancels
and collects, and, for a session launched with its stdin open, sends it further
messages.

Tokens reach the CLI only through environment variables: the MCP configuration
written to disk names the variable, never its value.

Each launch is isolated as
:class:`~hyperloom.orchestrator.specialists.subprocess_.SpecialistSubprocessDispatcher`'s
are (:mod:`~hyperloom.orchestrator.specialists.launch_env`): as the account
``HYPERLOOM_SPECIALIST_USER`` names, with the workspace and ``cwd`` shared with
it; in a bubblewrap sandbox that writes the workspace and ``cwd`` and reads its
toolchain and ``readable``; or unconfined as before. Every file the runner writes
there is created without following a link the CLI left.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import CancelledError as ScopeCancelled
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol

from hyperloom.common.deadline import Deadline
from hyperloom.common.agent_isolation import AgentIsolation
from hyperloom.common.env_safety import RunAsError, redact_secret_values
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.session.session_paths import RESOURCES_RELATIVE_PATH
from hyperloom_common.fs_utils import (
    append_text_nofollow,
    create_nofollow,
    read_bytes_nofollow,
    read_text_nofollow,
    rewrite_lines_nofollow,
    write_text_nofollow,
)
from hyperloom_common.llm.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX

from .agent_cli_config import _build_specialist_env as build_specialist_env
from .agent_cli_config import SpecialistAgentUnavailableError
from .launch_env import (
    LaunchPlace,
    PreparedLaunch,
    SandboxedProcess,
    Spawn,
    agent_pid,
    codex_sandbox_off,
    prepare_launch,
    require_claude_login,
    resolve_launch_isolation,
    stage_mcp_config,
    stamp_reap_mark,
    without_codex_sandbox,
)
from .mcp_config import SPECIALIST_MCP_CONFIG_FILENAME
from .reap_watch import PROCESS_LOG_NAME, cli_alive, newest_activity
from .resources import ContainerSnapshot, ProcessTreeSampler, dir_usage_bytes, read_container_cgroup
from .subprocess_ import (
    ReapKillUnconfirmed,
    ReapWatch,
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _codex_mcp_config,
    _write_private_codex_config,
    clear_wall_budget_extension,
)

log = logging.getLogger(__name__)

__all__ = [
    "CONTAINER_RECORD_KIND",
    "DEFAULT_MCP_TOKEN_ENV",
    "LOST_AFTER_SEC",
    "SPECIALIST_RECORD_KIND",
    "AgentConfigError",
    "AgentEnded",
    "AgentHandle",
    "AgentLaunchSpec",
    "AgentRunner",
    "AgentStatus",
    "Beat",
    "HeartbeatSink",
    "InteractiveAgentRunner",
    "Liveness",
    "LocalAgentRunner",
    "McpEndpoint",
    "RunnerHost",
    "build_specialist_env",
]

#: How long a CLI may stay silent. Past it with no tool call in flight, the reap loop ends the CLI and ``poll`` reports
#: it lost; ``poll`` also counts a beat and the session's start as life within it.
LOST_AFTER_SEC: float = 300.0

#: The ``kind`` of each line the runner appends to :data:`RESOURCES_RELATIVE_PATH`: one specialist's run, and one
#: reading of the container.
SPECIALIST_RECORD_KIND = "specialist_resources"
CONTAINER_RECORD_KIND = "container_resources"

#: Environment variable that carries a specialist's MCP bearer token when its endpoint names no other.
DEFAULT_MCP_TOKEN_ENV = "HYPERLOOM_SPECIALIST_MCP_TOKEN"

#: Tree CPU seconds gained in one sampling period that count as "still working".
ACTIVITY_CPU_SECONDS = 0.1

CLAUDE_CONFIG_DIRNAME = ".claude-config"
PROMPT_FILE = "prompt.md"
ISOLATED = "isolated"
UNKNOWN = "unknown"
STATE_LOST = "lost"
UTF8 = "utf-8"
PRIVATE_FILE_MODE = 0o600
AGENT_RUNTIME_FILENAME = "agent_runtime.json"
_PLUGIN_ENV_PREFIXES = ("CLAUDE_CODE_PLUGIN", "CLAUDE_PLUGIN")
_MCP_BLOCK_BEGIN = "# >>> hyperloom agent_runner mcp"
_MCP_BLOCK_END = "# <<< hyperloom agent_runner mcp"
_SERVER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AgentConfigError(ValueError):
    """The launch spec cannot be turned into a safe session."""


class AgentEnded(RuntimeError):
    """A message was sent to a session whose CLI has already exited."""


#: Refreshes a session's liveness; safe to call from any thread, a no-op once the session is over.
Beat = Callable[[], None]

#: Receives the session's ``beat`` at launch. The MCP service calls it periodically while a call made with
#: that session's token is in flight, so ``poll`` does not judge a session lost while it waits on the call. The reap loop
#: does not read the beat. An error it raises fails the launch, once the CLI it started is ended.
HeartbeatSink = Callable[[Beat], None]


@dataclass(frozen=True)
class McpEndpoint:
    name: str
    """Server name in the CLI configuration."""
    url: str
    """MCP service address."""
    token_env: str = DEFAULT_MCP_TOKEN_ENV
    """Environment variable holding the bearer token; the token itself lives only there."""


@dataclass(frozen=True)
class AgentLaunchSpec:
    task_id: str
    workspace: Path
    """Holds prompt.md (stdin), system_prompt.md, worktree/, and the per-session CLI config."""
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    gpus: int
    deadline_unix: float
    """Absolute wall-clock end of the session."""
    done_paths: tuple[Path, ...]
    heartbeat_path: Path
    mcp_servers: tuple[McpEndpoint, ...] = ()
    claude_config_dir: Literal["isolated", "inherit"] = ISOLATED
    backend: str = ""
    """``claude`` or ``codex``; empty infers it from ``argv[0]``."""
    heartbeat_sink: HeartbeatSink | None = None
    """Called once at launch with the session's ``beat`` function; an error it raises fails the launch."""
    stdin_open: bool = False
    """Keep the CLI's stdin open after prompt.md, so :meth:`LocalAgentRunner.send` reaches the same session later; the
    CLI ends once :meth:`LocalAgentRunner.close_stdin` closes it."""
    readable: tuple[Path, ...] = ()
    """Paths the CLI reads besides its toolchain (references, a model, a lease area): bound read-only when it runs in
    a bubblewrap sandbox; nothing changes in another tier."""


@dataclass(frozen=True)
class AgentStatus:
    state: Literal["running", "exited", "lost"]
    returncode: int | None
    last_heartbeat_unix: float | None
    timed_out: bool = False
    error: str = ""


@dataclass
class AgentHandle:
    task_id: str
    pid: int
    started_unix: float
    session: Any = field(repr=False, default=None)


class AgentRunner(Protocol):
    """Launches, polls, cancels and collects agent CLI sessions."""

    name: str

    def launch(self, spec: AgentLaunchSpec) -> Awaitable[AgentHandle]:
        """Start the CLI ``spec`` describes and return the handle of its session. A launch that fails once the CLI has
        started ends it, and what it started, before the error is raised."""

    def poll(self, handle: AgentHandle) -> Awaitable[AgentStatus]:
        """Whether the session is running, has exited, or is lost. A session reads as ended once its CLI's root has
        exited, which can be before the teardown has swept what the CLI started; :meth:`collect` waits for that."""

    def cancel(self, handle: AgentHandle) -> Awaitable[None]:
        """Kill the CLI's process tree. When this returns the CLI's root is dead, but a process the CLI detached can
        still run until the session's teardown has swept the launch's reap mark; :meth:`collect` returns only after
        that sweep."""

    def collect(self, handle: AgentHandle) -> Awaitable[None]:
        """Record an ended session once its teardown, the reap-mark sweep included, has finished; a session whose CLI
        still runs is refused."""


class InteractiveAgentRunner(AgentRunner, Protocol):
    """A runner that keeps a session's stdin open when its spec asks, so one CLI takes message after message."""

    def send(self, handle: AgentHandle, data: bytes) -> Awaitable[None]:
        """Write ``data`` to the session's stdin."""

    def close_stdin(self, handle: AgentHandle) -> Awaitable[None]:
        """Close the session's stdin; its CLI ends once it has read what was sent."""


def _infer_backend(spec: AgentLaunchSpec) -> str:
    backend = spec.backend.strip().lower()
    if not backend:
        backend = AGENT_BACKEND_CODEX if "codex" in Path(spec.argv[0]).name.lower() else AGENT_BACKEND_CLAUDE
    if backend not in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX):
        raise AgentConfigError(f"backend {backend!r} is not claude or codex")
    return backend


def _validate_spec(spec: AgentLaunchSpec, now_unix: float) -> None:
    if not spec.argv:
        raise AgentConfigError("argv is empty")
    if spec.deadline_unix <= now_unix:
        raise AgentConfigError("deadline_unix is already in the past")
    for endpoint in spec.mcp_servers:
        if not _SERVER_NAME_RE.fullmatch(endpoint.name):
            raise AgentConfigError(
                f"MCP server name {endpoint.name!r} is not a letter followed by letters, digits, _ or -"
            )
        if not _ENV_NAME_RE.fullmatch(endpoint.token_env):
            raise AgentConfigError(f"token_env {endpoint.token_env!r} is not an environment variable name")
        token = spec.env.get(endpoint.token_env)
        if not token:
            raise AgentConfigError(f"token environment variable {endpoint.token_env!r} is not set in spec.env")
        if "\n" in token or "\r" in token:  # no header carries one, and the log is scrubbed line by line
            raise AgentConfigError(f"token environment variable {endpoint.token_env!r} holds a line break")
    names = [endpoint.name for endpoint in spec.mcp_servers]
    if len(set(names)) != len(names):
        raise AgentConfigError("duplicate MCP server names")


def _write_mcp_json(spec: AgentLaunchSpec) -> Path:
    """Write the ``{"mcpServers": ...}`` file; headers reference the token by variable name only."""
    servers = {
        endpoint.name: {
            "type": "http",
            "url": endpoint.url,
            "headers": {"Authorization": "Bearer ${" + endpoint.token_env + "}"},
        }
        for endpoint in spec.mcp_servers
    }
    text = json.dumps({"mcpServers": servers}, indent=2, sort_keys=True) + "\n"
    spec.workspace.mkdir(parents=True, exist_ok=True)
    return write_text_nofollow(
        spec.workspace, ("runtime", SPECIALIST_MCP_CONFIG_FILENAME), text, mode=PRIVATE_FILE_MODE
    )


def _strip_mcp_block(text: str) -> str:
    """Remove a block this module wrote on an earlier launch, so a relaunch does not duplicate tables."""
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == _MCP_BLOCK_BEGIN:
            skipping = True
        elif line.strip() == _MCP_BLOCK_END:
            skipping = False
        elif not skipping:
            out.append(line)
    return "\n".join(out)


def _wire_codex(spec: AgentLaunchSpec, env: dict[str, str], mcp_json: Path) -> None:
    """Append the ``[mcp_servers.*]`` tables to the task's private Codex home and add the private header variables."""
    lines, additions = _codex_mcp_config(
        str(mcp_json), source=dict(env), child_env=dict(env), protected_env_names=frozenset()
    )
    codex_home = Path(env["CODEX_HOME"])
    config = codex_home / "config.toml"
    existing = _strip_mcp_block(read_text_nofollow(config)) if config.exists() else ""
    for endpoint in spec.mcp_servers:
        if re.search(rf"^\[mcp_servers\.{re.escape(endpoint.name)}[.\]]", existing, re.MULTILINE):
            raise AgentConfigError(f"Codex config already defines MCP server {endpoint.name!r}")
    block = [_MCP_BLOCK_BEGIN, *lines, _MCP_BLOCK_END]
    _write_private_codex_config(
        codex_home=codex_home,
        developer_instructions="",
        mcp_lines=[*existing.splitlines(), "", *block],
    )
    env.update(additions)


def _wire_claude(spec: AgentLaunchSpec, env: dict[str, str], argv: list[str], mcp_json: Path | None) -> None:
    if mcp_json is not None:
        if "--mcp-config" in argv:
            raise AgentConfigError("argv already carries --mcp-config; pass the servers through spec.mcp_servers")
        argv.extend(["--mcp-config", str(mcp_json)])
    for name in [key for key in env if key.startswith(_PLUGIN_ENV_PREFIXES)]:
        del env[name]
    if spec.claude_config_dir == ISOLATED:
        config_dir = spec.workspace / CLAUDE_CONFIG_DIRNAME
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings = config_dir / "settings.json"
        if not settings.exists():
            write_text_nofollow(config_dir, (settings.name,), json.dumps({"enabledPlugins": {}}) + "\n")
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)


def _read_init_message(process_log: Path) -> dict[str, Any] | None:
    """The first ``system``/``init`` row of a Claude ``stream-json`` log, if any."""
    try:
        with process_log.open("r", encoding=UTF8, errors="replace") as handle:
            for line in handle:
                if '"init"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("type") == "system" and row.get("subtype") == "init":
                    return row
    except OSError:
        return None
    return None


def _probe_version(argv0: str, env: Mapping[str, str]) -> str:
    try:
        done = subprocess.run(
            [argv0, "--version"], env=dict(env), capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout.strip() else UNKNOWN


@dataclass
class _Session:
    spec: AgentLaunchSpec
    backend: str
    proc: Any
    env_tokens: tuple[str, ...]
    sampler: ProcessTreeSampler
    started_mono: float
    started_unix: float
    deadline: Deadline
    config_claim: str | None
    concurrent_at_start: int
    oom_kill_at_start: int | None
    reap_mark: str
    prepared: PreparedLaunch
    version: str = UNKNOWN
    task: asyncio.Task[None] | None = None
    sampler_task: asyncio.Task[None] | None = None
    outcome: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    torn_down: bool = False
    finished_mono: float | None = None
    collected: bool = False


def _ended(session: _Session) -> bool:
    """Whether the session's supervision has finished or its CLI has exited."""
    return (session.task is not None and session.task.done()) or session.proc.poll() is not None


def _ended_status(session: _Session, heartbeat: float | None) -> AgentStatus:
    """The status of an ended session, from what the reap loop recorded; ``lost`` when it ended the CLI as silent."""
    outcome = session.outcome
    returncode = outcome.get("exit_code", session.proc.poll())
    error = str(outcome.get("error", ""))
    if outcome.get("stale_heartbeat"):
        return AgentStatus(STATE_LOST, returncode, heartbeat, error=error)
    return AgentStatus("exited", returncode, heartbeat, timed_out=bool(outcome.get("timed_out")), error=error)


def _close_stdin(session: _Session) -> None:
    stdin = session.proc.stdin if session.spec.stdin_open else None
    if stdin is not None and not stdin.closed:
        with contextlib.suppress(OSError):  # a CLI that already ended leaves a broken pipe to flush
            stdin.close()


@dataclass(frozen=True)
class RunnerHost:
    """The clocks, the process launcher and the kernel file trees a runner reads; every field is a test seam."""

    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    popen: Callable[..., Any] = subprocess.Popen
    cgroup_root: Path = Path("/sys/fs/cgroup")
    proc_root: Path = Path("/proc")
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(frozen=True)
class Liveness:
    """When a session counts as alive: CPU work of ``activity_cpu_seconds`` in one sampling period beats its heartbeat,
    and ``lost_after_seconds`` is how long its CLI may be silent. ``poll`` and the reap loop apply
    :func:`~hyperloom.orchestrator.specialists.reap_watch.cli_alive` over that window, the CLI's output or a tool call it
    started after that output; ``poll`` also counts a beat, which the reap loop does not read."""

    activity_cpu_seconds: float = ACTIVITY_CPU_SECONDS
    lost_after_seconds: float = LOST_AFTER_SEC


@dataclass(frozen=True)
class _Spawned:
    """An agent CLI just started: its process, the MCP tokens and the reap mark its environment carries, its launch as
    its tier prepared it, and its CLI version."""

    proc: Any
    tokens: tuple[str, ...]
    reap_mark: str
    prepared: PreparedLaunch
    version: str


def _launch_isolation() -> AgentIsolation:
    """How to isolate the CLI; a launch that must be isolated and cannot be is refused."""
    try:
        return resolve_launch_isolation(None)
    except RunAsError as exc:
        raise AgentConfigError(f"specialist isolation refused: {exc}") from exc


def _refuse_stale_done_paths(spec: AgentLaunchSpec) -> None:
    stale = [str(path) for path in spec.done_paths if path.exists()]
    if stale:
        raise AgentConfigError(
            f"{stale[0]} is left over from an earlier session and would end this one at once; consume it first"
        )


class LocalAgentRunner:
    """Run agent CLIs as local subprocesses of this process (the default implementation)."""

    name = "local"

    def __init__(
        self,
        *,
        session_dir: Path,
        poll_interval_seconds: float = 5.0,
        version_probe: Callable[[str, Mapping[str, str]], str] = _probe_version,
        liveness: Liveness = Liveness(),
        host: RunnerHost = RunnerHost(),
    ) -> None:
        self._session_dir = Path(session_dir)
        self._poll_interval = float(poll_interval_seconds)
        self._lost_after = float(liveness.lost_after_seconds)
        self._monotonic = host.monotonic
        self._wall = host.wall
        self._popen = host.popen
        self._cgroup_root = host.cgroup_root
        self._proc_root = host.proc_root
        self._version_probe = version_probe
        self._dispatcher = SpecialistSubprocessDispatcher(
            SpecialistSubprocessConfig(
                poll_interval_seconds=self._poll_interval, heartbeat_stale_seconds=self._lost_after
            )
        )
        self._sleep = host.sleep
        self._activity_cpu = float(liveness.activity_cpu_seconds)
        self._live: dict[int, _Session] = {}
        self._claims: set[str] = set()
        self._max_concurrent = 0

    # Socket interface
    async def launch(self, spec: AgentLaunchSpec) -> AgentHandle:
        _validate_spec(spec, self._wall())
        backend = _infer_backend(spec)
        _refuse_stale_done_paths(spec)
        clear_wall_budget_extension(spec.task_id)  # an extension left by an earlier run must not move this deadline
        isolation = _launch_isolation()
        env = dict(spec.env)
        argv = list(spec.argv)
        claim = self._prepare(spec, backend, env, argv, isolation)
        try:
            spawned = await self._spawned(spec, argv, env, isolation)
        except BaseException:
            self._release(claim)
            raise
        session = self._session(spec, backend, spawned, claim)
        try:
            self._write_runtime(session, plugins=None)
            if spec.heartbeat_sink is not None:
                spec.heartbeat_sink(lambda: self._beat(session))
        except BaseException:
            await self._abandon(session)
            raise
        self._supervise_from_now(session)
        return AgentHandle(
            task_id=spec.task_id, pid=spawned.proc.pid, started_unix=session.started_unix, session=session
        )

    async def _spawned(
        self, spec: AgentLaunchSpec, argv: list[str], env: dict[str, str], isolation: AgentIsolation
    ) -> _Spawned:
        spec.workspace.mkdir(parents=True, exist_ok=True)
        tokens = tuple(env[e.token_env] for e in spec.mcp_servers)
        reap_mark = stamp_reap_mark(env)  # after the caller's env, so a mark it carried cannot join two launches
        version = await asyncio.to_thread(
            self._version_probe, spec.argv[0], {k: v for k, v in env.items() if v not in tokens}
        )
        private = (spec.workspace / PROMPT_FILE, spec.workspace / PROCESS_LOG_NAME)
        # The child works in ``cwd``, so that is the tree it must be able to write.
        place = LaunchPlace(spec.workspace, spec.cwd, private, spec.readable, spec.gpus > 0)
        try:
            prepared = await asyncio.to_thread(prepare_launch, isolation, place, argv, env)
        except RunAsError as exc:
            raise AgentConfigError(f"specialist isolation refused: {exc}") from exc
        try:
            proc = prepared.started(self._spawn(spec, prepared.local(argv, env)))
        except BaseException:
            prepared.release()
            raise
        return _Spawned(proc, tokens, reap_mark, prepared, version)

    def _session(self, spec: AgentLaunchSpec, backend: str, spawned: _Spawned, claim: str | None) -> _Session:
        """The session of the CLI just ``spawned``, not yet supervised."""
        proc = spawned.proc
        started_mono = self._monotonic()
        started_unix = self._wall()
        return _Session(
            spec=spec,
            backend=backend,
            proc=proc,
            env_tokens=spawned.tokens,
            sampler=ProcessTreeSampler(proc.pid, proc_root=self._proc_root),
            started_mono=started_mono,
            started_unix=started_unix,
            deadline=Deadline.after(spec.deadline_unix - started_unix, now=started_mono),
            config_claim=claim,
            concurrent_at_start=0,
            oom_kill_at_start=read_container_cgroup(self._cgroup_root).oom_kill,
            reap_mark=spawned.reap_mark,
            prepared=spawned.prepared,
            version=spawned.version,
        )

    def _supervise_from_now(self, session: _Session) -> None:
        """Count ``session`` live, and sample and supervise it from now on."""
        task_id = session.spec.task_id
        self._live[session.proc.pid] = session
        session.concurrent_at_start = len(self._live)
        self._max_concurrent = max(self._max_concurrent, len(self._live))
        session.sampler_task = asyncio.create_task(self._sample_loop(session), name=f"agent-sampler-{task_id}")
        session.task = asyncio.create_task(self._supervise(session), name=f"agent-{task_id}")
        session.task.add_done_callback(lambda task: self._on_supervision_done(session, task))

    async def _abandon(self, session: _Session) -> None:
        """End a session whose launch failed after its CLI started, before it was supervised: its tree and every
        process carrying its reap mark go as at the end of supervision, a beat its sink kept does nothing, and its
        config dir claim is released as a failed spawn's is."""
        session.finished_mono = self._monotonic()
        session.torn_down = True
        try:
            await asyncio.to_thread(self._tear_down, session)
        finally:
            _close_stdin(session)
            self._release(session.config_claim)

    async def poll(self, handle: AgentHandle) -> AgentStatus:
        session: _Session = handle.session
        heartbeat = newest_activity((session.spec.heartbeat_path, session.spec.workspace / PROCESS_LOG_NAME))
        if not _ended(session):
            silent = await self._silent(session, heartbeat)
            # The look at the CLI runs on a thread, while the reap loop may end the session; a deadline kill takes
            # the tool down first, so a look taken during it sees silence. How the session ended outranks that look.
            if not _ended(session):
                if silent:
                    error = f"no heartbeat for more than {self._lost_after:.0f}s"
                    return AgentStatus(STATE_LOST, None, heartbeat, error=error)
                return AgentStatus("running", None, heartbeat)
        return _ended_status(session, heartbeat)

    async def _silent(self, session: _Session, heartbeat: float | None) -> bool:
        """Whether nothing shows the session alive. A beat, the CLI's process log or the session's start within the
        window does; past it, so does what the reap loop counts: the CLI's other output, or a tool it waits on."""
        now = self._wall()
        if now - (heartbeat or session.started_unix) <= self._lost_after:
            return False
        alive = await asyncio.to_thread(
            cli_alive, session.spec.workspace, agent_pid(session.proc), window_sec=self._lost_after, now_unix=now
        )
        return not alive

    async def cancel(self, handle: AgentHandle) -> None:
        session: _Session = handle.session
        session.cancelled = True
        if session.proc.poll() is None:
            await asyncio.to_thread(self._dispatcher._kill, session.proc)

    async def send(self, handle: AgentHandle, data: bytes) -> None:
        """Write ``data`` to the stdin of a session launched with ``stdin_open``; raises :class:`AgentEnded` once its
        CLI has exited."""
        session: _Session = handle.session
        stdin = session.proc.stdin if session.spec.stdin_open else None
        if stdin is None:
            raise AgentConfigError(f"task {session.spec.task_id}: stdin is not open")
        if stdin.closed or session.proc.poll() is not None:
            raise AgentEnded(f"task {session.spec.task_id}: its CLI has ended")

        def write() -> None:
            stdin.write(data)
            stdin.flush()

        try:
            await asyncio.to_thread(write)
        except (BrokenPipeError, ValueError) as exc:  # ValueError: the pipe was closed while this write waited
            raise AgentEnded(f"task {session.spec.task_id}: its CLI ended while a message was sent") from exc

    async def close_stdin(self, handle: AgentHandle) -> None:
        """Close a ``stdin_open`` session's stdin: its CLI ends once it has read what was sent."""
        _close_stdin(handle.session)

    async def collect(self, handle: AgentHandle) -> None:
        session: _Session = handle.session
        if session.collected:
            return
        if session.proc.poll() is None:
            raise RuntimeError(f"task {session.spec.task_id}: still running; cancel it before collect")
        if session.task is not None:
            await asyncio.wait_for(asyncio.shield(session.task), timeout=self._poll_interval * 4 + 60.0)
        process_log = session.spec.workspace / PROCESS_LOG_NAME
        await asyncio.to_thread(self._scrub_log, process_log, session.env_tokens)
        init = _read_init_message(process_log)
        self._write_runtime(session, plugins=init)
        self._append_record(self._specialist_record(session))
        self._append_record(self._container_record())
        session.collected = True

    # Internals
    def _prepare(
        self, spec: AgentLaunchSpec, backend: str, env: dict[str, str], argv: list[str], isolation: AgentIsolation
    ) -> str | None:
        claim: str | None = None
        if backend == AGENT_BACKEND_CLAUDE and spec.claude_config_dir == ISOLATED:
            claim = str((spec.workspace / CLAUDE_CONFIG_DIRNAME).resolve())
            if claim in self._claims:
                raise AgentConfigError(f"Claude Code config dir {claim} is already used by a live session")
            self._claims.add(claim)
        try:
            mcp_json = _write_mcp_json(spec) if spec.mcp_servers else None
            if backend == AGENT_BACKEND_CODEX:
                env["CODEX_HOME"] = str(spec.workspace / ".codex")  # always private to this task, as in the dispatcher
                if mcp_json is not None:
                    _wire_codex(spec, env, mcp_json)
                if codex_sandbox_off(isolation):
                    argv[:] = without_codex_sandbox(argv)
            else:
                _wire_claude(spec, env, argv, mcp_json)
                _require_login(isolation, env)
            stage_mcp_config(isolation, spec.workspace, argv)
            if spec.gpus == 0:
                for name in GPU_MASK_ENV_NAMES:
                    env.setdefault(name, "")
        except BaseException:
            self._release(claim)
            raise
        return claim

    def _release(self, claim: str | None) -> None:
        if claim is not None:
            self._claims.discard(claim)

    def _spawn(self, spec: AgentLaunchSpec, spawn: Spawn) -> Any:
        prompt = _read_prompt(spec.workspace / PROMPT_FILE)
        if spec.stdin_open:
            stdin: Any = subprocess.PIPE
        elif prompt is None:
            stdin = subprocess.DEVNULL
        else:
            stdin = tempfile.TemporaryFile()
            stdin.write(prompt)
            stdin.seek(0)
        try:
            # Created fresh, never through a name a previous (possibly demoted) child left in the workspace.
            with create_nofollow(spec.workspace, (PROCESS_LOG_NAME,), mode=PRIVATE_FILE_MODE) as out:
                proc = self._popen(
                    spawn.argv,
                    stdin=stdin,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=spawn.env,
                    cwd=str(spec.cwd),
                    start_new_session=True,
                    **spawn.kwargs,
                )
        finally:
            if hasattr(stdin, "close"):
                stdin.close()
        if spec.stdin_open and prompt is not None:
            proc.stdin.write(prompt)
            proc.stdin.flush()
        return proc

    async def _supervise(self, session: _Session) -> None:
        spec = session.spec
        try:
            session.outcome = await self._dispatcher._reap_loop(
                session.proc,
                ReapWatch(workspace=spec.workspace, done_files=tuple(spec.done_paths), task_id=spec.task_id),
                deadline=session.deadline,
                started=session.started_mono,
            )
            self._unset_sandbox(session)
        except ScopeCancelled as exc:  # the caller's cancel scope fired: a stop request, not a failure
            session.cancelled = True
            session.outcome = {"exit_code": session.proc.poll(), "error": f"cancelled: {exc}"}
        except ReapKillUnconfirmed as exc:  # the loop ended the CLI as timed out or silent; that stays the outcome
            log.exception("agent %s: reap loop failed", spec.task_id)
            error = f"{exc.outcome['error']}; reap loop failed: {exc!r}"
            session.outcome = {**exc.outcome, "exit_code": session.proc.poll(), "error": error}
        except Exception as exc:
            log.exception("agent %s: reap loop failed", spec.task_id)
            session.outcome = {"exit_code": session.proc.poll(), "error": f"reap loop failed: {exc!r}"}
        finally:
            # Supervision is over: stop sampling, then end what is left of the session off the event loop.
            session.torn_down = True
            if session.sampler_task is not None:
                session.sampler_task.cancel()
            await asyncio.to_thread(self._tear_down, session)

    @staticmethod
    def _unset_sandbox(session: _Session) -> None:
        """Record a sandbox bubblewrap could not set up as that, not as an exit of the CLI, which never ran."""
        proc = session.proc
        unset = (
            proc.setup_failure(session.spec.workspace / PROCESS_LOG_NAME) if isinstance(proc, SandboxedProcess) else ""
        )
        if unset:
            session.outcome = {
                **session.outcome,
                "exit_code": None,
                "error": f"the CLI's sandbox could not be set up: {unset}",
            }

    def _on_supervision_done(self, session: _Session, _task: asyncio.Task[None]) -> None:
        """Runs however the supervisor ended, and tears the session down when it was cancelled before its first step."""
        if not session.torn_down:
            self._tear_down(session)
        _close_stdin(session)
        session.finished_mono = self._monotonic()
        if session.sampler_task is not None:
            session.sampler_task.cancel()
        session.sampler.sample()
        self._release(session.config_claim)
        self._live.pop(session.proc.pid, None)

    def _tear_down(self, session: _Session) -> None:
        """End what is left of the session: the CLI's tree if its root still runs, then every process that carries the
        session's reap mark, which a CLI that exited on its own leaves running and a tree kill does not reach once it
        has detached."""
        self._ensure_dead(session)
        try:
            self._dispatcher._reap_marked(session.reap_mark, session.prepared)
        finally:
            session.prepared.release()

    def _ensure_dead(self, session: _Session) -> None:
        """Kill the CLI's tree when the reap loop ended with its root still running; what it detached is the sweep's."""
        if session.proc.poll() is not None:
            return
        try:
            self._dispatcher._kill(session.proc)
        except Exception as exc:  # recorded for poll(); the caller decides what to do
            log.exception("agent %s: tree cleanup unconfirmed: %r", session.spec.task_id, exc)
            session.outcome["error"] = f"{session.outcome.get('error', '')}; tree cleanup unconfirmed: {exc!r}".strip(
                "; "
            )
        session.outcome["exit_code"] = session.proc.poll()

    async def _sample_loop(self, session: _Session) -> None:
        """Sample the process tree; CPU work in the tree is evidence the session is still working."""
        previous = 0.0
        while True:
            session.sampler.sample()
            gained = session.sampler.cpu_seconds - previous
            previous = session.sampler.cpu_seconds
            if gained >= self._activity_cpu:
                self._beat(session)
            await self._sleep(self._poll_interval)

    def _beat(self, session: _Session) -> None:
        """Refresh the mtime of the heartbeat file, which ``poll`` reads; the reap loop reads only the CLI's own output
        (:func:`~hyperloom.orchestrator.specialists.reap_watch.specialist_activity_paths`)."""
        if session.finished_mono is not None:
            return
        now = self._wall()
        path = session.spec.heartbeat_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # O_NONBLOCK: a FIFO the demoted account planted fails the open instead of hanging the beat.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
            try:
                os.utime(fd, (now, now))
            finally:
                os.close(fd)
        except OSError:
            log.debug("agent %s: cannot refresh the heartbeat file", session.spec.task_id, exc_info=True)

    @staticmethod
    def _scrub_log(process_log: Path, secrets: tuple[str, ...]) -> None:
        """Take the session's tokens, and every value :func:`redact_secret_values` recognises, out of the log in one
        rewrite, as :func:`~hyperloom.common.env_safety.redact_file_in_place` does with the patterns alone: the copy
        is staged in a file created fresh at ``PRIVATE_FILE_MODE``, and a link at the log is refused. A token holds no
        line break (the launch refuses one), so a line-by-line replacement misses none. A log that cannot be
        rewritten is left as it is, and said so.
        """
        values = [value for value in secrets if value]

        def scrub(line: str) -> str:
            for value in values:
                line = line.replace(value, "[REDACTED]")
            return redact_secret_values(line)

        try:
            rewrite_lines_nofollow(process_log, scrub, mode=PRIVATE_FILE_MODE)
        except FileNotFoundError:
            return
        except OSError:
            log.warning("agent: %s was not scrubbed", process_log, exc_info=True)

    def _write_runtime(self, session: _Session, *, plugins: dict[str, Any] | None) -> None:
        spec = session.spec
        payload: dict[str, Any] = {
            "backend": session.backend,
            "executable": Path(spec.argv[0]).name,
            "version": session.version,
            "config_dir_mode": spec.claude_config_dir if session.backend == AGENT_BACKEND_CLAUDE else None,
            "plugins": UNKNOWN,
            "mcp_servers": [endpoint.name for endpoint in spec.mcp_servers],
        }
        if plugins is not None and isinstance(plugins.get("plugins"), list):
            payload["plugins"] = plugins["plugins"]
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        write_text_nofollow(spec.workspace, (AGENT_RUNTIME_FILENAME,), text, mode=PRIVATE_FILE_MODE)

    def _ended_by(self, session: _Session) -> str:
        outcome = session.outcome
        code = outcome.get("exit_code")
        now_oom = read_container_cgroup(self._cgroup_root).oom_kill
        oom_delta = (
            now_oom - session.oom_kill_at_start if now_oom is not None and session.oom_kill_at_start is not None else 0
        )
        if outcome.get("timed_out"):
            return "timeout"
        if outcome.get("stale_heartbeat"):
            return STATE_LOST
        if session.cancelled:
            return "cancelled"
        if oom_delta > 0 and code in (-9, 137):
            return "oom_killed"
        if code == 0 or any(path.exists() for path in session.spec.done_paths):
            return "normal"
        return "failed"

    def _specialist_record(self, session: _Session) -> dict[str, Any]:
        spec = session.spec
        end = session.finished_mono if session.finished_mono is not None else self._monotonic()
        return {
            "kind": SPECIALIST_RECORD_KIND,
            "task_id": spec.task_id,
            "backend": session.backend,
            "location": "local",
            "gpus": spec.gpus,
            "wall_seconds": round(end - session.started_mono, 3),
            "peak_rss_bytes": session.sampler.peak_rss_bytes,
            "cpu_seconds": round(session.sampler.cpu_seconds, 3),
            "worktree_disk_bytes": dir_usage_bytes(spec.workspace / "worktree"),
            "selfcheck_venv_disk_bytes": dir_usage_bytes(spec.workspace / "selfcheck_venv"),
            "concurrent_specialists_at_start": session.concurrent_at_start,
            "ended_by": self._ended_by(session),
            "ts_unix": self._wall(),
        }

    def _append_record(self, record: Mapping[str, Any]) -> None:
        """Append one record to the session's resource records, refusing a link planted anywhere below the session."""
        line = json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        append_text_nofollow(self._session_dir, RESOURCES_RELATIVE_PATH.parts, line)

    def _container_record(self) -> dict[str, Any]:
        snapshot: ContainerSnapshot = read_container_cgroup(self._cgroup_root)
        return {
            "kind": CONTAINER_RECORD_KIND,
            **snapshot.as_fields(),
            "max_concurrent_specialists": self._max_concurrent,
            "ts_unix": self._wall(),
        }


def _require_login(isolation: AgentIsolation, env: Mapping[str, str]) -> None:
    """Refuse a claude CLI that could not log in where it runs (:func:`~.launch_env.require_claude_login`)."""
    try:
        require_claude_login(isolation, env)
    except SpecialistAgentUnavailableError as exc:
        raise AgentConfigError(str(exc)) from exc


def _read_prompt(path: Path) -> bytes | None:
    """The bytes of ``prompt.md``, ``None`` when there is none; a symlink (or FIFO, or device) there is refused, never
    followed: a demoted child of an earlier session could have left one pointing at a file only the runner may read."""
    try:
        return read_bytes_nofollow(path)
    except FileNotFoundError:
        return None
