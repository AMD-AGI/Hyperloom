# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AI work socket: launch, poll, cancel and collect agent CLI sessions.

The local implementation spawns the argv the existing command builders produced,
and hands the existing reap loop an absolute deadline, so a long-lived session is
not cut by the 4 hour cap that only applies to an unbounded run. Heartbeat rules,
environment allowlist and tree teardown are the existing ones. How a session that
ended or was lost is restarted and resumed from the branch log is the specialist
runtime's decision (multi-branch design 4.4); this module only launches, polls,
cancels and collects, and, for a session launched with its stdin open, sends it
further messages.

Tokens reach the CLI only through environment variables: the MCP configuration
written to disk names the variable, never its value.

``HYPERLOOM_SPECIALIST_USER`` starts the CLI as that account, as the legacy
launcher does: the workspace and ``cwd`` are shared with it, and every file the
runner writes there is created without following a link the account left.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import time
from concurrent.futures import CancelledError as ScopeCancelled
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol

from hyperloom.common.deadline import Deadline
from hyperloom.common.env_safety import RunAs, RunAsError, redact_file_in_place
from hyperloom.common.fs_utils import create_nofollow, read_text_nofollow, write_text_nofollow
from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES

from .launch_env import prepare_demote, resolve_launch_user, stage_mcp_config
from .mcp_config import SPECIALIST_MCP_CONFIG_FILENAME
from .resources import (
    CONTAINER_RECORD_KIND,
    RESOURCES_RELATIVE_PATH,
    SPECIALIST_RECORD_KIND,
    ContainerSnapshot,
    ProcessTreeSampler,
    ResourceLog,
    dir_usage_bytes,
    read_container_cgroup,
)
from .subprocess_ import (
    ReapWatch,
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _codex_mcp_config,
    _write_private_codex_config,
    clear_wall_budget_extension,
)
from .subprocess_ import _build_specialist_env as build_specialist_env

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MCP_TOKEN_ENV",
    "LONG_RUNNING_SESSION_SEC",
    "LOST_AFTER_SEC",
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

#: No heartbeat or process-log activity for this long means the session is lost.
LOST_AFTER_SEC: float = 300.0

#: Minimum session length for a long-lived (one per branch) specialist.
LONG_RUNNING_SESSION_SEC: float = 6 * 60 * 60.0

#: Variable name the tool layer uses for a specialist's MCP token (``grant_access()``).
DEFAULT_MCP_TOKEN_ENV = "HYPERLOOM_SPECIALIST_MCP_TOKEN"

#: Tree CPU seconds gained in one sampling period that count as "still working".
ACTIVITY_CPU_SECONDS = 0.1

CLAUDE_CONFIG_DIRNAME = ".claude-config"
PROCESS_LOG = "process.log"
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
#: that session's token is in flight, so a session blocked in a long tool call is not judged lost.
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
    """Absolute wall-clock end of the session; a long-lived specialist asks for at least 6 hours."""
    done_paths: tuple[Path, ...]
    heartbeat_path: Path
    mcp_servers: tuple[McpEndpoint, ...] = ()
    claude_config_dir: Literal["isolated", "inherit"] = ISOLATED
    backend: str = ""
    """``claude`` or ``codex``; empty infers it from ``argv[0]``."""
    heartbeat_sink: HeartbeatSink | None = None
    """Called once at launch with the session's ``beat`` function."""
    stdin_open: bool = False
    """Keep the CLI's stdin open after prompt.md, so :meth:`LocalAgentRunner.send` reaches the same session later; the
    CLI ends once :meth:`LocalAgentRunner.close_stdin` closes it."""


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
    name: str

    def launch(self, spec: AgentLaunchSpec) -> Awaitable[AgentHandle]: ...

    def poll(self, handle: AgentHandle) -> Awaitable[AgentStatus]: ...

    def cancel(self, handle: AgentHandle) -> Awaitable[None]: ...

    def collect(self, handle: AgentHandle) -> Awaitable[None]: ...


class InteractiveAgentRunner(AgentRunner, Protocol):
    """A runner that keeps a session's stdin open when its spec asks, so one CLI takes message after message."""

    def send(self, handle: AgentHandle, data: bytes) -> Awaitable[None]: ...

    def close_stdin(self, handle: AgentHandle) -> Awaitable[None]: ...


def _infer_backend(spec: AgentLaunchSpec) -> str:
    backend = (spec.backend or "").strip().lower()
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
        if not spec.env.get(endpoint.token_env):
            raise AgentConfigError(f"token environment variable {endpoint.token_env!r} is not set in spec.env")
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
    version: str = UNKNOWN
    task: asyncio.Task[None] | None = None
    sampler_task: asyncio.Task[None] | None = None
    outcome: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    finished_mono: float | None = None
    collected: bool = False


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
    and ``lost_after_seconds`` with no heartbeat and no process-log growth loses it."""

    activity_cpu_seconds: float = ACTIVITY_CPU_SECONDS
    lost_after_seconds: float = LOST_AFTER_SEC


@dataclass(frozen=True)
class _Spawned:
    """An agent CLI just started: its process, the MCP tokens its environment carries, and its CLI version."""

    proc: Any
    tokens: tuple[str, ...]
    version: str


def _launch_user() -> RunAs | None:
    """The account ``HYPERLOOM_SPECIALIST_USER`` names, or ``None``; one that cannot be used refuses the launch."""
    try:
        return resolve_launch_user(None)
    except RunAsError as exc:
        raise AgentConfigError(f"specialist user refused: {exc}") from exc


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
        self._log = ResourceLog(Path(session_dir) / RESOURCES_RELATIVE_PATH)
        self._poll_interval = float(poll_interval_seconds)
        self._lost_after = float(liveness.lost_after_seconds)
        self._monotonic = host.monotonic
        self._wall = host.wall
        self._popen = host.popen
        self._cgroup_root = Path(host.cgroup_root)
        self._proc_root = Path(host.proc_root)
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
        run_as = _launch_user()
        env = dict(spec.env)
        argv = list(spec.argv)
        claim = self._prepare(spec, backend, env, argv, run_as)
        try:
            spawned = await self._spawned(spec, argv, env, run_as)
        except BaseException:
            self._release(claim)
            raise
        session = self._supervised(spec, backend, spawned, claim)
        if spec.heartbeat_sink is not None:
            try:
                spec.heartbeat_sink(lambda: self._beat(session))
            except Exception:
                log.exception(
                    "agent %s: heartbeat sink failed; MCP calls will not keep the session alive", spec.task_id
                )
        return AgentHandle(
            task_id=spec.task_id, pid=spawned.proc.pid, started_unix=session.started_unix, session=session
        )

    async def _spawned(
        self, spec: AgentLaunchSpec, argv: list[str], env: dict[str, str], run_as: RunAs | None
    ) -> _Spawned:
        spec.workspace.mkdir(parents=True, exist_ok=True)
        tokens = tuple(env[e.token_env] for e in spec.mcp_servers)
        version = await asyncio.to_thread(
            self._version_probe, spec.argv[0], {k: v for k, v in env.items() if v not in tokens}
        )
        private = (spec.workspace / PROMPT_FILE, spec.workspace / PROCESS_LOG)
        # The child works in ``cwd``, so that is the tree it must be able to write.
        demote = await asyncio.to_thread(prepare_demote, run_as, spec.workspace, spec.cwd, env, private)
        command = argv if run_as is None else run_as.command(argv)
        return _Spawned(self._spawn(spec, command, env, demote), tokens, version)

    def _supervised(self, spec: AgentLaunchSpec, backend: str, spawned: _Spawned, claim: str | None) -> _Session:
        """The live session of the CLI just ``spawned``, sampled and supervised from now on."""
        proc = spawned.proc
        started_mono = self._monotonic()
        started_unix = self._wall()
        session = _Session(
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
            version=spawned.version,
        )
        self._live[proc.pid] = session
        session.concurrent_at_start = len(self._live)
        self._max_concurrent = max(self._max_concurrent, len(self._live))
        self._write_runtime(session, plugins=None)
        session.sampler_task = asyncio.create_task(self._sample_loop(session), name=f"agent-sampler-{spec.task_id}")
        session.task = asyncio.create_task(self._supervise(session), name=f"agent-{spec.task_id}")
        session.task.add_done_callback(lambda task: self._on_supervision_done(session, task))
        return session

    async def poll(self, handle: AgentHandle) -> AgentStatus:
        session: _Session = handle.session
        heartbeat = self._last_activity(session)
        outcome = session.outcome
        task_done = session.task is not None and session.task.done()
        if task_done or session.proc.poll() is not None:
            returncode = outcome.get("exit_code", session.proc.poll())
            if outcome.get("stale_heartbeat"):
                return AgentStatus(STATE_LOST, returncode, heartbeat, error=str(outcome.get("error", "")))
            return AgentStatus(
                "exited",
                returncode,
                heartbeat,
                timed_out=bool(outcome.get("timed_out")),
                error=str(outcome.get("error", "")),
            )
        if self._wall() - (heartbeat or session.started_unix) > self._lost_after:
            return AgentStatus(STATE_LOST, None, heartbeat, error=f"no heartbeat for more than {self._lost_after:.0f}s")
        return AgentStatus("running", None, heartbeat)

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
        process_log = session.spec.workspace / PROCESS_LOG
        await asyncio.to_thread(self._scrub_log, process_log, session.env_tokens)
        init = _read_init_message(process_log)
        self._write_runtime(session, plugins=init)
        self._log.append(self._specialist_record(session))
        self._log.append(self._container_record())
        session.collected = True

    async def finish_campaign(self) -> None:
        """Write the closing container record, with the highest specialist concurrency seen."""
        self._log.append(self._container_record())

    # Internals
    def _prepare(
        self, spec: AgentLaunchSpec, backend: str, env: dict[str, str], argv: list[str], run_as: RunAs | None
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
            else:
                _wire_claude(spec, env, argv, mcp_json)
            stage_mcp_config(run_as, spec.workspace, argv)
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

    def _spawn(self, spec: AgentLaunchSpec, argv: list[str], env: dict[str, str], demote: dict[str, Any]) -> Any:
        prompt = spec.workspace / PROMPT_FILE
        if spec.stdin_open:
            stdin: Any = subprocess.PIPE
        else:
            stdin = prompt.open("rb") if prompt.exists() else subprocess.DEVNULL
        try:
            # Created fresh, never through a name a previous (possibly demoted) child left in the workspace.
            with create_nofollow(spec.workspace, (PROCESS_LOG,), mode=PRIVATE_FILE_MODE) as out:
                proc = self._popen(
                    argv,
                    stdin=stdin,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(spec.cwd),
                    start_new_session=True,
                    **demote,
                )
        finally:
            if hasattr(stdin, "close"):
                stdin.close()
        if spec.stdin_open and prompt.exists():
            proc.stdin.write(prompt.read_bytes())
            proc.stdin.flush()
        return proc

    async def _supervise(self, session: _Session) -> None:
        spec = session.spec
        try:
            session.outcome = await self._dispatcher._reap_loop(
                session.proc,
                ReapWatch(
                    workspace=spec.workspace,
                    done_files=tuple(spec.done_paths),
                    heartbeat_file=spec.heartbeat_path,
                    task_id=spec.task_id,
                ),
                deadline=session.deadline,
                started=session.started_mono,
            )
        except ScopeCancelled as exc:  # the caller's cancel scope fired: a stop request, not a failure
            session.cancelled = True
            session.outcome = {"exit_code": session.proc.poll(), "error": f"cancelled: {exc}"}
        except Exception as exc:
            log.exception("agent %s: reap loop failed", spec.task_id)
            session.outcome = {"exit_code": session.proc.poll(), "error": f"reap loop failed: {exc!r}"}

    def _on_supervision_done(self, session: _Session, _task: asyncio.Task[None]) -> None:
        """Runs however the supervisor ended, even if it was cancelled before its first step."""
        self._ensure_dead(session)
        _close_stdin(session)
        session.finished_mono = self._monotonic()
        if session.sampler_task is not None:
            session.sampler_task.cancel()
        session.sampler.sample()
        self._release(session.config_claim)
        self._live.pop(session.proc.pid, None)

    def _ensure_dead(self, session: _Session) -> None:
        """However the reap loop ended, no process of the session may outlive its supervision."""
        if session.proc.poll() is not None:
            return
        try:
            self._dispatcher._kill(session.proc)
        except Exception as exc:  # noqa: BLE001 -- the failure is recorded for poll(), the caller decides what to do
            log.error("agent %s: tree cleanup unconfirmed: %r", session.spec.task_id, exc)
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
        """Refresh the mtime of the heartbeat file, which is what the reap loop and ``poll`` read."""
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

    def _last_activity(self, session: _Session) -> float | None:
        newest: float | None = None
        for path in (session.spec.heartbeat_path, session.spec.workspace / PROCESS_LOG):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        return newest

    @staticmethod
    def _scrub_log(process_log: Path, secrets: tuple[str, ...]) -> None:
        """Take the session's tokens out of the log, then run the existing pattern redaction.

        The literal tokens go first, through a private temporary file: the existing redaction writes its own
        temporary file with default permissions before it restricts them, and a token that no pattern recognises
        must not be in it.
        """
        if not process_log.exists():
            return
        values = [value for value in secrets if value]
        if values:
            text = process_log.read_text(encoding=UTF8, errors="replace")
            scrubbed = text
            for value in values:
                scrubbed = scrubbed.replace(value, "[REDACTED]")
            if scrubbed != text:
                temporary = process_log.with_name(process_log.name + ".scrub")
                with create_nofollow(temporary.parent, (temporary.name,), mode=PRIVATE_FILE_MODE) as handle:
                    handle.write(scrubbed)
                os.replace(temporary, process_log)
        redact_file_in_place(process_log, mode=PRIVATE_FILE_MODE)

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

    def _container_record(self) -> dict[str, Any]:
        snapshot: ContainerSnapshot = read_container_cgroup(self._cgroup_root)
        return {
            "kind": CONTAINER_RECORD_KIND,
            **snapshot.as_fields(),
            "max_concurrent_specialists": self._max_concurrent,
            "ts_unix": self._wall(),
        }
