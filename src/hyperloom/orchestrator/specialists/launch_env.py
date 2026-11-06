# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist launch decides before it spawns: the child's environment, GPU visibility and isolation.

Every launch is isolated as :func:`resolve_launch_isolation` chooses (:mod:`hyperloom.common.agent_isolation`): as
another account (``HYPERLOOM_SPECIALIST_USER``), in a bubblewrap sandbox, or unconfined as before where neither works
and the run does not require isolation. :func:`prepare_launch` readies the chosen tier and returns the
:class:`PreparedLaunch` that gives the spawn's arguments, finds the launch's marked processes, and is released once
the launch has ended.

In a sandbox (:func:`specialist_confinement`) the specialist writes its workspace and worktree, where its home, CLI
state and compiler caches are pointed, and its own ``/tmp`` and ``/dev/shm``; it reads its toolchain, its worktree's
git metadata, its model and references; the credential locations inside what it reads are masked; a GPU task gets
the GPU devices. The claude CLI cannot read its own login there, so it needs one in its environment
(:func:`require_claude_login`); codex runs with its own sandbox off (:func:`codex_sandbox_off`), since bubblewrap
already confines it and codex's own, a nested bubblewrap, could not map a user namespace inside.

bubblewrap ends its sandbox at once on a SIGTERM of its own, without passing it on, and exits before the agent ran
when it cannot set the sandbox up. So the process a sandboxed launch gives (:class:`SandboxedProcess`) asks the agent
to stop by TERMing the sandbox's marked processes, and tells a setup failure from the agent's exit by what bubblewrap
wrote on ``--json-status-fd``. A Ray GPU actor's launch is built from this host's probe and paths, so under bubblewrap
it is for a single-node Ray cluster only; its status goes to a file the actor's shell opens.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from hyperloom_common.bwrap import NESTED_SAFE, BwrapPolicy, BwrapPolicyError, BwrapStatus, Scratch
from hyperloom_common.fs_utils import read_bytes_nofollow, read_text_nofollow, write_text_nofollow
from hyperloom.common.agent_isolation import (
    AccountIsolation,
    AgentIsolation,
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
    choose_agent_isolation,
)
from hyperloom.common.confinement import Confinement, credential_paths, toolchain_paths
from hyperloom.common.env_safety import RunAs
from hyperloom.common.proctree import (
    REAP_MARK_ENV,
    TERM_GRACE_SEC,
    descendants,
    marked_pids,
    proc_identity,
    process_uids,
    signal_processes,
)
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots
from hyperloom_common.llm.llm_attribution import inject_env as inject_attribution_env

from .agent_cli_config import SpecialistAgentUnavailableError, _build_specialist_env, load_mcp_servers

log = logging.getLogger(__name__)

_GPU_IDS_ENV = "INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"
_AITER_JIT_ENV = "INFERENCE_OPTIMIZER_AITER_JIT_DIR"
_MCP_FLAG = "--mcp-config"
_CACHE_DIR = ".cache"
_AITER_JIT_DIR = "aiter_jit"

#: The device nodes a GPU specialist's sandbox binds, where the host has them.
GPU_DEVICES = (Path("/dev/kfd"), Path("/dev/dri"))
#: Where a sandboxed specialist's home is, in its workspace, unless its ``HOME`` is in a tree it writes already.
SANDBOX_HOME = ".home"
#: Variables naming a directory a sandboxed CLI or compiler writes, each with where it goes in the workspace when it
#: names one outside the trees the specialist writes: the frameworks' installed trees are read-only in the sandbox,
#: and a cache shared with the app would let an agent decide what a later measurement runs. Unset, each tool falls
#: back to a place under the home or ``/tmp``, which the sandbox writes.
SANDBOX_DIRECTORIES = {
    "CLAUDE_CONFIG_DIR": (".claude-config",),
    "CODEX_HOME": (".codex",),
    "XDG_CACHE_HOME": (_CACHE_DIR,),
    "TRITON_CACHE_DIR": (_CACHE_DIR, "triton"),
    "TORCHINDUCTOR_CACHE_DIR": (_CACHE_DIR, "torchinductor"),
    "AITER_JIT_DIR": (_CACHE_DIR, _AITER_JIT_DIR),
    _AITER_JIT_ENV: (_CACHE_DIR, _AITER_JIT_DIR),
}
#: The temporary-directory variables, which name the sandbox's own ``/tmp``.
TEMP_VARIABLES = ("TMPDIR", "TMP", "TEMP")
#: The app's own trees, masked wherever a sandbox's binds would show them.
APP_ONLY_VARIABLES = ("HYPERLOOM_STATE_DIR", "HYPERLOOM_BUILD_STORE", "HYPERLOOM_BUILD_TRUST")
#: Any one of these in its environment logs the claude CLI in.
CLAUDE_TOKEN_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
_BEDROCK = "CLAUDE_CODE_USE_BEDROCK"
_BEDROCK_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
#: The files a Bedrock-backed claude CLI may take its AWS credentials from; bound read-only, never masked.
BEDROCK_FILE_VARIABLES = ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE")
_TRUE = frozenset({"1", "true", "yes", "on"})
CLAUDE_LOGIN_REFUSAL = (
    "the claude CLI cannot read its own login inside the agent's bubblewrap sandbox: give it ANTHROPIC_API_KEY, "
    "ANTHROPIC_AUTH_TOKEN or CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token` prints one), or CLAUDE_CODE_USE_BEDROCK=1 "
    "with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or with AWS_SHARED_CREDENTIALS_FILE or AWS_CONFIG_FILE naming "
    "a file, which the sandbox then reads"
)
CODEX_SANDBOX_OFF = (
    "codex runs with its own sandbox off in the agent's bubblewrap sandbox: bubblewrap confines it, and codex's own "
    "sandbox, a nested bubblewrap, cannot map a user namespace there"
)
_CODEX_SANDBOX_FLAGS = ("--sandbox", "-s")
_CODEX_SANDBOX_SWITCHES = ("--full-auto",)
CODEX_BYPASS_FLAG = "--dangerously-bypass-approvals-and-sandbox"
#: The descriptor a Ray actor's shell opens the status file on, and how: bubblewrap's own exit codes start at 1.
_ACTOR_STATUS_FD = 3
_OPEN_STATUS = 'exec 3>>"$0" && exec "$@"'
_BWRAP_LINE = "bwrap:"
_POLL_SEC = 0.05
#: Bounds the scan for an account's marked processes; walking a few thousand /proc entries takes well under a second.
_MARK_SCAN_TIMEOUT_SEC = 30.0
#: Prints the pids whose environment holds the entry in ``$1``; exits 3 when ``grep`` cannot read NUL-separated input.
LIST_MARKED_SCRIPT = (
    'grep -qzxF -- x /dev/null; [ "$?" -eq 1 ] || exit 3; '
    'for p in /proc/[0-9]*; do grep -qzxF -- "$1" "$p/environ" 2>/dev/null && echo "${p#/proc/}"; done; exit 0'
)


@dataclass(frozen=True)
class SpecialistLaunch:
    """A resolved launch: argv, child env, how it is isolated, backend, the task's prompt and log files, the launch as
    its tier prepared it, and the reap mark its environment carries."""

    cmd: list[str]
    env: dict[str, str]
    isolation: AgentIsolation
    backend: str
    prompt_file: Path
    process_log: Path
    prepared: PreparedLaunch
    reap_mark: str


def stamp_reap_mark(env: dict[str, str]) -> str:
    """Put a reap mark unique to one launch in ``env``, in place, and return it.

    Every process the child starts inherits the mark however it detaches, so a sweep for it after the launch ends
    finds every process that still carries it, and none of another launch's. One that replaced its environment (an
    exec with an empty one, say) carries no mark, and the sweep does not find it.
    """
    reap_mark = uuid.uuid4().hex
    env[REAP_MARK_ENV] = reap_mark
    return reap_mark


def compose_launch_env(task_id: str) -> tuple[dict[str, str], str]:
    """Compose the child env for one launch; return it with the launch's reap mark."""
    # Compose a minimal env. Provider credentials are inherited by default
    # for compatibility with deployments that authenticate the agent CLI
    # via env; set HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=0 to disable.
    env = _build_specialist_env()
    # Keep the spawned CLI off its non-essential traffic and auto-updater.
    from hyperloom_common.llm.llm_stability_env import apply_llm_stability_env

    apply_llm_stability_env(env)
    reap_mark = stamp_reap_mark(env)
    # The child spends against the gateway, so tag it or its spend lands
    # under no component at all. The task tag is what attributes the
    # requests of a child that dies before its result row -- the only
    # place such a child's token usage survives is the gateway's log.
    inject_attribution_env(env, component="specialist", operation="run_agent", task=task_id)
    return env, reap_mark


def apply_gpu_env(env: dict[str, str], workspace: Path, gpu_ids: Sequence[int], gpu_lease: object | None) -> None:
    """Set the GPU and compiler-cache environment of a specialist, in place.

    Args:
        env: The child environment.
        workspace: The task workspace, home of the private compiler caches of a CPU specialist.
        gpu_ids: Cards pinned for the task (the logical view under Ray's mask when ``gpu_lease`` is set).
        gpu_lease: The Ray GPU lease when the specialist runs in a Ray actor, else ``None``.
    """
    if gpu_lease is not None:
        # Ray-managed GPU execution: Ray sets *_VISIBLE_DEVICES in
        # the actor's worker; never let the caller env pin them (that would
        # override Ray's card assignment). ``gpu_ids`` here is the logical
        # 0..N-1 view the specialist sees under Ray's mask — kept only as the
        # informational count env for specialist tooling.
        for var in GPU_MASK_ENV_NAMES:
            env.pop(var, None)
        if gpu_ids:
            env[_GPU_IDS_ENV] = ",".join(str(g) for g in gpu_ids)
    elif gpu_ids:
        visible = ",".join(str(g) for g in gpu_ids)
        env["HIP_VISIBLE_DEVICES"] = visible
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["ROCR_VISIBLE_DEVICES"] = visible
        env[_GPU_IDS_ENV] = visible
    else:
        # CPU specialists: hide all GPUs and use workspace-local compiler caches.
        for var in GPU_MASK_ENV_NAMES:
            env[var] = ""
        cache_root = workspace / _CACHE_DIR
        aiter_jit = str(cache_root / _AITER_JIT_DIR)
        env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
        env["AITER_JIT_DIR"] = aiter_jit
        env[_AITER_JIT_ENV] = aiter_jit


def resolve_launch_isolation(gpu_lease: object | None) -> AgentIsolation:
    """How to isolate a specialist launched now (the module doc); one in a Ray GPU actor (``gpu_lease`` set) is never
    started as another account. Reads the environment, the capability set and bubblewrap's probe; changes nothing.

    Raises:
        IsolationRefused: The specialist must be isolated (a hosted run, or ``HYPERLOOM_SPECIALIST_USER`` set) and no
            tier works here; a :class:`~hyperloom.common.env_safety.RunAsError`.
    """
    return choose_agent_isolation(allow_account=gpu_lease is None)


@dataclass(frozen=True)
class LaunchPlace:
    """Where a specialist runs and what it reaches besides its toolchain.

    ``workspace`` holds its prompt, log and CLI state; ``worktree`` is the tree it works in (``None``: the workspace);
    ``private_files`` are the app's files in the workspace another account must not read; ``readable`` the other
    paths it reads (a local model, references); ``gpu`` whether it gets the GPU devices; ``mcp_config`` the operator's
    MCP configuration, whose stdio servers it starts; ``git_admin`` the worktree's admin directory as its repository
    records it (:func:`~hyperloom_common.git_safety.recorded_admin_dir`), whose repository a sandbox shows read-only
    (``None``: no git metadata is shown; what the worktree's ``.git`` names never is).
    """

    workspace: Path
    worktree: Path | None = None
    private_files: tuple[Path, ...] = ()
    readable: tuple[Path, ...] = ()
    gpu: bool = False
    mcp_config: str | None = None
    git_admin: Path | None = None

    @property
    def cwd(self) -> Path:
        """The directory the specialist starts in."""
        return self.worktree or self.workspace

    @property
    def writable(self) -> tuple[Path, ...]:
        """The trees the specialist writes."""
        return tuple(dict.fromkeys(path for path in (self.worktree, self.workspace) if path is not None))


@dataclass(frozen=True)
class Spawn:
    """The ``Popen`` arguments of one launch: its command line, its environment and the other keyword arguments."""

    argv: list[str]
    env: dict[str, str]
    kwargs: dict[str, Any]


@dataclass
class LaunchStatus:
    """Where bubblewrap writes its status (``--json-status-fd``): a local launch's pipe (``read_fd``, and
    ``write_fd`` until the launch has its own copy), or, for a Ray actor's, ``path`` in a directory only the app
    writes."""

    read_fd: int = -1
    write_fd: int = -1
    path: Path | None = None
    data: bytes = b""

    @classmethod
    def pipe(cls) -> LaunchStatus:
        """A pipe whose write end the launch inherits."""
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        return cls(read_fd, write_fd)

    @classmethod
    def file(cls) -> LaunchStatus:
        """A file name in a fresh directory of mode 0700."""
        return cls(path=Path(tempfile.mkdtemp(prefix="hl-agent-status-")) / "status.json")

    def started(self) -> None:
        """Close this process's copy of the write end, once the launch holds its own."""
        if self.write_fd >= 0:
            os.close(self.write_fd)
            self.write_fd = -1

    def read(self) -> BwrapStatus:
        """What bubblewrap has written so far."""
        if self.path is not None:
            try:
                self.data = read_bytes_nofollow(self.path)
            except OSError:
                pass
        elif self.read_fd >= 0:
            self.data += _drained(self.read_fd)
        try:
            return BwrapStatus.parse(self.data)
        except (ValueError, TypeError, KeyError):
            return BwrapStatus(None, None)

    def close(self) -> None:
        """Close the pipe, or remove the file and its directory; what was read stays readable."""
        self.read()
        self.started()
        if self.read_fd >= 0:
            os.close(self.read_fd)
            self.read_fd = -1
        if self.path is not None:
            shutil.rmtree(self.path.parent, ignore_errors=True)
            self.path = None


def _drained(fd: int) -> bytes:
    """Everything that can be read from the non-blocking ``fd`` now."""
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _bwrap_line(process_log: Path) -> str:
    """bubblewrap's last line in the launch's log, where its stderr goes; empty when there is none."""
    try:
        text = read_bytes_nofollow(process_log, limit=1 << 20).decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith(_BWRAP_LINE)]
    return lines[-1] if lines else ""


class SandboxedProcess:
    """A launch's bubblewrap process (``process``: a ``Popen``, or a Ray actor's stand-in) as the reap loop supervises
    it; every other attribute is the process's own.

    bubblewrap ends the sandbox when the agent ends, exiting with the agent's code (``128 + n`` for a death by signal
    ``n``), and on a SIGTERM of its own at once, without passing it on: :meth:`stop_agent` TERMs the sandbox's
    processes that carry the launch's mark instead, so the agent's own stop runs. :meth:`setup_failure` tells a
    sandbox bubblewrap could not set up from an agent that ran and failed.
    """

    def __init__(self, process: Any, mark: str, status: LaunchStatus) -> None:
        self.process = process
        self._mark = mark
        self._status = status

    def __getattr__(self, name: str) -> Any:
        return getattr(self.process, name)

    @property
    def agent_pid(self) -> int:
        """The agent's own process, whose children are the tools it runs (``pid`` is bubblewrap's, whose pid 1
        starts the agent); bubblewrap's until the sandbox has started it."""
        init = self._status.read().child_pid
        if init is None:
            return self.process.pid
        try:
            started = [pid for pid, _ in descendants(init) if (proc_identity(pid) or (0,))[0] == init]
        except OSError:
            started = []
        return min(started, default=self.process.pid)

    def stop_agent(self, grace_sec: float = TERM_GRACE_SEC) -> bool:
        """SIGTERM the sandbox's marked processes and wait up to ``grace_sec`` for bubblewrap to end with the agent.

        Returns:
            bool: Whether bubblewrap exited on its own, which ends every process of the sandbox; ``False`` at once
            when none carries the mark, and when a signal ended bubblewrap (its sandbox may outlive it).
        """
        found = [(pid, seen[1]) for pid in marked_pids(REAP_MARK_ENV, self._mark) if (seen := proc_identity(pid))]
        if not found:
            return False
        signal_processes(found, signal.SIGTERM)
        deadline = time.monotonic() + grace_sec
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(_POLL_SEC)
        code = self.process.poll()
        return code is not None and code >= 0

    def setup_failure(self, process_log: Path) -> str:
        """Why the sandbox could not be set up, from bubblewrap's line in ``process_log``, when bubblewrap failed
        (exited non-zero) without running the agent; empty when the agent ran, or bubblewrap still runs, exited 0 (its
        status unread: it fails with 1) or was ended by a signal."""
        code = self.process.poll()
        if code is None or code <= 0 or self._status.read().exit_code is not None:
            return ""
        return _bwrap_line(process_log) or f"bubblewrap exited {code} before the agent started"


def agent_pid(process: Any) -> int:
    """The pid of the agent CLI ``process`` runs, whose children are its tools: a sandbox's agent process, else the
    launched process itself."""
    return process.agent_pid if isinstance(process, SandboxedProcess) else process.pid


def marked_pids_as(run_as: RunAs, mark: str) -> list[int]:
    """The pids carrying ``mark``, read as ``run_as``: the app's uid cannot read a demoted child's environ.

    The scan runs as a shell under that account, because it cannot import Hyperloom from the app's tree. Any
    process of that account can write into the scan's stdout, so only pids the account owns are believed.
    """
    try:
        cp = subprocess.run(
            run_as.command(["sh", "-c", LIST_MARKED_SCRIPT, "sh", f"{REAP_MARK_ENV}={mark}"]),
            capture_output=True,
            text=True,
            timeout=_MARK_SCAN_TIMEOUT_SEC,
            check=False,
            env={"PATH": os.defpath},
            cwd="/",
            **run_as.popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("specialist reap: cannot list marked processes as %s: %s", run_as.user, exc)
        return []
    if cp.returncode != 0:
        log.warning("specialist reap: listing marked processes as %s exited %d", run_as.user, cp.returncode)
    listed = (int(pid) for pid in cp.stdout.split() if pid.isdigit())
    return [pid for pid in listed if process_uids(pid) == (run_as.uid, run_as.uid)]


@dataclass
class PreparedLaunch:
    """One launch made ready in its tier: ``popen_kwargs`` start it as another account; ``policy`` is its sandbox's,
    with ``scratch`` (its ``/tmp`` and ``/dev/shm``) bound in it.

    :meth:`local` or :meth:`actor` gives the spawn's arguments and :meth:`started` the process to supervise;
    :meth:`marked_pids` finds what carries the launch's reap mark; :meth:`release` removes what the launch left here
    once it has ended.
    """

    isolation: AgentIsolation
    popen_kwargs: dict[str, Any] = field(default_factory=dict)
    policy: BwrapPolicy | None = None
    scratch: Scratch | None = None
    status: LaunchStatus | None = None
    mark: str = ""

    def local(self, cmd: Sequence[str], env: Mapping[str, str]) -> Spawn:
        """The ``Popen`` arguments that start ``cmd`` with ``env`` here. In a sandbox the reap mark is set inside
        only, and bubblewrap's status goes to a pipe whose write end ``kwargs`` passes on.

        Raises:
            IsolationRefused: This bubblewrap must not launch the sandbox's policy.
        """
        if isinstance(self.isolation, AccountIsolation):
            return Spawn(self.isolation.run_as.command(cmd), dict(env), dict(self.popen_kwargs))
        if not isinstance(self.isolation, BwrapIsolation) or self.policy is None:
            return Spawn(list(cmd), dict(env), {})
        self.status = LaunchStatus.pipe()
        argv, child_env = self._sandboxed(self.isolation, cmd, env, self.policy, self.status.write_fd)
        return Spawn(argv, child_env, {"pass_fds": (self.status.write_fd,)})

    def actor(self, cmd: Sequence[str], env: Mapping[str, str]) -> Spawn:
        """The arguments that start ``cmd`` with ``env`` in a Ray GPU actor on this host: built from this host's
        probe and paths, so under bubblewrap for a single-node cluster only. There a shell opens the status file on
        descriptor 3 and execs bubblewrap, which keeps no ``--die-with-parent`` (the actor's thread is not this one).

        Raises:
            IsolationRefused: This bubblewrap must not launch the sandbox's policy.
        """
        if not isinstance(self.isolation, BwrapIsolation) or self.policy is None:
            return self.local(cmd, env)
        self.status = LaunchStatus.file()
        policy = replace(self.policy, die_with_parent=False)
        argv, child_env = self._sandboxed(self.isolation, cmd, env, policy, _ACTOR_STATUS_FD)
        return Spawn(["/bin/sh", "-c", _OPEN_STATUS, str(self.status.path), *argv], child_env, {})

    def _sandboxed(
        self,
        isolation: BwrapIsolation,
        cmd: Sequence[str],
        env: Mapping[str, str],
        policy: BwrapPolicy,
        status_fd: int,
    ) -> tuple[list[str], dict[str, str]]:
        """bubblewrap's command line and environment for ``cmd``: the reap mark moves inside, and is kept for
        :meth:`started`."""
        self.mark = env.get(REAP_MARK_ENV, "")
        return isolation.launch(cmd, policy, env, marks=(REAP_MARK_ENV,), status_fd=status_fd)

    def started(self, process: Any) -> Any:
        """The process to supervise for ``process``, just started with :meth:`local` or :meth:`actor`."""
        if self.status is None:
            return process
        self.status.started()
        return SandboxedProcess(process, self.mark, self.status)

    def marked_pids(self, mark: str) -> list[int]:
        """The processes that carry the launch's reap mark ``mark``: read as the account it ran as, or as the app
        (which owns a sandbox's user namespace, so it reads every environment in it)."""
        if isinstance(self.isolation, AccountIsolation):
            return marked_pids_as(self.isolation.run_as, mark)
        return marked_pids(REAP_MARK_ENV, mark)

    def release(self) -> None:
        """Close the status, and remove the sandbox's scratch with what the agent left in it."""
        if self.status is not None:
            self.status.close()
        scratch, self.scratch = self.scratch, None
        if scratch is not None:
            try:
                scratch.remove()
            except OSError:
                log.warning("specialist: cannot remove the sandbox scratch %s", scratch, exc_info=True)


def prepare_launch(
    isolation: AgentIsolation, place: LaunchPlace, cmd: Sequence[str], env: dict[str, str]
) -> PreparedLaunch:
    """Ready ``isolation`` for one launch of ``cmd`` in ``place``, changing ``env`` in place.

    As another account: the workspace and worktree are shared with it, the private files set back to 0600, and the
    identity variables are its. In a bubblewrap sandbox: ``HOME``, the CLI state and compiler caches named outside
    the trees it writes, and the temporary directory, are pointed inside it (:data:`SANDBOX_DIRECTORIES`), and its
    confinement is derived (:func:`specialist_confinement`). Unconfined: nothing.

    Raises:
        IsolationRefused: The sandbox's confinement cannot be built: a path it reads is reached through a tree it
            writes.
        SpecialistAgentUnavailableError: ``place.mcp_config`` cannot be read.
        OSError: The account's share, or the sandbox's home or scratch, could not be made.
    """
    if isinstance(isolation, AccountIsolation):
        return PreparedLaunch(isolation, popen_kwargs=_demote(isolation.run_as, place, env))
    if not isinstance(isolation, BwrapIsolation):
        return PreparedLaunch(isolation)
    _point_into_sandbox(place, env)
    confinement = specialist_confinement(place, cmd[0], env)
    scratch = Scratch.create()
    try:
        policy = confinement.policy(scratch=scratch, chdir=place.cwd)
    except BwrapPolicyError as refused:
        scratch.remove()
        raise IsolationRefused(refused.problems) from None
    return PreparedLaunch(isolation, policy=_scratch_where_free(policy, isolation.probe.version), scratch=scratch)


def _demote(run_as: RunAs, place: LaunchPlace, env: dict[str, str]) -> dict[str, Any]:
    """Make the specialist's tree writable by ``run_as`` and return its ``Popen`` kwargs.

    The workspace is shared with the sticky bit so the child cannot delete the app's own files; the private files
    (the prompt and the log) are set back to 0600. A worktree outside the workspace is shared as well.
    """
    workspace, worktree = place.workspace, place.worktree
    run_as.share_tree(workspace, sticky=True)
    for private in place.private_files:
        if private.is_file() and not private.is_symlink():
            private.chmod(0o600)
    if worktree is not None and workspace not in worktree.parents:
        run_as.share_tree(worktree)
    if worktree is not None:
        run_as.share_worktree_git(worktree)
    run_as.apply_env(env, safe_dirs=[worktree] if worktree is not None else ())
    return run_as.popen_kwargs()


def _at_or_under(path: str, root: str) -> bool:
    path, root = os.path.normpath(path), os.path.normpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def _point_into_sandbox(place: LaunchPlace, env: dict[str, str]) -> None:
    """Point ``env``'s home, CLI state and caches named outside the trees the specialist writes into its workspace,
    and its temporary directory at the sandbox's ``/tmp``; make the home."""
    trees = [str(tree) for tree in place.writable]

    def outside(value: str) -> bool:
        return not any(_at_or_under(value, tree) for tree in trees)

    if outside(env.get("HOME", "") or "/"):
        home = place.workspace / SANDBOX_HOME
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        env["HOME"] = str(home)
    for name, parts in SANDBOX_DIRECTORIES.items():
        if env.get(name) and outside(env[name]):
            env[name] = str(place.workspace.joinpath(*parts))
    for name in TEMP_VARIABLES:
        env[name] = "/tmp"


def _scratch_where_free(policy: BwrapPolicy, version: tuple[int, int, int] | None) -> BwrapPolicy:
    """``policy`` with a fresh tmpfs at ``/tmp`` or ``/dev/shm`` instead of its bound scratch where something is
    mounted below it and this bubblewrap is older than :data:`~hyperloom_common.bwrap.NESTED_SAFE` (there a mount
    below a directory an agent wrote may follow a link it left): a workspace under the host's ``/tmp``, say."""
    if version is not None and version >= NESTED_SAFE:
        return policy
    mounted = [PurePosixPath(path) for path in (*policy.read_only, *policy.writable, *policy.masked, *policy.devices)]

    def free(point: str) -> bool:
        return not any(PurePosixPath(point) in path.parents for path in mounted)

    return replace(policy, tmp=policy.tmp if free("/tmp") else None, shm=policy.shm if free("/dev/shm") else None)


def specialist_confinement(place: LaunchPlace, argv0: str, env: Mapping[str, str]) -> Confinement:
    """What a specialist running ``argv0`` with ``env`` in ``place`` reaches in its sandbox.

    It writes its workspace and worktree. It reads its toolchain (:func:`~hyperloom.common.confinement.toolchain_paths`,
    which names its model and benchmark trees too), the git metadata of its worktree, ``place.readable`` (a local
    model, references), the framework source roots, the programs of the stdio MCP servers it starts, and the AWS files
    a Bedrock login names. The credential locations those show are masked, and so are the app's state directory and
    build store; a GPU task gets the GPU devices.

    Raises:
        SpecialistAgentUnavailableError: ``place.mcp_config`` cannot be read.
    """
    writable = place.writable
    given = bedrock_files(env)
    readable = [
        *toolchain_paths(env, argv0, writable=writable),
        *_git_dirs(place.git_admin),
        *_readable_extras(place),
        *given,
    ]
    for server in _stdio_servers(place.mcp_config):
        readable.extend(toolchain_paths(env, server, writable=writable))
    readable = list(dict.fromkeys(readable))
    masked = credential_paths(env, given=given, allowed=(*readable, *writable), writable=writable)
    app_only = [Path(value) for value in map(os.environ.get, APP_ONLY_VARIABLES) if value and os.path.isabs(value)]
    return Confinement(
        writable=writable,
        readable=tuple(readable),
        masked=(*masked, *app_only),
        devices=tuple(device for device in GPU_DEVICES if place.gpu and device.exists()),
    )


def _readable_extras(place: LaunchPlace) -> list[Path]:
    """``place.readable`` and the framework source roots that exist, but not those in a tree the specialist writes,
    which shows them already."""
    found = [*place.readable, *map(Path, resolve_kernel_search_roots())]
    trees = [str(tree) for tree in place.writable]
    return [
        path
        for path in found
        if path.is_absolute() and path.exists() and not any(_at_or_under(str(path), tree) for tree in trees)
    ]


def _git_dirs(admin: Path | None) -> tuple[Path, ...]:
    """What git in a linked worktree whose admin directory is ``admin`` reads besides the tree: its repository's git
    directory, which holds ``admin``, and the object stores its ``alternates`` file names. Nothing without ``admin``:
    where the agent runs as the app's uid, the directory the worktree's ``.git`` names may be one it made, ownership
    and all, so that file decides nothing here."""
    if admin is None:
        return ()
    common = admin.parent.parent
    try:
        alternates = read_text_nofollow(common / "objects" / "info" / "alternates").splitlines()
    except (OSError, UnicodeDecodeError):
        alternates = []
    return (common, *(Path(line.strip()) for line in alternates if os.path.isabs(line.strip())))


def _stdio_servers(mcp_config: str | None) -> tuple[str, ...]:
    """The commands of the stdio MCP servers in ``mcp_config``, which run inside the sandbox; says once which."""
    servers = [server for server in load_mcp_servers(mcp_config) if not server.is_http]
    if servers:
        _log_once(
            logging.WARNING,
            "the stdio MCP servers "
            + ", ".join(f"{server.name} ({server.command})" for server in servers)
            + " run inside the specialist's bubblewrap sandbox: their programs are bound, and nothing else they read "
            "is unless the specialist's confinement shows it",
        )
    return tuple(server.command for server in servers)


@functools.lru_cache(maxsize=64)
def _log_once(level: int, message: str) -> None:
    log.log(level, "%s", message)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE


def bedrock_files(env: Mapping[str, str]) -> tuple[Path, ...]:
    """The AWS files a Bedrock-backed claude CLI takes its credentials from, as ``env`` names them (absolute and
    present); none unless ``CLAUDE_CODE_USE_BEDROCK`` is on. A sandbox binds them read-only and does not mask them."""
    if not _truthy(env.get(_BEDROCK)):
        return ()
    named = (env.get(name, "") for name in BEDROCK_FILE_VARIABLES)
    return tuple(Path(value) for value in named if os.path.isabs(value) and os.path.isfile(value))


def require_claude_login(isolation: AgentIsolation, env: Mapping[str, str]) -> None:
    """Refuse a claude CLI in a bubblewrap sandbox that has no login it can read there: its own (under the app's
    home or ``CLAUDE_CONFIG_DIR``) is not bound. A token in ``env`` logs it in, and so does Bedrock with its keys
    in ``env`` or a credentials file ``env`` names (:func:`bedrock_files`).

    Raises:
        SpecialistAgentUnavailableError: :data:`CLAUDE_LOGIN_REFUSAL`.
    """
    if not isinstance(isolation, BwrapIsolation) or any(env.get(name) for name in CLAUDE_TOKEN_VARIABLES):
        return
    if _truthy(env.get(_BEDROCK)) and (all(env.get(name) for name in _BEDROCK_KEYS) or bedrock_files(env)):
        return
    raise SpecialistAgentUnavailableError(CLAUDE_LOGIN_REFUSAL)


def codex_sandbox_off(isolation: AgentIsolation) -> bool:
    """Whether codex must run with its own sandbox off: in a bubblewrap sandbox (:data:`CODEX_SANDBOX_OFF`, said once)."""
    if isinstance(isolation, BwrapIsolation):
        _log_once(logging.INFO, CODEX_SANDBOX_OFF)
        return True
    return False


def without_codex_sandbox(argv: Sequence[str]) -> list[str]:
    """``argv`` of a codex CLI with its own sandbox off: ``--sandbox``/``-s`` with its mode and ``--full-auto``
    dropped, and :data:`CODEX_BYPASS_FLAG` after ``exec`` (after the program when there is none)."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg in _CODEX_SANDBOX_FLAGS:
            skip = True
        elif not arg.startswith("--sandbox=") and arg not in _CODEX_SANDBOX_SWITCHES:
            out.append(arg)
    if CODEX_BYPASS_FLAG not in out:
        at = out.index("exec") + 1 if "exec" in out else min(1, len(out))
        out.insert(at, CODEX_BYPASS_FLAG)
    return out


def stage_mcp_config(isolation: AgentIsolation, workspace: Path, cmd: list[str]) -> None:
    """Point ``--mcp-config`` in ``cmd`` at a copy in the workspace, in place, for a child that cannot read the
    configured file: one run as another account, or in a sandbox, which does not bind the app's runtime directory.

    The configured file sits in the app's runtime directory at 0600. The copy goes in the task workspace, and is made
    before :func:`prepare_launch` shares that workspace, so the account can read it. Nothing changes for an
    unconfined child or a ``cmd`` that carries no ``--mcp-config``.
    """
    if isinstance(isolation, NoIsolation) or _MCP_FLAG not in cmd:
        return
    at = cmd.index(_MCP_FLAG) + 1
    try:
        text = read_bytes_nofollow(cmd[at]).decode("utf-8")
        cmd[at] = str(write_text_nofollow(workspace, ("specialist_mcp.json",), text, mode=0o600))
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecialistAgentUnavailableError(
            f"cannot stage the MCP config {cmd[at]} for the specialist: {exc}"
        ) from exc
