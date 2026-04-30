"""MultiCLILauncher — generic version of marathon's ``write_pane_script``.

Takes a list of :class:`AgentCard` and produces:

1. One per-agent restart-loop bash script under
   ``$SESSION_DIR/.multicli/run_pane_<name>.sh``. Each script wraps a
   ``while ... claude --print --continue ...`` loop that:

       * sources a generated ``.env`` file with the run-wide config
       * loads the agent's ``system_prompt.md`` (base64-injected)
       * passes ``--add-dir`` for $SESSION_DIR + $AGENT_DIR + each extra
       * detects ``$STOP_FILE`` / ``MAX_RESTARTS`` to bail cleanly
       * sleeps ``backoff_seconds`` between restarts then re-enters
         with ``--continue`` (Claude) or with ``conversation.jsonl``
         re-injected into the prompt header (Codex)

2. Optionally a tmux session that drives them in parallel.

   - One ``tmux new-session`` plus ``new-window`` per agent.
   - The launcher writes ``$SESSION_DIR/.multicli/tmux_session_name``
     so callers can ``tmux attach`` later.

Reference
---------

* ``marathon/launcher/run.sh`` lines 419–477 — the original
  ``write_pane_script`` we generalise here.
* ``marathon/launcher/run.sh`` lines 496–503 — the tmux orchestration we
  reuse.

The launcher is **side-effect-free at construction**: nothing happens
until you call :meth:`stage` (writes scripts) or :meth:`launch` (starts
tmux + scripts). This makes it test-friendly: tests call ``stage`` then
inspect the generated files.
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .agent_card import AgentCard


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class LauncherError(RuntimeError):
    """Raised when the launcher cannot stage or start an agent."""


DEFAULT_TMUX_SESSION = "io-multicli"
WORK_SUBDIR = ".multicli"
LOG_SUBDIR = "logs"


# ---------------------------------------------------------------------------
@dataclass
class StagedAgent:
    """A per-agent script + log path produced by :meth:`MultiCLILauncher.stage`.

    Fields are absolute paths so callers can ``tmux send-keys`` them
    directly without knowing the launcher's working directory.

    When :meth:`MultiCLILauncher.launch_subprocess` started the agent,
    the resulting :class:`subprocess.Popen` handle is also stashed on
    :attr:`process` so the Conductor can poll status / send signals.
    """

    name: str
    role: str
    backend: str
    pane_script: Path
    log_file: Path
    stop_file: Path
    inbox_path: Path
    outbox_path: Path
    agent_dir: Path
    process: subprocess.Popen | None = None


@dataclass
class MultiCLILauncher:
    """Generates the bash scripts + tmux session that drive multi-CLI agents.

    Attributes:
        session_dir:   $SESSION_DIR — per-run mutable root.
        cards:         Mapping ``{name: AgentCard}`` of agents to launch.
        env:           Extra env passed into the per-pane ``.env`` file.
        extra_dirs:    Additional ``--add-dir`` arguments for each Claude
                       pane (e.g. ``InferenceX``, ``BASE_DIR``).
        agent_root:    Where ``agents/<name>/`` lives so the launcher can
                       ``--add-dir`` each agent's private assets.
        tmux_session:  tmux session name (default ``io-multicli``).
        claude_bin / codex_bin: PATH-resolved binary names (overridable
                                for tests).
    """

    session_dir: Path
    cards: Mapping[str, AgentCard]
    env: Mapping[str, str] = field(default_factory=dict)
    extra_dirs: Sequence[Path] = field(default_factory=tuple)
    agent_root: Path | None = None
    tmux_session: str = DEFAULT_TMUX_SESSION
    claude_bin: str = "claude"
    codex_bin: str = "codex"
    initial_user_msg: str = (
        "Begin your role from the agent_card workflow. Read inbox.jsonl, "
        "process new envelopes after the persisted seq cursor, then write "
        "any responses to outbox.jsonl per the protocol."
    )
    resume_user_msg: str = (
        "Continue. Re-read $SESSION_DIR/state.json + inbox.jsonl tail; pick "
        "up at your last persisted cursor and emit any pending intents."
    )
    allowed_tools: str = (
        "Bash Read Write Edit MultiEdit Glob Grep TodoWrite Task WebSearch WebFetch"
    )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @property
    def work_dir(self) -> Path:
        return Path(self.session_dir) / WORK_SUBDIR

    @property
    def log_dir(self) -> Path:
        return Path(self.session_dir) / LOG_SUBDIR

    @property
    def env_file(self) -> Path:
        return self.work_dir / ".env"

    def pane_script_path(self, name: str) -> Path:
        return self.work_dir / f"run_pane_{name}.sh"

    def stop_file_path(self, name: str) -> Path:
        return Path(self.session_dir) / f"STOP_AGENT_{name}"

    def log_file_path(self, name: str) -> Path:
        return self.log_dir / f"{name}.log"

    def agent_dir(self, card: AgentCard) -> Path:
        """Return the per-session agent dir (inbox/outbox + cursor live here)."""
        return Path(self.session_dir) / "agents" / card.name

    # ------------------------------------------------------------------
    # Stage — write all scripts to disk; do NOT spawn anything
    # ------------------------------------------------------------------
    def stage(self) -> dict[str, StagedAgent]:
        if not self.cards:
            raise LauncherError("no agent cards to stage")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.session_dir, "agents").mkdir(parents=True, exist_ok=True)

        self._write_env_file()

        staged: dict[str, StagedAgent] = {}
        for name, card in self.cards.items():
            if not card.enabled:
                continue
            agent_dir = self.agent_dir(card)
            agent_dir.mkdir(parents=True, exist_ok=True)
            inbox = agent_dir / card.inbox_filename
            outbox = agent_dir / card.outbox_filename
            inbox.touch(exist_ok=True)
            outbox.touch(exist_ok=True)

            script_path = self._write_pane_script(card)
            staged[name] = StagedAgent(
                name=name,
                role=card.role,
                backend=card.backend,
                pane_script=script_path,
                log_file=self.log_file_path(name),
                stop_file=self.stop_file_path(name),
                inbox_path=inbox,
                outbox_path=outbox,
                agent_dir=agent_dir,
            )
        return staged

    # ------------------------------------------------------------------
    # Launch — actually start tmux (best-effort; raises LauncherError if
    # tmux is not installed)
    # ------------------------------------------------------------------
    def launch(self, *, kill_existing: bool = True) -> dict[str, StagedAgent]:
        """Stage scripts and start a tmux session running each pane.

        ``kill_existing`` first ``tmux kill-session`` to avoid stale
        windows from a previous run.
        """
        staged = self.stage()
        if not _have_tmux():
            raise LauncherError(
                "tmux not found on PATH; install tmux or invoke each pane "
                "script manually under your own supervisor"
            )
        if kill_existing:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.tmux_session],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        first = True
        for name, agent in staged.items():
            if first:
                subprocess.run(
                    ["tmux", "new-session", "-d", "-s", self.tmux_session,
                     "-n", name, "-c", str(self.work_dir)],
                    check=True,
                )
                first = False
            else:
                subprocess.run(
                    ["tmux", "new-window", "-t", self.tmux_session,
                     "-n", name, "-c", str(self.work_dir)],
                    check=True,
                )
            subprocess.run(
                ["tmux", "send-keys", "-t", f"{self.tmux_session}:{name}",
                 f"bash {shlex.quote(str(agent.pane_script))}; exit", "C-m"],
                check=True,
            )
        # Persist the session name so external tools can attach.
        (self.work_dir / "tmux_session_name").write_text(
            self.tmux_session, encoding="utf-8"
        )
        return staged

    def launch_subprocess(self) -> dict[str, StagedAgent]:
        """Stage scripts and start each pane as a detached subprocess.

        Unlike :meth:`launch` (which needs tmux), this method uses
        :class:`subprocess.Popen` directly. Best for unattended /
        headless runs and CI; tmux remains the right choice when an
        operator wants to ``tmux attach`` and watch panes interactively.

        Each child:

        * starts in its own session (``start_new_session=True``) so a
          SIGINT to the Conductor doesn't propagate immediately;
        * sends stdout+stderr to the same per-agent log file that the
          tmux path uses, so monitor.sh and tail -f keep working;
        * is held in :attr:`StagedAgent.process` so the Conductor can
          poll, signal, or wait on it.
        """
        staged = self.stage()
        for name, agent in staged.items():
            log_handle = agent.log_file.open("a", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    ["bash", str(agent.pane_script)],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(self.work_dir),
                )
            except OSError:
                log_handle.close()
                log.exception("launcher: subprocess.Popen failed for %s", name)
                raise LauncherError(
                    f"failed to spawn agent {name!r}; check {agent.log_file}"
                )
            # NOTE: log_handle stays open for the lifetime of the child;
            # GC closing it would EBADF the writer. We attach it to the
            # process object so it survives as long as the process does.
            proc._io_logs_keepalive = log_handle  # type: ignore[attr-defined]
            agent.process = proc
            log.info(
                "launcher: spawned %s as pid=%d (script=%s log=%s)",
                name, proc.pid, agent.pane_script, agent.log_file,
            )
        return staged

    def request_stop_all(self, staged: Mapping[str, StagedAgent]) -> None:
        """Drop a STOP_AGENT_<name> sentinel for every staged agent.

        The pane scripts check this file at the top of each restart loop
        iteration and bail before re-entering ``claude --print``.
        """
        for name, agent in staged.items():
            try:
                agent.stop_file.parent.mkdir(parents=True, exist_ok=True)
                agent.stop_file.touch(exist_ok=True)
            except OSError:
                log.exception("launcher: failed to drop STOP file for %s", name)

    def wait_for_exit(
        self,
        staged: Mapping[str, StagedAgent],
        *,
        timeout_s: float = 30.0,
        kill_after_timeout: bool = True,
    ) -> dict[str, int | None]:
        """Wait for every spawned subprocess to exit, returning ``{name: rc}``.

        Only the subprocess path is supported (tmux panes own their own
        process tree). When ``kill_after_timeout`` is True (default), any
        still-running children are SIGKILLed after ``timeout_s`` so the
        Conductor process never deadlocks on a wedged agent.
        """
        import os
        import signal
        import time

        deadline = time.monotonic() + max(0.0, timeout_s)
        rcs: dict[str, int | None] = {}
        for name, agent in staged.items():
            if agent.process is None:
                rcs[name] = None
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                rcs[name] = agent.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                if kill_after_timeout:
                    # Kill the whole process group (start_new_session=True
                    # gives every agent its own sid); covers grand-children
                    # the bash restart-loop may have spawned.
                    try:
                        os.killpg(agent.process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        rcs[name] = agent.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        rcs[name] = None
                    log.warning(
                        "launcher: agent %s did not exit after STOP; SIGKILL pid=%d",
                        name, agent.process.pid,
                    )
                else:
                    rcs[name] = None
        return rcs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _write_env_file(self) -> None:
        """Write ``$WORK_DIR/.env`` with run-wide vars + each card extra.

        We lean on bash's ``set -a; source .env; set +a`` to export every
        line — same convention as marathon's launcher.
        """
        merged: dict[str, str] = {
            "SESSION_DIR": str(self.session_dir),
            "MULTICLI_WORK_DIR": str(self.work_dir),
            "MULTICLI_LOG_DIR": str(self.log_dir),
        }
        if self.agent_root is not None:
            merged["INFERENCE_OPTIMIZER_AGENTS_ROOT"] = str(self.agent_root)
        merged.update({k: str(v) for k, v in self.env.items()})

        lines = [
            "# Auto-generated by MultiCLILauncher; consumed by per-pane scripts.",
        ]
        for k, v in sorted(merged.items()):
            # Use shlex.quote so values with spaces / quotes survive bash sourcing.
            lines.append(f"{k}={shlex.quote(v)}")
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(self.env_file, 0o600)
        except OSError:
            pass

    def _read_system_prompt(self, card: AgentCard) -> str:
        path = card.system_prompt_path
        if not path.is_file():
            log.warning(
                "launcher: agent %s has no system_prompt.md at %s; using stub",
                card.name, path,
            )
            return (
                f"# {card.name} (stub)\n"
                f"You are the {card.role} agent. Read inbox.jsonl, write "
                f"intents to outbox.jsonl per the multi-CLI A2A protocol.\n"
            )
        return path.read_text(encoding="utf-8")

    def _write_pane_script(self, card: AgentCard) -> Path:
        if card.backend == "claude":
            body = self._compose_claude_pane(card)
        elif card.backend == "codex":
            body = self._compose_codex_pane(card)
        elif card.backend == "mock":
            body = self._compose_mock_pane(card)
        elif card.backend == "mock-cli":
            body = self._compose_mock_cli_pane(card)
        else:  # pragma: no cover — schema rejects this earlier
            raise LauncherError(f"unknown backend {card.backend!r} for {card.name}")
        path = self.pane_script_path(card.name)
        path.write_text(body, encoding="utf-8")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        return path

    # ------------------------------------------------------------------
    # Per-backend pane templates
    # ------------------------------------------------------------------
    def _add_dir_args(self, card: AgentCard) -> str:
        agent_dir = self.agent_dir(card)
        dirs = [str(self.session_dir), str(agent_dir), str(card.card_dir)]
        # v0.4 MVP — triage is the cross-layer health watcher and needs to
        # Read sibling agents' outbox/inbox jsonl files. Give it the
        # parent ``$SESSION_DIR/agents/`` so Claude SDK path-safety lets
        # it tail every sibling. See standalone_agent_design §13.9.5.
        if card.role == "triage":
            siblings_root = str(self.session_dir / "agents")
            if siblings_root not in dirs:
                dirs.append(siblings_root)
        for extra in self.extra_dirs:
            extra_str = str(extra)
            if extra_str and extra_str not in dirs:
                dirs.append(extra_str)
        # Each --add-dir on its own continuation line so the script is
        # easy to read in logs.
        return " \\\n        ".join(
            f"--add-dir {shlex.quote(d)}" for d in dirs
        )

    def _compose_claude_pane(self, card: AgentCard) -> str:
        prompt = self._read_system_prompt(card)
        sp_b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        max_restarts = card.restart_policy.max_restarts
        backoff = card.restart_policy.backoff_seconds
        continue_flag = "--continue" if card.restart_policy.continue_flag else ""
        log_file = self.log_file_path(card.name)
        stop_file = self.stop_file_path(card.name)
        agent_dir = self.agent_dir(card)
        initial_msg = shlex.quote(self.initial_user_msg)
        resume_msg = shlex.quote(self.resume_user_msg)
        add_dirs = self._add_dir_args(card)
        env_file = self.env_file
        claude = self.claude_bin
        allowed_tools = self.allowed_tools

        return f"""#!/usr/bin/env bash
# Auto-generated pane launcher for agent '{card.name}' (backend=claude).
#
# Restart-loop strategy mirrors marathon/launcher/run.sh lines 419-472:
# the inner `claude --print` is one-shot; the outer `while` adds
# `--continue` from the second iteration onwards to resume the prior
# conversation history (Anthropic SDK tracks it per working directory).
set -a
[ -f "{env_file}" ] && source "{env_file}"
set +a

cd "{self.work_dir}"

SYSTEM_PROMPT=$(base64 -d <<'B64'
{sp_b64}
B64
)

LOG="{log_file}"
STOP_FILE="{stop_file}"
AGENT_DIR="{agent_dir}"
MAX_RESTARTS={max_restarts}
ATTEMPT=0
CONTINUE_FLAG=""
USER_MSG={initial_msg}

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Iseconds)] [agent:{card.name}] launcher starting" >> "$LOG"

while [ ! -f "$STOP_FILE" ] && [ $ATTEMPT -lt $MAX_RESTARTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[$(date -Iseconds)] [agent:{card.name}] attempt=$ATTEMPT continue=$CONTINUE_FLAG" >> "$LOG"
    {claude} --print \\
        --output-format stream-json --verbose \\
        $CONTINUE_FLAG \\
        --permission-mode dontAsk \\
        {add_dirs} \\
        --allowedTools {shlex.quote(allowed_tools)} \\
        --system-prompt "$SYSTEM_PROMPT" \\
        $USER_MSG \\
        >> "$LOG" 2>&1 < /dev/null
    EXIT=$?
    echo "[$(date -Iseconds)] [agent:{card.name}] claude exit=$EXIT" >> "$LOG"
    [ -f "$STOP_FILE" ] && break
    sleep {backoff}
    CONTINUE_FLAG={shlex.quote(continue_flag)}
    USER_MSG={resume_msg}
done
echo "[$(date -Iseconds)] [agent:{card.name}] launcher exiting" >> "$LOG"
"""

    def _compose_codex_pane(self, card: AgentCard) -> str:
        """Codex restart-loop with explicit conversation log injection.

        Codex CLI lacks ``--continue``; we therefore prepend the current
        ``conversation.jsonl`` (truncated to a sane head if huge) to each
        run's prompt so the model sees the prior turns.

        The actual conversation.jsonl maintenance lives in Phase 3
        (codex_continuity todo) — this template ships the reload hook
        so the file is consulted from day one.
        """
        prompt = self._read_system_prompt(card)
        sp_b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        max_restarts = card.restart_policy.max_restarts
        backoff = card.restart_policy.backoff_seconds
        log_file = self.log_file_path(card.name)
        stop_file = self.stop_file_path(card.name)
        agent_dir = self.agent_dir(card)
        env_file = self.env_file
        codex = self.codex_bin
        conv_log_name = (card.extra or {}).get("conversation_log", "conversation.jsonl")

        return f"""#!/usr/bin/env bash
# Auto-generated pane launcher for agent '{card.name}' (backend=codex).
#
# Codex CLI has no `--continue`. We approximate persistence by re-reading
# ``conversation.jsonl`` on every restart and prepending it to the prompt.
set -a
[ -f "{env_file}" ] && source "{env_file}"
set +a

cd "{self.work_dir}"

SYSTEM_PROMPT=$(base64 -d <<'B64'
{sp_b64}
B64
)

LOG="{log_file}"
STOP_FILE="{stop_file}"
AGENT_DIR="{agent_dir}"
CONV_LOG="$AGENT_DIR/{conv_log_name}"
MAX_RESTARTS={max_restarts}
ATTEMPT=0
mkdir -p "$(dirname "$LOG")" "$AGENT_DIR"
touch "$CONV_LOG"
echo "[$(date -Iseconds)] [agent:{card.name}] launcher starting" >> "$LOG"

while [ ! -f "$STOP_FILE" ] && [ $ATTEMPT -lt $MAX_RESTARTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[$(date -Iseconds)] [agent:{card.name}] attempt=$ATTEMPT" >> "$LOG"
    PROMPT_FILE=$(mktemp)
    {{
      echo "$SYSTEM_PROMPT"
      echo ""
      echo "==== prior conversation (oldest -> newest) ===="
      cat "$CONV_LOG"
      echo "==== end conversation ===="
    }} > "$PROMPT_FILE"
    {codex} --prompt-file "$PROMPT_FILE" \\
        >> "$LOG" 2>&1 < /dev/null
    EXIT=$?
    rm -f "$PROMPT_FILE"
    echo "[$(date -Iseconds)] [agent:{card.name}] codex exit=$EXIT" >> "$LOG"
    [ -f "$STOP_FILE" ] && break
    sleep {backoff}
done
echo "[$(date -Iseconds)] [agent:{card.name}] launcher exiting" >> "$LOG"
"""

    def _compose_mock_pane(self, card: AgentCard) -> str:
        """Test-only mock pane that just heart-beats into outbox.jsonl.

        Used by the multi-cli e2e harness to validate the launcher +
        Router glue without needing a real Claude/Codex install.
        """
        max_restarts = card.restart_policy.max_restarts
        backoff = card.restart_policy.backoff_seconds
        log_file = self.log_file_path(card.name)
        stop_file = self.stop_file_path(card.name)
        agent_dir = self.agent_dir(card)
        env_file = self.env_file
        outbox = agent_dir / card.outbox_filename
        return f"""#!/usr/bin/env bash
# Auto-generated MOCK pane launcher for agent '{card.name}'. Test-only.
set -a
[ -f "{env_file}" ] && source "{env_file}"
set +a

LOG="{log_file}"
STOP_FILE="{stop_file}"
OUTBOX="{outbox}"
MAX_RESTARTS={max_restarts}
ATTEMPT=0
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUTBOX")"
touch "$OUTBOX"
echo "[$(date -Iseconds)] [agent:{card.name}] mock launcher starting" >> "$LOG"

while [ ! -f "$STOP_FILE" ] && [ $ATTEMPT -lt $MAX_RESTARTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    SEQ=$(($(wc -l < "$OUTBOX") + 1))
    TS=$(date -Iseconds)
    MSG_ID=$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4().hex)')
    echo "{{\\"kind\\":\\"intent\\",\\"msg_id\\":\\"$MSG_ID\\",\\"seq\\":$SEQ,\\"ts\\":\\"$TS\\",\\"from_agent\\":\\"{card.name}\\",\\"to_agent\\":\\"conductor\\",\\"intent_type\\":\\"send_message\\",\\"payload\\":{{\\"topic\\":\\"heartbeat\\",\\"attempt\\":$ATTEMPT}}}}" >> "$OUTBOX"
    echo "[$TS] [agent:{card.name}] mock heartbeat attempt=$ATTEMPT" >> "$LOG"
    sleep {backoff}
done
echo "[$(date -Iseconds)] [agent:{card.name}] mock launcher exiting" >> "$LOG"
"""

    def _compose_mock_cli_pane(self, card: AgentCard) -> str:
        """Pane that drives the in-tree :mod:`mock_agent` module.

        Unlike ``mock`` (one-shot heartbeat), ``mock-cli`` actually
        polls the inbox + responds via outbox using the canonical
        envelope schema — so it exercises the complete subprocess +
        Router round-trip in CI without needing a real Claude install.

        Extra knobs come from ``card.extra``:

            mock_cli_args:        list[str]   appended verbatim to the
                                              python -m mock_agent CLI
            mock_cli_python:      str         override interpreter
            mock_cli_env:         dict        extra env injected before
                                              the python invocation
        """
        log_file = self.log_file_path(card.name)
        stop_file = self.stop_file_path(card.name)
        agent_dir = self.agent_dir(card)
        env_file = self.env_file
        extra = card.extra or {}
        py = str(extra.get("mock_cli_python") or sys.executable or "python3")
        cli_args = extra.get("mock_cli_args") or []
        if not isinstance(cli_args, list):
            raise LauncherError(
                f"agent {card.name}: extra.mock_cli_args must be a list, "
                f"got {type(cli_args).__name__}"
            )
        cli_extra_env = extra.get("mock_cli_env") or {}
        if not isinstance(cli_extra_env, dict):
            raise LauncherError(
                f"agent {card.name}: extra.mock_cli_env must be a mapping, "
                f"got {type(cli_extra_env).__name__}"
            )
        # Build a quoted CLI tail that survives being injected into the
        # generated bash script.
        argv_tail = " ".join(shlex.quote(str(a)) for a in cli_args)
        env_lines = "\n".join(
            f"export {k}={shlex.quote(str(v))}" for k, v in cli_extra_env.items()
        )
        # Note: subprocess.Popen inherits PYTHONPATH from the parent env
        # by default, and self._write_env_file() also exposes whatever
        # the operator passed via launcher_env. We deliberately do NOT
        # emit a hardcoded ``export PYTHONPATH=...`` line here — that
        # used to clobber the absolute-path PYTHONPATH the Conductor
        # injected via .env with whatever relative value happened to be
        # in os.environ at script-write time.
        return f"""#!/usr/bin/env bash
# Auto-generated MOCK-CLI pane launcher for agent '{card.name}'.
# Drives mock_agent.py as a real subprocess that follows the multi-cli
# A2A protocol — used by e2e tests + smoke runs without a real Claude.
set -a
[ -f "{env_file}" ] && source "{env_file}"
set +a
{env_lines}

LOG="{log_file}"
STOP_FILE="{stop_file}"
AGENT_DIR="{agent_dir}"
mkdir -p "$(dirname "$LOG")" "$AGENT_DIR"
echo "[$(date -Iseconds)] [agent:{card.name}] mock-cli launcher starting" >> "$LOG"

{shlex.quote(py)} -m inference_optimizer.orchestrator.multi_cli.mock_agent \\
    --agent-name {shlex.quote(card.name)} \\
    --session-dir "$SESSION_DIR" \\
    {argv_tail} \\
    >> "$LOG" 2>&1 < /dev/null

EXIT=$?
echo "[$(date -Iseconds)] [agent:{card.name}] mock-cli exit=$EXIT" >> "$LOG"
"""


# ---------------------------------------------------------------------------
def _have_tmux() -> bool:
    return shutil.which("tmux") is not None


__all__ = [
    "DEFAULT_TMUX_SESSION",
    "LauncherError",
    "LOG_SUBDIR",
    "MultiCLILauncher",
    "StagedAgent",
    "WORK_SUBDIR",
]
