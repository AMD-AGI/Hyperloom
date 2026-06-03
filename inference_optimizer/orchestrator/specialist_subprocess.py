"""Subprocess-based specialist dispatcher — PR-A2 (Arbor-into-Hyperloom).

The original v0.8 M5 :class:`SpecialistRunner` invoked an in-process
``Backend`` (:class:`ClaudeBackend`) to drive the specialist LLM loop.
That shape:

* couples the specialist's claude-agent-sdk subprocess to the
  orchestrator reactor — a hung agent stalls the coordinator;
* shares filesystem access with the orchestrator's CWD, so a misbehaving
  specialist could write into the main framework_source_roots;
* offers no per-agent process.log / heartbeat granularity.

This module ports Arbor's per-specialist subprocess dispatch shape
(``arbor/src/arbor/dispatch.py::_dispatch_via_cli``) to Hyperloom:

1. The runner creates a per-task git worktree under
   ``runs/specialist/<task_id>/worktree/`` (rooted at
   ``INFERENCEX_PATH`` / the first framework source root).
2. Spawns ``claude --print --output-format stream-json --verbose ...``
   with ``--add-dir <worktree>`` and ``--add-dir <session_dir>`` so the
   agent's write tools are scoped to its own workspace.
3. The agent writes ``specialist_done.json`` + optional patches under
   ``worktree/patches/`` as its exit signal (in addition to / instead of
   the in-process ``emit_intent`` MCP path).
4. SpecialistRunner harvests either signal — done-file OR captured
   ``specialist_done`` intent — to build the final
   :class:`SpecialistRunResult`.

The runner keeps the in-process Backend path (``backend_factory`` arg)
intact for unit tests that drive specialists with ``MockBackend``;
production cli wiring uses the subprocess path via
:class:`SpecialistSubprocessConfig`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecialistSubprocessConfig:
    """Static config for spawning claude subprocesses per specialist.

    Captured once at CLI boot from operator flags / env; the same instance
    is reused for every specialist dispatch in the session. Per-task
    state (workspace, worktree, gap, domain) is passed at run time via
    :meth:`SpecialistSubprocessDispatcher.run`.
    """

    claude_executable: str = "claude"
    """Path / name of the claude CLI binary. Default looks it up on $PATH."""

    model: str = ""
    """Claude model id (e.g. ``claude-opus-4-7``). Empty = SDK default."""

    permission_mode: str = "auto"
    """claude-cli ``--permission-mode``. ``auto`` matches Arbor."""

    framework_source_roots: tuple[str, ...] = ()
    """Roots used to seed ``git worktree add`` and as ``--add-dir`` parents.

    The first existing root becomes the worktree base; the rest are
    surfaced as additional ``--add-dir`` entries so the agent can read
    them (Read / Grep / Glob) without being able to write through them
    (writes still need the worktree).
    """

    mcp_config_path: str | None = None
    """Optional path to a JSON file holding ``{"mcpServers": {...}}``."""

    output_format: str = "stream-json"
    """``--output-format`` flag. ``stream-json`` matches Arbor + lets
    a future reaper inspect tool_use blocks if needed."""

    extra_claude_args: tuple[str, ...] = ()
    """Operator escape hatch — appended verbatim to the claude command."""

    per_turn_max_seconds: float = 600.0
    """Wall-clock cap PER LLM turn. Multiplied by ``max_turns`` to get
    the per-task hard timeout; the dispatcher kills the subprocess past
    that wall-clock."""

    poll_interval_seconds: float = 5.0
    """How often the reaper polls done.json / process exit / heartbeat."""

    heartbeat_stale_seconds: float = 300.0
    """If the agent stops writing heartbeat.json for this long, treat
    it as stale and kill the subprocess (matches Arbor's 5-min cap)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class SpecialistSubprocessResult:
    """Outcome of one specialist subprocess invocation.

    The SpecialistRunner translates this into its own
    :class:`SpecialistRunResult`; this type stays internal to the
    subprocess dispatch shape so the per-protocol bits don't leak.
    """

    done_payload: dict[str, Any] | None = None
    """Parsed ``specialist_done.json`` content, or None when the file
    never appeared. The runner falls back to ``build_empty_specialist_done``
    in the None case."""

    exit_code: int | None = None
    """Subprocess exit code (None when killed before exit)."""

    elapsed_seconds: float = 0.0

    timed_out: bool = False
    """True when the dispatcher killed the subprocess past the
    ``max_turns * per_turn_max_seconds`` ceiling."""

    stale_heartbeat: bool = False
    """True when the heartbeat went stale and the dispatcher killed
    the subprocess."""

    process_log_path: str = ""
    patches: list[str] = field(default_factory=list)
    """Worktree-relative patch paths discovered under
    ``runs/specialist/<task_id>/worktree/patches/``."""

    error: str = ""


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------
def _pick_worktree_base(roots: tuple[str, ...]) -> Path | None:
    """Return the first ``roots`` entry that looks like a git checkout.

    Args:
        roots (tuple[str, ...]): Candidate directory paths to probe, in
            priority order.

    Returns:
        Path | None: The first existing directory with a ``.git`` marker,
            or None when none qualify — the runner then runs the specialist
            without an isolated worktree (``--add-dir <workspace>`` only,
            still safer than the in-process path).
    """
    for r in roots:
        p = Path(r)
        if not p.is_dir():
            continue
        # Either a worktree (``.git`` is a file pointing to gitdir) or a
        # bare-ish repo (``.git`` is a dir).
        git_marker = p / ".git"
        if git_marker.exists():
            return p
    return None


def _setup_worktree(
    base: Path, worktree_path: Path, branch: str,
) -> tuple[Path | None, str]:
    """Create a fresh git worktree at ``worktree_path`` branched off
    ``base``'s HEAD.

    Best-effort: on any git error we return ``(None, err)`` so the caller
    can decide whether to abort (PR-A2 default: proceed without isolation)
    or hard-fail. An existing path is logged and reused.

    Args:
        base (Path): Git checkout the worktree is branched off.
        worktree_path (Path): Target directory for the new worktree.
        branch (str): Branch name to create with ``git worktree add -b``.

    Returns:
        tuple[Path | None, str]: ``(worktree_dir, error_message)`` —
            ``worktree_dir`` is None on failure with the reason in the
            string; on success the string is empty.
    """
    if worktree_path.exists():
        # Resume / retry: try to remove if stale; safer to just reuse
        # if it's already a worktree. The agent's iron rules tell it
        # to start fresh, so a stale worktree from a previous attempt
        # is rare. We log + reuse.
        log.warning(
            "specialist worktree already exists at %s; reusing",
            worktree_path,
        )
        return worktree_path, ""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "-C", str(base), "worktree", "add",
        "-b", branch, str(worktree_path),
    ]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git worktree add failed to spawn: {exc!r}"
    if cp.returncode != 0:
        return None, (
            f"git worktree add rc={cp.returncode}: "
            f"stderr={cp.stderr.strip()[:400]!r}"
        )
    return worktree_path, ""


def _teardown_worktree(base: Path | None, worktree_path: Path) -> None:
    """Best-effort cleanup of a specialist worktree.

    Called only by the runner on the REVERT / synth-empty path. The
    KEEP path leaves the worktree in place so ``integrate_patch`` can
    pull patches out of it. Tries ``git worktree remove`` first, then
    falls back to ``rmtree`` if the directory survives.

    Args:
        base (Path | None): The base git checkout, used for
            ``git worktree remove``; None / missing repo skips the git step.
        worktree_path (Path): The worktree directory to remove.
    """
    if not worktree_path.exists():
        return
    if base is not None and (base / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(base), "worktree", "remove", "--force",
                 str(worktree_path)],
                capture_output=True, text=True, timeout=30.0, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    # Fall back to plain rm -rf if the worktree dir survived (e.g.
    # base repo went away).
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class SpecialistSubprocessDispatcher:
    """Spawn + reap one claude subprocess for a specialist task.

    Designed to be reusable across many specialist tasks in a session.
    The dispatcher owns no per-task state — every :meth:`run` call
    receives the task's workspace + prompts and returns a
    :class:`SpecialistSubprocessResult`.
    """

    def __init__(self, config: SpecialistSubprocessConfig):
        """Store the static spawn config for reuse across dispatches.

        Args:
            config (SpecialistSubprocessConfig): Session-wide config
                captured at CLI boot; reused for every :meth:`run` call.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        *,
        task_id: str,
        workspace: Path,
        worktree: Path | None,
        worktree_base: Path | None,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: tuple[str, ...],
        max_turns: int,
    ) -> SpecialistSubprocessResult:
        """Spawn a claude subprocess, reap it, return the parsed result.

        Args:
            task_id (str): Task identifier used for logging / workspace
                layout.
            workspace (Path): ``runs/specialist/<task_id>/`` — where
                prompt.md, process.log, heartbeat.json, and
                specialist_done.json live.
            worktree (Path | None): Per-task git worktree (None when
                worktree setup failed; the dispatcher still spawns claude
                but the agent has no write-isolated tree, only
                ``--add-dir <workspace>``).
            worktree_base (Path | None): The base checkout the worktree was
                branched off (used by callers for teardown).
            system_prompt (str): System prompt assembled by
                :func:`specialist_prompt_builder.build_specialist_prompts`.
            user_prompt (str): User prompt from the same builder.
            allowed_tools (tuple[str, ...]): Per-task tool whitelist
                (post-:meth:`SpecialistRunner._resolve_tools`).
            max_turns (int): Hard cap on LLM turns; multiplied by the
                config's ``per_turn_max_seconds`` for the wall-clock ceiling.

        Returns:
            SpecialistSubprocessResult: Parsed outcome — done payload (if
                any), exit code, timing, timeout / stale-heartbeat flags,
                process log path, and discovered patches.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        prompt_file = workspace / "prompt.md"
        process_log = workspace / "process.log"
        # Where to look for ``specialist_done.json``.
        #
        # The specialist prompt
        # (``specialist_prompt_builder._section_output_protocol``) tells
        # the agent to ``Write {workspace_path}/specialist_done.json``,
        # and ``SpecialistRunner._prepare`` sets
        # ``workspace_path = worktree or workspace``. So when a per-task
        # git worktree was provisioned, the canonical write target is
        # ``<worktree>/specialist_done.json``; without a worktree it
        # collapses to ``<workspace>/specialist_done.json``.
        #
        # We poll both locations (worktree first when set, workspace as
        # a fallback) so:
        #   * production runs with a worktree (the path advertised in
        #     the prompt) are picked up — the original bug was that
        #     the dispatcher only polled the parent workspace and timed
        #     out into a spurious ``empty=true`` synth even though the
        #     specialist had written a full proposal_set into the
        #     worktree;
        #   * legacy / test fakes that still write the done-file at the
        #     workspace root keep working without changes.
        done_candidates: list[Path] = []
        if worktree is not None:
            done_candidates.append(worktree / "specialist_done.json")
        done_candidates.append(workspace / "specialist_done.json")
        # Primary path is what the prompt advertises; the reap loop
        # falls back to the workspace copy below.
        done_file = done_candidates[0]
        heartbeat_file = workspace / "heartbeat.json"

        # 1. Write the prompt file. We collapse system + user into a
        # single system-prompt-file because the claude CLI's
        # ``--system-prompt-file`` flag overrides the default system
        # message. The ``-p`` argument carries the "go execute" kickoff.
        combined = (
            "<!-- system_prompt -->\n"
            + system_prompt
            + "\n<!-- user_prompt -->\n"
            + user_prompt
        )
        prompt_file.write_text(combined, encoding="utf-8")

        # 2. Compose the claude command.
        cmd = self._build_claude_cmd(
            prompt_file=prompt_file,
            workspace=workspace,
            worktree=worktree,
            allowed_tools=allowed_tools,
        )

        # 3. Compose the env. Pass through the parent env (so
        # ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / CLAUDE_* propagate).
        env = os.environ.copy()
        # The specialist subprocess must NOT inherit the orchestrator's
        # HIP/CUDA_VISIBLE_DEVICES — Inv-3 (single tenant GPU) is
        # preserved by never giving specialists serving GPUs (PR-A2
        # decision: CPU-only specialists; PR-A3+ may add a GPU pool).
        for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES"):
            env.pop(var, None)

        # 4. Spawn.
        proc_started = time.monotonic()
        log_fh = process_log.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(worktree or workspace),
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as exc:
            log_fh.close()
            return SpecialistSubprocessResult(
                done_payload=None,
                exit_code=None,
                elapsed_seconds=0.0,
                process_log_path=str(process_log),
                error=f"failed to spawn claude subprocess: {exc!r}",
            )

        # 5. Reap loop — poll done-file / exit / heartbeat staleness /
        #    overall timeout.
        max_seconds = float(max_turns) * float(self.config.per_turn_max_seconds)
        try:
            outcome = await self._reap_loop(
                proc=proc,
                workspace=workspace,
                done_files=tuple(done_candidates),
                heartbeat_file=heartbeat_file,
                max_seconds=max_seconds,
                started=proc_started,
            )
        finally:
            log_fh.close()

        # 6. Patches: scan worktree/patches/ (Arbor convention).
        patches = self._collect_patches(worktree, workspace)

        # 7. Parse done.json (best-effort) — pick the first candidate
        #    that exists. Agents listening to the prompt write to the
        #    worktree; legacy fakes / the no-worktree fallback write to
        #    the workspace root.
        done_payload = None
        for cand in done_candidates:
            if cand.exists():
                done_payload = self._read_done(cand)
                if done_payload is not None:
                    done_file = cand
                    break

        return SpecialistSubprocessResult(
            done_payload=done_payload,
            exit_code=outcome["exit_code"],
            elapsed_seconds=outcome["elapsed"],
            timed_out=outcome["timed_out"],
            stale_heartbeat=outcome["stale_heartbeat"],
            process_log_path=str(process_log),
            patches=patches,
            error=outcome["error"],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_claude_cmd(
        self, *,
        prompt_file: Path,
        workspace: Path,
        worktree: Path | None,
        allowed_tools: tuple[str, ...],
    ) -> list[str]:
        """Assemble the ``claude`` CLI argv for a specialist subprocess.

        Builds the flag list (output format, permission mode, system
        prompt file, tool whitelist, mcp config, ``--add-dir`` entries,
        operator escape-hatch args). ``emit_intent`` is dropped from the
        tool whitelist since the subprocess has no in-process MCP server.

        Args:
            prompt_file (Path): Combined system+user prompt file passed via
                ``--system-prompt-file``.
            workspace (Path): Task workspace surfaced as an ``--add-dir``.
            worktree (Path | None): Write-isolated worktree surfaced as the
                first ``--add-dir`` when present.
            allowed_tools (tuple[str, ...]): Per-task tool whitelist.

        Returns:
            list[str]: The full command argv to spawn.
        """
        cfg = self.config
        cmd: list[str] = [
            cfg.claude_executable,
            "--print",
            "--output-format", cfg.output_format,
            "--verbose",
            "--permission-mode", cfg.permission_mode,
            "--system-prompt-file", str(prompt_file),
            "-p",
            "Execute the task in your system prompt. Work autonomously. "
            "Write specialist_done.json as your absolute last action.",
        ]
        if cfg.model:
            cmd.extend(["--model", cfg.model])
        # Tool whitelist. Drop bare ``emit_intent`` because the
        # subprocess has no in-process MCP server; the agent
        # exits via writing specialist_done.json.
        tools_filtered = [t for t in allowed_tools if t != "emit_intent"]
        if tools_filtered:
            cmd.extend(["--allowedTools", ",".join(tools_filtered)])
        if cfg.mcp_config_path:
            cmd.extend(["--mcp-config", cfg.mcp_config_path])
        # --add-dir order: worktree first (where writes go), workspace
        # second (where the agent dumps done.json), then framework roots.
        add_dirs: list[str] = []
        if worktree is not None:
            add_dirs.append(str(worktree))
        add_dirs.append(str(workspace))
        for r in cfg.framework_source_roots:
            if r and Path(r).is_dir() and r not in add_dirs:
                add_dirs.append(r)
        for d in add_dirs:
            cmd.extend(["--add-dir", d])
        if cfg.extra_claude_args:
            cmd.extend(list(cfg.extra_claude_args))
        return cmd

    async def _reap_loop(
        self,
        *,
        proc: subprocess.Popen,
        workspace: Path,
        done_files: tuple[Path, ...],
        heartbeat_file: Path,
        max_seconds: float,
        started: float,
    ) -> dict[str, Any]:
        """Poll the subprocess until it finishes, stalls, or times out.

        Each tick checks (in order): a done-file at any candidate path
        (graceful exit with a short grace window), natural process exit,
        heartbeat staleness, and the hard wall-clock cap. Stale / timed-out
        runs are killed via :meth:`_kill`.

        Args:
            proc (subprocess.Popen): The running claude subprocess.
            workspace (Path): Task workspace (reserved for context).
            done_files (tuple[Path, ...]): Candidate done-file paths to poll.
            heartbeat_file (Path): Heartbeat file whose mtime gauges liveness.
            max_seconds (float): Hard wall-clock ceiling for the run.
            started (float): ``time.monotonic()`` value at spawn time.

        Returns:
            dict[str, Any]: Outcome with ``exit_code``, ``elapsed``,
                ``timed_out``, ``stale_heartbeat``, and ``error`` keys.
        """
        cfg = self.config
        outcome: dict[str, Any] = {
            "exit_code": None,
            "elapsed": 0.0,
            "timed_out": False,
            "stale_heartbeat": False,
            "error": "",
        }
        last_heartbeat_seen: float = started

        while True:
            await asyncio.sleep(cfg.poll_interval_seconds)
            now = time.monotonic()
            elapsed = now - started
            outcome["elapsed"] = elapsed

            # done.json appeared at any candidate path — graceful exit
            # even if the subprocess is still cleaning up. Give it a
            # few seconds to terminate cleanly, then move on.
            if any(p.exists() for p in done_files):
                # Allow up to 30s grace for the agent to finalise output.
                grace_until = now + 30.0
                while time.monotonic() < grace_until and proc.poll() is None:
                    await asyncio.sleep(2.0)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

            # Process exited on its own.
            if proc.poll() is not None:
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = elapsed
                break

            # Heartbeat staleness check (advisory — only fires after the
            # agent has at least once written a heartbeat).
            if heartbeat_file.exists():
                try:
                    hb_mtime = heartbeat_file.stat().st_mtime
                except OSError:
                    hb_mtime = 0.0
                # convert mtime to monotonic-equivalent age in wall seconds
                age = max(0.0, time.time() - hb_mtime)
                if age <= cfg.heartbeat_stale_seconds:
                    last_heartbeat_seen = now

            if (now - last_heartbeat_seen) > cfg.heartbeat_stale_seconds:
                outcome["stale_heartbeat"] = True
                outcome["error"] = (
                    f"heartbeat stale for {now - last_heartbeat_seen:.0f}s "
                    f"(> {cfg.heartbeat_stale_seconds:.0f}s threshold)"
                )
                self._kill(proc)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

            # Hard wall-clock cap.
            if elapsed > max_seconds:
                outcome["timed_out"] = True
                outcome["error"] = (
                    f"specialist subprocess exceeded "
                    f"{max_seconds:.0f}s wall-clock cap"
                )
                self._kill(proc)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

        return outcome

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        """Tear down a claude subprocess.

        Kills the whole process group (SIGTERM, then SIGKILL after a 5s
        grace) so child SDK / curl invocations die with it. No-op if the
        process already exited.

        Args:
            proc (subprocess.Popen): The subprocess to terminate.
        """
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        # Give SIGTERM 5s before SIGKILL.
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _collect_patches(
        worktree: Path | None, workspace: Path,
    ) -> list[str]:
        """Discover patch files written by the specialist.

        Scans both ``worktree/patches/`` and ``workspace/patches/``
        (defense in depth — the agent may write to either) for ``*.patch``
        and ``*.diff`` files.

        Args:
            worktree (Path | None): Per-task worktree, or None.
            workspace (Path): Task workspace.

        Returns:
            list[str]: Discovered patch file paths.
        """
        out: list[str] = []
        for base in (worktree, workspace):
            if base is None:
                continue
            patches_dir = base / "patches"
            if not patches_dir.is_dir():
                continue
            for ext in ("*.patch", "*.diff"):
                for p in sorted(patches_dir.glob(ext)):
                    out.append(str(p))
        return out

    @staticmethod
    def _read_done(done_file: Path) -> dict[str, Any] | None:
        """Parse a ``specialist_done.json`` file, unwrapping intent envelopes.

        Tolerates missing files and parse errors (logged, returns None).
        When the file holds a ``specialist_done`` intent envelope, the
        inner ``payload`` is merged with the outer keys so callers always
        see a flat dict.

        Args:
            done_file (Path): Path to the candidate done-file.

        Returns:
            dict[str, Any] | None: The parsed (and possibly unwrapped)
                payload, or None when missing / unparseable / not a dict.
        """
        if not done_file.exists():
            return None
        try:
            data = json.loads(done_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "specialist_done.json parse failed at %s: %r", done_file, exc,
            )
            return None
        if not isinstance(data, dict):
            log.warning(
                "specialist_done.json at %s is not a dict (%r); ignoring",
                done_file, type(data).__name__,
            )
            return None
        if (
            str(data.get("intent_type") or "") == "specialist_done"
            and isinstance(data.get("payload"), dict)
        ):
            inner = data["payload"]
            merged: dict[str, Any] = {}
            for k, v in data.items():
                if k in ("intent_type", "payload"):
                    continue
                merged[k] = v
            for k, v in inner.items():
                merged[k] = v
            log.info(
                "_read_done: unwrapped specialist_done intent envelope at %s "
                "(proposal_set_len=%d, empty=%s)",
                done_file,
                len(inner.get("proposal_set") or [])
                if isinstance(inner.get("proposal_set"), list) else 0,
                inner.get("empty"),
            )
            return merged
        return data


__all__ = [
    "SpecialistSubprocessConfig",
    "SpecialistSubprocessDispatcher",
    "SpecialistSubprocessResult",
    "_pick_worktree_base",
    "_setup_worktree",
    "_teardown_worktree",
]
