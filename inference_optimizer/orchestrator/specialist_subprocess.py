# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Subprocess-based specialist dispatcher — PR-A2 (Arbor-into-Hyperloom).

Per-task git worktree under ``runs/specialist/<task_id>/worktree/``, a
``claude --print --output-format stream-json`` subprocess scoped via
``--add-dir``, and a ``specialist_done.json`` (+ ``worktree/patches/``) exit
signal harvested into the final :class:`SpecialistRunResult`. The in-process
Backend path (``backend_factory``) stays for unit tests; production uses the
subprocess path.
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


# Configuration
@dataclass(frozen=True)
class SpecialistSubprocessConfig:
    """Static config for spawning claude subprocesses per specialist.

    Captured once at CLI boot; the same instance is reused for every
    specialist dispatch. Per-task state is passed at run time via
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
    read-only ``--add-dir`` entries (writes still need the worktree).
    """

    mcp_config_path: str | None = None
    """Optional path to a JSON file holding ``{"mcpServers": {...}}``."""

    output_format: str = "stream-json"
    """``--output-format`` flag; ``stream-json`` matches Arbor."""

    extra_claude_args: tuple[str, ...] = ()
    """Operator escape hatch — appended verbatim to the claude command."""

    per_turn_max_seconds: float = 600.0
    """Wall-clock cap PER LLM turn; multiplied by ``max_turns`` to get the
    per-task hard timeout."""

    poll_interval_seconds: float = 5.0
    """How often the reaper polls done.json / process exit / heartbeat."""

    heartbeat_stale_seconds: float = 300.0
    """If the agent stops writing heartbeat.json for this long, treat
    it as stale and kill the subprocess (matches Arbor's 5-min cap)."""


# Result
@dataclass
class SpecialistSubprocessResult:
    """Outcome of one specialist subprocess invocation.

    The SpecialistRunner translates this into its own
    :class:`SpecialistRunResult`.
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


# Worktree management
def _pick_worktree_base(roots: tuple[str, ...]) -> Path | None:
    """Return the first ``roots`` entry that looks like a git checkout.

    Falls back to None when none exist — the runner then runs the
    specialist without an isolated worktree.
    """
    for r in roots:
        p = Path(r)
        if not p.is_dir():
            continue
        # ``.git`` may be a file (worktree) or a dir (repo).
        git_marker = p / ".git"
        if git_marker.exists():
            return p
    return None


def _setup_worktree(
    base: Path, worktree_path: Path, branch: str,
) -> tuple[Path | None, str]:
    """Create a fresh git worktree at ``worktree_path`` branched off
    ``base``'s HEAD. Returns (worktree_dir, error_message).

    Best-effort: on git error returns ``(None, err)`` so the caller can
    proceed without isolation (PR-A2 default) or hard-fail.
    """
    if worktree_path.exists():
        # Resume / retry: reuse an existing worktree (stale ones are rare).
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

    Called only on the REVERT / synth-empty path; the KEEP path leaves the
    worktree in place so ``integrate_patch`` can pull patches out of it.
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
    # Fall back to plain rm -rf if the worktree dir survived.
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


# Dispatcher
class SpecialistSubprocessDispatcher:
    """Spawn + reap one claude subprocess for a specialist task.

    Reusable across many specialist tasks; owns no per-task state.
    """

    def __init__(self, config: SpecialistSubprocessConfig):
        self.config = config

    # Public entry point
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
        gpu_ids: tuple[int, ...] = (),
    ) -> SpecialistSubprocessResult:
        """Spawn a claude subprocess, reap it, return the parsed result.

        Parameters
        ----------
        workspace: ``runs/specialist/<task_id>/`` — where prompt.md,
            process.log, heartbeat.json, specialist_done.json live.
        worktree: per-task git worktree (None when worktree setup
            failed; the dispatcher still spawns claude but the agent
            has no write-isolated tree, only ``--add-dir <workspace>``).
        system_prompt / user_prompt: assembled by
            :func:`specialist_prompt_builder.build_specialist_prompts`.
        allowed_tools: per-task tool whitelist (post-:meth:`SpecialistRunner._resolve_tools`).
        max_turns: hard cap on LLM turns; multiplied by config's
            ``per_turn_max_seconds`` for the wall-clock ceiling.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        prompt_file = workspace / "prompt.md"
        process_log = workspace / "process.log"
        # specialist_done.json write target is ``worktree or workspace``;
        # poll worktree first (prompt-advertised path), workspace as
        # fallback for legacy/test fakes that write at the workspace root.
        done_candidates: list[Path] = []
        if worktree is not None:
            done_candidates.append(worktree / "specialist_done.json")
        done_candidates.append(workspace / "specialist_done.json")
        done_file = done_candidates[0]
        heartbeat_file = workspace / "heartbeat.json"

        # Write the prompt file (system + user collapsed into one
        # --system-prompt-file; -p carries the kickoff).
        combined = (
            "<!-- system_prompt -->\n"
            + system_prompt
            + "\n<!-- user_prompt -->\n"
            + user_prompt
        )
        prompt_file.write_text(combined, encoding="utf-8")

        cmd = self._build_claude_cmd(
            prompt_file=prompt_file,
            workspace=workspace,
            worktree=worktree,
            allowed_tools=allowed_tools,
        )

        # Compose the env (pass through parent so API keys propagate).
        env = os.environ.copy()
        if gpu_ids:
            visible = ",".join(str(g) for g in gpu_ids)
            env["HIP_VISIBLE_DEVICES"] = visible
            env["CUDA_VISIBLE_DEVICES"] = visible
            env["ROCR_VISIBLE_DEVICES"] = visible
            env["INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"] = visible
        else:
            # CPU specialists must not inherit serving GPU visibility.
            for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES"):
                env.pop(var, None)

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

        # Reap loop — poll done-file / exit / heartbeat staleness / timeout.
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

        # Patches: scan worktree/patches/ (Arbor convention).
        patches = self._collect_patches(worktree, workspace)

        # Parse done.json (best-effort) — first existing candidate.
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

    # Internals
    def _build_claude_cmd(
        self, *,
        prompt_file: Path,
        workspace: Path,
        worktree: Path | None,
        allowed_tools: tuple[str, ...],
    ) -> list[str]:
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
        # Drop ``emit_intent``: the subprocess has no in-process MCP server
        # and exits via writing specialist_done.json instead.
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
        cfg = self.config
        outcome: dict[str, Any] = {
            "exit_code": None,
            "elapsed": 0.0,
            "timed_out": False,
            "stale_heartbeat": False,
            "error": "",
        }
        last_heartbeat_seen: float = started
        # The subprocess streams stream-json (model tokens / tool calls) to
        # process.log; its mtime is a reliable "still working" signal even
        # when the agent never self-writes heartbeat.json.
        process_log = workspace / "process.log"

        while True:
            await asyncio.sleep(cfg.poll_interval_seconds)
            now = time.monotonic()
            elapsed = now - started
            outcome["elapsed"] = elapsed

            # done.json appeared — graceful exit with up to 30s grace for
            # the agent to terminate cleanly.
            if any(p.exists() for p in done_files):
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

            # Liveness check. The subprocess counts as alive if EITHER the
            # agent refreshed heartbeat.json OR it is still streaming output
            # to process.log (model tokens / tool calls). Relying on
            # heartbeat.json alone reaps productive specialists that stay in
            # a single long tool-call turn without self-writing a heartbeat
            # (common under gateway latency) — the original cause of
            # 100%-stale_heartbeat specialist failures. The hard wall-clock
            # cap below still bounds genuinely hung / runaway subprocesses.
            for activity_file in (heartbeat_file, process_log):
                try:
                    if not activity_file.exists():
                        continue
                    a_mtime = activity_file.stat().st_mtime
                except OSError:
                    continue
                if max(0.0, time.time() - a_mtime) <= cfg.heartbeat_stale_seconds:
                    last_heartbeat_seen = now
                    break

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
        """Tear down a claude subprocess. Kills the whole process group
        so child SDK / curl invocations die with it."""
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
        """Return sorted patch paths discovered under worktree/patches/
        and workspace/patches/ (defense in depth — the agent may write
        to either)."""
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
