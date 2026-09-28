# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay of a round's enablement setup commands (installs) before its patches apply and its server boots."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...enablement.recipe.setup_allowlist import is_allowlisted_setup_command, sanitize_setup_command
from ...enablement.recipe.setup_ledger import build_execution_row
from ..cancel_channel import cancel_scope_listener, stop_was_asked_for

log = logging.getLogger(__name__)

SETUP_CMD_MAX = 12  # cap on distinct setup commands per integrate
SETUP_CMD_TIMEOUT_SEC = 1800  # 30 min per install command


def with_skipped_setup_reason(reason: str, setup_result: dict[str, Any]) -> str:
    """Append the allowlist rejections to a round's ``reason``.

    A rejected setup command was only ever a ``log.warning``. Downstream saw the
    round's outcome with no link to the cause, so the same authoring attempt was
    re-dispatched until the budget ran out -- each round proposing the same fix
    and each round having it silently dropped. Naming the rejection in the reason
    is what lets the next round (or an operator) see that the proposal was never
    the problem.

    Args:
        reason: The round's existing reason text.
        setup_result: The :func:`run_setup_commands` result.

    Returns:
        ``reason`` unchanged when nothing was rejected, else ``reason`` with a
        one-line summary of the rejected commands appended.
    """
    # ``run_setup_commands`` already stores the sanitised form, so for every
    # production caller this is a no-op. Applied again anyway: the lesson of the
    # gap this closes is that a safety step placed at the call sites protects
    # the call sites that exist, and the sanitiser is idempotent.
    skipped = [sanitize_setup_command(c) for c in (setup_result.get("skipped") or []) if str(c).strip()]
    if not skipped:
        return reason
    listed = "; ".join(skipped[:SETUP_CMD_MAX])
    if len(skipped) > SETUP_CMD_MAX:
        listed += f"; (+{len(skipped) - SETUP_CMD_MAX} more)"
    note = f"{len(skipped)} setup command(s) were REJECTED by the install-only allowlist and never ran: {listed}"
    return f"{reason} ({note})" if reason else note


def resolve_setup_commands(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> list[str]:
    """Resolve the ordered, deduped enablement setup commands to replay.

    Sources (in order; deduped preserving first occurrence): base commands
    stacked from prior rounds (``params['enablement_setup_commands']``) then the
    current specialist's ``specialist_done.setup_commands``. Non-string / blank
    entries are dropped; the list is capped at :data:`SETUP_CMD_MAX`.

    Args:
        params: The integrate_patch task params.
        done_payload: The specialist ``specialist_done`` payload (may be None).

    Returns:
        list[str]: Ordered unique candidate setup commands (pre-allowlist).
    """
    out: list[str] = []
    seen: set[str] = set()
    sources: list[Any] = []
    base = params.get("enablement_setup_commands")
    if isinstance(base, list):
        sources.extend(base)
    if isinstance(done_payload, dict):
        dp = done_payload.get("setup_commands")
        if isinstance(dp, list):
            sources.extend(dp)
    for c in sources:
        s = str(c or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= SETUP_CMD_MAX:
            break
    return out


def setup_command_sources(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> dict[str, str]:
    """Map each candidate command to whether it was inherited or proposed here.

    A command replayed from the durable base and one this round's specialist
    proposed carry different replay meaning, and the resolved list dedups them
    into one string.
    """
    inherited = {str(c or "").strip() for c in (params.get("enablement_setup_commands") or [])}
    sources: dict[str, str] = {cmd: "inherited" for cmd in inherited if cmd}
    proposed = (done_payload or {}).get("setup_commands") or []
    for raw in proposed:
        cmd = str(raw or "").strip()
        if cmd and cmd not in sources:
            sources[cmd] = "proposed"
    return sources


def execute_setup_command(cmd: str, *, cwd: Path, env: dict[str, str], log_path: Path) -> bool:
    """Run one allowlisted setup command, appending its output to the replay log.

    Returns:
        True when the command exited zero. A non-zero install is recorded but
        does not hard-fail the integration -- the subsequent boot/gate is the
        source of truth for runnability.
    """
    log.info("integrate_patch: enablement setup replay: %s", cmd)
    try:
        proc = subprocess.run(  # nosec B602 - allowlisted install-only shell command.
            cmd,
            shell=True,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=SETUP_CMD_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("integrate_patch: enablement setup errored (%s) for: %s", type(exc).__name__, cmd)
        return False
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}\n(rc={proc.returncode})\n\n")
    except OSError:
        # Logging is best-effort.
        pass
    if proc.returncode != 0:
        log.warning("integrate_patch: enablement setup rc=%d for: %s", proc.returncode, cmd)
    return proc.returncode == 0


def run_setup_commands(
    commands: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    sources: dict[str, str] | None = None,
    round_task_id: str = "",
    seq_start: int = 0,
    on_execution: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay allowlisted enablement setup commands (installs) before boot.

    Blocking (serial ``subprocess.run``, 1800s cap); call via ``asyncio.to_thread``.
    Cancel is checked between commands; a command already in ``subprocess.run``
    is not killed. Cancelling the await unwinds integrate and does not continue
    to apply patches.

    Runs each allowlisted command non-interactively with a per-command timeout,
    appending combined output to ``<log_dir>/enablement_setup.log``. Commands
    that fail the allowlist are skipped (never executed). A non-zero install is
    recorded but does NOT hard-fail the integration — the subsequent boot/gate
    is the source of truth for runnability.

    Args:
        commands: Candidate setup commands (already deduped / capped).
        cwd: Working directory for the commands.
        log_dir: Directory to write ``enablement_setup.log`` into.

    Args (continued):
        sources: ``{cmd: "inherited"|"proposed"}`` for the ledger rows.
        round_task_id: The round the executions belong to.
        seq_start: Highest ledger ``seq`` already durable, so occurrence
            identity stays monotonic across rounds.

    Returns:
        dict[str, Any]: ``{"applied", "skipped", "failed", "executions"}`` where
        ``applied`` are the allowlisted commands that ran (rc==0) and
        ``executions`` is one ledger row per ATTEMPTED command.
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    executions: list[dict[str, Any]] = []
    if not commands:
        return {"applied": applied, "skipped": skipped, "failed": failed, "executions": executions}
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Logging is best-effort.
        pass
    log_path = log_dir / "enablement_setup.log"
    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    def _record(cmd: str, index: int, outcome: str) -> None:
        row = build_execution_row(
            seq=int(seq_start) + len(executions) + 1,
            round_task_id=round_task_id,
            cmd_index=index,
            cmd=cmd,
            source=(sources or {}).get(str(cmd).strip(), "proposed"),
            outcome=outcome,
            env=env,
        )
        executions.append(row)
        if on_execution is not None:
            # Persisted HERE, not after the await returns. Cancelling the await
            # unwinds the caller while this thread and its in-flight subprocess
            # carry on, so a row handed back through the return value is lost
            # for a command that actually ran -- and the ledger is what says a
            # round installed into the shared venv at all.
            on_execution(row)

    with cancel_scope_listener():
        for cmd_index, cmd in enumerate(commands):
            # Checked between commands, as upstream does: a command already
            # inside subprocess.run is not killed. Commands never reached are
            # recorded nowhere -- the ledger states what ran, not what was planned.
            if stop_was_asked_for():
                log.info("integrate_patch: enablement setup replay stopped after cancel")
                break
            if not is_allowlisted_setup_command(cmd):
                # Sanitised HERE, not at the reporting sites. This list is copied
                # verbatim into every result payload that carries
                # ``setup_commands_skipped``, and a rejected command is LLM-written
                # text that can hold a bearer token or a credentialed URL. Doing it
                # at the four call sites protects those four; doing it at the source
                # protects the fifth as well.
                safe_cmd = sanitize_setup_command(cmd)
                skipped.append(safe_cmd)
                # Also carried into the round's ``reason`` by
                # with_skipped_setup_reason: a warning alone left the caller with an
                # outcome and no link to the cause, so the same proposal was
                # re-authored and re-dropped until the budget ran out. The log is a
                # disk-backed surface too, so it gets the sanitised form as well.
                log.warning("integrate_patch: skipping non-allowlisted enablement setup command: %s", safe_cmd)
                _record(cmd, cmd_index, "skipped")
                continue
            if execute_setup_command(cmd, cwd=cwd, env=env, log_path=log_path):
                applied.append(cmd)
                _record(cmd, cmd_index, "applied")
            else:
                failed.append(cmd)
                _record(cmd, cmd_index, "failed")
    return {"applied": applied, "skipped": skipped, "failed": failed, "executions": executions}
