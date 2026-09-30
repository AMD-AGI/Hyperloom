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

from ...enablement.recipe.credentials import sanitize_command_text
from ...enablement.recipe.setup_allowlist import is_allowlisted_setup_command
from ...enablement.recipe.setup_ledger import CMD_SANITIZED_CHARS, build_execution_row
from ..cancel_channel import cancel_scope_listener, stop_was_asked_for

log = logging.getLogger(__name__)

SETUP_CMD_MAX = 12  # cap on distinct setup commands per integrate
SETUP_CMD_TIMEOUT_SEC = 1800  # 30 min per install command


def setup_report_fields(reason: str, setup_result: dict[str, Any]) -> dict[str, Any]:
    """The ``reason`` and ``setup_commands_*`` fields of a round's result.

    Allowlist rejections are named in the reason, not only logged. A rejection
    that was only a ``log.warning`` left downstream with the round's outcome and
    no link to the cause, so the same proposal was re-authored and re-dropped
    until the budget ran out.

    Args:
        reason: The round's own reason text.
        setup_result: The :func:`run_setup_commands` result.
    """
    applied = list(setup_result.get("applied") or [])
    skipped = list(setup_result.get("skipped") or [])
    if skipped:
        note = (
            f"{len(skipped)} setup command(s) were REJECTED by the install-only allowlist "
            f"and never ran: {'; '.join(skipped)}"
        )
        reason = f"{reason} ({note})" if reason else note
    return {"reason": reason, "setup_commands_applied": applied, "setup_commands_skipped": skipped}


def resolve_setup_commands(*, params: dict[str, Any], done_payload: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Resolve the ordered, deduped setup commands to replay, each with its source.

    Base commands stacked from prior rounds (``params['enablement_setup_commands']``)
    come first as ``"inherited"``, then the current specialist's
    ``specialist_done.setup_commands`` as ``"proposed"``; a command in both stays
    ``"inherited"``. Non-list sources and blank entries are dropped, and the list
    is capped at :data:`SETUP_CMD_MAX`.

    Returns:
        ``(cmd, source)`` pairs in replay order (pre-allowlist).
    """
    resolved: dict[str, str] = {}
    for source, raw in (
        ("inherited", params.get("enablement_setup_commands")),
        ("proposed", (done_payload or {}).get("setup_commands")),
    ):
        for item in raw if isinstance(raw, list) else []:
            cmd = str(item or "").strip()
            if cmd:
                resolved.setdefault(cmd, source)
    return list(resolved.items())[:SETUP_CMD_MAX]


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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}\n(rc={proc.returncode})\n\n")
    except OSError as exc:
        log.warning("integrate_patch: enablement setup output not written to %s (%s)", log_path, exc)
    if proc.returncode != 0:
        log.warning("integrate_patch: enablement setup rc=%d for: %s", proc.returncode, cmd)
    return proc.returncode == 0


def run_setup_commands(
    commands: list[tuple[str, str]],
    *,
    cwd: Path,
    log_dir: Path,
    round_task_id: str,
    seq_start: int,
    on_execution: Callable[[dict[str, Any]], None],
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
        commands: :func:`resolve_setup_commands` output.
        cwd: Working directory for the commands.
        log_dir: Directory to write ``enablement_setup.log`` into.
        round_task_id: The round the executions belong to.
        seq_start: Highest ledger ``seq`` already durable, so occurrence
            identity stays monotonic across rounds.
        on_execution: Persists one ledger row as its command finishes.

    Returns:
        dict[str, Any]: ``{"applied", "skipped", "failed", "executions"}`` where
        ``applied`` are the allowlisted commands that ran (rc==0), ``skipped``
        the rejected ones in sanitized form, and ``executions`` one ledger row
        per ATTEMPTED command.
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    executions: list[dict[str, Any]] = []
    log_path = log_dir / "enablement_setup.log"
    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    def _record(cmd: str, index: int, source: str, outcome: str) -> None:
        row = build_execution_row(
            seq=seq_start + len(executions) + 1,
            round_task_id=round_task_id,
            cmd_index=index,
            cmd=cmd,
            source=source,
            outcome=outcome,
            env=env,
        )
        executions.append(row)
        # Persisted HERE, not after the await returns. Cancelling the await
        # unwinds the caller while this thread and its in-flight subprocess
        # carry on, so a row handed back through the return value is lost
        # for a command that actually ran -- and the ledger is what says a
        # round installed into the shared venv at all.
        on_execution(row)

    with cancel_scope_listener():
        for cmd_index, (cmd, source) in enumerate(commands):
            # Checked between commands, as upstream does: a command already
            # inside subprocess.run is not killed. Commands never reached are
            # recorded nowhere -- the ledger states what ran, not what was planned.
            if stop_was_asked_for():
                log.info("integrate_patch: enablement setup replay stopped after cancel")
                break
            if not is_allowlisted_setup_command(cmd):
                # A rejected command is LLM-written text that can hold a bearer
                # token or a credentialed URL, and this list is copied into every
                # result payload and from there into the journal, the report and
                # the KB, so it is only ever stored sanitized.
                safe_cmd = sanitize_command_text(cmd, clip=CMD_SANITIZED_CHARS)
                skipped.append(safe_cmd)
                log.warning("integrate_patch: skipping non-allowlisted enablement setup command: %s", safe_cmd)
                _record(cmd, cmd_index, source, "skipped")
                continue
            if execute_setup_command(cmd, cwd=cwd, env=env, log_path=log_path):
                applied.append(cmd)
                _record(cmd, cmd_index, source, "applied")
            else:
                failed.append(cmd)
                _record(cmd, cmd_index, source, "failed")
    return {"applied": applied, "skipped": skipped, "failed": failed, "executions": executions}
