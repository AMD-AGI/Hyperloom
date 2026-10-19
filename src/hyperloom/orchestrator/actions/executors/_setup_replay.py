# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay of a round's enablement setup commands (installs) before its patches apply and its server boots."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.session.session_paths import runs_dir

from ...enablement.recipe.credentials import sanitize_command_text
from ...enablement.recipe.setup_allowlist import is_allowlisted_setup_command
from ...enablement.recipe.setup_ledger import CMD_SANITIZED_CHARS, build_execution_row
from ..cancel_channel import cancel_scope_listener, stop_was_asked_for

if TYPE_CHECKING:
    from ._integrate_attempt import IntegrateAttempt

log = logging.getLogger(__name__)

SETUP_CMD_MAX = 12  # cap on distinct setup commands per integrate
SETUP_CMD_TIMEOUT_SEC = 1800  # 30 min per install command


def refused_setup_commands(commands: list[tuple[str, str]]) -> list[str]:
    """The commands the install-only allowlist refuses, in replay order.

    Sanitized because a refused command is LLM-written text that can hold a
    bearer token, and this list reaches the result payload, the journal and the
    next round's mandate.
    """
    return [
        sanitize_command_text(cmd, clip=CMD_SANITIZED_CHARS)
        for cmd, _ in commands
        if not is_allowlisted_setup_command(cmd)
    ]


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
) -> list[str]:
    """Replay enablement setup commands (installs) before boot.

    Blocking (serial ``subprocess.run``, 1800s cap); call via ``asyncio.to_thread``.
    Cancel is checked between commands; a command already in ``subprocess.run``
    is not killed. Cancelling the await unwinds integrate and does not continue
    to apply patches.

    Runs each command non-interactively with a per-command timeout, appending
    combined output to ``<log_dir>/enablement_setup.log``. A non-zero install is
    recorded but does NOT hard-fail the integration — the subsequent boot/gate
    is the source of truth for runnability.

    Args:
        commands: :func:`resolve_setup_commands` output, already cleared by
            :func:`refused_setup_commands`.
        cwd: Working directory for the commands.
        log_dir: Directory to write ``enablement_setup.log`` into.
        round_task_id: The round the executions belong to.
        seq_start: Highest ledger ``seq`` already durable, so occurrence
            identity stays monotonic across rounds.
        on_execution: Persists one ledger row as its command finishes.

    Returns:
        list[str]: The commands that ran with rc==0, in replay order. Every
        ATTEMPTED command reaches ``on_execution`` as a ledger row, which is
        where a failed one is recorded.
    """
    applied: list[str] = []
    recorded = 0
    log_path = log_dir / "enablement_setup.log"
    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    def _record(cmd: str, index: int, source: str, outcome: str) -> None:
        nonlocal recorded
        recorded += 1
        # Persisted HERE, not after the await returns: cancelling the await
        # unwinds the caller while this thread and its in-flight subprocess
        # carry on, and the ledger is what says a round installed into the
        # shared venv at all.
        on_execution(
            build_execution_row(
                seq=seq_start + recorded,
                round_task_id=round_task_id,
                cmd_index=index,
                cmd=cmd,
                source=source,
                outcome=outcome,
                env=env,
            )
        )

    with cancel_scope_listener():
        for cmd_index, (cmd, source) in enumerate(commands):
            # Checked between commands, as upstream does: a command already
            # inside subprocess.run is not killed. Commands never reached are
            # recorded nowhere -- the ledger states what ran, not what was planned.
            if stop_was_asked_for():
                log.info("integrate_patch: enablement setup replay stopped after cancel")
                break
            if execute_setup_command(cmd, cwd=cwd, env=env, log_path=log_path):
                applied.append(cmd)
                _record(cmd, cmd_index, source, "applied")
            else:
                _record(cmd, cmd_index, source, "failed")
    return applied


def _append_setup_execution(shared_state: Any, row: dict[str, Any], *, session_dir: Path) -> None:
    """Append one execution row to the durable, append-only ledger."""
    shared_state.enablement.append_setup_execution(row)
    try:
        shared_state.save(session_dir)
    except OSError:
        # The rows are on the in-memory state either way, and the rearm saves again.
        log.debug("integrate_patch: save after setup ledger append failed", exc_info=True)


async def replay_setup_commands(
    attempt: IntegrateAttempt,
    *,
    params: dict[str, Any],
    session_dir: Path,
) -> dict[str, Any] | None:
    """Replay this round's setup commands, recording on ``attempt`` the ones that ran.

    Returns an ``apply_failed`` result when any command is not package/tool
    installation, in which case none of them ran: a partial replay would boot an
    environment the round never proposed and grade its patches against it.
    ``None`` means the round is clear to apply its patches.
    """
    commands = resolve_setup_commands(params=params, done_payload=attempt.done_payload)
    refused = refused_setup_commands(commands)
    if refused:
        return {
            "status": "apply_failed",
            "enablement": True,
            "error_class": "setup_commands_refused",
            "error": (
                f"{len(refused)} setup command(s) are not package/tool installation and were "
                f"refused by the install-only allowlist; none of this round's setup commands "
                f"ran: {'; '.join(refused)}"
            ),
            "reason": "setup commands refused by the install-only allowlist",
            "setup_commands_refused": refused,
            "specialist_task_id": attempt.specialist_task_id,
            "patches_applied": [],
            "patches_reverted": [],
        }
    if commands:
        shared_state = attempt.shared_state
        attempt.setup_commands_applied = await asyncio.to_thread(
            run_setup_commands,
            commands,
            cwd=session_dir,
            log_dir=runs_dir(session_dir, "integrate_patch", attempt.task_id),
            round_task_id=attempt.specialist_task_id,
            seq_start=shared_state.enablement.last_execution_seq(),
            on_execution=lambda row: _append_setup_execution(shared_state, row, session_dir=session_dir),
        )
    return None
