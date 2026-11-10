# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``setup_executions``: one durable row per attempted setup execution.

Recorded where the commands run rather than where the round reports, because
several exits after setup has already mutated the shared venv return without the
outcome lists -- and one returns without the enablement flag at all, which the
lane's rearm ignores entirely.

No row stores a command verbatim: this is the one field that records a *failed*
execution, whose text is in no durable field today, so keeping it verbatim would
open a credential sink the pre-existing ``setup.cmd`` exposure does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .credentials import (
    classify_credential_class,
    detect_credential_channels,
    installer_class,
    sanitize_command_text,
)
from .steps import command_digest

#: Clip for every stored sanitized command, so a single install naming a hundred
#: packages cannot crowd out its neighbours.
CMD_SANITIZED_CHARS = 160

OUTCOMES: tuple[str, ...] = ("applied", "failed")


@dataclass(frozen=True)
class SetupExecution:
    """One attempted setup command, before it is a ledger row.

    Attributes:
        seq: Monotonic execution identity; a row is never matched by its text.
        round_task_id: The round the execution belongs to.
        cmd_index: Position within the round's resolved command list.
        cmd: The verbatim command, used only for its digest and sanitized form.
        source: ``"proposed"`` (this round's own payload) or ``"inherited"``.
        outcome: One of :data:`OUTCOMES`.
    """

    seq: int
    round_task_id: str
    cmd_index: int
    cmd: str
    source: str
    outcome: str


def build_execution_row(
    execution: SetupExecution,
    env: Mapping[str, str] | None,
    fs_root: Path | str = "/",
) -> dict[str, Any]:
    """Build one durable ledger row for one attempted execution.

    Args:
        execution: The attempted command and where it ran in the round.
        env: The environment the command was given, classified by channel name.
        fs_root: Filesystem root the ambient channel locations are probed under.

    Raises:
        ValueError: When ``outcome`` is outside the recorded vocabulary.
    """
    cmd = execution.cmd
    outcome = execution.outcome
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    return {
        "seq": int(execution.seq),
        "round_task_id": str(execution.round_task_id or ""),
        "cmd_index": int(execution.cmd_index),
        "cmd_sanitized": sanitize_command_text(cmd, clip=CMD_SANITIZED_CHARS),
        "cmd_digest": command_digest(cmd),
        "source": str(execution.source),
        "outcome": outcome,
        "round_disposition": "unreported",
        "at_accepted_round": False,
        "present_at_final_launch": False,
        "replayed_at_final_launch": False,
        "credential_class": classify_credential_class(cmd),
        "credential_channels": detect_credential_channels(env, fs_root=fs_root),
        "installer": installer_class(cmd),
    }
