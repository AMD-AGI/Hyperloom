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

from pathlib import Path
from typing import Any, Mapping

from .credentials import (
    classify_credential_class,
    detect_credential_channels,
    installer_class,
    sanitize_command_text,
)
from .steps import command_digest

#: Clip for the sanitized form, mirroring the executor's own rejection clip so a
#: single install naming a hundred packages cannot crowd out its neighbours.
_CMD_SANITIZED_CHARS = 160

OUTCOMES: tuple[str, ...] = ("applied", "failed", "skipped")


def build_execution_row(
    *,
    seq: int,
    round_task_id: str,
    cmd_index: int,
    cmd: str,
    source: str,
    outcome: str,
    env: Mapping[str, str] | None,
    fs_root: Path | str = "/",
) -> dict[str, Any]:
    """Build one durable ledger row for one attempted execution.

    Args:
        seq: Monotonic execution identity; a row is never matched by its text.
        round_task_id: The round the execution belongs to.
        cmd_index: Position within the round's resolved command list.
        cmd: The verbatim command, used only for its digest and sanitized form.
        source: ``"proposed"`` (this round's own payload) or ``"inherited"``.
        outcome: One of :data:`OUTCOMES`.
        env: The environment the command was given, classified by channel name.
        fs_root: Filesystem root the ambient channel locations are probed under.

    Raises:
        ValueError: When ``outcome`` is outside the recorded vocabulary.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    return {
        "seq": int(seq),
        "round_task_id": str(round_task_id or ""),
        "cmd_index": int(cmd_index),
        "cmd_sanitized": sanitize_command_text(cmd, clip=_CMD_SANITIZED_CHARS),
        "cmd_digest": command_digest(cmd),
        "source": str(source),
        "outcome": outcome,
        "round_disposition": "unreported",
        "at_accepted_round": False,
        "present_at_final_launch": False,
        "replayed_at_final_launch": False,
        "credential_class": classify_credential_class(cmd),
        "credential_channels": detect_credential_channels(env, fs_root=fs_root),
        "installer": installer_class(cmd),
    }


def mark_round_disposition(
    ledger: list[dict[str, Any]],
    *,
    round_task_id: str,
    disposition: str,
    accepted: bool,
) -> list[dict[str, Any]]:
    """Record a round's outcome onto its rows, its membership and its presence.

    Presence is per occurrence and succession is a separate fact: an entry *is*
    an entry of the accepted round, while another occurrence of the same digest
    merely reproduces its effect. Keying both off the command string would let a
    failed occurrence be certified by a later successful one.

    Only one round terminates the lane, so an accepted round takes presence away
    from whichever earlier round held it: leaving it standing would let a stale
    row answer for a command the validated launch never ran.
    """
    rows = [dict(row) for row in ledger]
    for row in rows:
        own = str(row.get("round_task_id") or "") == str(round_task_id or "")
        if own:
            row["round_disposition"] = str(disposition)
        if own or accepted:
            # Membership, not effect: a round whose every reached command failed
            # is still the round the lane accepted.
            row["at_accepted_round"] = bool(own and accepted)
            row["present_at_final_launch"] = bool(own and accepted and str(row.get("outcome")) == "applied")
    if not accepted:
        return rows
    replayed = {str(row.get("cmd_digest") or "") for row in rows if row.get("present_at_final_launch")}
    for row in rows:
        row["replayed_at_final_launch"] = str(row.get("cmd_digest") or "") in replayed
    return rows
