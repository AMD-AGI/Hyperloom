# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge forge-say``: leave a message for a running forge-loop."""

from __future__ import annotations

import click

from kernelforge.loop.operator_inbox import CLEAR, post_message


@click.command("forge-say")
@click.option(
    "--workspace",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Workspace of the campaign to message",
)
@click.option("--clear", is_flag=True, help="Withdraw every earlier message instead of posting one")
@click.argument("message", nargs=-1)
def forge_say(workspace: str, clear: bool, message: tuple[str, ...]) -> None:
    """Post MESSAGE to the campaign at --workspace.

    The loop reads it at the start of its next iteration and keeps it in front of the planners, the supervisor
    and the implementers, ahead of their own rulings, until --clear withdraws it.
    """
    text = CLEAR if clear else " ".join(message).strip()
    if not text:
        raise click.UsageError("give a message, or --clear")
    if clear and message:
        raise click.UsageError("--clear takes no message")
    click.echo(f"posted {post_message(workspace, text)}")
