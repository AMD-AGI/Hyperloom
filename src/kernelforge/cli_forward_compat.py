# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forward-compatible option parsing for the entry points a consumer drives.

KernelForge and its consumers ship independently, so a caller can be ahead of
the installed producer and pass an option this version never declared. Click's
default is to abort during argument parsing, which turns one stale flag into a
dead campaign: the child exits 2 before any work starts, and the caller learns
only the return code.

``TolerantCommand`` accepts such a run instead. The unrecognized tokens are
dropped, named on stderr, and recorded on the command's result document under
``ignored_cli_options``, so the mismatch is a reported fact rather than
something inferred from an exit status.

The tolerance cannot distinguish version skew from a typo. ``--max-hour 6`` is
dropped exactly like an option from a future release, and the campaign then
runs on the ``--max-hours`` default for an hour instead of six. That is the
accepted cost of not failing the run, and it is why the dropped tokens are
reported back rather than only logged: a caller that cares must read
``ignored_cli_options`` and decide for itself.
"""

from __future__ import annotations

import sys

import click

# Click shares ``Context.meta`` with the parent group, so the command body reads
# back exactly what parsing recorded.
_META_KEY = "kernelforge.ignored_cli_options"

RESULT_FIELD = "ignored_cli_options"


class TolerantCommand(click.Command):
    """A command that reports unrecognized options instead of aborting on them."""

    def __init__(self, *args, **kwargs) -> None:
        context_settings = dict(kwargs.pop("context_settings", None) or {})
        # Both settings are required: ignoring the option alone still leaves its
        # tokens as extra arguments, which click rejects on a plain Command.
        context_settings["ignore_unknown_options"] = True
        context_settings["allow_extra_args"] = True
        super().__init__(*args, context_settings=context_settings, **kwargs)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        remaining = super().parse_args(ctx, args)
        ignored = list(ctx.args)
        ctx.meta[_META_KEY] = ignored
        # Shell completion parses the same argv; warning there would corrupt it.
        if ignored and not ctx.resilient_parsing:
            print(
                f"warning: ignoring {len(ignored)} unrecognized command-line "
                f"token(s): {' '.join(ignored)}",
                file=sys.stderr,
            )
            print(
                "warning: a misspelled option is dropped the same way an unknown "
                f"one is, so this run may proceed on a default it was meant to "
                f"override; the dropped tokens are reported as "
                f"{RESULT_FIELD!r} in the result",
                file=sys.stderr,
            )
        return remaining


def ignored_cli_options() -> list[str]:
    """Return the unrecognized tokens the running invocation dropped."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return []
    return list(ctx.meta.get(_META_KEY, ()))


def stamp_ignored_cli_options(
    result: dict,
    ignored: list[str] | None = None,
) -> dict:
    """Record the dropped tokens on a result document, in place.

    Written only when something was dropped, so a consumer parsing a conforming
    call never has to learn a key that call does not produce.
    """
    dropped = ignored_cli_options() if ignored is None else list(ignored)
    if dropped:
        result[RESULT_FIELD] = dropped
    return result
