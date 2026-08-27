# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the forward-compatible option handling shared by the CLI entries.

The per-command behaviour lives with each command's own contract tests; what is
pinned here is the mechanism and, above all, its scope: tolerance is granted to
the two entry points a separate repository drives, and to nothing else.
"""

from __future__ import annotations

import click
from click.testing import CliRunner

from kernel_agents.cli import main
from kernel_agents.cli_forward_compat import (
    RESULT_FIELD,
    TolerantCommand,
    ignored_cli_options,
    stamp_ignored_cli_options,
)

# The entry points a consumer in another repository invokes, and therefore the
# only ones that can be handed an option from a release this one predates.
TOLERANT_COMMANDS = {"forge-loop", "forge-rewrite-by-flydsl"}


def test_only_the_cross_repo_entry_points_tolerate_unknown_options():
    tolerant = {
        name
        for name, command in main.commands.items()
        if isinstance(command, TolerantCommand)
    }

    assert tolerant == TOLERANT_COMMANDS


def test_every_other_command_still_fails_on_an_unknown_option():
    # An interactive command has a human to read the error, so a typo there must
    # stay fatal rather than silently selecting a default.
    @main.command("strict-probe")
    def probe():
        pass

    try:
        result = CliRunner().invoke(main, ["strict-probe", "--not-an-option"])
    finally:
        del main.commands["strict-probe"]

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_unknown_options_are_dropped_and_reported():
    @main.command("tolerance-probe", cls=TolerantCommand)
    @click.option("--declared", default="")
    def probe(declared):
        click.echo(f"declared={declared}")
        click.echo(f"ignored={ignored_cli_options()}")

    try:
        result = CliRunner().invoke(
            main,
            ["tolerance-probe", "--declared", "kept", "--undeclared", "dropped"],
        )
    finally:
        del main.commands["tolerance-probe"]

    assert result.exit_code == 0
    # The declared option still binds; only the unknown pair is removed.
    assert "declared=kept" in result.output
    assert "ignored=['--undeclared', 'dropped']" in result.output
    assert "--undeclared" in result.stderr


def test_shell_completion_parsing_stays_silent():
    """Completion parses the same argv; a warning there would corrupt its output."""

    @main.command("silent-probe", cls=TolerantCommand)
    def probe():
        pass

    command = main.commands["silent-probe"]
    try:
        ctx = click.Context(
            command,
            resilient_parsing=True,
            ignore_unknown_options=True,
            allow_extra_args=True,
        )
        command.parse_args(ctx, ["--undeclared"])
    finally:
        del main.commands["silent-probe"]

    assert ctx.meta["kernel_agents.ignored_cli_options"] == ["--undeclared"]


def test_ignored_options_outside_a_cli_invocation_is_empty():
    assert ignored_cli_options() == []


def test_a_conforming_call_leaves_the_result_document_untouched():
    document = {"success": True}

    stamp_ignored_cli_options(document, [])

    assert document == {"success": True}
    assert RESULT_FIELD not in document


def test_dropped_tokens_are_recorded_on_the_result_document():
    document = {"success": True}

    stamp_ignored_cli_options(document, ["--e2e-pct", "3.2"])

    assert document[RESULT_FIELD] == ["--e2e-pct", "3.2"]
