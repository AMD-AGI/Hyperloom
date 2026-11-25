# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""D7 over llama.cpp: a native server has no Python parser, so a launch is held to the flags its own ``--help``
lists. The fixture is the first line of each option in the ``--help`` of llama-server 1.0.0-dev (build 12439)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from inference_suite.contracts.launch import Launch, LaunchArg, LayerSource
from inference_suite.parser_check import HelpOption, RefusedFlag, check, help_options
from inference_suite.tests.test_compose import _llamacpp_compose

_HELP = Path(__file__).parent / "fixtures" / "llamacpp" / "llama-server-help.txt"


def _server(root: Path) -> dict[str, str]:
    """A stand-in ``llama-server`` that prints the captured ``--help``."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    program = bin_dir / "llama-server"
    program.write_text(f'#!/bin/sh\n/bin/cat "{_HELP}"\n')
    program.chmod(0o755)
    return {"PATH": str(bin_dir)}


def _with(launch: Launch, *args: tuple[str, tuple[str, ...]]) -> Launch:
    added = tuple(LaunchArg(flag, values, LayerSource.CANDIDATE) for flag, values in args)
    return replace(launch, args=launch.args + added)


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    (tmp_path / "m.gguf").write_text("")
    composed = _llamacpp_compose(tmp_path)
    assert isinstance(composed, Launch), composed
    # The shared compose fixture's user, candidate and runner layers carry sglang flags.
    own = (LayerSource.BASE, LayerSource.GUARD)
    return replace(composed, args=tuple(arg for arg in composed.args if arg.source in own))


_SMALL_HELP = """\
----- common params -----

-t,    --threads N                      number of CPU threads to use during generation (default:
                                        -1)
                                        (env: SERVER_ARG_THREADS)
--jinja, --no-jinja                     whether to use the chat template engine
--old-size N                            the argument has been removed. use --ctx-size
"""


def test_an_entry_is_one_option_with_every_spelling_and_its_wrapped_description_joined():
    threads, jinja, old = help_options(_SMALL_HELP)

    assert threads == HelpOption(
        spellings=("-t", "--threads"),
        help="N number of CPU threads to use during generation (default: -1) (env: SERVER_ARG_THREADS)",
        removed=False,
    )
    assert jinja.spellings == ("--jinja", "--no-jinja") and not jinja.removed
    assert old == HelpOption(("--old-size",), "N the argument has been removed. use --ctx-size", removed=True)


def test_a_launch_of_listed_flags_passes(tmp_path, launch):
    env = _server(tmp_path)
    assert check(_with(launch, ("-np", ("8",)), ("-kvu", ())), "unused", env, cwd=tmp_path) is None


def test_flags_the_server_does_not_take_are_refused_naming_their_layer(tmp_path, launch):
    env = _server(tmp_path)
    refusal = check(_with(launch, ("--no-mmap", ()), ("--mlock", ()), ("-np", ("8",))), "unused", env, cwd=tmp_path)

    assert refusal is not None
    assert refusal.flags == (
        RefusedFlag("--no-mmap", LayerSource.CANDIDATE),
        RefusedFlag("--mlock", LayerSource.CANDIDATE),
    )
    assert "--mlock" in refusal.detail


def test_a_removed_flag_is_refused_with_what_replaced_it(tmp_path, launch):
    env = _server(tmp_path)
    refusal = check(_with(launch, ("--draft-max", ("8",))), "unused", env, cwd=tmp_path)

    assert refusal is not None
    assert refusal.flags == (RefusedFlag("--draft-max", LayerSource.CANDIDATE),)
    assert "--spec-draft-n-max" in refusal.detail
