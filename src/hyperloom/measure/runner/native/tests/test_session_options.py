# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A single-branch session keeps the option list of the server it built, read once against its frozen base launch."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure.runner.native.server_options import SERVER_OPTIONS_FILE
from hyperloom.measure.runner.native.session_options import record_session_options, session_options
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base
from inference_suite import FrameworkProbeError

_HELP = """\
-c,    --ctx-size N                     size of the prompt context (default: 0)
--jinja, --no-jinja                     whether to use the chat template engine
--keep N                                number of tokens to keep from the initial prompt
"""


def _server(root: Path, text: str = _HELP, *, code: int = 0) -> dict[str, str]:
    """A stand-in ``llama-server`` on a ``PATH`` of its own that prints ``text`` for any argument and names the
    ``SERVER_LIBS`` it ran with."""
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    (root / "help.txt").write_text(text, encoding="utf-8")
    program = bin_dir / "llama-server"
    ran = f'echo "$SERVER_LIBS" > "{root / "ran"}"'
    program.write_text(f'#!/bin/sh\n/bin/cat "{root / "help.txt"}"\n{ran}\nexit {code}\n', encoding="utf-8")
    program.chmod(0o755)
    return {"PATH": str(bin_dir)}


def _session(root: Path, args: tuple[str, ...], env: dict[str, str] | None = None) -> Path:
    session = root / "session"
    session.mkdir()
    freeze_base(session, replace(base_launch(), framework="llamacpp", args=args, env=env or {}, golden_al=None))
    return session


def test_an_option_the_runner_sets_on_every_launch_is_not_left_unset(tmp_path: Path) -> None:
    session = _session(tmp_path, ())
    listed = _HELP + "-m,    --model FNAME                    model path to load\n--port PORT   port to listen on\n"

    record_session_options(session, "llamacpp", _server(tmp_path, listed))

    inventory = session_options(session).inventory
    assert inventory.option("--model").set_by_base and inventory.option("--port").set_by_base
    assert [option.name for option in inventory.unset] == ["--ctx-size", "--no-jinja", "--keep"]


def test_the_server_is_asked_under_the_environment_its_launch_sets(tmp_path: Path) -> None:
    session = _session(tmp_path, (), {"SERVER_LIBS": "/opt/libs"})

    record_session_options(session, "llamacpp", {**_server(tmp_path), "SERVER_LIBS": "/usr/lib"})

    assert (tmp_path / "ran").read_text().strip() == "/opt/libs"


def test_a_later_leg_keeps_the_list_the_session_started_with(tmp_path: Path) -> None:
    session = _session(tmp_path, ())
    record_session_options(session, "llamacpp", _server(tmp_path))

    again = record_session_options(session, "llamacpp", _server(tmp_path, "--other N   another option\n"))

    assert again == session / SERVER_OPTIONS_FILE
    unset = session_options(session).inventory.unset
    assert [option.name for option in unset] == ["--ctx-size", "--no-jinja", "--keep"]


def test_a_framework_whose_server_is_the_image_s_keeps_no_list(tmp_path: Path) -> None:
    session = _session(tmp_path, ())

    assert record_session_options(session, "sglang", _server(tmp_path)) is None
    assert session_options(session) is None
    assert not (session / SERVER_OPTIONS_FILE).exists()


def test_a_server_that_prints_no_help_is_the_probe_s_failure_and_leaves_no_list(tmp_path: Path) -> None:
    session = _session(tmp_path, ())

    with pytest.raises(FrameworkProbeError):
        record_session_options(session, "llamacpp", _server(tmp_path, code=3))

    assert session_options(session) is None
