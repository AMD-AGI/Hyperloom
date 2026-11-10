# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``HYPERLOOM_LEDGER_DIR`` that is not an absolute path ends the commands that open a session's coordinator database
with one line and exit code 2, at the entry, before anything is read or started: not a traceback from deep in a run."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import hyperloom.cli as hyperloom_cli
from hyperloom.inference_optimizer.session.paths import ENV_LEDGER_DIR

_ERROR = "ERROR: HYPERLOOM_LEDGER_DIR must be an absolute path, not 'relative/ledgers'\n"
_COMMANDS = [
    pytest.param(["optimize", "--model", "/models/m", "--framework", "sglang"], id="optimize"),
    pytest.param(["optimize", "--resume-from", "/workspace/s"], id="optimize-resume"),
    pytest.param(["recover", "--session-dir", "/workspace/s"], id="recover"),
    pytest.param(["session", "events", "/workspace/s"], id="session-events"),
]


@pytest.mark.parametrize("argv", _COMMANDS)
def test_a_relative_ledger_directory_is_one_clean_line_and_exit_code_2_at_the_entry(monkeypatch, capsys, argv):
    monkeypatch.setenv(ENV_LEDGER_DIR, "relative/ledgers")
    assert hyperloom_cli.main(argv) == 2
    seen = capsys.readouterr()
    assert (seen.out, seen.err) == ("", _ERROR)


def test_the_console_entry_exits_2_with_the_line_and_no_traceback_in_a_real_process():
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\nfrom hyperloom.cli import main\nsys.exit(main(sys.argv[1:]))",
            "optimize",
            "--model",
            "/models/m",
            "--framework",
            "sglang",
        ],
        env={**os.environ, ENV_LEDGER_DIR: "relative/ledgers"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert (done.returncode, done.stdout, done.stderr) == (2, "", _ERROR)


def test_an_absolute_ledger_directory_is_not_this_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    assert (
        hyperloom_cli.main(["session", "events", str(tmp_path)]) == 2
    )  # no database in that directory: the tool's own message
    assert "HYPERLOOM_LEDGER_DIR" not in capsys.readouterr().err


def _a_session_that_ran_with_a_credential(workspace) -> object:
    """A resumable session (manifest and state) whose pins hold a credential and that counted a tick."""
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer.session.manifest import write_manifest
    from hyperloom.orchestrator.state.shared_state import SharedState

    session = workspace / "m" / "20261110T000000Z-ab12cd34"
    session.mkdir(parents=True)
    argv = ["optimize", "--model", "/models/m", "--framework", "sglang"]
    write_manifest(session, args=_build_parser().parse_args(argv), session_id="m_20261110T000000Z_ab12cd34")
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": "hf_live_value_123"}
    state.tick = 3
    state.save(session)
    return session


def _in_a_new_container(tmp_path, monkeypatch, workspace):
    """A session that ran with a credential in one container, seen from another that has no ledger for it."""
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv(ENV_LEDGER_DIR, raising=False)
    first = tmp_path / "container-1"
    first.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(first))
    session = _a_session_that_ran_with_a_credential(workspace)
    second = tmp_path / "container-2"
    second.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(second))
    return session


def _one_line_naming_the_ledger_directory(error: str) -> bool:
    lines = error.strip().splitlines()
    return len(lines) == 1 and "HYPERLOOM_LEDGER_DIR" in lines[0] and "start a new session" in lines[0]


def test_a_resume_without_a_ledger_is_one_line_and_exit_code_2_from_the_resume_start_of_the_cli(
    tmp_path, monkeypatch, capsys
):
    """``optimize --resume-from`` on a session that holds a credential and has run, in a container with no ledger for
    it: the start of the resume ends the run with the guard's one line and exit code 2, before anything is started."""
    import argparse

    from hyperloom.inference_optimizer.cli import _open_resumed_session

    session = _in_a_new_container(tmp_path, monkeypatch, tmp_path / "workspace")
    with pytest.raises(SystemExit) as ended:
        _open_resumed_session(argparse.Namespace(resume_from=str(session), session_dir=None))
    seen = capsys.readouterr()
    assert ended.value.code == 2 and seen.out == ""
    assert seen.err.startswith("ERROR: --resume-from this session holds a credential")
    assert _one_line_naming_the_ledger_directory(seen.err)


def test_a_refusal_from_deep_in_a_run_is_ended_at_the_entry_the_same_way(tmp_path, monkeypatch, capsys):
    """Should the coordinator be reached without the guard (the credential is only exported, so the guard saw none), it
    refuses the same resume itself; ``hyperloom optimize`` turns that into the same line and exit code 2."""
    import types

    from hyperloom.orchestrator.loop.coordinator import Coordinator, default_role_registry

    session = _in_a_new_container(tmp_path, monkeypatch, tmp_path / "workspace")
    backends = {name: object() for name in default_role_registry()}

    def run(argv):
        Coordinator(session, backends=backends)
        return 0

    monkeypatch.setitem(sys.modules, "ledger_probe", types.SimpleNamespace(main=run))
    monkeypatch.setitem(hyperloom_cli._COMMANDS, "optimize", ("ledger_probe", (), "stand-in for the optimizer"))
    assert hyperloom_cli.main(["optimize"]) == 2
    seen = capsys.readouterr()
    assert seen.out == "" and seen.err.startswith("ERROR: this session holds a credential")
    assert _one_line_naming_the_ledger_directory(seen.err)
