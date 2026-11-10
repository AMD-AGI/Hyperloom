# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``HYPERLOOM_LEDGER_DIR`` that is not an absolute path ends the commands that open a session's coordinator database
with one line and exit code 2, at the entry, before anything is read or started: not a traceback from deep in a run."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hyperloom.cli import main
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
    assert main(argv) == 2
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
    assert main(["session", "events", str(tmp_path)]) == 2  # no database in that directory: the tool's own message
    assert "HYPERLOOM_LEDGER_DIR" not in capsys.readouterr().err
