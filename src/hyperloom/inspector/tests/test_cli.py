# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inspector.cli import resolve_session_dir, run_inspect


def _session(path, mtime):
    path.mkdir(parents=True)
    (path / "state.json").write_text("{}")
    os.utime(path / "state.json", (mtime, mtime))
    return path


def test_session_dir_is_used_as_is(tmp_path):
    s = _session(tmp_path / "s1", 1000)
    assert resolve_session_dir(s) == s


def test_model_dir_picks_newest_session(tmp_path):
    _session(tmp_path / "MiMo" / "sessions" / "20260925T105719Z-old", 1000)
    new = _session(tmp_path / "MiMo" / "sessions" / "20260925T143055Z-new", 2000)
    _session(tmp_path / "MiMo" / ".hidden" / "x", 3000)
    assert resolve_session_dir(tmp_path / "MiMo") == new
    assert resolve_session_dir(tmp_path) == new


def test_a_session_just_started_is_the_newest(tmp_path):
    _session(tmp_path / "m" / "20260925T105719Z-old", 1000)
    started = tmp_path / "m" / "20260925T143055Z-new"
    started.mkdir(parents=True)
    (started / "manifest.json").write_text("{}")
    os.utime(started / "manifest.json", (2000, 2000))
    assert resolve_session_dir(tmp_path) == started


def test_no_session_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_session_dir(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        resolve_session_dir(tmp_path / "missing")


def test_parser_defaults_and_missing_session(tmp_path, capsys):
    args = _build_parser().parse_args(["inspect", "--session-dir", str(tmp_path)])
    assert (args.command, args.host, args.port, args.replay, args.speed) == ("inspect", "127.0.0.1", 8765, False, 1.0)
    assert (args.history_dir, args.no_history) == (None, False)
    assert run_inspect(args) == 2
    assert "no session" in capsys.readouterr().err


def test_module_entry_point_shares_the_options(tmp_path, capsys):
    from hyperloom.inspector.__main__ import main

    assert main(["--session-dir", str(tmp_path), "--port", "0"]) == 2
    assert "no session" in capsys.readouterr().err


def test_optimizer_cli_runs_the_inspect_subcommand(tmp_path, capsys):
    from hyperloom.inference_optimizer.cli import main

    assert main(["inspect", "--session-dir", str(tmp_path), "--port", "0", "-v"]) == 2
    assert "no session" in capsys.readouterr().err


def test_hyperloom_command_runs_the_inspect_subcommand(tmp_path, capsys):
    from hyperloom.cli import main

    assert main(["inspect", "--session-dir", str(tmp_path), "--port", "0"]) == 2
    assert "no session" in capsys.readouterr().err


def test_hyperloom_help_lists_the_inspect_command(capsys):
    from hyperloom.cli import main

    assert main(["--help"]) == 0
    assert "  inspect " in capsys.readouterr().out
