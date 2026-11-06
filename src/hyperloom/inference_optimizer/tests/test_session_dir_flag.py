# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``optimize --session-dir``, the hosted session-directory name, and the application-private archive directory."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom_common.tests.shared_accounts import other_accounts
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.breakdown import session_package
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.session import paths


_SRC = str(Path(paths.__file__).parents[3])


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.delenv(paths.ENV_CURRENT_SESSION_DIR, raising=False)
    return tmp_path


def _parse(*argv):
    return _build_parser().parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_the_flag_is_absent_by_default(workspace):
    assert _parse().session_dir is None


def test_a_session_dir_below_the_workspace_is_accepted(workspace):
    assert _parse("--session-dir", str(workspace / "m" / "s1")).session_dir == workspace / "m" / "s1"


@pytest.mark.parametrize("value", ["relative/dir", "/elsewhere/entirely", "WORKSPACE", "WORKSPACE/../out"])
def test_a_session_dir_that_is_not_below_the_workspace_is_refused(workspace, value, capsys):
    value = value.replace("WORKSPACE", str(workspace))
    with pytest.raises(SystemExit) as raised:
        _parse("--session-dir", value)
    assert raised.value.code == 2
    assert "--session-dir" in capsys.readouterr().err


@pytest.mark.parametrize("present", [["manifest.json"], ["state.json"], ["manifest.json", "state.json"]])
def test_a_directory_that_already_holds_a_session_is_refused_at_parse_time(workspace, present, capsys):
    target = workspace / "m" / "s"
    target.mkdir(parents=True)
    for name in present:
        (target / name).write_text("{}")
    with pytest.raises(SystemExit) as raised:
        _parse("--session-dir", str(target))
    assert raised.value.code == 2 and "resume it with --resume-from" in capsys.readouterr().err
    with pytest.raises(ValueError, match="already holds a session"):
        paths.make_session_dir("m", target=target)


def test_a_directory_with_only_the_launchers_files_is_accepted(workspace):
    target = workspace / "m" / "s"
    (target / "archive").mkdir(parents=True)
    assert _parse("--session-dir", str(target)).session_dir == target
    assert paths.make_session_dir("m", target=target) == target


def test_recover_keeps_its_own_session_dir_flag(workspace, tmp_path):
    args = _build_parser().parse_args(["recover", "--session-dir", "/anywhere/at/all"])
    assert str(args.session_dir) == "/anywhere/at/all"


def test_make_session_dir_creates_exactly_the_target_and_its_skeleton(workspace):
    target = workspace / "model" / "run-1"
    made = paths.make_session_dir("model", target=target)
    assert made == target
    assert (target / "storage").is_dir() and (workspace / "logs").is_dir()
    assert sorted(p.name for p in (workspace / "model").iterdir()) == ["run-1"]
    assert os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(target)


def test_make_session_dir_refuses_a_target_outside_the_workspace(workspace, tmp_path_factory):
    with pytest.raises(ValueError, match="not below the workspace"):
        paths.make_session_dir("m", target=tmp_path_factory.mktemp("other") / "s")


def test_make_session_dir_without_a_target_still_generates_a_name(workspace):
    made = paths.make_session_dir("Some/Model:X")
    assert made.parent == workspace / "Model_X"
    assert made.name.count("-") == 1


def test_resume_refuses_a_session_dir_flag(workspace, capsys):
    args = argparse.Namespace(resume_from=str(workspace / "m" / "s"), session_dir=workspace / "m" / "t")
    with pytest.raises(SystemExit) as raised:
        cli._open_resumed_session(args)
    assert raised.value.code == 2
    assert "cannot be combined with --resume-from" in capsys.readouterr().err


# -- the hosted session directory --------------------------------------------------------------------------------
_CREATED = datetime(2026, 10, 23, 8, 30, 5, tzinfo=timezone.utc)


def test_the_hosted_session_dir_is_a_function_of_the_run(workspace):
    first = paths.hosted_session_dir("/shared/models/Qwen3-8B", _CREATED, "run-1")
    assert first == paths.hosted_session_dir("/shared/models/Qwen3-8B", _CREATED, "run-1")
    assert first.parent == workspace / "Qwen3-8B"
    assert first.name.startswith("20261023T083005Z-") and len(first.name) == len("20261023T083005Z-") + 8
    assert first != paths.hosted_session_dir("/shared/models/Qwen3-8B", _CREATED, "run-2")


def test_the_hosted_session_dir_reads_the_creation_time_in_utc(workspace):
    other_zone = _CREATED.astimezone(timezone(timedelta(hours=8)))
    assert paths.hosted_session_dir("/m", other_zone, "r") == paths.hosted_session_dir("/m", _CREATED, "r")


# -- the archive directory ---------------------------------------------------------------------------------------
def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_archive_root_is_shared_by_the_runs_group_and_closed_to_everyone_else(tmp_path):
    root = paths.ensure_archive_root(tmp_path / "session")
    assert root == paths.archive_root(tmp_path / "session") == tmp_path / "session" / "archive"
    assert (_mode(root), _mode(tmp_path / "session")) == (0o2770, 0o2775)


def test_an_archive_root_an_earlier_leg_made_is_taken_as_it_is(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive").chmod(0o750)
    assert _mode(paths.ensure_archive_root(tmp_path)) == 0o750, "never chmodded"


def test_an_archive_root_any_account_can_write_is_refused_and_left_as_it_was(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive").chmod(0o777)
    with pytest.raises(PermissionError, match="can be written by any account"):
        paths.ensure_archive_root(tmp_path)
    assert _mode(tmp_path / "archive") == 0o777


@pytest.mark.parametrize("planted", ["link", "file"])
def test_an_archive_root_that_is_not_a_directory_is_refused(tmp_path, planted):
    (tmp_path / "elsewhere").mkdir()
    archive = tmp_path / "archive"
    archive.symlink_to(tmp_path / "elsewhere") if planted == "link" else archive.write_text("")
    with pytest.raises(PermissionError, match="is not a directory"):
        paths.ensure_archive_root(tmp_path)


_LEG = """
import json, sys
import session_paths
try:
    print(json.dumps(str(session_paths.ensure_archive_root(session_paths.Path(sys.argv[1])))))
except PermissionError as exc:
    print(json.dumps({"refused": str(exc)}))
"""


def test_another_accounts_archive_root_is_taken_when_this_account_can_write_it_and_refused_naming_it_when_not():
    with other_accounts() as accounts:
        accounts.add_module(Path(paths.__file__), "session_paths")
        session = accounts.shared_dir("user-data") / "session"
        root = json.loads(accounts.run(accounts.first, f"import sys\nsys.argv[1:] = [{str(session)!r}]\n{_LEG}"))
        assert (os.stat(root).st_uid, os.stat(root).st_gid, _mode(root)) == (accounts.first, accounts.group, 0o2770)

        assert (
            json.loads(accounts.run(accounts.second, f"import sys\nsys.argv[1:] = [{str(session)!r}]\n{_LEG}")) == root
        )
        refused = json.loads(accounts.run(accounts.outsider, f"import sys\nsys.argv[1:] = [{str(session)!r}]\n{_LEG}"))
        assert refused == {
            "refused": f"this account cannot write the archive {root}: EACCES (owner {accounts.first}:{accounts.group} 2770)"
        }


def test_the_session_skeleton_is_group_shared_whatever_the_umask(workspace):
    code = (
        "import os, sys\nos.umask(0o077)\nsys.path.insert(0, sys.argv[1])\n"
        "from hyperloom.inference_optimizer.session import paths\n"
        "paths.ensure_session_skeleton(paths.Path(sys.argv[2]))"
    )
    session = workspace / "m" / "s1"
    subprocess.run([sys.executable, "-I", "-c", code, _SRC, str(session)], check=True, timeout=120)
    made = [session.parent, session, *session.rglob("*")]
    assert all((session / sub).is_dir() for sub in paths._SESSION_SKELETON)
    assert {path: _mode(path) for path in made} == dict.fromkeys(made, 0o2775)


def test_the_session_package_never_walks_into_private_subdirectories(tmp_path):
    (tmp_path / "archive" / "r").mkdir(parents=True)
    (tmp_path / "archive" / "r" / "result.json").write_text("{}")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "final.json").write_text("{}")
    (tmp_path / "nested" / "archive").mkdir(parents=True)
    (tmp_path / "nested" / "archive" / "kept.txt").write_text("x")
    found = {p.relative_to(tmp_path).as_posix() for p in session_package._iter_session_files(tmp_path)}
    assert found == {"reports/final.json", "nested/archive/kept.txt"}
    assert paths.PRIVATE_SESSION_SUBDIRS == ("archive",)
