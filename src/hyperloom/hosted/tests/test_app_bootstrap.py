# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bootstrap's refusals: which inputs end a run for good (a final exit.json) and which are left to a restart."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASH = shutil.which("bash") or "bash"
SCRIPT = Path(__file__).resolve().parent.parent / "app_bootstrap.sh"
_SHA = "0123456789abcdef0123456789abcdef01234567"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return done.stdout.strip()


@pytest.fixture
def source(tmp_path) -> Path:
    """A git repository with one commit and no lock file."""
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("x")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "c")
    return repo


def _bootstrap(tmp_path, **env):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    full = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HYPERLOOM_STATE_DIR": str(state),
        "USER_DATA_PATH": str(tmp_path / "data"),
    }
    full.update(env)
    done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True, env=full, timeout=120)
    written = state / "exit.json"
    return done.returncode, (json.loads(written.read_text()) if written.exists() else None), done.stderr


def _expect_final(code, exit_file, stderr, message):
    assert code == 2
    assert exit_file["final"] is True and exit_file["returncode"] == 2 and message in exit_file["error"], stderr


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": "main"},
            "HYPERLOOM_SOURCE_SHA must be a full 40-character",
        ),
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": _SHA.upper()},
            "HYPERLOOM_SOURCE_SHA must be a full 40-character",
        ),
        ({"HYPERLOOM_SOURCE_SHA": _SHA}, "HYPERLOOM_SOURCE_REPO is not set"),
        ({"HYPERLOOM_SOURCE_REPO": "https://", "HYPERLOOM_SOURCE_SHA": _SHA}, "neither OWNER/NAME nor a git URL"),
        (
            {"HYPERLOOM_SOURCE_REPO": "https://host-without-a-path", "HYPERLOOM_SOURCE_SHA": _SHA},
            "neither OWNER/NAME nor a git URL",
        ),
        ({"HYPERLOOM_SOURCE_REPO": "not a repo", "HYPERLOOM_SOURCE_SHA": _SHA}, "neither OWNER/NAME nor a git URL"),
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": _SHA, "USER_DATA_PATH": "relative"},
            "USER_DATA_PATH is not an absolute path",
        ),
    ],
)
def test_an_input_the_script_cannot_work_with_ends_the_run_for_good(tmp_path, env, message):
    _expect_final(*_bootstrap(tmp_path, **env), message)


def test_a_commit_the_repository_does_not_have_ends_the_run_for_good(tmp_path, source):
    _expect_final(
        *_bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=_SHA), "has no such commit"
    )


def test_a_repository_that_cannot_be_reached_is_left_to_a_restart(tmp_path):
    code, exit_file, stderr = _bootstrap(
        tmp_path, HYPERLOOM_SOURCE_REPO=str(tmp_path / "missing"), HYPERLOOM_SOURCE_SHA=_SHA
    )
    assert code == 1 and exit_file is None and "cannot fetch the source commit" in stderr


def test_the_script_leaves_no_build_directory_behind_when_it_fails(tmp_path, source):
    _bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=_SHA)
    assert not list((tmp_path / "data" / ".hyperloom").glob("build.*"))


_STUB = """#!/bin/bash
# A stand-in for the framework Python: answers the two probes and records every other call.
if [[ "$1" == "-c" ]]; then
  case "$2" in *base_prefix*) echo 1 ;; *) echo "$FAKE_PREFIX" ;; esac
  exit 0
fi
echo "$*" >>"$FAKE_LOG"
if [[ " $* " == *" wheel "* ]]; then
  while [[ "$1" != "-w" ]]; do shift; done
  : >"$2/hyperloom-0-py3-none-any.whl"
fi
"""


def _framework_run(tmp_path, source, **env):
    stub = tmp_path / "framework" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text(_STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls"
    code, exit_file, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO=str(source),
        HYPERLOOM_SOURCE_SHA=_git(source, "rev-parse", "HEAD"),
        PYTHON=str(stub),
        FAKE_PREFIX=str(tmp_path / "framework"),
        FAKE_LOG=str(log),
        **env,
    )
    return code, stderr, stub, (log.read_text().splitlines() if log.exists() else [])


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_only_the_wheel_goes_into_the_framework_python_and_it_runs_the_entry(tmp_path, source):
    code, stderr, stub, calls = _framework_run(tmp_path, source)
    assert code == 0, stderr
    installs = [c for c in calls if " install " in c]
    assert len(installs) == 1, calls  # the wheel itself: no lock, no dependency
    assert "install -q --force-reinstall --no-deps --no-index" in installs[0] and installs[0].endswith(".whl")
    assert not [c for c in calls if "--require-hashes" in c or " -r " in c]
    assert [c for c in calls if " wheel " in c and "--no-deps" in c]
    assert calls[-1] == "-m hyperloom hosted run"  # exec'd with the framework Python
    assert not (tmp_path / "state" / "venv").exists()  # no environment of its own
    assert (tmp_path / "framework" / ".hyperloom-install").read_text().strip() == _git(source, "rev-parse", "HEAD")


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_restart_of_the_same_commit_installs_nothing_again(tmp_path, source):
    _framework_run(tmp_path, source)
    (tmp_path / "calls").unlink()
    code, _, _, calls = _framework_run_again(tmp_path, source)
    assert code == 0 and calls == ["-m hyperloom hosted run"]


def _framework_run_again(tmp_path, source):
    stub = tmp_path / "framework" / "bin" / "python"
    log = tmp_path / "calls"
    code, _, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO=str(source),
        HYPERLOOM_SOURCE_SHA=_git(source, "rev-parse", "HEAD"),
        PYTHON=str(stub),
        FAKE_PREFIX=str(tmp_path / "framework"),
        FAKE_LOG=str(log),
    )
    return code, stderr, stub, log.read_text().splitlines()


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_without_any_python_the_run_ends_for_good(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "mv").symlink_to(shutil.which("mv"))  # the only tool the refusal itself needs
    _expect_final(
        *_bootstrap(tmp_path, PATH=str(empty), HYPERLOOM_SOURCE_REPO="o/r", HYPERLOOM_SOURCE_SHA=_SHA),
        "no framework Python was found",
    )
