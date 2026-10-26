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
    done = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=full, timeout=120)
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


def test_a_commit_without_a_lock_file_ends_the_run_for_good(tmp_path, source):
    sha = _git(source, "rev-parse", "HEAD")
    _expect_final(
        *_bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=sha), "no requirements/app.lock"
    )


def test_a_repository_that_cannot_be_reached_is_left_to_a_restart(tmp_path):
    code, exit_file, stderr = _bootstrap(
        tmp_path, HYPERLOOM_SOURCE_REPO=str(tmp_path / "missing"), HYPERLOOM_SOURCE_SHA=_SHA
    )
    assert code == 1 and exit_file is None and "cannot fetch the source commit" in stderr


def test_the_script_leaves_no_build_directory_behind_when_it_fails(tmp_path, source):
    _bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=_SHA)
    assert not list((tmp_path / "data" / ".hyperloom").glob("build.*"))
