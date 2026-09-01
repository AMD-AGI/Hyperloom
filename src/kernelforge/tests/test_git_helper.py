# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The single git entry point every caller in the repository runs through."""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

from kernelforge.llm.git import DEFAULT_TIMEOUT_SEC, GitError, git, git_async


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "--quiet", cwd=root)
    git("config", "user.email", "t@t", cwd=root)
    git("config", "user.name", "t", cwd=root)
    (root / "kernel.py").write_text("VALUE = 1\n")
    git("add", "kernel.py", cwd=root)
    git("commit", "-m", "base", cwd=root)
    return root


def test_a_failed_command_raises_with_gits_own_words(tmp_path):
    root = _repo(tmp_path)

    with pytest.raises(GitError) as raised:
        git("rev-parse", "--verify", "refs/heads/absent", cwd=root)

    assert "git rev-parse --verify refs/heads/absent failed" in str(raised.value)
    assert isinstance(raised.value, subprocess.CalledProcessError)


def test_a_tolerated_failure_is_returned_rather_than_raised(tmp_path):
    root = _repo(tmp_path)

    result = git("rev-parse", "--verify", "--quiet", "refs/heads/absent", cwd=root, check=False)

    assert result.returncode != 0
    assert result.stdout == ""


def test_bytes_mode_keeps_paths_that_are_not_utf8(tmp_path):
    root = _repo(tmp_path)
    (root / os.fsdecode(b"weird\xff.py")).write_text("x = 1\n")
    git("add", "-A", cwd=root)

    listed = git("ls-files", "-z", cwd=root, text=False).stdout

    assert b"weird\xff.py" in listed


def test_the_environment_overlay_extends_rather_than_replaces(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setenv("FORGE_GIT_HELPER_PROBE", "inherited")

    result = git(
        "-c",
        'alias.probe=!printf \'%s %s\' "$FORGE_GIT_HELPER_PROBE" "$OVERLAID"',
        "probe",
        cwd=root,
        env={"OVERLAID": "overlaid"},
    )

    assert result.stdout.strip() == "inherited overlaid"


def test_input_reaches_the_command(tmp_path):
    root = _repo(tmp_path)
    patch = (
        "diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )

    git("apply", "-", cwd=root, input=patch)

    assert (root / "kernel.py").read_text() == "VALUE = 2\n"


def test_a_wedged_command_is_bounded_by_the_timeout(tmp_path):
    """The sync path takes the group down too: subprocess.run would kill git
    and leave whatever an alias started holding the pipes it inherited."""
    root = _repo(tmp_path)
    marker = f"forge-sync-{uuid.uuid4().hex[:12]}"

    with pytest.raises(subprocess.TimeoutExpired):
        git("-c", f"alias.wait=!sleep 30 # {marker}", "wait", cwd=root, timeout=0.2)

    assert marker not in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


async def test_the_async_entry_reports_the_same_result_shape(tmp_path):
    root = _repo(tmp_path)

    completed = await git_async("rev-parse", "HEAD", cwd=root)
    tolerated = await git_async("symbolic-ref", "--quiet", "refs/heads/absent", cwd=root, check=False)

    assert len(completed.stdout.strip()) == 40
    assert tolerated.returncode != 0
    with pytest.raises(GitError):
        await git_async("rev-parse", "--verify", "refs/heads/absent", cwd=root)


def test_the_default_timeout_is_generous_enough_for_real_plumbing():
    """A default that trips on a large worktree would abort correct runs."""
    assert DEFAULT_TIMEOUT_SEC >= 120


def test_a_failure_in_bytes_mode_still_reads_as_words(tmp_path):
    root = _repo(tmp_path)

    with pytest.raises(GitError) as raised:
        git("rev-parse", "--verify", "refs/heads/absent", cwd=root, text=False)

    assert "Needed a single revision" in str(raised.value)


async def test_a_cancelled_await_takes_the_git_process_with_it(tmp_path):
    """A lane giving up mid-clone must not race its own directory removal."""
    root = _repo(tmp_path)
    # Tagged uniquely: the assertion reads the whole process table, and a
    # sibling test or a parallel shard sleeping too would otherwise answer it.
    marker = f"forge-cancel-{uuid.uuid4().hex[:12]}"

    task = asyncio.ensure_future(git_async("-c", f"alias.wait=!sleep 30 # {marker}", "wait", cwd=root))
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        # Bounded so a cancellation that does not take fails the test rather
        # than hanging it.
        await asyncio.wait_for(task, timeout=5)
    assert marker not in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


async def test_a_wedged_await_is_bounded_too(tmp_path):
    """``asyncio.TimeoutError`` names the right class on every version: before
    3.11 it is not the builtin one, which is what let a timeout slip past the
    kill on 3.10."""
    root = _repo(tmp_path)
    marker = f"forge-wedged-{uuid.uuid4().hex[:12]}"

    with pytest.raises(asyncio.TimeoutError):
        await git_async("-c", f"alias.wait=!sleep 30 # {marker}", "wait", cwd=root, timeout=0.2)

    assert marker not in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


def test_a_replaced_file_keeps_the_permissions_it_had(tmp_path):
    """The temp file is created owner-only; a replaced driver must not come back
    less readable than the one it replaced."""
    from kernelforge.durable_io import atomic_write_bytes

    driver = tmp_path / "forge_driver.py"
    driver.write_bytes(b"old\n")
    driver.chmod(0o755)

    atomic_write_bytes(driver, b"new\n")

    assert driver.read_bytes() == b"new\n"
    assert stat.S_IMODE(driver.stat().st_mode) == 0o755


def test_a_file_that_did_not_exist_is_published_owner_only(tmp_path):
    from kernelforge.durable_io import atomic_write_bytes

    fresh = tmp_path / "fresh.json"
    atomic_write_bytes(fresh, b"{}\n")

    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600
