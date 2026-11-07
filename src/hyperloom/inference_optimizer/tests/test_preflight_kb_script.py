# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``preflight_kb.sh`` on shared storage: ``runtime/`` is ``2755`` (the installer's), ``recipe_kb/`` ``2775`` and its
marker ``664`` whatever umask it runs under, and a marker another account left (``0600``, not writable) is replaced by
rename instead of failing the write."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hyperloom_common.tests.shared_accounts import Accounts, other_accounts

_SCRIPT = Path(__file__).resolve().parents[1] / "assets" / "preflight_kb.sh"
#: What the script records with no probe to run (``SKIP_PR_PROBE``): the recipe KB has none, and the PR probe is skipped.
_SKIPPED = {"kb_reachable": False, "pr_reachable": False, "kb_skipped": True, "pr_skipped": True}


def _run(
    user_data: Path, *, umask: int, script: Path = _SCRIPT, account: tuple[int, int] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the script under ``umask`` with the probe skipped, as this account or as ``account`` (uid, group)."""
    run_as = []
    if account is not None:
        uid, group = account
        run_as = ["setpriv", f"--reuid={uid}", f"--regid={uid}", f"--groups={group}", "--"]
    command = [*run_as, "bash", "-c", f'umask {umask:o} && exec bash "$0"', str(script)]
    env = {"PATH": os.environ["PATH"], "USER_DATA_PATH": str(user_data), "SKIP_PR_PROBE": "1"}
    return subprocess.run(command, capture_output=True, text=True, timeout=60, env=env, check=False)


def _shared_copy(accounts: Accounts) -> Path:
    """The script where every account can run it (the checkout may sit in a directory only root can enter)."""
    script = accounts.root / "preflight_kb.sh"
    shutil.copyfile(_SCRIPT, script)
    script.chmod(0o755)
    return script


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _setgid(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o2775)
    if not path.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return path


@pytest.mark.parametrize("umask", [0o077, 0o022])
def test_the_directories_are_2775_and_the_marker_664_whatever_the_umask(tmp_path, umask):
    user_data = _setgid(tmp_path / "claw")
    done = _run(user_data, umask=umask)
    assert (done.returncode, done.stderr) == (0, "")
    assert (_mode(user_data / "runtime"), _mode(user_data / "runtime" / "recipe_kb")) == (0o2755, 0o2775)
    kb = user_data / "runtime" / "recipe_kb"
    assert sorted(path.name for path in kb.iterdir()) == [".kb_preflight.json"], "no temporary file is left behind"
    assert _mode(kb / ".kb_preflight.json") == 0o664
    assert json.loads((kb / ".kb_preflight.json").read_text()) == _SKIPPED


def test_a_marker_of_mode_0600_is_replaced_by_rename_not_written_in_place(tmp_path):
    user_data = _setgid(tmp_path / "claw")
    kb = user_data / "runtime" / "recipe_kb"
    kb.mkdir(parents=True)
    marker = kb / ".kb_preflight.json"
    marker.write_text('{"kb_reachable":true,"pr_reachable":true,"kb_skipped":false,"pr_skipped":false}')
    marker.chmod(0o400)
    (kb / "the-old-inode").hardlink_to(marker)
    done = _run(user_data, umask=0o077)
    assert (done.returncode, done.stderr) == (0, "")
    assert (_mode(marker), json.loads(marker.read_text())) == (0o664, _SKIPPED)
    assert json.loads((kb / "the-old-inode").read_text())["pr_reachable"] is True
    assert sorted(path.name for path in kb.iterdir()) == [".kb_preflight.json", "the-old-inode"]


def test_a_link_at_the_marker_is_replaced_and_nothing_is_written_through_it(tmp_path):
    user_data = _setgid(tmp_path / "claw")
    kb = user_data / "runtime" / "recipe_kb"
    kb.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("not the marker")
    (kb / ".kb_preflight.json").symlink_to(victim)
    done = _run(user_data, umask=0o077)
    assert (done.returncode, done.stderr) == (0, "")
    marker = kb / ".kb_preflight.json"
    assert (marker.is_symlink(), _mode(marker), json.loads(marker.read_text())) == (False, 0o664, _SKIPPED)
    assert victim.read_text() == "not the marker"


def test_a_marker_the_next_account_of_the_group_cannot_write_is_still_replaced_by_it(tmp_path):
    with other_accounts() as accounts:
        script, user_data = _shared_copy(accounts), accounts.shared_dir("claw")
        first = (accounts.first, accounts.group)
        assert _run(user_data, umask=0o077, script=script, account=first).returncode == 0
        marker = user_data / "runtime" / "recipe_kb" / ".kb_preflight.json"
        assert (marker.stat().st_uid, _mode(marker.parent.parent), _mode(marker.parent)) == (
            accounts.first,
            0o2755,
            0o2775,
        )
        marker.chmod(0o600)  # as the other account's umask leaves a file it made without this script
        done = _run(user_data, umask=0o077, script=script, account=(accounts.second, accounts.group))
        assert (done.returncode, done.stderr) == (0, "")
        assert (marker.stat().st_uid, marker.stat().st_gid, _mode(marker)) == (accounts.second, accounts.group, 0o664)
        assert json.loads(marker.read_text()) == _SKIPPED


def test_an_account_that_cannot_write_the_directory_says_so_and_leaves_the_marker(tmp_path):
    with other_accounts() as accounts:
        script, user_data = _shared_copy(accounts), accounts.shared_dir("claw")
        assert _run(user_data, umask=0o022, script=script).returncode == 0
        marker = user_data / "runtime" / "recipe_kb" / ".kb_preflight.json"
        before = (marker.stat().st_ino, marker.read_text())
        done = _run(user_data, umask=0o022, script=script, account=(accounts.outsider, accounts.outsider))
        assert done.returncode == 0
        assert f"cannot write a marker in {marker.parent}" in done.stderr
        assert (marker.stat().st_ino, marker.read_text()) == before
