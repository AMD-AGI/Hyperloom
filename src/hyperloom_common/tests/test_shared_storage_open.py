# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opening in place what other accounts of a group open too (a lock, a ledger), and appending to a JSONL file of the
group: a new file is ``664`` whatever the umask, one that exists is never truncated, chmodded or replaced."""

from __future__ import annotations

import errno
import fcntl
import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common import io as io_module
from hyperloom_common import shared_storage
from hyperloom_common.io import SHARED, JsonLineFormat, append_jsonl, append_shared_jsonl, atomic_write_json
from hyperloom_common.shared_storage import open_shared, open_shared_append
from hyperloom_common.tests.shared_accounts import other_accounts

_SRC = str(Path(shared_storage.__file__).parents[1])


def _here(code: str, *, umask: int) -> str:
    preamble = f"import os, sys\nos.umask({umask:#o})\nsys.path.insert(0, {_SRC!r})\n"
    done = subprocess.run([sys.executable, "-I", "-c", preamble + code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _open_then_close(path: Path, access: int) -> None:
    descriptor = open_shared(path, access)
    os.close(descriptor)


@pytest.mark.parametrize("umask", [0o077, 0o027, 0o002])
def test_a_new_file_is_664_whatever_the_umask(tmp_path, umask):
    _here(
        "import os\nfrom pathlib import Path\nfrom hyperloom_common.shared_storage import open_shared, open_shared_append\n"
        f"os.close(open_shared(Path({str(tmp_path / 'lock')!r}), os.O_RDWR))\n"
        f"os.close(open_shared_append(Path({str(tmp_path / 'ledger.jsonl')!r})))\n",
        umask=umask,
    )
    assert (_mode(tmp_path / "lock"), _mode(tmp_path / "ledger.jsonl")) == (0o664, 0o664)


def test_a_file_that_exists_is_opened_as_it_is(tmp_path):
    path = tmp_path / "lock"
    path.write_text("held by the first leg")
    path.chmod(0o640)
    inode = path.stat().st_ino
    descriptor = open_shared(path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.read(descriptor, 100) == b"held by the first leg"
    finally:
        os.close(descriptor)
    assert (path.stat().st_ino, _mode(path), path.read_text()) == (inode, 0o640, "held by the first leg")


def test_a_symlink_is_never_followed_to_create_or_open(tmp_path):
    (tmp_path / "target").write_text("kept")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    for name in ("link", "dangling"):
        with pytest.raises(OSError) as raised:
            _open_then_close(tmp_path / name, os.O_RDWR)
        assert raised.value.errno == errno.ELOOP, name
    assert not (tmp_path / "nowhere").exists()
    assert (tmp_path / "target").read_text() == "kept"


def test_a_file_this_account_cannot_open_raises_permission_error_and_is_not_replaced(tmp_path):
    if os.geteuid() == 0:
        with other_accounts() as accounts:
            shared = accounts.shared_dir("shared")
            path = shared / "lock"
            path.write_text("the first account's")
            os.chown(path, accounts.first, accounts.group)
            path.chmod(0o600)
            out = accounts.run(
                accounts.second,
                "import os\nfrom pathlib import Path\nfrom hyperloom_common.shared_storage import open_shared\n"
                "try:\n"
                f"    open_shared(Path({str(path)!r}), os.O_RDWR)\n"
                "except PermissionError as exc:\n"
                "    print('EACCES', exc.errno)\n",
            )
            assert out.split() == ["EACCES", str(errno.EACCES)]
            assert (path.read_text(), _mode(path), path.stat().st_uid) == ("the first account's", 0o600, accounts.first)
        return
    path = tmp_path / "lock"
    path.write_text("not ours to open")
    path.chmod(0o000)
    inode = path.stat().st_ino
    with pytest.raises(PermissionError):
        _open_then_close(path, os.O_RDWR)
    assert (path.stat().st_ino, _mode(path)) == (inode, 0o000)


def test_appends_land_after_what_is_there(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for line in ("one", "two"):
        with os.fdopen(open_shared_append(path), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    assert path.read_text() == "one\ntwo\n"


def test_append_shared_jsonl_makes_2775_parents_and_a_664_file_and_leaves_an_existing_file_alone(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    path = shared / "a" / "b" / "events.jsonl"
    _here(
        "from pathlib import Path\nfrom hyperloom_common.io import append_shared_jsonl\n"
        f"append_shared_jsonl(Path({str(path)!r}), {{'n': 1}})\n",
        umask=0o077,
    )
    assert (_mode(path.parent.parent), _mode(path.parent), _mode(path)) == (0o2775, 0o2775, 0o664)
    path.chmod(0o640)
    append_shared_jsonl(path, {"n": 2})
    assert [json.loads(line)["n"] for line in path.read_text().splitlines()] == [1, 2]
    assert _mode(path) == 0o640


def test_append_jsonl_is_unchanged(tmp_path):
    path = tmp_path / "plain" / "events.jsonl"
    append_jsonl(path, {"n": 1}, make_parents=True)
    assert path.read_text() == '{"n": 1}\n'


def test_shared_atomic_write_replaces_a_file_another_account_made_unwritable(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}")
    path.chmod(0o400)
    (tmp_path / "old-inode").hardlink_to(path)
    atomic_write_json(path, {"leg": 2}, options=SHARED)
    assert (_mode(path), json.loads(path.read_text())) == (0o664, {"leg": 2})
    assert (tmp_path / "old-inode").read_text() == "{}", "replaced by rename, never written in place"


def test_share_parents_makes_the_parents_2775_and_leaves_the_file_owner_only(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    path = shared / "a" / "b" / "secret.json"
    _here(
        "from pathlib import Path\nfrom hyperloom_common.io import WriteOptions, atomic_write_json\n"
        f"atomic_write_json(Path({str(path)!r}), {{'k': 1}}, options=WriteOptions(make_parents=True, share_parents=True))\n",
        umask=0o077,
    )
    assert (_mode(path.parent.parent), _mode(path.parent), _mode(path)) == (0o2775, 0o2775, 0o600)


# -- the contract the other branches that use these functions rely on ------------------------------------------------
def test_open_shared_append_returns_a_descriptor_that_appends_and_is_not_inherited(tmp_path):
    path = tmp_path / "ledger.jsonl"
    descriptor = open_shared_append(path)
    try:
        assert isinstance(descriptor, int)
        assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & (os.O_ACCMODE | os.O_APPEND) == os.O_WRONLY | os.O_APPEND
        assert os.get_inheritable(descriptor) is False
        os.write(descriptor, b"one\n")
    finally:
        os.close(descriptor)
    descriptor = open_shared_append(path)
    try:
        os.write(descriptor, b"two\n")
    finally:
        os.close(descriptor)
    assert path.read_text() == "one\ntwo\n"


def test_open_shared_append_never_follows_a_link(tmp_path):
    (tmp_path / "target").write_text("kept")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(OSError) as raised:
        os.close(open_shared_append(tmp_path / "link"))
    assert raised.value.errno == errno.ELOOP
    assert (tmp_path / "target").read_text() == "kept"


def test_append_shared_jsonl_has_the_signature_of_the_branch_that_calls_it_with_fsync(tmp_path, monkeypatch):
    parameters = inspect.signature(append_shared_jsonl).parameters
    assert [(name, parameter.kind.name, parameter.default) for name, parameter in parameters.items()] == [
        ("path", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("row", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("fsync", "KEYWORD_ONLY", False),
        ("fmt", "KEYWORD_ONLY", JsonLineFormat()),
    ]
    synced = []
    monkeypatch.setattr(io_module.os, "fsync", synced.append)
    path = tmp_path / "events.jsonl"
    append_shared_jsonl(path, {"n": 1})
    assert synced == []
    append_shared_jsonl(path, {"n": 2}, fsync=True)
    assert len(synced) == 1 and isinstance(synced[0], int)
    assert [json.loads(line)["n"] for line in path.read_text().splitlines()] == [1, 2]
