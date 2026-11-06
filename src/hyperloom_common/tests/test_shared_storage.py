# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared-storage helper: directories ``2775`` and files ``664`` whatever the umask, the group from a setgid parent,
nothing that exists chmodded or rewritten, and a writability probe that names each directory it cannot write in. Then
the same with the accounts it is for: two of one group taking turns on local disk, one outside the group, and one in
the single-id user namespace its container is."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import hyperloom_common
from hyperloom_common import shared_storage
from hyperloom_common.shared_storage import make_shared_dirs, setgid_warnings, writable_problems
from hyperloom_common.tests.shared_accounts import other_accounts

_SRC = str(Path(hyperloom_common.__file__).parents[1])


def _here(code: str, *, umask: int) -> str:
    """Run ``code`` as this account in a process of its own whose umask is ``umask`` (this process's is never changed)."""
    preamble = f"import os, sys\nos.umask({umask:#o})\nsys.path.insert(0, {_SRC!r})\n"
    done = subprocess.run([sys.executable, "-I", "-c", preamble + code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _setgid_parent(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o2775)
    if not path.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return path


# -- one account --------------------------------------------------------------------------------------------------
def test_each_directory_made_under_umask_022_is_2775_and_takes_the_setgid_parents_group(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    existing = shared / "existing"
    existing.mkdir()
    existing.chmod(0o750)
    _here(
        f"from hyperloom_common.shared_storage import make_shared_dirs\nmake_shared_dirs({str(existing / 'a' / 'b')!r})",
        umask=0o022,
    )
    assert _mode(existing) == 0o750, "a directory that was there is left as it was"
    for made in (existing / "a", existing / "a" / "b"):
        assert _mode(made) == 0o2775, made
        assert made.stat().st_gid == shared.stat().st_gid


def test_a_new_shared_file_is_664_whatever_the_umask_and_one_already_there_is_replaced_by_rename(tmp_path):
    old = tmp_path / "run.pid"
    old.write_text("previous leg")
    old.chmod(0o600)
    (tmp_path / "the-old-inode").hardlink_to(old)
    _here(
        "from hyperloom_common.shared_storage import create_shared_file, write_shared_bytes\n"
        "from pathlib import Path\n"
        f"create_shared_file(Path({str(tmp_path / 'new.log')!r}))\n"
        f"create_shared_file(Path({str(old)!r}))\n"
        f"write_shared_bytes(Path({str(tmp_path / 'launch.sh')!r}), b'#!/bin/sh', mode=0o775)\n",
        umask=0o077,
    )
    assert _mode(tmp_path / "new.log") == 0o664 and (tmp_path / "new.log").read_bytes() == b""
    assert _mode(old) == 0o664 and old.read_bytes() == b""
    assert (tmp_path / "the-old-inode").read_text() == "previous leg", "replaced, never truncated in place"
    assert _mode(tmp_path / "launch.sh") == 0o775
    assert sorted(path.name for path in tmp_path.iterdir()) == ["launch.sh", "new.log", "run.pid", "the-old-inode"]


def test_a_file_where_a_directory_belongs_is_not_taken_for_one(tmp_path):
    (tmp_path / "taken").write_text("")
    for below in (tmp_path / "taken", tmp_path / "taken" / "below"):
        with pytest.raises(FileExistsError, match="taken'$"):
            make_shared_dirs(below)


def test_writable_problems_name_each_directory_with_its_errno_and_owner_and_leave_nothing(tmp_path):
    good, missing, file = tmp_path / "good", tmp_path / "missing", tmp_path / "file"
    good.mkdir()
    file.write_text("")
    info = file.stat()
    assert writable_problems({"GOOD": good, "MISSING": missing, "FILE": file}) == [
        f"MISSING {missing}: ENOENT",
        f"FILE {file}: ENOTDIR (owner {info.st_uid}:{info.st_gid} {stat.S_IMODE(info.st_mode):04o})",
    ]
    assert list(good.iterdir()) == []


def test_a_directory_this_account_cannot_write_in_is_named_eacces(tmp_path):
    if os.geteuid() == 0:
        with other_accounts() as accounts:
            closed = accounts.root / "closed"
            closed.mkdir(mode=0o755)
            out = accounts.run(
                accounts.outsider,
                "import json\nfrom hyperloom_common.shared_storage import writable_problems\n"
                f"print(json.dumps(writable_problems({{'CLOSED': {str(closed)!r}}})))",
            )
            assert json.loads(out) == [f"CLOSED {closed}: EACCES (owner 0:0 0755)"]
        return
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o555)
    assert writable_problems({"CLOSED": closed}) == [
        f"CLOSED {closed}: EACCES (owner {os.getuid()}:{os.getgid()} 0555)"
    ]


def test_a_new_directory_that_did_not_keep_setgid_is_logged_naming_it(tmp_path, monkeypatch, caplog):
    real_fstat = os.fstat

    def as_if_not_in_the_group(descriptor):
        info = real_fstat(descriptor)
        return os.stat_result((info.st_mode & ~stat.S_ISGID, *tuple(info)[1:]))

    monkeypatch.setattr(shared_storage.os, "fstat", as_if_not_in_the_group)
    with caplog.at_level(logging.WARNING, logger=shared_storage.__name__):
        make_shared_dirs(tmp_path / "a" / "b")
    assert [record.args[0] for record in caplog.records] == [tmp_path / "a", tmp_path / "a" / "b"]
    assert "did not keep the setgid bit" in caplog.records[0].getMessage()


def test_setgid_warnings_name_each_key_directory_without_setgid(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    plain = tmp_path / "plain"
    plain.mkdir()
    warnings = setgid_warnings({"SHARED": shared, "PLAIN": plain, "MISSING": tmp_path / "missing"})
    assert len(warnings) == 1 and warnings[0].startswith(f"PLAIN {plain} is not setgid")
    assert f"chmod g+s {plain}" in warnings[0]


# -- other accounts ------------------------------------------------------------------------------------------------
_LEG = """
from pathlib import Path
from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom_common.shared_storage import make_shared_dirs
shared = Path({shared!r})
make_shared_dirs(shared / "session" / "branch")
atomic_write_text(shared / "session" / "state.json", "first", options=WriteOptions(shared=True))
(shared / "plain-dir").mkdir()
(shared / "plain.txt").write_text("first")
"""

_NEXT_LEG = """
import errno, json
from pathlib import Path
from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom_common.shared_storage import make_shared_dirs, writable_problems
shared = Path({shared!r})
seen = {{}}
def attempt(name, action):
    try:
        action()
        seen[name] = "ok"
    except OSError as exc:
        seen[name] = errno.errorcode[exc.errno]
attempt("append to its file", lambda: (shared / "session" / "state.json").open("a").write("+"))
attempt("new file in its directory", lambda: (shared / "session" / "branch" / "next").write_text("x"))
attempt("new directory in its directory", lambda: make_shared_dirs(shared / "session" / "branch" / "dev"))
attempt(
    "replace its file", lambda: atomic_write_text(shared / "session" / "state.json", "second", options=WriteOptions(shared=True))
)
attempt("new file in its umask-022 directory", lambda: (shared / "plain-dir" / "x").write_text("x"))
attempt("rewrite its umask-022 file", lambda: (shared / "plain.txt").write_text("x"))
attempt("chmod its directory", lambda: (shared / "session").chmod(0o2777))
seen["probe"] = writable_problems({{"SESSION": shared / "session", "PLAIN": shared / "plain-dir"}})
print(json.dumps(seen))
"""


def test_two_accounts_of_one_group_continue_each_others_tree_and_nothing_else(tmp_path):
    with other_accounts() as accounts:
        shared = accounts.shared_dir("shared")
        accounts.run(accounts.first, _LEG.format(shared=str(shared)), umask=0o022)
        for made in (shared / "session", shared / "session" / "branch"):
            assert (_mode(made), made.stat().st_uid, made.stat().st_gid) == (0o2775, accounts.first, accounts.group)
        assert _mode(shared / "plain-dir") == 0o2755, "outside the helper the umask decides"

        seen = json.loads(accounts.run(accounts.second, _NEXT_LEG.format(shared=str(shared)), umask=0o022))

        assert seen.pop("probe") == [
            f"PLAIN {shared / 'plain-dir'}: EACCES (owner {accounts.first}:{accounts.group} 2755)"
        ]
        assert seen == {
            "append to its file": "ok",
            "new file in its directory": "ok",
            "new directory in its directory": "ok",
            "replace its file": "ok",
            "new file in its umask-022 directory": "EACCES",
            "rewrite its umask-022 file": "EACCES",
            "chmod its directory": "EPERM",
        }
        state = shared / "session" / "state.json"
        assert (state.read_text(), _mode(state), state.stat().st_uid) == ("second", 0o664, accounts.second)
        dev = shared / "session" / "branch" / "dev"
        assert (_mode(dev), dev.stat().st_uid, dev.stat().st_gid) == (0o2775, accounts.second, accounts.group)


def test_an_account_outside_the_group_gets_no_setgid_and_is_warned(tmp_path):
    with other_accounts() as accounts:
        shared = accounts.shared_dir("shared")
        shared.chmod(0o2777)  # so that an account outside the group can make anything in it at all
        err = accounts.run(
            accounts.outsider,
            "import logging, sys\nlogging.basicConfig(stream=sys.stdout)\n"
            "from pathlib import Path\nfrom hyperloom_common.shared_storage import make_shared_dirs\n"
            f"make_shared_dirs(Path({str(shared / 'outside')!r}))",
        )
        made = shared / "outside"
        assert (_mode(made), made.stat().st_gid) == (0o775, accounts.group), "chmod 2775 came out 775"
        assert f"{made} did not keep the setgid bit" in err


_IN_THE_CONTAINER = """
import errno, json, os
from pathlib import Path
from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom_common.shared_storage import make_shared_dirs, writable_problems
shared = Path({shared!r})
seen = {{"ids": [os.getuid(), os.getgid(), sorted(os.getgroups())], "root": list(os.stat(shared))[4:6]}}
made = make_shared_dirs(shared / "from-the-container" / "x")
atomic_write_text(made / "f.txt", "x", options=WriteOptions(shared=True))
seen["inside"] = [list(os.stat(path))[4:6] for path in (made, made / "f.txt")]
try:
    os.chown(made, -1, {group})
except OSError as exc:
    seen["chgrp"] = errno.errorcode[exc.errno]
seen["probe"] = writable_problems({{"SHARED": shared, "THEIRS": shared / "theirs", "CLOSED": shared / "closed"}})
print(json.dumps(seen))
"""

_OTHER_ACCOUNTS_DIRECTORIES = """
from pathlib import Path
from hyperloom_common.shared_storage import make_shared_dirs
make_shared_dirs(Path({theirs!r}))
Path({closed!r}).mkdir()
"""


@pytest.mark.parametrize("container", ["maps written from outside", "unshare --user --map-root-user"])
def test_in_a_single_id_user_namespace_the_helper_still_makes_the_groups_2775_and_664(tmp_path, container):
    """The container's root is one host account and nothing else is mapped: the maps a container runtime writes for an
    account, or the ones ``unshare --user --map-root-user`` writes for root."""
    with other_accounts() as accounts:
        shared = accounts.shared_dir("shared")
        theirs, closed = shared / "theirs", shared / "closed"
        accounts.run(accounts.second, _OTHER_ACCOUNTS_DIRECTORIES.format(theirs=str(theirs), closed=str(closed)))

        code = _IN_THE_CONTAINER.format(shared=str(shared), group=accounts.group)
        if container == "maps written from outside":
            host_uid, root_seen_as = accounts.first, 65534
            seen = json.loads(accounts.run_in_single_id_namespace(accounts.first, code, umask=0o022))
        else:
            host_uid, root_seen_as = 0, 0
            seen = json.loads(accounts.run_as_root_in_single_id_namespace(code, umask=0o022))

        overflow = 65534
        assert seen["ids"] == [0, 0, [overflow]], "the container's root, its group unmapped"
        assert seen["root"] == [root_seen_as, overflow], "an unmapped owner or group shows as the overflow id"
        assert seen["inside"] == [[0, overflow], [0, overflow]]
        assert seen["chgrp"] == "EINVAL", "the group cannot even be named inside: only a setgid parent can give it"
        assert seen["probe"] == [f"CLOSED {closed}: EACCES (owner {overflow}:{overflow} 2755)"], (
            "another account's directory is judged by a real write, not by its owner: the group's one is written, the "
            "one without group write is refused even to the container's root"
        )
        made = shared / "from-the-container"
        for path, mode in ((made, 0o2775), (made / "x", 0o2775), (made / "x" / "f.txt", 0o664)):
            assert (_mode(path), path.stat().st_uid, path.stat().st_gid) == (mode, host_uid, accounts.group)


def test_the_helper_needs_nothing_but_the_standard_library():
    imported = {
        line.split()[1].split(".")[0]
        for line in Path(shared_storage.__file__).read_text().splitlines()
        if line.startswith(("import ", "from "))
    }
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


def test_io_shared_writes_are_664_whatever_the_umask_and_make_their_parents_2775(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    target = shared / "events" / "session.json"
    _here(
        "from hyperloom_common.io import WriteOptions, atomic_write_text\n"
        f"atomic_write_text({str(target)!r}, '{{}}', options=WriteOptions(make_parents=True, shared=True, mode=0o600))",
        umask=0o077,
    )
    assert (_mode(target.parent), _mode(target)) == (0o2775, 0o664)


def test_a_file_appended_to_in_turn_is_made_664_whatever_the_umask_and_is_appended_to_not_replaced(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    log, link = shared / "events.log", shared / "link.log"
    link.symlink_to(tmp_path / "elsewhere.log")
    out = _here(
        "import os\n"
        "from pathlib import Path\n"
        "from hyperloom_common.shared_storage import open_shared_append\n"
        f"for text in (b'one\\n', b'two\\n'):\n"
        f"    fd = open_shared_append(Path({str(log)!r}))\n"
        "    os.write(fd, text)\n"
        "    os.close(fd)\n"
        "try:\n"
        f"    open_shared_append(Path({str(link)!r}))\n"
        "except OSError as exc:\n"
        "    print(exc.errno)\n",
        umask=0o077,
    )
    assert (_mode(log), log.read_bytes()) == (0o664, b"one\ntwo\n")
    assert int(out) == errno.ELOOP and not (tmp_path / "elsewhere.log").exists(), "never through a link"


_APPENDING_LEG = """
import os
from pathlib import Path
from hyperloom_common.shared_storage import open_shared_append
shared = Path({shared!r})
os.write(log := open_shared_append(shared / "events.log"), b"leg 1\\n")
os.close(log)
with open(shared / "plain.log", "a") as plain:
    plain.write("leg 1\\n")
"""

_NEXT_APPENDING_LEG = """
import errno, json, os
from pathlib import Path
from hyperloom_common.shared_storage import open_shared_append
shared = Path({shared!r})
seen = {{}}
def append(path):
    descriptor = open_shared_append(path)
    os.write(descriptor, b"leg 2\\n")
    os.close(descriptor)
def attempt(name, action):
    try:
        action()
        seen[name] = "ok"
    except OSError as exc:
        seen[name] = errno.errorcode[exc.errno]
attempt("append to its log", lambda: append(shared / "events.log"))
attempt("append to its umask-022 log", lambda: append(shared / "plain.log"))
print(json.dumps(seen))
"""


def test_the_next_account_of_the_group_appends_to_what_the_first_made_for_appending_and_to_nothing_else(tmp_path):
    with other_accounts() as accounts:
        shared = accounts.shared_dir("shared")
        accounts.run(accounts.first, _APPENDING_LEG.format(shared=str(shared)), umask=0o022)

        seen = json.loads(accounts.run(accounts.second, _NEXT_APPENDING_LEG.format(shared=str(shared)), umask=0o022))

        assert seen == {"append to its log": "ok", "append to its umask-022 log": "EACCES"}
        assert (shared / "events.log").read_text() == "leg 1\nleg 2\n"
