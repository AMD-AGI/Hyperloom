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
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom_common import shared_storage
from hyperloom_common.shared_storage import make_shared_dirs, setgid_warnings, writable_problems
from hyperloom_common.tests.shared_accounts import other_accounts

_SRC = str(Path(shared_storage.__file__).parents[1])


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

        probe = seen.pop("probe")
        assert probe == [f"PLAIN {shared / 'plain-dir'}: EACCES (owner {accounts.first}:{accounts.group} 2755)"]
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


# -- a link a member of the group put on the way ------------------------------------------------------------------
@pytest.fixture
def unshared(tmp_path: Path) -> Iterator[Path | None]:
    """A new directory no directory above which another account can write, or ``None`` where there is none: ``/tmp``
    can be written by all, so what is refused below it may be refused for ``/tmp`` rather than for the directory a test
    makes the group's."""
    home = Path.home()
    if any(os.stat(path).st_mode & (stat.S_IWGRP | stat.S_IWOTH) for path in (home, *home.parents)):
        yield None
        return
    base = Path(tempfile.mkdtemp(prefix=".hl-shared-storage-test-", dir=home))
    try:
        yield base
    finally:
        shutil.rmtree(base)


def _planted(tmp_path: Path) -> tuple[Path, Path]:
    """A directory the group writes (``2775``) holding a link ``planted`` to a directory only its owner may enter,
    with a file ``kept`` in it: what a member of the group can do to the account that writes there next."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (private / "kept.json").write_text("kept\n")
    (shared / "planted").symlink_to(private)
    return shared, private


@pytest.mark.parametrize(
    "write",
    [
        "write_shared_bytes(target, b'theirs')",
        "write_shared_text(target, 'theirs')",
        "copy_shared_file(source, target)",
        "os.close(open_shared_replacement(target))",
        "os.close(create_new_shared_file(target.with_name('new.json')))",
        "create_shared_file(target.with_name('new.json'))",
        "make_shared_dirs(target.with_name('new'))",
        "atomic_write_json(target, {}, options=WriteOptions(shared=True, make_parents=True))",
    ],
)
def test_a_link_at_a_directory_of_the_path_is_never_followed_below_a_directory_the_group_writes(
    unshared, tmp_path, write
):
    base = unshared or tmp_path  # below /tmp the refusal holds all the more
    shared, private = _planted(base)
    source = base / "source"
    source.write_text("theirs")
    target = shared / "planted" / "kept.json"
    namespace = {"os": os, "target": target, "source": source, **vars(shared_storage)}
    exec("from hyperloom_common.io import WriteOptions, atomic_write_json", namespace)
    with pytest.raises(OSError, match="a link in a directory other accounts can write is not followed") as raised:
        exec(write, namespace)
    assert raised.value.errno == errno.ELOOP and raised.value.filename == str(shared / "planted")
    assert sorted(path.name for path in private.iterdir()) == ["kept.json"]
    assert (private / "kept.json").read_text() == "kept\n"


def test_a_link_deeper_below_a_directory_the_group_writes_is_not_followed_either(unshared, tmp_path):
    """A member of the group can make a directory of its own there, writable by itself alone, and put the link in it."""
    shared, private = _planted(unshared or tmp_path)
    theirs = shared / "theirs"
    theirs.mkdir(mode=0o755)
    (theirs / "planted").symlink_to(private)
    with pytest.raises(OSError, match="not followed"):
        shared_storage.write_shared_bytes(theirs / "planted" / "kept.json", b"theirs")
    assert (private / "kept.json").read_text() == "kept\n"


def test_a_link_below_directories_only_their_owners_write_is_followed(unshared):
    if unshared is None:
        pytest.skip("the home directory or one above it can be written by other accounts")
    (unshared / "real").mkdir()
    (unshared / "via").symlink_to("real")
    shared_storage.write_shared_bytes(unshared / "via" / "kept.json", b"ours")
    assert (unshared / "real" / "kept.json").read_bytes() == b"ours"


@pytest.mark.parametrize("link", ["/var/run", "/lib", "/bin"])
def test_a_link_where_no_account_but_its_owner_can_put_one_is_followed(link):
    if not os.path.islink(link) or os.stat(os.path.dirname(link)).st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        pytest.skip(f"{link} is not a link in a directory only its owner writes on this host")
    descriptor = shared_storage.open_directory(Path(link))
    try:
        assert os.path.samestat(os.fstat(descriptor), os.stat(link))
    finally:
        os.close(descriptor)


def test_open_directory_names_a_missing_directory_and_a_file_on_the_way(tmp_path):
    (tmp_path / "file").write_text("")
    with pytest.raises(FileNotFoundError) as missing:
        shared_storage.open_directory(tmp_path / "missing" / "below")
    assert missing.value.filename == str(tmp_path / "missing")
    with pytest.raises(NotADirectoryError) as file:
        shared_storage.open_directory(tmp_path / "file" / "below")
    assert file.value.filename == str(tmp_path / "file")


# -- a new file at its final name ---------------------------------------------------------------------------------
def test_a_new_shared_file_has_its_mode_before_it_has_its_name(tmp_path, monkeypatch):
    """A process killed between the two must not leave the file at its name with a mode only its owner may read."""
    seen = []
    real_link = os.link

    def watching_link(source, target, **kwargs):
        seen.append(_mode(Path(f"/proc/self/fd/{kwargs['src_dir_fd']}") / source))
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(shared_storage.os, "link", watching_link)
    os.close(shared_storage.create_new_shared_file(tmp_path / "0000000001.jsonl"))
    assert seen == [0o664]
    assert _mode(tmp_path / "0000000001.jsonl") == 0o664
    assert [path.name for path in tmp_path.iterdir()] == ["0000000001.jsonl"], "its own name is gone"


@pytest.mark.parametrize("links", ["made", "not made by the file system"])
def test_a_new_shared_file_never_takes_a_name_that_is_there_and_never_writes_through_a_link(
    tmp_path, monkeypatch, links
):
    if links != "made":

        def no_links(*_args, **_kwargs):
            raise PermissionError(errno.EPERM, "links are not made here")

        monkeypatch.setattr(shared_storage.os, "link", no_links)
    outside = tmp_path / "outside"
    outside.write_text("kept\n")
    taken = tmp_path / "taken.jsonl"
    taken.symlink_to(outside)
    with pytest.raises(FileExistsError):
        shared_storage.create_new_shared_file(taken)
    assert outside.read_text() == "kept\n" and taken.is_symlink()
    descriptor = shared_storage.create_new_shared_file(tmp_path / "free.jsonl")
    os.write(descriptor, b"x")
    os.close(descriptor)
    assert (tmp_path / "free.jsonl").read_bytes() == b"x" and _mode(tmp_path / "free.jsonl") == 0o664
    assert sorted(path.name for path in tmp_path.iterdir()) == ["free.jsonl", "outside", "taken.jsonl"]
