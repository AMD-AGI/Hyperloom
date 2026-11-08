# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared-storage helper: below a setgid, group-writable directory others cannot write, directories ``2775`` and
files ``664`` whatever the umask, the group from the setgid parent; anywhere else only the umask decides; nothing that
exists chmodded or rewritten, and a writability probe that names each directory it cannot write in. Then
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
from hyperloom_common.tests.shared_accounts import other_accounts, theirs

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
    existing.chmod(0o2770)
    _here(
        f"from hyperloom_common.shared_storage import make_shared_dirs\nmake_shared_dirs({str(existing / 'a' / 'b')!r})",
        umask=0o022,
    )
    assert _mode(existing) == 0o2770, "a directory that was there is left as it was"
    for made in (existing / "a", existing / "a" / "b"):
        assert _mode(made) == 0o2775, made
        assert made.stat().st_gid == shared.stat().st_gid


def test_a_new_shared_file_is_664_whatever_the_umask_and_one_already_there_is_replaced_by_rename(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    old = shared / "run.pid"
    old.write_text("previous leg")
    old.chmod(0o600)
    (shared / "the-old-inode").hardlink_to(old)
    _here(
        "from hyperloom_common.shared_storage import write_shared_bytes\n"
        "from pathlib import Path\n"
        f"write_shared_bytes(Path({str(shared / 'new.log')!r}), b'')\n"
        f"write_shared_bytes(Path({str(old)!r}), b'')\n"
        f"write_shared_bytes(Path({str(shared / 'launch.sh')!r}), b'#!/bin/sh', mode=0o775)\n",
        umask=0o077,
    )
    assert _mode(shared / "new.log") == 0o664 and (shared / "new.log").read_bytes() == b""
    assert _mode(old) == 0o664 and old.read_bytes() == b""
    assert (shared / "the-old-inode").read_text() == "previous leg", "replaced, never truncated in place"
    assert _mode(shared / "the-old-inode") == 0o600, "the inode it replaced is never chmodded"
    assert _mode(shared / "launch.sh") == 0o775
    assert sorted(path.name for path in shared.iterdir()) == ["launch.sh", "new.log", "run.pid", "the-old-inode"]


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
    shared = _setgid_parent(tmp_path / "shared")
    set_up = os.stat(shared)

    def as_if_not_in_the_group(descriptor):
        info = real_fstat(descriptor)
        if (info.st_dev, info.st_ino) == (set_up.st_dev, set_up.st_ino):
            return info  # the shared directory ops set up keeps its setgid bit; only what this account makes loses it
        return os.stat_result((info.st_mode & ~stat.S_ISGID, *tuple(info)[1:]))

    monkeypatch.setattr(shared_storage.os, "fstat", as_if_not_in_the_group)
    with caplog.at_level(logging.WARNING, logger=shared_storage.__name__):
        make_shared_dirs(shared / "a" / "b")
    assert [record.args[0] for record in caplog.records] == [shared / "a", shared / "a" / "b"]
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


_SWAP_FOR_A_LINK = """
import os
os.rename({shared!r}, {shared!r} + ".old")
os.symlink({target!r}, {shared!r})
"""

_WRITE_THROUGH_IT = """
import errno, json
from pathlib import Path
from hyperloom_common.shared_storage import write_shared_bytes
try:
    write_shared_bytes(Path({path!r}), b"from the owner of the home, through this account")
except OSError as exc:
    print(json.dumps([errno.errorcode[exc.errno], exc.filename]))
else:
    print(json.dumps("written"))
"""


def _closed_to_others_parent() -> Path | None:
    """A directory this account can write in whose every directory up to ``/`` is root's and written by root alone."""
    for candidate in map(Path, ("/srv", "/opt", "/var/lib", "/usr/local")):
        chain = (candidate, *candidate.parents)
        try:
            closed = all(
                (info := os.stat(path)).st_uid == 0 and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                for path in chain
            )
        except OSError:
            continue
        if closed and os.access(candidate, os.W_OK):
            return candidate
    return None


@pytest.fixture
def closed_to_others() -> Iterator[Path]:
    """A new ``0755`` directory of root's whose every directory above is root's and written by root alone: below
    ``/tmp`` a link would be refused for ``/tmp`` already, whoever owns the directories after it."""
    parent = _closed_to_others_parent()
    if parent is None:
        pytest.skip("no directory here whose way is root's alone")
    base = Path(tempfile.mkdtemp(prefix=".hl-shared-storage-test-", dir=parent))
    try:
        base.chmod(0o755)
        yield base
    finally:
        shutil.rmtree(base)


def test_a_link_in_a_directory_another_account_owns_is_not_followed_though_only_that_account_can_write_it(
    closed_to_others,
):
    """A member's own home (``0755``, only that member writes it) holding the shared tree: the member can swap the tree
    for a link to a directory only the next account may enter, and so must not be followed there."""
    with other_accounts() as accounts:
        home = closed_to_others / "home-of-first"
        home.mkdir(mode=0o755)
        os.chown(home, accounts.first, accounts.group)
        shared = home / "shared"
        shared.mkdir()
        os.chown(shared, accounts.first, accounts.group)
        shared.chmod(0o2775)
        private = closed_to_others / "private-of-second"
        private.mkdir(mode=0o700)
        settings = private / "settings.json"
        settings.write_text("orig")
        settings.chmod(0o600)
        os.chown(settings, accounts.second, accounts.second)
        os.chown(private, accounts.second, accounts.second)
        accounts.run(accounts.first, _SWAP_FOR_A_LINK.format(shared=str(shared), target=str(private)))
        assert shared.is_symlink() and shared.lstat().st_uid == accounts.first

        seen = json.loads(accounts.run(accounts.second, _WRITE_THROUGH_IT.format(path=str(shared / "settings.json"))))

        assert seen == ["ELOOP", str(shared)]
        assert (settings.read_text(), _mode(settings), settings.stat().st_uid) == ("orig", 0o600, accounts.second)
        assert sorted(path.name for path in private.iterdir()) == ["settings.json"]


@pytest.mark.parametrize(("home_mode", "followed"), [(0o755, True), (0o775, False)], ids=["0755", "0775"])
def test_a_directory_of_the_overflow_id_is_trusted_as_roots_and_only_its_mode_decides(
    closed_to_others, home_mode, followed
):
    """The overflow id is root as a user namespace shows it, mapped there or not, so its directories are trusted as
    root's: a real account of that id (the host's ``nobody``) is trusted with them too, the tradeoff that lets a
    container pass host directories. Its link is followed where only it writes the directory, and not where the group
    can write it."""
    if os.geteuid() != 0:
        pytest.skip("only root can give a directory and a link to the overflow id")
    nobody = int(Path("/proc/sys/kernel/overflowuid").read_text())
    home = closed_to_others / "home-of-nobody"
    home.mkdir()
    home.chmod(home_mode)
    target = closed_to_others / "target"
    target.mkdir()
    (home / "link").symlink_to(target)
    os.lchown(home / "link", nobody, nobody)
    os.chown(home, nobody, nobody)

    if followed:
        shared_storage.write_shared_bytes(home / "link" / "f.json", b"x")
        assert (target / "f.json").read_bytes() == b"x"
        return
    with pytest.raises(OSError, match="is not followed past a directory another account can write or owns") as raised:
        shared_storage.write_shared_bytes(home / "link" / "f.json", b"x")
    assert (raised.value.errno, raised.value.filename) == (errno.ELOOP, str(home / "link"))
    assert list(target.iterdir()) == []


_PASS_THE_HOSTS = """
import errno, json, os
from pathlib import Path
from hyperloom_common.shared_storage import open_directory, write_shared_bytes
seen = []
def attempt(action):
    try:
        action()
    except OSError as exc:
        seen.append([errno.errorcode[exc.errno], exc.filename])
    else:
        seen.append("ok")
for path in {paths!r}:
    attempt(lambda: write_shared_bytes(Path(path), b"from the container"))
for path in ("/var/run", "/lib"):
    attempt(lambda: os.close(open_directory(Path(path))))
print(json.dumps(seen))
"""

#: The subuid range of a rootless container or a Kubernetes pod without host users: it maps the overflow id itself.
_SUBUIDS = (1, 300000, 65536)


def test_in_a_namespace_of_several_ranges_host_directories_are_passed_and_a_mapped_accounts_are_not(
    closed_to_others,
):
    """Where the namespace maps a range of ids besides its root, host root still shows as the overflow id, now a
    mapped one: a root link in a directory only root writes is followed, and ``/var/run`` and ``/lib`` open. A
    directory of another account the namespace maps is that account's, and its link is not followed."""
    with other_accounts() as accounts:
        target = closed_to_others / "target"
        target.mkdir()
        os.chown(target, accounts.first, accounts.group)
        target.chmod(0o2775)
        rootdir = closed_to_others / "rootdir"
        rootdir.mkdir(mode=0o755)
        (rootdir / "ln").symlink_to(target)
        mapped = _SUBUIDS[1] + 5
        home = closed_to_others / "home-of-a-mapped-account"
        home.mkdir(mode=0o755)
        (home / "ln").symlink_to(target)
        os.lchown(home / "ln", mapped, mapped)
        os.chown(home, mapped, mapped)
        followed, refused = rootdir / "ln" / "f.json", home / "ln" / "refused.json"

        seen = json.loads(
            accounts.run_in_namespace(
                accounts.first, _PASS_THE_HOSTS.format(paths=[str(followed), str(refused)]), more=(_SUBUIDS,)
            )
        )

        assert seen == ["ok", ["ELOOP", str(home / "ln")], "ok", "ok"]
        assert sorted(path.name for path in target.iterdir()) == ["f.json"]


_WRITE_THROUGH_EACH = """
import errno, json
from pathlib import Path
from hyperloom_common.shared_storage import write_shared_bytes
seen = []
for path in {paths!r}:
    try:
        write_shared_bytes(Path(path), b"from the container")
    except OSError as exc:
        seen.append([errno.errorcode[exc.errno], exc.filename])
    else:
        seen.append("written")
print(json.dumps(seen))
"""


def test_in_a_single_id_namespace_root_links_on_a_way_only_root_writes_are_followed_and_others_still_are_not(
    closed_to_others,
):
    """In a container whose root is the account, root's directories (``/``, ``/var``) look as owned by the overflow id,
    as does every other account's: their owner cannot be told, so their mode alone decides. A host link in a
    directory only root writes is followed; one in a directory others can write still is not, whoever it looks like."""
    with other_accounts() as accounts:
        target = closed_to_others / "target"
        target.mkdir()
        os.chown(target, accounts.first, accounts.group)
        target.chmod(0o2775)
        closed = closed_to_others / "closed"
        closed.mkdir(mode=0o755)
        (closed / "link").symlink_to(target)
        open_to_all = closed_to_others / "open-to-all"
        open_to_all.mkdir()
        open_to_all.chmod(0o1777)
        (open_to_all / "link").symlink_to(target)
        followed, refused = closed / "link" / "followed.json", open_to_all / "link" / "refused.json"

        seen = json.loads(
            accounts.run_in_single_id_namespace(
                accounts.first, _WRITE_THROUGH_EACH.format(paths=[str(followed), str(refused)])
            )
        )

        assert seen == ["written", ["ELOOP", str(open_to_all / "link")]]
        assert sorted(path.name for path in target.iterdir()) == ["followed.json"]
        assert (followed.read_bytes(), followed.stat().st_uid) == (b"from the container", accounts.first)


def test_an_account_outside_the_group_gets_no_setgid_and_is_warned(tmp_path):
    with other_accounts() as accounts:
        shared = accounts.shared_dir("shared")
        os.chown(shared, accounts.outsider, accounts.group)  # so that an account outside the group can make anything
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


def _planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A directory the group writes (``2775``) holding another account's link ``planted`` to a directory only its owner
    may enter, with a file ``kept`` in it: what a member of the group can do to the account that writes there next."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (private / "kept.json").write_text("kept\n")
    (shared / "planted").symlink_to(private)
    theirs(shared / "planted", monkeypatch)
    return shared, private


@pytest.mark.parametrize(
    "write",
    [
        "write_shared_bytes(target, b'theirs')",
        "copy_shared_file(source, target)",
        "os.close(open_shared_replacement(target))",
        "os.close(create_new_shared_file(target.with_name('new.json')))",
        "with replace_shared_file(target) as handle: handle.write(b'theirs')",
        "make_shared_dirs(target.with_name('new'))",
        "atomic_write_json(target, {}, options=WriteOptions(shared=True, make_parents=True))",
    ],
)
def test_a_link_at_a_directory_of_the_path_is_never_followed_below_a_directory_the_group_writes(
    unshared, tmp_path, monkeypatch, write
):
    base = unshared or tmp_path  # below /tmp the refusal holds all the more
    shared, private = _planted(base, monkeypatch)
    source = base / "source"
    source.write_text("theirs")
    target = shared / "planted" / "kept.json"
    namespace = {"os": os, "target": target, "source": source, **vars(shared_storage)}
    exec("from hyperloom_common.io import WriteOptions, atomic_write_json", namespace)
    with pytest.raises(OSError, match="is not followed past a directory another account can write or owns") as raised:
        exec(write, namespace)
    assert raised.value.errno == errno.ELOOP and raised.value.filename == str(shared / "planted")
    assert sorted(path.name for path in private.iterdir()) == ["kept.json"]
    assert (private / "kept.json").read_text() == "kept\n"


def test_a_link_deeper_below_a_directory_the_group_writes_is_not_followed_either(unshared, tmp_path, monkeypatch):
    """A member of the group can make a directory of its own there, writable by itself alone, and put the link in it."""
    shared, private = _planted(unshared or tmp_path, monkeypatch)
    deeper = shared / "theirs"
    deeper.mkdir(mode=0o755)
    (deeper / "planted").symlink_to(private)
    theirs(deeper / "planted", monkeypatch)
    with pytest.raises(OSError, match="not followed"):
        shared_storage.write_shared_bytes(deeper / "planted" / "kept.json", b"theirs")
    assert (private / "kept.json").read_text() == "kept\n"


def test_a_link_of_this_accounts_own_below_a_directory_the_group_writes_is_followed(tmp_path):
    """No other account can make a link that is this account's (or root's): a user's own link to a bigger disk."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    (tmp_path / "disk").mkdir()
    (shared / "data").symlink_to(tmp_path / "disk")
    shared_storage.write_shared_bytes(shared / "data" / "kept.json", b"ours")
    assert (tmp_path / "disk" / "kept.json").read_bytes() == b"ours"


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


def test_a_directory_that_cannot_be_made_is_named_by_its_whole_path(tmp_path, monkeypatch):
    def refused(name, *_args, **_kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", name)

    monkeypatch.setattr(shared_storage.os, "mkdir", refused)
    with pytest.raises(PermissionError) as raised:
        make_shared_dirs(tmp_path / "new" / "below")
    assert (raised.value.errno, raised.value.filename) == (errno.EACCES, str(tmp_path / "new"))


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

    shared = _setgid_parent(tmp_path / "shared")
    monkeypatch.setattr(shared_storage.os, "link", watching_link)
    os.close(shared_storage.create_new_shared_file(shared / "0000000001.jsonl"))
    assert seen == [0o664]
    assert _mode(shared / "0000000001.jsonl") == 0o664
    assert [path.name for path in shared.iterdir()] == ["0000000001.jsonl"], "its own name is gone"


@pytest.mark.parametrize("links", ["made", "not made by the file system"])
def test_a_new_shared_file_never_takes_a_name_that_is_there_and_never_writes_through_a_link(
    tmp_path, monkeypatch, links
):
    if links != "made":

        def no_links(*_args, **_kwargs):
            raise PermissionError(errno.EPERM, "links are not made here")

        monkeypatch.setattr(shared_storage.os, "link", no_links)
    shared = _setgid_parent(tmp_path / "shared")
    outside = shared / "outside"
    outside.write_text("kept\n")
    taken = shared / "taken.jsonl"
    taken.symlink_to(outside)
    with pytest.raises(FileExistsError):
        os.close(shared_storage.create_new_shared_file(taken))
    assert outside.read_text() == "kept\n" and taken.is_symlink()
    descriptor = shared_storage.create_new_shared_file(shared / "free.jsonl")
    try:
        os.write(descriptor, b"x")
    finally:
        os.close(descriptor)
    assert (shared / "free.jsonl").read_bytes() == b"x" and _mode(shared / "free.jsonl") == 0o664
    assert sorted(path.name for path in shared.iterdir()) == ["free.jsonl", "outside", "taken.jsonl"]


def test_past_a_link_of_this_accounts_own_the_way_of_its_target_decides_what_is_followed(unshared, monkeypatch):
    """A user's own link from the shared tree to a disk of its own: below that disk's directories, which no other
    account can write, any link is followed again."""
    if unshared is None:
        pytest.skip("the home directory or one above it can be written by other accounts")
    shared = unshared / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    disk = unshared / "disk"
    (disk / "real").mkdir(parents=True)
    (shared / "data").symlink_to(disk)
    (disk / "inner").symlink_to("real")
    theirs(disk / "inner", monkeypatch)
    shared_storage.write_shared_bytes(shared / "data" / "inner" / "kept.json", b"ours")
    assert (disk / "real" / "kept.json").read_bytes() == b"ours"


# -- shared only where the storage is set up to be shared -----------------------------------------------------------
_EVERY_HELPER = """
import json, os, stat
from pathlib import Path
from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom_common import shared_storage as s
parent = Path({parent!r})
s.make_shared_dirs(parent / "chain" / "a" / "b")
s.make_shared_dirs(parent / "existing" / "below")
s.make_shared_dirs(parent / "archive", mode=0o2770)
s.write_shared_bytes(parent / "new.log", b"")
s.write_shared_bytes(parent / "old.log", b"")
s.write_shared_bytes(parent / "bytes", b"x")
s.write_shared_bytes(parent / "credential", b"secret", mode=0o600)
s.write_shared_bytes(parent / "text", b"x")
os.close(s.open_shared_replacement(parent / "replacement.log"))
os.close(s.create_new_shared_file(parent / "segment.jsonl"))
s.copy_shared_file(parent / "text", parent / "copy")
with s.replace_shared_file(parent / "archive.zip") as handle:
    handle.write(b"PK")
atomic_write_text(parent / "io" / "a" / "state.json", "x", options=WriteOptions(make_parents=True, shared=True))
modes = {{}}
for path in sorted(parent.rglob("*")):
    modes[str(path.relative_to(parent))] = stat.S_IMODE(os.lstat(path).st_mode)
print(json.dumps(modes))
"""


def _every_helper_under(parent: Path, *, umask: int) -> dict[str, int]:
    (parent / "existing").mkdir()
    (parent / "existing").chmod(0o700)
    (parent / "old.log").write_text("old")
    (parent / "old.log").chmod(0o640)
    (parent / "the-old-inode").hardlink_to(parent / "old.log")
    return json.loads(_here(_EVERY_HELPER.format(parent=str(parent)), umask=umask))


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_below_a_setgid_group_writable_directory_every_helper_widens_whatever_the_umask(tmp_path, umask):
    modes = _every_helper_under(_setgid_parent(tmp_path / "shared"), umask=umask)
    assert modes == {
        "archive": 0o2770,
        "archive.zip": 0o664,
        "bytes": 0o664,
        "chain": 0o2775,
        "chain/a": 0o2775,
        "chain/a/b": 0o2775,
        "copy": 0o664,
        "credential": 0o600,
        "existing": 0o700,
        "existing/below": 0o775 & ~umask,  # judged at the nearest existing directory: not setgid, so not shared
        "io": 0o2775,
        "io/a": 0o2775,
        "io/a/state.json": 0o664,
        "new.log": 0o664,
        "old.log": 0o664,
        "replacement.log": 0o664,
        "segment.jsonl": 0o664,
        "text": 0o664,
        "the-old-inode": 0o640,
    }


# 0o002 leaves group write: a parent made in a setgid directory then looks shared, but what it holds must not widen
@pytest.mark.parametrize("umask", [0o022, 0o077, 0o002])
@pytest.mark.parametrize(
    "parent_mode", [0o755, 0o775, 0o2755, 0o2777, 0o1777], ids=["0755", "0775", "2755", "2777", "1777"]
)
def test_anywhere_else_every_helper_leaves_what_it_makes_to_the_umask(tmp_path, umask, parent_mode):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(parent_mode)
    if stat.S_IMODE(parent.stat().st_mode) != parent_mode:
        pytest.skip("this filesystem does not keep the mode for the test user")
    modes = _every_helper_under(parent, umask=umask)
    inherited = parent_mode & stat.S_ISGID  # the kernel, not the helper, gives a new directory a setgid parent's bit
    directory, file = (0o775 & ~umask) | inherited, 0o664 & ~umask
    assert modes == {
        "archive": (0o770 & ~umask) | inherited,
        "archive.zip": file,
        "bytes": file,
        "chain": directory,
        "chain/a": directory,
        "chain/a/b": directory,
        "copy": file,
        "credential": 0o600,
        "existing": 0o700,
        "existing/below": 0o775 & ~umask,
        "io": directory,
        "io/a": directory,
        "io/a/state.json": 0o600,
        "new.log": file,
        "old.log": file,
        "replacement.log": file,
        "segment.jsonl": file,
        "text": file,
        "the-old-inode": 0o640,
    }


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "private"])
def test_a_link_at_the_final_name_is_refused_or_replaced_never_followed(tmp_path, shared):
    if shared:
        parent = _setgid_parent(tmp_path / "parent")
    else:
        parent = tmp_path / "parent"
        parent.mkdir(mode=0o755)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / "file").write_text("theirs")
    (elsewhere / "file").chmod(0o600)
    (parent / "dir-link").symlink_to(elsewhere)
    (parent / "dangling").symlink_to(tmp_path / "nowhere")
    (parent / "file-link").symlink_to(elsewhere / "file")
    with pytest.raises(FileExistsError):
        make_shared_dirs(parent / "dangling")
    shared_storage.write_shared_bytes(parent / "file-link", b"")
    assert not (parent / "file-link").is_symlink(), "the link itself is replaced by rename"
    with shared_storage.replace_shared_file(parent / "dir-link") as handle:
        handle.write(b"x")
    assert (parent / "dir-link").read_bytes() == b"x" and not (parent / "dir-link").is_symlink()
    assert ((elsewhere / "file").read_text(), _mode(elsewhere / "file"), _mode(elsewhere)) == ("theirs", 0o600, 0o700)
    assert not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "private"])
def test_a_replacement_whose_block_raises_leaves_the_file_there_and_no_new_one(tmp_path, shared):
    parent = _setgid_parent(tmp_path / "parent") if shared else tmp_path
    target = parent / "archive.zip"
    target.write_bytes(b"earlier")
    with pytest.raises(RuntimeError, match="writer failed"):
        with shared_storage.replace_shared_file(target) as handle:
            handle.write(b"half")
            raise RuntimeError("writer failed")
    assert target.read_bytes() == b"earlier"
    assert sorted(path.name for path in parent.iterdir()) == ["archive.zip"]


def test_a_link_to_shared_storage_is_not_taken_for_it(tmp_path):
    shared = _setgid_parent(tmp_path / "shared")
    (tmp_path / "link").symlink_to(shared)
    _here(
        "from pathlib import Path\nfrom hyperloom_common import shared_storage as s\n"
        f"s.make_shared_dirs(Path({str(tmp_path / 'link' / 'a')!r}))\n"
        f"s.write_shared_bytes(Path({str(tmp_path / 'link' / 'f')!r}), b'x')\n",
        umask=0o022,
    )
    assert (_mode(shared / "a") & 0o777, _mode(shared / "f")) == (0o755, 0o644), "judged by lstat: a link is not shared"


def test_parent_is_shared_judges_the_nearest_existing_directory_as_the_walk_reaches_it(tmp_path, monkeypatch):
    shared = _setgid_parent(tmp_path / "shared")
    (shared / "dangling").symlink_to(shared / "nowhere")  # its target would be shared storage, were it there
    (tmp_path / "link").symlink_to(shared)
    (shared / "planted").symlink_to(shared)
    theirs(shared / "planted", monkeypatch)
    assert shared_storage.parent_is_shared(shared / "new")
    assert shared_storage.parent_is_shared(shared / "missing" / "deeper" / "new"), "judged at the nearest that exists"
    assert not shared_storage.parent_is_shared(tmp_path / "new")
    assert not shared_storage.parent_is_shared(tmp_path / "link" / "new"), "a link stands at that directory's name"
    assert not shared_storage.parent_is_shared(tmp_path / "link" / "below" / "new"), "nearest existing: the link"
    make_shared_dirs(shared / "below")
    assert shared_storage.parent_is_shared(tmp_path / "link" / "below" / "new"), "past the link its target decides"
    assert not shared_storage.parent_is_shared(shared / "dangling" / "new"), "a link to nothing is no shared storage"
    assert not shared_storage.parent_is_shared(shared / "planted" / "new"), "a link the walk refuses: not shared"
