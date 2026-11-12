# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the blackboard leaves for the run's next leg, which may be another account of the group: below storage set up
to be shared (setgid, group-writable, closed to others) the segments and logs that leg appends to are ``664`` and
the files it reads ``644``, in ``2775`` folders, whatever the umask of the leg that made them. Anywhere else every
participant runs as the session's user: a segment and a published file stay owner-only, as they always were, and the
logs and folders are what the umask leaves them. No append follows a link a member of the group put at the name.

The test process's umask keeps group write off what a plain create makes, so each mode below is the writer's own."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard import Role
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.learning_search import Retrieval, RetrievalLog, RetrievalSource
from hyperloom_multibranch.blackboard.local.files import publish_once
from hyperloom_multibranch.blackboard.local.segments import Entry, EntryKind, SegmentWriter
from hyperloom_multibranch.blackboard.rsi.frontier import FrontierScoring
from hyperloom_multibranch.blackboard.rsi.ladder import Trial, record_trial
from hyperloom_multibranch.blackboard.rsi.menu import DirectionMenu, MenuLog
from hyperloom_multibranch.blackboard.rsi.outcomes import OutcomePool, SpecialistOutcomes
from hyperloom_multibranch.blackboard.rsi.playbook import PlaybookPool, Retired

WRITER = "a.h.1.0000"


def _umask() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        return next(int(line.split()[1], 8) for line in status if line.startswith("Umask:"))


@pytest.fixture(autouse=True)
def _a_umask_that_keeps_group_write_off() -> None:
    assert _umask() & 0o020, "the test needs a umask that keeps group write off what a plain create makes"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture
def shared(tmp_path: Path) -> Path:
    """``tmp_path`` set up to be shared: setgid, group-writable, closed to others."""
    tmp_path.chmod(0o2775)
    if not tmp_path.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return tmp_path


def _entry(seq: int) -> Entry:
    return Entry(
        f"{WRITER}:{seq}", seq, float(seq), WRITER, Role.TOOLBOX, "t", EntryKind.TOOL_RESULT, "s1", "", "x", {}
    )


def _segment(directory: Path) -> Path:
    writer = SegmentWriter(directory, WRITER)
    writer.append(_entry)
    return writer.path


#: What stays owner-only outside shared storage (relative to the writers' root): the published files and a segment.
_OWNER_ONLY_ELSEWHERE = (
    Path("claims") / "a.json",
    Path("outcomes") / "s1.json",
    Path("playbooks") / "retired" / "p1.json",
    Path("playbooks") / "trials" / "s1.json",
    Path("segments") / f"{WRITER}.jsonl",
)


def _every_writer(root: Path, snapshot: BlackboardSnapshot) -> dict[Path, int]:
    """Run each blackboard writer of the session's shared state once below ``root``; what each made, and its mode
    below shared storage."""
    segment = _segment(root / "segments")
    publish_once(root / "claims" / "a.json", b"{}\n")
    RetrievalLog(root / "reports" / "learnings.jsonl").record(
        snapshot, Retrieval("s1", RetrievalSource.TOOL, "fp8"), ()
    )
    MenuLog(root / "menus" / "menus.jsonl").record(DirectionMenu("s1", "c1", 0.0, control=False))
    FrontierScoring._append(root / "frontier" / "frontier.jsonl", b"{}\n")
    OutcomePool(root / "outcomes").record(SpecialistOutcomes("s1"))
    playbooks = PlaybookPool(root)
    playbooks.retire(Retired("p1", 1.0, "e"))
    record_trial(playbooks, Trial("s1", "p1", "h0"))
    return {
        root / "segments": 0o2775,
        segment: 0o664,
        root / "claims": 0o2775,
        root / "claims" / "a.json": 0o644,
        root / "reports": 0o2775,
        root / "reports" / "learnings.jsonl": 0o664,
        root / "menus": 0o2775,
        root / "menus" / "menus.jsonl": 0o664,
        root / "frontier": 0o2775,
        root / "frontier" / "frontier.jsonl": 0o664,
        root / "outcomes": 0o2775,
        root / "outcomes" / "s1.json": 0o644,
        root / "playbooks": 0o2775,
        root / "playbooks" / "retired": 0o2775,
        root / "playbooks" / "retired" / "p1.json": 0o644,
        root / "playbooks" / "trials": 0o2775,
        root / "playbooks" / "trials" / "s1.json": 0o644,
    }


def test_below_shared_storage_every_writer_leaves_its_file_and_folder_to_the_group(
    shared: Path, snapshot: BlackboardSnapshot
) -> None:
    made = _every_writer(shared, snapshot)

    assert {path: oct(_mode(path)) for path in made} == {path: oct(mode) for path, mode in made.items()}


@pytest.mark.parametrize("parent_mode", [0o755, 0o2755, 0o2777, 0o1777], ids=oct)
def test_outside_shared_storage_every_writer_keeps_what_it_makes_as_private_as_before(
    tmp_path: Path, snapshot: BlackboardSnapshot, parent_mode: int
) -> None:
    """A directory that is not setgid, group-writable and closed to others is no shared storage: there a published
    file and a segment are owner-only, and each other writer makes what a plain create under this umask would, never
    widening it."""
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(parent_mode)

    made = _every_writer(root, snapshot)

    umask = _umask()
    owner_only = {root / name for name in _OWNER_ONLY_ELSEWHERE}
    assert owner_only <= made.keys()
    for path, shared_mode in made.items():
        inherited = parent_mode & stat.S_ISGID if path.is_dir() else 0  # mkdir keeps a setgid parent's bit
        expected = 0o600 if path in owner_only else shared_mode & ~umask & 0o777 | inherited
        assert _mode(path) == expected, (path, oct(_mode(path)))


def test_a_segment_already_there_is_appended_to_not_replaced(shared: Path) -> None:
    path = _segment(shared / "segments")
    inode = path.stat().st_ino

    _segment(shared / "segments")

    assert path.stat().st_ino == inode and len(path.read_bytes().splitlines()) == 2


@pytest.mark.parametrize(
    "append",
    [
        pytest.param(lambda path: _segment(path.parent), id="segment"),
        pytest.param(lambda path: FrontierScoring._append(path, b"{}\n"), id="frontier"),
    ],
)
def test_an_append_never_goes_through_a_link_a_member_of_the_group_put_at_its_name(
    shared: Path, append: Callable[[Path], object], caplog: pytest.LogCaptureFixture
) -> None:
    target = shared / "elsewhere"
    link = shared / f"{WRITER}.jsonl"
    link.symlink_to(target)

    try:
        append(link)
    except OSError as refused:
        assert refused.errno == errno.ELOOP
    else:
        assert "could not append the frontier scores" in caplog.text, "the frontier log only warns"

    assert not target.exists(), "nothing is written through the link"


def test_a_torn_tail_is_cut_only_in_the_segment_never_through_a_link_at_its_name(shared: Path) -> None:
    segments = shared / "segments"
    segments.mkdir()
    elsewhere = shared / "elsewhere"
    elsewhere.write_bytes(b"whole\ntorn")
    (segments / f"{WRITER}.jsonl").symlink_to(elsewhere)

    with pytest.raises(OSError) as refused:
        _segment(segments)

    assert refused.value.errno == errno.ELOOP
    assert elsewhere.read_bytes() == b"whole\ntorn", "the link's target is never cut"
