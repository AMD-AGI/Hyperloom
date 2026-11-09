# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session leaves for the run's next leg, which may be another account of the group: below storage set
up to be shared (setgid, group-writable, closed to others) its directories ``2775`` and the files that leg reads,
appends to or replaces ``664``, whatever the umask of the leg that made them; anywhere else as private as the umask
leaves them, as a plain create would.

The test process's umask keeps group write off what a plain create makes, so each mode below is the writer's own."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom_multibranch import cli
from hyperloom_multibranch.blackboard.local.files import publish_once
from hyperloom_multibranch.blackboard.local.segments import Entry, EntryKind, SegmentWriter
from hyperloom_multibranch.contracts.blackboard import Role
from hyperloom_multibranch.scheduling import hosted_run
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.hosted_run import (
    EXIT_FILE,
    SESSION_FILE,
    HostedExit,
    HostedRefused,
    record_session,
    recorded_session,
    write_final_exit,
)
from hyperloom_multibranch.scheduling.layout import SESSION_SUBDIRS, create_session
from hyperloom_multibranch.scheduling.manifest import SessionManifest, code_version, read_manifest
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.toolbox.submission_log import append_line, complete_lines

WORKLOAD_ARGS = [
    "--model", "/models/Llama-3-8B", "--framework", "vllm", "--tp", "2", "--isl", "1024", "--osl", "512",
    "--conc", "16", "--precision", "fp8", "--workload", "inferencex", "--gpu-type", "mi300x",
]  # fmt: skip


@pytest.fixture(autouse=True)
def _a_umask_that_keeps_group_write_off() -> None:
    with open("/proc/self/status", encoding="ascii") as status:
        umask = next(int(line.split()[1], 8) for line in status if line.startswith("Umask:"))
    assert umask & 0o020, "the test needs a umask that keeps group write off what a plain create makes"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _umask() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        return next(int(line.split()[1], 8) for line in status if line.startswith("Umask:"))


@pytest.fixture
def shared(tmp_path: Path) -> Path:
    """``tmp_path`` set up to be shared: setgid, group-writable, closed to others."""
    tmp_path.chmod(0o2775)
    if not tmp_path.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return tmp_path


def test_the_session_directory_and_each_of_its_subdirectories_are_the_group_s(shared: Path) -> None:
    tmp_path = shared
    layout = create_session(tmp_path / "workspace", "Llama-3-8B", Clock())

    for directory in (layout.root.parent, layout.root, *(layout.root / name for name in SESSION_SUBDIRS)):
        assert _mode(directory) == 0o2775, directory


def test_session_json_and_exit_json_are_read_by_the_next_leg_and_replaced_by_it(shared: Path) -> None:
    tmp_path = shared
    record_session(tmp_path, tmp_path / "session")
    write_final_exit(tmp_path, HostedExit(0, stop_reason="time_exhausted"))

    assert _mode(tmp_path / SESSION_FILE) == 0o664 and _mode(tmp_path / EXIT_FILE) == 0o664
    assert recorded_session(tmp_path) == tmp_path / "session"


def test_a_session_json_this_account_cannot_read_ends_the_run_naming_it(
    shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = shared
    record_session(tmp_path, tmp_path / "session")

    def refused(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(hosted_run.Path, "read_text", refused)

    with pytest.raises(HostedRefused, match=rf"{tmp_path / SESSION_FILE} \(owner \d+:\d+ 0664\)"):
        recorded_session(tmp_path)


def test_the_manifest_a_later_leg_finishes_the_session_from_is_the_group_s(shared: Path) -> None:
    tmp_path = shared
    settings = cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS]), lambda _tp: 1)
    manifest = SessionManifest(
        code_version=code_version(),
        rule_pack=dict(rule_pack_identity()),
        session_id="MB-x",
        started_at=0.0,
        deadline=3600.0,
        settings=settings,
        metric=settings.workload.metric,
        policy_versions={"frontier": "default"},
    )

    manifest.write(tmp_path / "MB-manifest.json")

    assert _mode(tmp_path / "MB-manifest.json") == 0o664
    assert read_manifest(tmp_path / "MB-manifest.json") == manifest


def test_the_logs_a_later_leg_appends_to_are_the_group_s(shared: Path) -> None:
    tmp_path = shared
    handler = cli._file_handler(tmp_path / "MB-scheduler.log")
    handler.close()
    registry = ProcessRegistry(tmp_path / "MB-pids.jsonl", Clock())
    registry.record_scheduler()
    writer = SegmentWriter(tmp_path / "segments", "a.h.1.0000")
    writer.append(
        lambda seq: Entry(
            f"{writer.writer}:{seq}",
            seq,
            0.0,
            writer.writer,
            Role.TOOLBOX,
            "t",
            EntryKind.TOOL_RESULT,
            "s1",
            "",
            "x",
            {},
        )
    )
    append_line(tmp_path / "toolbox" / "submissions.jsonl", {"type": "accepted"})

    for appended in (
        tmp_path / "MB-scheduler.log",
        tmp_path / "MB-pids.jsonl",
        writer.path,
        tmp_path / "toolbox" / "submissions.jsonl",
    ):
        assert _mode(appended) == 0o664, appended
    assert _mode(tmp_path / "segments") == 0o2775
    assert complete_lines(tmp_path / "toolbox" / "submissions.jsonl") == ['{"type": "accepted"}']


def test_a_published_file_is_readable_by_the_group_and_written_by_none_and_its_folder_is_the_group_s(
    shared: Path,
) -> None:
    tmp_path = shared
    published = publish_once(tmp_path / "claims" / "a.json", b"{}\n")

    assert published

    assert _mode(tmp_path / "claims" / "a.json") == 0o644 and _mode(tmp_path / "claims") == 0o2775


def _every_writer(root: Path) -> dict[Path, int]:
    """Run each writer of the session's shared state once below ``root``; what each made, and the mode below shared
    storage that a plain create under the umask narrows elsewhere."""
    layout = create_session(root / "workspace", "Llama-3-8B", Clock())
    record_session(root / "state", root / "session")
    handler = cli._file_handler(root / "logs" / "MB-scheduler.log")
    handler.close()
    ProcessRegistry(root / "pids" / "MB-pids.jsonl", Clock()).record_scheduler()
    writer = SegmentWriter(root / "segments", "a.h.1.0000")
    writer.append(
        lambda seq: Entry(
            f"{writer.writer}:{seq}",
            seq,
            0.0,
            writer.writer,
            Role.TOOLBOX,
            "t",
            EntryKind.TOOL_RESULT,
            "s1",
            "",
            "x",
            {},
        )
    )
    append_line(root / "toolbox" / "submissions.jsonl", {"type": "accepted"})
    publish_once(root / "claims" / "a.json", b"{}\n")
    return {
        layout.root: 0o2775,
        root / "state" / SESSION_FILE: 0o600,  # a replaced file is owner-only outside shared storage
        root / "logs" / "MB-scheduler.log": 0o664,
        root / "pids" / "MB-pids.jsonl": 0o664,
        writer.path: 0o664,
        root / "segments": 0o2775,
        root / "toolbox" / "submissions.jsonl": 0o664,
        root / "toolbox": 0o2775,
        root / "claims" / "a.json": 0o644,
        root / "claims": 0o2775,
    }


@pytest.mark.parametrize("parent_mode", [0o755, 0o2755, 0o2777, 0o1777], ids=oct)
def test_outside_shared_storage_every_writer_keeps_what_it_makes_as_private_as_the_umask(
    tmp_path: Path, parent_mode: int
) -> None:
    """A directory that is not setgid, group-writable and closed to others is no shared storage: there each writer
    makes what a plain create under this umask would, never widening it."""
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(parent_mode)
    for path in (root / "logs", root / "pids", root / "state"):
        path.mkdir()

    made = _every_writer(root)

    umask = _umask()
    for path, shared_mode in made.items():
        inherited = parent_mode & stat.S_ISGID if path.is_dir() else 0  # mkdir keeps a setgid parent's bit
        assert _mode(path) == shared_mode & ~umask & 0o777 | inherited, (path, oct(_mode(path)))


def test_the_logs_a_later_leg_appends_to_are_never_opened_through_a_link_at_their_name(shared: Path) -> None:
    """A member of the group may have put a link at the name: each append refuses it and writes nothing through it."""
    target = shared / "elsewhere"
    names = ("MB-scheduler.log", "submissions.jsonl", "a.h.1.0000.jsonl")
    for name in names:
        (shared / name).symlink_to(target)

    for append in (
        lambda: cli._file_handler(shared / names[0]).close(),
        lambda: append_line(shared / names[1], {"type": "accepted"}),
        lambda: SegmentWriter(shared, "a.h.1.0000").append(
            lambda seq: Entry(
                f"a.h.1.0000:{seq}", seq, 0.0, "a.h.1.0000", Role.TOOLBOX, "t", EntryKind.TOOL_RESULT, "s1", "", "x", {}
            )
        ),
    ):
        with pytest.raises(OSError) as refused:
            append()
        assert refused.value.errno == errno.ELOOP
    assert not target.exists()
