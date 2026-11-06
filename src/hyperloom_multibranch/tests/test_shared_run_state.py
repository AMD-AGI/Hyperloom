# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session leaves for the run's next leg, which may be another account of the group: its directories
``2775`` and the files that leg reads, appends to or replaces ``664``, whatever the umask of the leg that made them.

The test process's umask keeps group write off what a plain create makes, so each mode below is the writer's own."""

from __future__ import annotations

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


def test_the_session_directory_and_each_of_its_subdirectories_are_the_group_s(tmp_path: Path) -> None:
    layout = create_session(tmp_path / "workspace", "Llama-3-8B", Clock())

    for directory in (layout.root.parent, layout.root, *(layout.root / name for name in SESSION_SUBDIRS)):
        assert _mode(directory) == 0o2775, directory


def test_session_json_and_exit_json_are_read_by_the_next_leg_and_replaced_by_it(tmp_path: Path) -> None:
    record_session(tmp_path, tmp_path / "session")
    write_final_exit(tmp_path, HostedExit(0, stop_reason="time_exhausted"))

    assert _mode(tmp_path / SESSION_FILE) == 0o664 and _mode(tmp_path / EXIT_FILE) == 0o664
    assert recorded_session(tmp_path) == tmp_path / "session"


def test_a_session_json_this_account_cannot_read_ends_the_run_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_session(tmp_path, tmp_path / "session")

    def refused(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(hosted_run.Path, "read_text", refused)

    with pytest.raises(HostedRefused, match=rf"{tmp_path / SESSION_FILE} \(owner \d+:\d+ 0664\)"):
        recorded_session(tmp_path)


def test_the_manifest_a_later_leg_finishes_the_session_from_is_the_group_s(tmp_path: Path) -> None:
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


def test_the_logs_a_later_leg_appends_to_are_the_group_s(tmp_path: Path) -> None:
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
    (tmp_path / "toolbox").mkdir()
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
    tmp_path: Path,
) -> None:
    assert publish_once(tmp_path / "claims" / "a.json", b"{}\n")

    assert _mode(tmp_path / "claims" / "a.json") == 0o644 and _mode(tmp_path / "claims") == 0o2775
