# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A record file is written whole on any file system and read only against the content hash the database holds."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from hyperloom_kb.records import RecordFiles
from hyperloom_kb.storage import canonical_experience_bytes, experience_content_hash
from hyperloom_kb.tests.test_schema import complete_experience


def _directory_flush_fails(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    flush = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(code, os.strerror(code))
        flush(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)


def test_a_record_is_kept_on_a_file_system_that_cannot_flush_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _directory_flush_fails(monkeypatch, errno.EINVAL)
    records = RecordFiles(tmp_path, "kb-test")
    experience = complete_experience()

    records.write(experience.id, canonical_experience_bytes(experience))

    assert records.read(experience.id, experience_content_hash(experience)) == experience


def test_a_directory_flush_that_fails_for_another_reason_fails_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _directory_flush_fails(monkeypatch, errno.EIO)
    experience = complete_experience()

    with pytest.raises(OSError, match=os.strerror(errno.EIO)):
        RecordFiles(tmp_path, "kb-test").write(experience.id, canonical_experience_bytes(experience))
