# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/nofollow``: the app never follows a link a less-privileged process planted."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.common.nofollow import (
    read_bytes_nofollow,
    read_text_nofollow,
    regular_files_nofollow,
    write_text_nofollow,
)


def test_reading_a_regular_file_returns_its_text(tmp_path):
    (tmp_path / "f").write_text("héllo", encoding="utf-8")
    assert read_text_nofollow(tmp_path / "f") == "héllo"
    assert read_bytes_nofollow(tmp_path / "f") == "héllo".encode()


def test_reading_a_symlink_is_refused_and_leaks_nothing(tmp_path):
    secret = tmp_path / "app_only.json"
    secret.write_text('{"k": 1}', encoding="utf-8")
    (tmp_path / "specialist_done.json").symlink_to(secret)
    with pytest.raises(OSError):
        read_text_nofollow(tmp_path / "specialist_done.json")


def test_reading_a_fifo_is_refused_instead_of_hanging(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(OSError):
        read_text_nofollow(tmp_path / "pipe")


def test_writing_creates_the_directories_and_the_file(tmp_path):
    out = write_text_nofollow(tmp_path, ("patches", "d.patch"), "diff\n")
    assert out == tmp_path / "patches" / "d.patch"
    assert out.read_text(encoding="utf-8") == "diff\n"


def test_a_symlinked_final_name_is_replaced_not_written_through(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep me", encoding="utf-8")
    (tmp_path / "patches").mkdir()
    (tmp_path / "patches" / "d.patch").symlink_to(victim)
    write_text_nofollow(tmp_path, ("patches", "d.patch"), "diff\n")
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert not (tmp_path / "patches" / "d.patch").is_symlink()
    assert (tmp_path / "patches" / "d.patch").read_text(encoding="utf-8") == "diff\n"


def test_a_hard_linked_final_name_is_replaced_not_written_through(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep me", encoding="utf-8")
    (tmp_path / "patches").mkdir()
    os.link(victim, tmp_path / "patches" / "d.patch")
    write_text_nofollow(tmp_path, ("patches", "d.patch"), "diff\n")
    assert victim.read_text(encoding="utf-8") == "keep me"


def test_a_symlinked_directory_on_the_way_is_refused_and_nothing_is_written(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "wt"
    base.mkdir()
    (base / "patches").symlink_to(outside)
    with pytest.raises(OSError):
        write_text_nofollow(base, ("patches", "d.patch"), "diff\n")
    assert list(outside.iterdir()) == []


def test_listing_skips_symlinked_entries_and_a_symlinked_directory(tmp_path):
    base = tmp_path / "wt"
    (base / "patches").mkdir(parents=True)
    (base / "patches" / "a.patch").write_text("a", encoding="utf-8")
    (base / "patches" / "b.diff").write_text("b", encoding="utf-8")
    (base / "patches" / "c.patch").symlink_to(tmp_path / "elsewhere")
    assert [p.name for p in regular_files_nofollow(base, "patches", (".patch",))] == ["a.patch"]
    other = tmp_path / "ws"
    other.mkdir()
    (other / "patches").symlink_to(base / "patches")
    assert list(regular_files_nofollow(other, "patches", (".patch", ".diff"))) == []
    assert list(regular_files_nofollow(tmp_path / "missing", "patches", (".patch",))) == []


def test_the_base_itself_may_be_reached_through_a_link(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real)
    write_text_nofollow(tmp_path / "alias", ("f",), "x")
    assert (real / "f").read_text(encoding="utf-8") == "x"
