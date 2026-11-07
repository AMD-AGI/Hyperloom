# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom_common/fs_utils`` (no-follow I/O): the app never follows a link a less-privileged process planted."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom_common.fs_utils import (
    is_private_dir,
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


def test_a_link_above_the_base_is_followed_but_a_base_that_is_a_link_is_refused(tmp_path: Path):
    real = tmp_path / "real"
    (real / "worktree" / "patches").mkdir(parents=True)
    (real / "worktree" / "patches" / "a.patch").write_text("a", encoding="utf-8")
    (tmp_path / "alias").symlink_to(real)
    write_text_nofollow(tmp_path / "alias" / "worktree", ("f",), "x")
    assert (real / "worktree" / "f").read_text(encoding="utf-8") == "x"

    # What an agent writing the workspace leaves: its worktree renamed, a link to another directory in its place.
    swapped = tmp_path / "ws" / "worktree"
    swapped.parent.mkdir()
    swapped.symlink_to(real / "worktree")
    with pytest.raises(OSError):
        write_text_nofollow(swapped, ("patches", "_worktree_diff.patch"), "diff\n")
    assert sorted(p.name for p in (real / "worktree" / "patches").iterdir()) == ["a.patch"]
    assert list(regular_files_nofollow(swapped, "patches", (".patch",))) == []


def test_rewriting_lines_never_follows_a_planted_staging_name(tmp_path):
    from hyperloom_common.fs_utils import rewrite_lines_nofollow

    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    log = tmp_path / "process.log"
    log.write_text("secret 1\nplain\n", encoding="utf-8")
    (tmp_path / "process.log.redacting").symlink_to(victim)
    rewrite_lines_nofollow(log, lambda line: line.replace("secret", "xxxxxx"))
    assert victim.read_text(encoding="utf-8") == "keep"
    assert log.read_text(encoding="utf-8") == "xxxxxx 1\nplain\n"
    assert not (tmp_path / "process.log.redacting").exists()
    assert (log.stat().st_mode & 0o777) == 0o600


def test_rewriting_an_unchanged_file_leaves_no_staging_file_and_follows_nothing(tmp_path):
    from hyperloom_common.fs_utils import rewrite_lines_nofollow

    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    log = tmp_path / "process.log"
    log.write_text("plain\n", encoding="utf-8")
    (tmp_path / "process.log.redacting").symlink_to(victim)
    rewrite_lines_nofollow(log, lambda line: line)
    assert victim.read_text(encoding="utf-8") == "keep"
    assert not os.path.lexists(tmp_path / "process.log.redacting")


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_rewriting_refuses_a_link_at_the_file_itself(tmp_path, kind):
    from hyperloom_common.fs_utils import rewrite_lines_nofollow

    victim = tmp_path / "victim"
    victim.write_text("secret\n", encoding="utf-8")
    log = tmp_path / "process.log"
    if kind == "symlink":
        log.symlink_to(victim)
    else:
        os.link(victim, log)
    with pytest.raises(OSError):
        rewrite_lines_nofollow(log, lambda line: "x\n")
    assert victim.read_text(encoding="utf-8") == "secret\n"
    assert not os.path.lexists(tmp_path / "process.log.redacting")


def test_creating_a_file_replaces_a_planted_link_and_never_writes_through_it(tmp_path):
    from hyperloom_common.fs_utils import create_nofollow

    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    (tmp_path / "process.log").symlink_to(victim)
    with create_nofollow(tmp_path, ("process.log",)) as handle:
        handle.write("new")
    assert victim.read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "process.log").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "process.log").is_symlink()


def test_appending_refuses_a_link_and_never_writes_through_it(tmp_path):
    from hyperloom_common.fs_utils import append_text_nofollow

    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    (tmp_path / "transcript.jsonl").symlink_to(victim)
    with pytest.raises(OSError):
        append_text_nofollow(tmp_path, ("transcript.jsonl",), "x\n")
    assert victim.read_text(encoding="utf-8") == "keep"
    (tmp_path / "ok.jsonl").write_text("a\n", encoding="utf-8")
    append_text_nofollow(tmp_path, ("ok.jsonl",), "b\n")
    assert (tmp_path / "ok.jsonl").read_text(encoding="utf-8") == "a\nb\n"


def test_reading_below_a_base_refuses_a_swapped_directory_or_file(tmp_path):
    from hyperloom_common.fs_utils import read_text_below_nofollow

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f").write_text("secret", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "d").symlink_to(outside)
    (base / "g").symlink_to(outside / "f")
    with pytest.raises(OSError):
        read_text_below_nofollow(base, ("d", "f"))
    with pytest.raises(OSError):
        read_text_below_nofollow(base, ("g",))


def test_a_private_directory_is_this_accounts_own_not_a_link_and_only_it_can_write_it(tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    mine.mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(mine)
    (tmp_path / "file").write_text("", encoding="utf-8")

    assert is_private_dir(mine)
    assert not is_private_dir(tmp_path / "link"), "a link to one is not"
    assert not is_private_dir(tmp_path / "file")
    assert not is_private_dir(tmp_path / "missing")
    for mode, private in ((0o755, True), (0o770, False), (0o707, False)):
        mine.chmod(mode)
        assert is_private_dir(mine) is private, oct(mode)
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    assert not is_private_dir(mine), "another account's"
