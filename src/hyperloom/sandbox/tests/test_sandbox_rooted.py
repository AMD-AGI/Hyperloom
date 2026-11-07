# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A path below a sandbox's root opened by one walk: the links in the root it may follow, and every way out it refuses."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.sandbox.rooted import MAX_LINKS, open_rooted
from hyperloom.sandbox.socket import SandboxDenied

ESCAPES = "resolves outside the sandbox"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """``fs`` with ``work/real.txt`` and ``work/dir/inner.txt``, beside an ``outside/secret.txt`` it must not reach;
    given by a path through a link (``given -> real``), so that its given and resolved paths differ."""
    real = tmp_path / "real"
    (real / "fs" / "work" / "dir").mkdir(parents=True)
    (real / "fs" / "work" / "real.txt").write_text("real")
    (real / "fs" / "work" / "dir" / "inner.txt").write_text("inner")
    (real / "outside").mkdir()
    (real / "outside" / "secret.txt").write_text("SECRET")
    (tmp_path / "given").symlink_to(real)
    return tmp_path / "given" / "fs"


def _read(root: Path, virtual: str) -> str:
    with os.fdopen(open_rooted(root, virtual, os.O_RDONLY), "rb") as handle:
        return handle.read().decode()


def test_a_path_with_no_link_is_opened(root: Path) -> None:
    assert _read(root, "/work/dir/inner.txt") == "inner"


@pytest.mark.parametrize(
    ("link", "target", "virtual"),
    [
        ("work/alias", "real.txt", "/work/alias"),
        ("work/up", "dir/../real.txt", "/work/up"),
        ("work/dir/back", "../real.txt", "/work/dir/back"),
        ("work/via", "dir", "/work/via/inner.txt"),
        ("work/given", "{root}/work/real.txt", "/work/given"),
        ("work/resolved", "{real}/work/real.txt", "/work/resolved"),
        ("work/top", "{root}", "/work/top/work/real.txt"),
    ],
)
def test_a_link_that_stays_below_the_root_is_followed(root: Path, link: str, target: str, virtual: str) -> None:
    (root / link).symlink_to(target.format(root=root, real=os.path.realpath(root)))
    expected = "inner" if virtual.endswith("inner.txt") else "real"
    assert _read(root, virtual) == expected


@pytest.mark.parametrize(
    ("target", "virtual"),
    [
        ("{outside}/secret.txt", "/work/link"),
        ("../../outside/secret.txt", "/work/link"),
        ("{outside}", "/work/link/secret.txt"),
        ("{root}/../outside/secret.txt", "/work/link"),
        ("{root}x/secret.txt", "/work/link"),
    ],
)
def test_a_link_that_leads_out_of_the_root_is_refused(root: Path, target: str, virtual: str) -> None:
    (root / "work" / "link").symlink_to(target.format(root=root, outside=root.parent / "outside"))
    with pytest.raises(SandboxDenied, match=ESCAPES):
        _read(root, virtual)


def test_a_link_out_of_the_root_is_refused_as_the_last_component_of_a_write_and_nothing_outside_changes(
    root: Path,
) -> None:
    secret = root.parent / "outside" / "secret.txt"
    (root / "work" / "link").symlink_to(secret)
    (root / "work" / "dirlink").symlink_to(root.parent / "outside")
    for virtual in ("/work/link", "/work/dirlink/new/file.txt"):
        with pytest.raises(SandboxDenied, match=ESCAPES):
            open_rooted(root, virtual, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, make_parents=True)
    assert secret.read_text() == "SECRET" and sorted(os.listdir(secret.parent)) == ["secret.txt"]


def test_parents_are_made_below_the_root_and_through_a_link_in_it(root: Path) -> None:
    (root / "work" / "via").symlink_to("dir")
    for virtual in ("/in/a/b/file.txt", "/work/via/c/file.txt"):
        os.close(open_rooted(root, virtual, os.O_WRONLY | os.O_CREAT, make_parents=True))
    assert (root / "in" / "a" / "b" / "file.txt").is_file() and (root / "work" / "dir" / "c" / "file.txt").is_file()
    with pytest.raises(FileNotFoundError):
        open_rooted(root, "/missing/file.txt", os.O_WRONLY | os.O_CREAT)


def test_a_loop_of_links_is_refused(root: Path) -> None:
    (root / "work" / "a").symlink_to("b")
    (root / "work" / "b").symlink_to("a")
    with pytest.raises(SandboxDenied, match=f"more than {MAX_LINKS} links"):
        _read(root, "/work/a")


def test_dot_dot_never_leaves_the_root_and_a_file_on_the_way_is_not_a_directory(root: Path) -> None:
    with pytest.raises(SandboxDenied, match=ESCAPES):
        _read(root, "/../outside/secret.txt")
    with pytest.raises(NotADirectoryError):
        _read(root, "/work/real.txt/x")
    descriptor = open_rooted(root, "/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert sorted(os.listdir(descriptor)) == ["work"]
    finally:
        os.close(descriptor)
