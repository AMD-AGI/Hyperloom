# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A compiler's index of its files names the directory it was written in: the cache stores it without (design 4.21).

Triton's group file is the real thing here (``triton_cache``), because the point is that what a test wrote in its own
directory resolves to existing files in the directory of the test that is given it. The refusals are asserted by what they
say: a group that names a file outside the cache is not the same refusal as one that is not a group at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.jit.locations import MAX_LOCATED_BYTES, NotPortable, localised, portable
from hyperloom.measure.jit.tests.triton_cache import PRODUCTS, put_kernel
from hyperloom.measure.jit_kinds import TRITON

HASH_DIR = "0123456789abcdef0123456789abcdef"
GROUP = f"{HASH_DIR}/__grp__kernel.json"


def _written_in(directory: Path, *, release: str = "3.2") -> bytes:
    put_kernel(directory, HASH_DIR, release=release)
    return (directory / GROUP).read_bytes()


def _group(children: dict[str, str]) -> bytes:
    return json.dumps({"child_paths": children}).encode()


def test_a_group_is_stored_with_every_path_relative_to_the_cache_directory(tmp_path) -> None:
    raw = _written_in(tmp_path / "sandbox" / "triton")

    stored = json.loads(portable(raw, tmp_path / "sandbox" / "triton", TRITON, GROUP))

    assert stored == {"child_paths": {name: f"{HASH_DIR}/{name}" for name in PRODUCTS}}
    assert str(tmp_path) not in portable(raw, tmp_path / "sandbox" / "triton", TRITON, GROUP).decode()


def test_the_same_kernel_compiled_in_two_directories_is_stored_as_the_same_bytes(tmp_path) -> None:
    one, two = tmp_path / "one" / "work" / "triton", tmp_path / "two" / "elsewhere" / "triton"

    assert _written_in(one) != _written_in(two), "what Triton wrote differs by directory"
    assert portable(_written_in(one), one, TRITON, GROUP) == portable(_written_in(two), two, TRITON, GROUP)


def test_a_stored_group_given_a_directory_names_the_files_there_the_way_triton_writes_it(tmp_path) -> None:
    stored = portable(_written_in(tmp_path / "one"), tmp_path / "one", TRITON, GROUP)

    local = json.loads(localised(stored, tmp_path / "two", TRITON, GROUP))

    assert local == {"child_paths": {name: f"{tmp_path / 'two'}/{HASH_DIR}/{name}" for name in PRODUCTS}}
    assert portable(localised(stored, tmp_path / "two", TRITON, GROUP), tmp_path / "two", TRITON, GROUP) == stored


def test_a_path_that_is_already_relative_is_kept(tmp_path) -> None:
    relative = {"kernel.json": f"{HASH_DIR}/kernel.json"}

    assert json.loads(portable(_group(relative), tmp_path, TRITON, GROUP)) == {"child_paths": relative}


def test_a_directory_that_is_reached_through_a_link_is_the_same_directory(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    raw = _written_in(real)

    assert (
        json.loads(portable(raw, tmp_path / "link", TRITON, GROUP))["child_paths"]["kernel.json"]
        == f"{HASH_DIR}/kernel.json"
    )


@pytest.mark.parametrize(
    "target",
    [
        "/somewhere/else/{hash}/kernel.json",
        "{root}/../outside/{hash}/kernel.json",
        "{root}/{hash}/kernel.pkl",
        "{root}/short/kernel.json",
        "{root}/{hash}/sub/kernel.json",
        "../{hash}/kernel.json",
    ],
)
def test_a_group_that_names_a_file_the_cache_does_not_hold_below_its_directory_is_refused(tmp_path, target) -> None:
    root = tmp_path / "triton"
    raw = _group({"kernel.json": target.format(root=root, hash=HASH_DIR)})

    with pytest.raises(NotPortable, match=r"kernel\.json: the path is not one of the files the triton cache holds"):
        portable(raw, root, TRITON, GROUP)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"not json", "not valid UTF-8 JSON"),
        (b"[]", "the top level must be a JSON object"),
        (b'{"child_paths": {}, "child_paths": {}}', "duplicate JSON key"),
        (b'{"other": {}}', "not a group"),
        (b'{"child_paths": {}, "more": 1}', "not a group"),
        (b'{"child_paths": "x"}', "not a group"),
        (b'{"child_paths": [1]}', "must name files by strings"),
        (b'{"child_paths": {"kernel.json": 3}}', "must name files by strings"),
        (b"{" + b" " * MAX_LOCATED_BYTES + b"}", "larger than a group of files is"),
    ],
)
def test_a_file_that_is_not_a_group_of_paths_is_refused_by_what_it_is(tmp_path, raw, message) -> None:
    with pytest.raises(NotPortable, match=message):
        portable(raw, tmp_path, TRITON, GROUP)
    with pytest.raises(NotPortable, match=message):
        localised(raw, tmp_path, TRITON, GROUP)


@pytest.mark.parametrize("stored", ["/etc/passwd", "../../etc/passwd", f"{HASH_DIR}/../x.json", "kernel.json", "a\\b"])
def test_a_stored_group_is_given_no_directory_when_a_path_in_it_is_not_one_the_kind_holds(tmp_path, stored) -> None:
    with pytest.raises(NotPortable, match=r"kernel\.json: .* is not a file of the triton cache"):
        localised(_group({"kernel.json": stored}), tmp_path, TRITON, GROUP)


def test_a_group_that_lists_names_beside_itself_needs_nothing_taken_out_or_put_in(tmp_path) -> None:
    """Current Triton writes bare names and joins them to the group's own directory: it is the same in every directory."""
    one, two = tmp_path / "one" / "triton", tmp_path / "two" / "triton"
    raw = _written_in(one, release="current")
    assert json.loads(raw) == {"child_paths": sorted(PRODUCTS)}

    stored = portable(raw, one, TRITON, GROUP)

    assert json.loads(stored) == {"child_paths": sorted(PRODUCTS)}
    assert stored == portable(_written_in(two, release="current"), two, TRITON, GROUP)
    assert localised(stored, two, TRITON, GROUP) == stored


@pytest.mark.parametrize(
    "name", ["../kernel.json", "other/kernel.json", "/etc/passwd", "kernel.pkl", "kernel.autotune.json"]
)
def test_a_group_that_lists_a_name_that_is_not_a_file_of_the_kernels_directory_is_refused(tmp_path, name) -> None:
    raw = json.dumps({"child_paths": ["kernel.json", name]}).encode()

    with pytest.raises(NotPortable, match="not a file of the triton cache beside the group"):
        portable(raw, tmp_path, TRITON, GROUP)
    with pytest.raises(NotPortable, match="not a file of the triton cache beside the group"):
        localised(raw, tmp_path, TRITON, GROUP)
