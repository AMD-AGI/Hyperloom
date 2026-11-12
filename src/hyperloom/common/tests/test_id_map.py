# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/id_map``: which ids this process's user namespace maps, and a map that cannot be read is never "mapped"."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.common.id_map import GID_MAP, UID_MAP, unmapped_ids
from hyperloom.common.tests._id_map_helpers import HOST_MAP, ONE_ID_MAP, use_id_maps


@pytest.fixture
def maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a uid map and a gid map (the gid map is the uid map unless given; ``None`` writes none) and have
    :func:`unmapped_ids` read them in place of this process's own; returns the two paths."""

    def write(uid_map: str | None, gid_map: str | None = None) -> tuple[Path, Path]:
        return use_id_maps(monkeypatch, tmp_path, uid_map, gid_map)

    return write


def test_the_maps_read_are_this_processs_own() -> None:
    assert (UID_MAP, GID_MAP) == (Path("/proc/self/uid_map"), Path("/proc/self/gid_map"))


def test_a_host_namespace_maps_every_id(maps) -> None:
    maps(HOST_MAP)
    assert unmapped_ids((0, 2000, 65534, 4294967294), (0, 2000, 65534)) is None


def test_a_one_id_namespace_maps_only_its_root(maps) -> None:
    paths = maps(ONE_ID_MAP)

    assert unmapped_ids((0,), (0,)) is None
    assert unmapped_ids((2000,), (2000,)) == (
        f"uid 2000, gid 2000 are not mapped in this process's user namespace "
        f"({paths[0]}: '0 1234 1'; {paths[1]}: '0 1234 1')"
    )
    assert unmapped_ids((65534,)) == (
        f"uid 65534 is not mapped in this process's user namespace ({paths[0]}: '0 1234 1')"
    )


def test_a_map_of_several_ranges_maps_what_one_of_them_covers_and_nothing_past_its_end(maps) -> None:
    paths = maps("0 1234 1\n1 100000 1999\n2000 300000 1\n", "0 1234 1\n1000 5000 1001\n")

    assert unmapped_ids((1, 1999, 2000), (1000, 2000)) is None
    assert unmapped_ids((2001,)) == (
        f"uid 2001 is not mapped in this process's user namespace "
        f"({paths[0]}: '0 1234 1; 1 100000 1999; 2000 300000 1')"
    )
    assert unmapped_ids((), (999, 2001)).startswith("gid 999, gid 2001 are not mapped")


def test_a_supplementary_group_the_gid_map_lacks_is_named_alone(maps) -> None:
    paths = maps(HOST_MAP, "0 0 1\n2000 2000 1\n")

    assert unmapped_ids((2000,), (2000, 3000, 2000)) == (
        f"gid 3000 is not mapped in this process's user namespace ({paths[1]}: '0 0 1; 2000 2000 1')"
    )


def test_an_empty_map_maps_nothing(maps) -> None:
    paths = maps("")
    assert unmapped_ids((0,)) == f"uid 0 is not mapped in this process's user namespace ({paths[0]}: 'empty')"


def test_an_unreadable_map_is_never_mapped_and_is_named(maps) -> None:
    _, missing = maps(HOST_MAP)
    missing.unlink()

    problem = unmapped_ids((2000,), (2000,))

    assert problem is not None and problem.startswith(f"cannot read {missing}: FileNotFoundError: "), problem


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("0 0\n", "0 0"),
        ("0 0 4294967295 1\n", "0 0 4294967295 1"),
        ("0 -1 5\n", "0 -1 5"),
        ("0 0 x\n", "0 0 x"),
        ("0 0 1\n\n", ""),
    ],
    ids=["two-fields", "four-fields", "signed", "not-a-number", "blank-line"],
)
def test_a_garbled_map_is_never_mapped_and_is_named_with_its_line(maps, text: str, line: str) -> None:
    paths = maps(text)

    assert unmapped_ids((0,)) == f"cannot parse {paths[0]}: {line!r} is not 'inside outside count'"


def test_a_map_that_is_not_text_is_never_mapped(maps) -> None:
    paths = maps(HOST_MAP)
    paths[0].write_bytes(b"0 0 \xff\n")

    problem = unmapped_ids((0,))

    assert problem is not None and problem.startswith(f"cannot read {paths[0]}: UnicodeDecodeError"), problem


def test_no_ids_reads_no_map(maps) -> None:
    maps(None)
    assert unmapped_ids((), ()) is None


_IN_NAMESPACE = """
import sys
sys.path.insert(0, sys.argv[1])
from hyperloom.common.id_map import unmapped_ids
print(unmapped_ids((0,), (0,)))
print(unmapped_ids((65534,), (65534,)))
"""


def test_in_a_real_one_id_user_namespace_the_kernels_maps_are_read() -> None:
    unshare = shutil.which("unshare")
    namespace = [str(unshare), "--user", "--map-root-user", "--"]
    if unshare is None or subprocess.run([*namespace, "true"], capture_output=True, check=False).returncode != 0:
        pytest.skip("this host cannot make a user namespace (unshare --user)")
    source = str(Path(__file__).resolve().parents[3])

    done = subprocess.run(
        [*namespace, sys.executable, "-I", "-c", _IN_NAMESPACE, source], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    root, nobody = done.stdout.splitlines()
    assert root == "None"
    assert nobody.startswith(
        "uid 65534, gid 65534 are not mapped in this process's user namespace (/proc/self/uid_map: '0 "
    )
