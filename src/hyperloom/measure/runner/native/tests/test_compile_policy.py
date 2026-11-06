# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bubblewrap sandbox of a compile, as a policy and the command line it becomes: only the work directory writable,
the toolchain, the child's source tree, the image's packages and the upstream build directory read-only, the CPU tree of
``/sys``, the base ``/dev`` nodes and no GPU, a network of its own, and the signing key absent or masked."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hyperloom_common.bwrap import BwrapPolicyError, BwrapProbe, interpreter_roots

from hyperloom.common import confinement
from hyperloom.common.confinement import PrivatePaths
from hyperloom.measure.runner.native.builder import BuildMachine, LaidTree
from hyperloom.measure.runner.native.isolated import SOURCE_ROOT, ChildInterpreter, compile_policy
from hyperloom.measure.runner.native.tests._native import BUILDER_IMAGE, CountingPreflight, fake_recipe
from hyperloom.measure.tests._recipes import IMAGE

PROBE = BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> PrivatePaths:
    """The app's private paths, empty for this test."""
    fresh = PrivatePaths()
    monkeypatch.setattr(confinement, "PRIVATE_PATHS", fresh)
    return fresh


def _laid(root: Path) -> LaidTree:
    """A laid tree whose build runs an image interpreter of its own (a link to this one in a directory nothing else
    binds)."""
    for name in ("work", "image/site-packages", "sources/fakepkg/native", "image-python/bin"):
        (root / name).mkdir(parents=True)
    (root / "image-python/bin/python3").symlink_to(sys.executable)
    machine = BuildMachine(IMAGE, BUILDER_IMAGE, "gfx942", str(root / "image-python/bin/python3"))
    recipe = fake_recipe(CountingPreflight())
    return LaidTree(recipe, machine, root / "sources/fakepkg/native", root / "image/site-packages", frozenset())


def _environment(work: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(work), "TMPDIR": str(work), "LC_ALL": "C.UTF-8"}


def _pairs(argv: list[str]) -> set[tuple[str, ...]]:
    return {tuple(argv[index : index + 3]) for index in range(len(argv))}


def _at(argv: list[str], flag: str, value: str) -> int:
    return next(index for index in range(len(argv) - 1) if argv[index : index + 2] == [flag, value])


def test_the_compile_writes_only_its_work_and_reads_its_toolchain_source_and_image(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert policy.writable == (str(work),)
    read_only = set(policy.read_only)
    assert {str(SOURCE_ROOT), str(laid.image_root), str(laid.tracked), "/usr", "/etc"} <= read_only
    assert set(interpreter_roots()) <= read_only, "the child's interpreter, wherever it is installed"
    image_bin = tmp_path / "image-python/bin"
    assert any(image_bin == Path(path) or Path(path) in image_bin.parents for path in read_only), "the image's python"
    assert not any(path == "/proc" or path.startswith(("/proc/", "/sys", "/dev")) for path in read_only)
    assert (policy.devices, policy.sys_paths, policy.tmp, policy.shm) == ((), ("/sys/devices/system/cpu",), None, None)
    assert (policy.own_network, policy.die_with_parent, policy.chdir) == (True, True, str(work))


def test_its_command_line_has_its_own_network_dies_with_its_parent_and_binds_no_gpu(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"
    child = ChildInterpreter()

    argv = compile_policy(work, child, laid, _environment(work)).argv(PROBE, child.command(work))

    flags = argv[: argv.index("--")]
    assert argv[0] == "/usr/bin/bwrap" and argv[argv.index("--") + 1 :] == child.command(work)
    assert {"--unshare-net", "--die-with-parent", "--unshare-pid", "--unshare-user", "--new-session"} <= set(flags)
    assert _at(flags, "--cap-drop", "ALL") >= 0
    pairs = _pairs(flags)
    assert ("--bind", str(work), str(work)) in pairs
    assert ("--ro-bind-try", "/sys/devices/system/cpu", "/sys/devices/system/cpu") in pairs
    assert ("--dev-bind-try", "/dev/null", "/dev/null") in pairs
    dev, null = _at(flags, "--tmpfs", "/dev"), _at(flags, "--dev-bind-try", "/dev/null")
    assert dev < null, "the base nodes are bound into a fresh /dev"
    assert "--dev" not in flags and not any("kfd" in arg or "/dev/dri" in arg for arg in flags)
    assert ("--ro-bind-try", "/sys", "/sys") not in pairs


def test_the_signing_key_is_absent_and_masked_where_a_bound_tree_holds_it(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"
    outside, inside = tmp_path / "service.key", laid.image_root / "service.key"
    for key in (outside, inside):
        key.write_text("not a key")
        registered.add(key, "the build service's signing key")

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert str(inside) in policy.masked
    bound = (*policy.read_only, *policy.writable)
    assert not any(str(outside) == path or str(outside).startswith(path.rstrip("/") + "/") for path in bound)
    assert str(outside) not in policy.masked, "nothing shows it, so there is nothing to mask"


def test_a_path_the_compile_reads_through_its_own_work_is_refused(tmp_path: Path, registered: PrivatePaths) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"
    (work / "image").symlink_to(laid.image_root)
    reached = LaidTree(laid.recipe, laid.machine, laid.tracked, work / "image", frozenset())

    with pytest.raises(BwrapPolicyError, match="reached through a tree the agent writes"):
        compile_policy(work, ChildInterpreter(), reached, _environment(work))
