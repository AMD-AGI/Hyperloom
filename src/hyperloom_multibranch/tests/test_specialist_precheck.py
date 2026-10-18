# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from hyperloom.measure.compile import ComponentLayout, PackageBase
from hyperloom_multibranch.specialist.precheck import CheckFailed, check_change
from hyperloom_multibranch.specialist.worktree import SourceBase

# A package below its upstream root, as sglang's python/sglang is.
LAYOUTS = {"demo": ComponentLayout("demo_pkg", "src/demo_pkg")}


@pytest.fixture
def rig(tmp_path):
    package = tmp_path / "site-packages" / "demo_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"from .config import VALUE\n")
    (package / "config.py").write_bytes(b"VALUE = 1\n")
    installed = PackageBase("demo_pkg", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed, layouts=LAYOUTS)
    base.build(["demo"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    return SimpleNamespace(worktree=worktree, installed=installed, source=worktree.path / "src" / "demo_pkg")


def _check(rig) -> int:
    return check_change(rig.worktree, "demo", LAYOUTS["demo"], rig.installed)


def test_a_change_that_compiles_and_imports_passes(rig):
    (rig.source / "config.py").write_bytes(b"VALUE = 2\n")
    (rig.source / "helper.py").write_bytes(b"HELP = 'more'\n")
    (rig.source / "__init__.py").write_bytes(b"from .config import VALUE\nfrom .helper import HELP\n")

    assert _check(rig) == 3


def test_an_untouched_worktree_imports_as_installed(rig):
    assert _check(rig) == 0


def test_a_file_that_does_not_compile_is_caught_before_the_import(rig):
    (rig.source / "config.py").write_bytes(b"VALUE = (\n")

    with pytest.raises(CheckFailed, match="src/demo_pkg/config.py does not compile"):
        _check(rig)


@pytest.mark.parametrize("breaks", ["a missing name", "a deleted module"])
def test_a_change_its_package_does_not_import_with_is_caught(rig, breaks):
    if breaks == "a deleted module":
        (rig.source / "config.py").unlink()
    else:
        (rig.source / "__init__.py").write_bytes(b"from .config import MISSING\n")

    with pytest.raises(CheckFailed, match="import demo_pkg failed with the change") as failed:
        _check(rig)
    assert "Error" in str(failed.value)
