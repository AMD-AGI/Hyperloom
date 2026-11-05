# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
from types import SimpleNamespace

import pytest

from hyperloom.common.outside_facts.component_layout import META_BASE, LayoutSection
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


# aiter/jit/core.py's resolution of its meta dir, reduced: a develop install keeps csrc beside the package, a wheel in
# site-packages/aiter_meta, and a meta dir without csrc falls back to the package's parent.
_AITER_CORE = b"""import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(f"{this_dir}/../../")
try:
    with open(f"{this_dir}/../install_mode") as f:
        develop = f.read().strip() == "develop"
except FileNotFoundError:
    develop = True
meta = root if develop else os.path.abspath(f"{root}/aiter_meta/")
if os.environ.get("AITER_META_DIR"):
    meta = os.path.abspath(os.environ["AITER_META_DIR"])
if not os.path.exists(os.path.join(meta, "csrc")):
    meta = root
sys.path.insert(0, meta)
"""


@pytest.mark.parametrize("install", ["develop", "wheel"])
def test_a_patched_aiter_imports_its_csrc_from_the_image(tmp_path, monkeypatch, install):
    monkeypatch.delenv("AITER_META_DIR", raising=False)
    root = tmp_path / ("aiter-repo" if install == "develop" else "site-packages")
    package = root / "aiter"
    (package / "jit").mkdir(parents=True)
    (package / "ops").mkdir()
    (package / "install_mode").write_text(install)
    (package / "jit" / "__init__.py").write_bytes(b"")
    (package / "jit" / "core.py").write_bytes(_AITER_CORE)
    (package / "ops" / "__init__.py").write_bytes(b"")
    (package / "ops" / "norm.py").write_bytes(b"EPS = 1e-6\n")
    (package / "__init__.py").write_bytes(b"from .jit import core\nfrom .ops import norm\nfrom csrc.pa import KERNEL\n")
    meta = root if install == "develop" else root / "aiter_meta"
    (meta / "csrc").mkdir(parents=True)
    (meta / "csrc" / "pa.py").write_bytes(b"KERNEL = 'pa'\n")
    layouts = {"aiter": ComponentLayout("aiter", "aiter")}
    installed = PackageBase("aiter", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed, layouts=layouts)
    base.build(["aiter"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / "aiter" / "ops" / "norm.py").write_bytes(b"EPS = 1e-5\n")

    assert check_change(worktree, "aiter", layouts["aiter"], installed) == 1


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout.decode().strip()


def test_a_change_to_the_source_beside_the_package_does_not_stop_the_check(tmp_path):
    checkout = tmp_path / "checkout"
    package = checkout / "demo_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    (checkout / "csrc").mkdir()
    (checkout / "csrc" / "kernel.cu").write_bytes(b"__global__ void k() {}\n")
    _git(checkout, "init", "--quiet")
    _git(checkout, "remote", "add", "origin", "https://example.invalid/demo")
    _git(checkout, "add", "--all")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "--no-gpg-sign", "-m", "demo")
    version = f"1.0.dev1+g{_git(checkout, 'rev-parse', '--short=9', 'HEAD')}"
    installed = PackageBase("demo_pkg", package, "demo-pkg", version)
    sections = (LayoutSection("src/demo_pkg"), LayoutSection("src/csrc", META_BASE, "csrc"))
    layouts = {"demo": ComponentLayout("demo_pkg", "src/demo_pkg", sections)}
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed, layouts=layouts)
    base.build(["demo"])
    worktree = base.worktree("s1", tmp_path / "s1")
    (worktree.path / "src" / "csrc" / "kernel.cu").write_bytes(b"__global__ void k() { return; }\n")
    (worktree.path / "src" / "demo_pkg" / "__init__.py").write_bytes(b"VALUE = 2\n")

    assert worktree.changed("demo").written == ("src/csrc/kernel.cu", "src/demo_pkg/__init__.py")
    assert check_change(worktree, "demo", layouts["demo"], installed) == 1
