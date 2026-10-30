# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import shutil
import subprocess

import pytest

from hyperloom.measure.compile import PackageBase, UnsupportedComponentError
from hyperloom_multibranch.specialist.worktree import SourceBase, Upstream, installed_upstream

KERNEL, HEADER = b"__global__ void k() {}\n", b"#pragma once\n"
AITER = "https://github.com/ROCm/aiter"


def _write(path, data=b"x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _aiter_install(root, meta, version=None):
    """aiter's package under ``root`` with build outputs a text patch never changes, and its csrc under ``meta``."""
    package = root / "aiter"
    _write(package / "__init__.py", b"from .jit import core\n")
    _write(package / "jit" / "core.py", b"AITER_CSRC_DIR = 'csrc'\n")
    _write(package / "jit" / "module_aiter_core.so", b"\x7fELF")
    _write(package / "jit" / "build" / "module_gemm" / "build" / "kernel.cu", KERNEL)
    _write(package / "jit" / "hyperloom_jit_backup" / "jit_build_1" / "module_gemm" / "log.txt")
    _write(package / "jit" / "flydsl_cache" / "combine_launch_0" / "kernel.txt")
    _write(meta / "csrc" / "kernels" / "gemm.cu", KERNEL)
    _write(meta / "csrc" / "include" / "gemm.h", HEADER)
    _write(meta / "csrc" / "cpp_itfs" / "__pycache__" / "utils.cpython-310.pyc", b"\x00compiled")
    _write(meta / "csrc" / "opus_gemm" / "gen_co" / "gfx950" / "opus_gemm_256x128.co", b"\x7fELF")
    _write(meta / "hsa" / "gfx950" / "gemm.co", b"\x7fELF")
    _write(meta / "hsa" / "codegen.py", b"ARCHS = ('gfx950',)\n")
    return PackageBase("aiter", package, "amd-aiter", version)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout.decode().strip()


def _develop_install(checkout):
    """aiter installed in develop mode from a git checkout, its version naming the checkout's commit."""
    _aiter_install(checkout, checkout)
    _git(checkout, "init", "--quiet")
    _git(checkout, "remote", "add", "origin", AITER)
    _git(checkout, "add", "--all")
    identity = ("-c", "user.name=t", "-c", "user.email=t@t")
    _git(checkout, *identity, "commit", "--quiet", "--no-gpg-sign", "-m", "aiter")
    commit = _git(checkout, "rev-parse", "HEAD")
    return PackageBase("aiter", checkout / "aiter", "amd-aiter", f"0.1.23.dev183+g{commit[:9]}.d20261008"), commit


def _tracked(path):
    listing = subprocess.run(["git", "ls-files"], cwd=path, capture_output=True, check=True).stdout.decode()
    return sorted(listing.split())


@pytest.fixture
def editable(tmp_path):
    return _develop_install(tmp_path / "aiter-checkout")[0]


@pytest.fixture
def worktree(tmp_path, editable):
    base = SourceBase(tmp_path / "source", locator=lambda _layout: editable)
    base.build(["aiter"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    return worktree


EXPECTED = ["aiter/__init__.py", "aiter/jit/core.py", "csrc/include/gemm.h", "csrc/kernels/gemm.cu", "hsa/codegen.py"]


def test_the_base_lays_out_aiter_s_package_and_its_csrc_without_build_outputs(tmp_path, editable):
    base = SourceBase(tmp_path / "source", locator=lambda _layout: editable)
    base.build(["aiter"])

    assert _tracked(base.path) == EXPECTED


def test_a_rebuilt_code_object_never_becomes_a_change(worktree):
    _write(worktree.path / "csrc" / "opus_gemm" / "gen_co" / "gfx950" / "opus_gemm_256x128.co", b"\x7fELF rebuilt")

    assert worktree.diff("aiter") == ""


def test_a_wheel_install_of_aiter_is_refused_as_it_names_no_full_upstream_commit(tmp_path):
    site = tmp_path / "site-packages"
    installed = _aiter_install(site, site / "aiter_meta", "0.1.23.dev183+ge7d2453f2.d20261008")
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)

    with pytest.raises(UnsupportedComponentError, match="no git checkout") as refused:
        base.build(["aiter"])
    assert "ge7d2453f2" in str(refused.value)
    assert installed_upstream(installed) is None


def test_a_develop_install_s_upstream_is_its_checkout_s_commit(tmp_path):
    installed, commit = _develop_install(tmp_path / "aiter-checkout")

    assert installed_upstream(installed) == Upstream(AITER, commit)


def test_an_aiter_without_meta_trees_lays_out_its_package_alone(tmp_path):
    checkout = tmp_path / "aiter-checkout"
    installed, _ = _develop_install(checkout)
    shutil.rmtree(checkout / "csrc")
    shutil.rmtree(checkout / "hsa")
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["aiter"])
    worktree = base.worktree("s1", tmp_path / "s1")
    _write(worktree.path / "aiter" / "ops.py", b"y = 2\n")

    assert _tracked(base.path) == ["aiter/__init__.py", "aiter/jit/core.py"]
    assert worktree.changed("aiter").written == ("aiter/ops.py",)


def _edit(worktree):
    csrc, jit = worktree.path / "csrc", worktree.path / "aiter" / "jit"
    _write(csrc / "kernels" / "gemm.cu", b"__global__ void k() { /* split-k 1 */ }\n")
    _write(csrc / "kernels" / "moe.cu", KERNEL)
    (csrc / "include" / "gemm.h").unlink()
    _write(csrc / "kernels" / "__pycache__" / "moe.cpython-310.pyc", b"\x00compiled")
    _write(jit / "build" / "module_moe" / "build" / "moe.cu", KERNEL)
    _write(jit / "hyperloom_jit_backup" / "jit_build_2" / "log.txt")
    _write(jit / "flydsl_cache" / "combine_launch_1" / "kernel.txt")
    _write(jit / "module_moe.so", b"\x7fELF")


def _files(patch):
    return sorted(line.split(" b/")[-1] for line in patch.splitlines() if line.startswith("diff --git"))


def test_a_change_to_aiter_covers_its_csrc_and_leaves_out_build_outputs(worktree):
    _edit(worktree)

    changes = worktree.changed("aiter")

    assert _files(worktree.diff("aiter")) == ["csrc/include/gemm.h", "csrc/kernels/gemm.cu", "csrc/kernels/moe.cu"]
    assert (changes.written, changes.deleted) == (
        ("csrc/kernels/gemm.cu", "csrc/kernels/moe.cu"),
        ("csrc/include/gemm.h",),
    )


def test_a_diff_from_a_commit_covers_aiter_s_csrc(worktree):
    root = worktree.commit()
    _edit(worktree)

    assert _files(worktree.diff_from(root, "aiter")) == [
        "csrc/include/gemm.h",
        "csrc/kernels/gemm.cu",
        "csrc/kernels/moe.cu",
    ]


def test_a_diff_between_two_stacks_covers_aiter_s_csrc(worktree):
    _edit(worktree)
    after = worktree.diff("aiter")

    assert _files(worktree.diff_between(worktree.commit(), [], after, "aiter")) == _files(after)


def test_a_sync_lays_a_csrc_patch_over_the_installed_sources(worktree):
    _write(worktree.path / "csrc" / "kernels" / "gemm.cu", b"// patched\n")
    landed = worktree.diff("aiter").encode()

    worktree.sync("c1", [landed])

    assert (worktree.path / "csrc" / "kernels" / "gemm.cu").read_bytes() == b"// patched\n"
    assert worktree.diff("aiter") == ""
