# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess

import pytest

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import COMPONENT_LAYOUTS, LocalBlobStore, PackageBase, apply_patch_stack
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.specialist.worktree import SourceBase, Upstream

VLLM = "https://github.com/vllm-project/vllm"
CONFIG, SCHEDULER = b"max_num_seqs = 128\n", b"policy = 'fcfs'\n"


@pytest.fixture
def installed(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    (package / "__pycache__").mkdir(parents=True)
    (package / "config.py").write_bytes(CONFIG)
    (package / "scheduler.py").write_bytes(SCHEDULER)
    (package / "__pycache__" / "config.cpython-312.pyc").write_bytes(b"\x00compiled")
    (package / "_C.so").write_bytes(b"\x7fELF")
    return PackageBase("vllm", package, None, None)


@pytest.fixture
def base(tmp_path, installed):
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["vllm"])
    return base


def test_the_base_holds_the_installed_sources_at_their_upstream_paths(base):
    tracked = [
        path for path in base.path.rglob("*") if path.is_file() and ".git" not in path.relative_to(base.path).parts
    ]
    assert sorted(path.relative_to(base.path).as_posix() for path in tracked) == ["vllm/config.py", "vllm/scheduler.py"]
    assert base.worktree("s1", base.path.parent / "s1").head() is None


def test_a_change_in_a_worktree_is_a_patch_the_toolbox_applies_to_the_installed_package(tmp_path, base, installed):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")
    (worktree.path / "vllm" / "lpm.py").write_bytes(b"policy = 'lpm'\n")

    patch = worktree.diff("vllm")

    blobs = LocalBlobStore(tmp_path / "blobs")
    ref = PatchRef("patch/vllm/change.patch", blobs.put(patch.encode()))
    stack = ComponentPatches("vllm", VLLM, "1" * 40, (ref,))
    root = apply_patch_stack(stack, COMPONENT_LAYOUTS["vllm"], installed, blobs)
    assert sorted(entry.path for entry in root.entries) == ["config.py", "lpm.py"]


def test_a_diff_from_a_commit_holds_what_was_committed_since_without_touching_the_worktree_s_index(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    root = worktree.commit()
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")
    subprocess.run(["git", "-c", "user.name=a", "-c", "user.email=a@a", "commit", "-qam", "keep"], cwd=worktree.path)
    (worktree.path / "vllm" / "scheduler.py").write_bytes(b"policy = 'lpm'\n")
    (worktree.path / "vllm" / "__pycache__").mkdir()
    (worktree.path / "vllm" / "__pycache__" / "scheduler.cpython-312.pyc").write_bytes(b"\x00compiled")
    status = ["git", "status", "--porcelain", "--ignored"]
    before = subprocess.run(status, cwd=worktree.path, capture_output=True).stdout

    patch = worktree.diff_from(root, "vllm")

    assert "+max_num_seqs = 256" in patch and "+policy = 'lpm'" in patch and "pyc" not in patch
    assert worktree.diff_from(worktree.commit(), "vllm").count("diff --git") == 1
    assert subprocess.run(status, cwd=worktree.path, capture_output=True).stdout == before


def test_a_sync_drops_the_edits_and_lays_the_head_patches_over_the_installed_sources(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")
    landed = worktree.diff("vllm").encode()
    (worktree.path / "vllm" / "scheduler.py").write_bytes(b"policy = 'unfinished'\n")

    worktree.sync("c1", [landed])

    assert worktree.head() == "c1"
    assert (worktree.path / "vllm" / "config.py").read_bytes() == b"max_num_seqs = 256\n"
    assert (worktree.path / "vllm" / "scheduler.py").read_bytes() == SCHEDULER
    assert worktree.diff("vllm") == ""


def test_a_sync_to_the_same_head_only_drops_the_edits(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 512\n")

    worktree.sync("c0", [])

    assert (worktree.head(), worktree.diff("vllm")) == ("c0", "")


def test_each_branch_edits_its_own_worktree(tmp_path, base):
    first, second = base.worktree("s1", tmp_path / "s1"), base.worktree("s2", tmp_path / "s2")
    first.sync("c1", [])
    (first.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")

    assert (second.path / "vllm" / "config.py").read_bytes() == CONFIG
    assert base.worktree("s1", tmp_path / "s1").head() == "c1"


def test_a_diff_from_a_commit_holds_new_files_under_a_nested_package_and_leaves_out_built_ones(tmp_path):
    package = tmp_path / "site-packages" / "sglang"
    (package / "srt").mkdir(parents=True)
    (package / "srt" / "a.py").write_bytes(b"x = 1\n")
    installed = PackageBase("sglang", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["sglang"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    root, srt = worktree.commit(), worktree.path / "python" / "sglang" / "srt"
    (srt / "tuned.csv").write_bytes(b"m,n,k\n")
    subprocess.run(["git", "add", "-A"], cwd=worktree.path, check=True)
    subprocess.run(["git", "-c", "user.name=a", "-c", "user.email=a@a", "commit", "-qm", "keep"], cwd=worktree.path)
    (srt / "kernel.py").write_bytes(b"y = 2\n")
    (srt / "__pycache__").mkdir()
    (srt / "__pycache__" / "kernel.cpython-310.pyc").write_bytes(b"\x00compiled")
    (srt / "ops.so").write_bytes(b"\x7fELF")

    patch = worktree.diff_from(root, "sglang")

    assert "b/python/sglang/srt/tuned.csv" in patch and "b/python/sglang/srt/kernel.py" in patch
    assert ".pyc" not in patch and "ops.so" not in patch


AOT = "python/sglang/kernels/aot"
SGLANG = "https://github.com/sgl-project/sglang.git"


def _native_base(tmp_path):
    """sgl-kernel installed as a wheel, and the native build's checkout of its sources at the image's commit."""
    package = tmp_path / "site-packages" / "sgl_kernel"
    package.mkdir(parents=True)
    (package / "gemm.py").write_bytes(b"installed = True\n")
    checkout = tmp_path / "sources" / "sgl_kernel" / ("2" * 40)
    for path, text in {
        "csrc/gemm.cu": b"int tile = 64;\n",
        "python/sgl_kernel/gemm.py": b"installed = False\n",
        "python/sgl_kernel/common_ops.so": b"\x7fELF",
        "CMakeLists.txt": b"project(sgl)\n",
    }.items():
        (checkout / AOT / path).parent.mkdir(parents=True, exist_ok=True)
        (checkout / AOT / path).write_bytes(text)
    tree = NativeTree(SourceRoot(AOT, checkout / AOT), SGLANG, "2" * 40)
    installed = PackageBase("sgl_kernel", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed, native={"sgl_kernel": tree})
    base.build(["sgl_kernel"])
    return base


def test_the_base_lays_the_native_tree_beside_the_installed_package_without_replacing_it(tmp_path):
    base = _native_base(tmp_path)

    assert (base.path / AOT / "csrc" / "gemm.cu").read_bytes() == b"int tile = 64;\n"
    assert (base.path / AOT / "python" / "sgl_kernel" / "gemm.py").read_bytes() == b"installed = True\n"
    assert not (base.path / AOT / "python" / "sgl_kernel" / "common_ops.so").exists()


def test_a_change_to_the_native_tree_is_in_the_component_s_diff_and_names_its_commit(tmp_path):
    base = _native_base(tmp_path)
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / AOT / "csrc" / "gemm.cu").write_bytes(b"int tile = 128;\n")
    (worktree.path / AOT / "python" / "sgl_kernel" / "gemm.py").write_bytes(b"installed = 2\n")

    patch = worktree.diff("sgl_kernel")

    assert f"b/{AOT}/csrc/gemm.cu" in patch and f"b/{AOT}/python/sgl_kernel/gemm.py" in patch
    assert worktree.native_upstream("sgl_kernel") == Upstream(SGLANG, "2" * 40)
    assert worktree.native_upstream("vllm") is None


def test_a_native_tree_inside_the_framework_s_tree_is_the_native_component_s_alone(tmp_path):
    """SGLang's develop install holds sgl-kernel's tree, so the framework lays it out first: a change there is
    sgl_kernel's, and the framework's diff leaves it out."""
    sglang = tmp_path / "checkout" / "python" / "sglang"
    for path in ("srt/server.py", "kernels/aot/csrc/gemm.cu", "kernels/aot/python/sgl_kernel/gemm.py"):
        (sglang / path).parent.mkdir(parents=True, exist_ok=True)
        (sglang / path).write_bytes(b"checkout = 1\n")
    kernel = tmp_path / "site-packages" / "sgl_kernel"
    kernel.mkdir(parents=True)
    (kernel / "gemm.py").write_bytes(b"installed = True\n")
    installed = {
        "sglang": PackageBase("sglang", sglang, None, None),
        "sgl_kernel": PackageBase("sgl_kernel", kernel, None, None),
    }
    tree = NativeTree(SourceRoot(AOT, sglang / "kernels" / "aot"), SGLANG, "2" * 40)
    base = SourceBase(
        tmp_path / "source", locator=lambda layout: installed[layout.import_name], native={"sgl_kernel": tree}
    )
    base.build(["sglang", "sgl_kernel"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / AOT / "csrc" / "gemm.cu").write_bytes(b"tile = 128\n")
    (worktree.path / "python" / "sglang" / "srt" / "server.py").write_bytes(b"checkout = 2\n")

    assert (worktree.path / AOT / "python" / "sgl_kernel" / "gemm.py").read_bytes() == b"installed = True\n"
    framework, native = worktree.diff("sglang"), worktree.diff("sgl_kernel")
    assert "srt/server.py" in framework and "kernels/aot" not in framework
    assert f"b/{AOT}/csrc/gemm.cu" in native and "srt/server.py" not in native
