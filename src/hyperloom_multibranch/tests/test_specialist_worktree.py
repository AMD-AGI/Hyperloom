# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess

import pytest

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import COMPONENT_LAYOUTS, LocalBlobStore, PackageBase, apply_patch_stack
from hyperloom_multibranch.specialist.worktree import SourceBase

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
