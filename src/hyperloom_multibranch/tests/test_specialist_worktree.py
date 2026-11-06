# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import COMPONENT_LAYOUTS, LocalBlobStore, PackageBase, apply_patch_stack
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.specialist import worktree as worktree_module
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
def base(tmp_path, installed, monkeypatch):
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
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


def test_a_diff_from_a_commit_holds_new_files_under_a_nested_package_and_leaves_out_built_ones(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "sglang"
    (package / "srt").mkdir(parents=True)
    (package / "srt" / "a.py").write_bytes(b"x = 1\n")
    installed = PackageBase("sglang", package, None, None)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
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


def _native_base(tmp_path, monkeypatch):
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
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source", native={"sgl_kernel": tree})
    base.build(["sgl_kernel"])
    return base


def test_the_base_lays_the_native_tree_beside_the_installed_package_without_replacing_it(tmp_path, monkeypatch):
    base = _native_base(tmp_path, monkeypatch)

    assert (base.path / AOT / "csrc" / "gemm.cu").read_bytes() == b"int tile = 64;\n"
    assert (base.path / AOT / "python" / "sgl_kernel" / "gemm.py").read_bytes() == b"installed = True\n"
    assert not (base.path / AOT / "python" / "sgl_kernel" / "common_ops.so").exists()


def test_a_change_to_the_native_tree_is_in_the_component_s_diff_and_names_its_commit(tmp_path, monkeypatch):
    base = _native_base(tmp_path, monkeypatch)
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / AOT / "csrc" / "gemm.cu").write_bytes(b"int tile = 128;\n")
    (worktree.path / AOT / "python" / "sgl_kernel" / "gemm.py").write_bytes(b"installed = 2\n")

    patch = worktree.diff("sgl_kernel")

    assert f"b/{AOT}/csrc/gemm.cu" in patch and f"b/{AOT}/python/sgl_kernel/gemm.py" in patch
    assert worktree.native_upstream("sgl_kernel") == Upstream(SGLANG, "2" * 40)
    assert worktree.native_upstream("vllm") is None


def test_a_native_tree_inside_the_framework_s_tree_is_the_native_component_s_alone(tmp_path, monkeypatch):
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
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda layout: installed[layout.import_name])
    base = SourceBase(tmp_path / "source", native={"sgl_kernel": tree})
    base.build(["sglang", "sgl_kernel"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    (worktree.path / AOT / "csrc" / "gemm.cu").write_bytes(b"tile = 128\n")
    (worktree.path / "python" / "sglang" / "srt" / "server.py").write_bytes(b"checkout = 2\n")

    assert (worktree.path / AOT / "python" / "sgl_kernel" / "gemm.py").read_bytes() == b"installed = True\n"
    framework, native = worktree.diff("sglang"), worktree.diff("sgl_kernel")
    assert "srt/server.py" in framework and "kernels/aot" not in framework
    assert f"b/{AOT}/csrc/gemm.cu" in native and "srt/server.py" not in native


def test_a_worktree_links_the_installed_package_s_compiled_extensions_without_tracking_them(
    tmp_path, installed, monkeypatch
):
    (installed.path / "_C.abi3.so").write_bytes(b"\x7fELF ext")
    (installed.path / "attn").mkdir()
    (installed.path / "attn" / "_fa_C.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF fa")
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
    base.build(["vllm"])
    worktree = base.worktree("s1", tmp_path / "s1")

    assert (worktree.path / "vllm" / "_C.abi3.so").read_bytes() == b"\x7fELF ext"
    assert (worktree.path / "vllm" / "attn" / "_fa_C.cpython-312-x86_64-linux-gnu.so").read_bytes() == b"\x7fELF fa"
    assert not (worktree.path / "vllm" / "_C.so").exists(), "a library that is no Python extension stays out"
    assert worktree.diff("vllm") == ""
    worktree.sync("c0", [])
    assert (worktree.path / "vllm" / "_C.abi3.so").exists(), "a sync keeps it"


def test_source_only_keeps_the_component_s_files_byte_for_byte_and_names_the_rest(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    own = "diff --git a/vllm/config.py b/vllm/config.py\n--- a/vllm/config.py\n+++ b/vllm/config.py\n@@ -1 +1 @@\n-a\n+b\r\n"
    scratch = "diff --git a/eval/run.log b/eval/run.log\nnew file mode 100644\n--- /dev/null\n+++ b/eval/run.log\n@@ -0,0 +1 @@\n+x\r y\n"
    near = "diff --git a/vllm_extra/x.py b/vllm_extra/x.py\n--- a/vllm_extra/x.py\n+++ b/vllm_extra/x.py\n@@ -1 +1 @@\n-1\n+2\n"

    kept, outside = worktree.source_only("vllm", scratch + own + near)

    assert kept == own
    assert outside == ("eval/run.log", "vllm_extra/x.py")


@pytest.fixture
def hostile_home(tmp_path, monkeypatch):
    """An operator whose git signs every commit, runs a pre-commit hook that refuses it, ignores ``lpm.py``, prefixes
    diff paths with c/ and i/, and seeds every new repository with that hook."""
    home = tmp_path / "home"
    hooks = home / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n")
    (hooks / "pre-commit").chmod(0o755)
    (home / "ignore").write_text("lpm.py\n")
    (home / ".gitconfig").write_text(
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n"
        f"[core]\n\thooksPath = {hooks}\n\texcludesFile = {home / 'ignore'}\n"
        f"[diff]\n\tmnemonicPrefix = true\n[init]\n\ttemplateDir = {home}\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "xdg"))
    return home


def test_the_operator_s_git_config_changes_nothing_a_worktree_builds_or_patches(
    tmp_path, installed, monkeypatch, hostile_home
):
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    base = SourceBase(tmp_path / "source")
    base.build(["vllm"])
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    root = worktree.commit()
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")
    (worktree.path / "vllm" / "lpm.py").write_bytes(b"policy = 'lpm'\n")

    patch = worktree.diff("vllm")

    assert "b/vllm/config.py" in patch and "b/vllm/lpm.py" in patch
    kept, outside = worktree.source_only("vllm", worktree.diff_from(root, "vllm"))
    assert ("b/vllm/config.py" in kept, "b/vllm/lpm.py" in kept, outside) == (True, True, ())
    worktree.sync("c1", [patch.encode()])
    assert (worktree.head(), (worktree.path / "vllm" / "lpm.py").exists()) == ("c1", True)


def _recording_filter(where: Path, ran: Path) -> Path:
    """A program that notes it ran in ``ran`` and passes its input through, as a filter or a text conversion would."""
    program = where / "recording-filter"
    program.write_text(f"#!/bin/sh\necho ran >> {ran}\ncat\n")
    program.chmod(0o755)
    return program


def _plant_programs(base_git: Path, tree: Path, ran: Path) -> None:
    """What a sandbox that writes the worktree and its repository can plant: a config naming a program for every hook
    of git it can name -- filter drivers under names of its own, an fsmonitor, a text conversion, a signing program on
    commit, a pager -- an attributes file that applies them to every path, and executable hooks."""
    program = _recording_filter(base_git.parent.parent, ran)
    marker = f"#!/bin/sh\necho ran >> {ran}\n"
    with (base_git / "config").open("a") as config:
        config.write(
            f'[filter "own"]\n\tclean = {program}\n\tsmudge = {program}\n'
            f'[filter "other.name"]\n\tprocess = {program}\n'
            f"[core]\n\tfsmonitor = {base_git / 'hooks' / 'fsmonitor'}\n\tpager = {program}\n"
            f'[diff "own"]\n\ttextconv = {program}\n\tcommand = {program}\n'
            f"[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = {base_git / 'hooks' / 'fsmonitor'}\n"
        )
    (base_git / "hooks").mkdir(exist_ok=True)
    for hook in ("fsmonitor", "pre-commit", "post-commit", "post-checkout", "reference-transaction"):
        (base_git / "hooks" / hook).write_text(marker)
        (base_git / "hooks" / hook).chmod(0o755)
    (tree / ".gitattributes").write_text("* filter=own diff=own\n*.py filter=other.name\n")


def _as_another_accounts(*trees: Path) -> None:
    for path in (path for tree in trees for path in (tree, *tree.rglob("*"))):
        os.lchown(path, 65534, 65534)


@pytest.mark.skipif(os.geteuid() != 0, reason="a repository of another owner needs root to chown")
def test_a_worktree_another_account_of_the_group_made_is_synced_and_diffed_with_no_program_of_it_run(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    ran = tmp_path / "ran"
    _plant_programs(base.path / ".git", worktree.path, ran)
    # The run's earlier leg, another account of the group, made the base and the worktree.
    _as_another_accounts(base.path, worktree.path)
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")

    patch = worktree.diff("vllm")
    worktree.sync("c1", [patch.encode()])

    assert "+max_num_seqs = 256" in patch
    assert worktree.head() == "c1"
    assert base.worktree("s2", tmp_path / "s2").head() is None
    assert not ran.exists(), ran.read_text()
