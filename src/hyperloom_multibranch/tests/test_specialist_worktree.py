# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    LocalBlobStore,
    PackageBase,
    UnsupportedComponentError,
    apply_patch_stack,
)
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.worktree import SourceBase, Upstream, installed_upstream, source_unkept
from hyperloom_multibranch.tests.agent_rig import ROOT

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
    # What a sandbox binds for the links is the base's own record, not what a worktree's links point at now.
    assert sorted(worktree.linked) == sorted(
        [installed.path / "_C.abi3.so", installed.path / "attn" / "_fa_C.cpython-312-x86_64-linux-gnu.so"]
    )
    (tmp_path / "elsewhere").write_bytes(b"chosen by whoever writes the worktree")
    (worktree.path / "vllm" / "_C.abi3.so").unlink()
    (worktree.path / "vllm" / "_C.abi3.so").symlink_to(tmp_path / "elsewhere")
    again = base.worktree("s1", tmp_path / "s1").linked
    assert installed.path / "_C.abi3.so" in again and tmp_path / "elsewhere" not in again


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


def test_the_base_and_each_worktree_are_the_group_s_to_write_whatever_the_umask(tmp_path, installed, monkeypatch):
    with open("/proc/self/status", encoding="ascii") as status:
        umask = next(int(line.split()[1], 8) for line in status if line.startswith("Umask:"))
    assert umask & 0o020, "the test needs a umask that keeps group write off what a plain create makes"
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    branches = tmp_path / "MB-branches"
    branches.mkdir()
    branches.chmod(0o2775)
    base = SourceBase(tmp_path / "MB-source")
    base.build(["vllm"])

    worktree = base.worktree("s1", branches / "s1" / "worktree")
    worktree.sync("c0", [])

    def group_writes(path: Path) -> bool:
        return path.lstat().st_mode & 0o020 == 0o020

    assert stat.S_IMODE(base.path.stat().st_mode) == 0o2775
    git_dirs = [base.path / ".git" / name for name in ("objects", "refs", "worktrees")]
    assert all(map(group_writes, git_dirs)), [oct(path.stat().st_mode) for path in git_dirs]
    assert stat.S_IMODE((branches / "s1").stat().st_mode) == 0o2775, "the branch's directory, for its dev sandboxes"
    assert group_writes(worktree.path) and group_writes(worktree.path / "vllm" / "config.py")


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


def test_no_program_a_sandbox_writes_into_the_worktree_s_repository_runs_as_the_app(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    root = worktree.commit()
    ran = tmp_path / "ran"
    _plant_programs(base.path / ".git", worktree.path, ran)
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")

    patch = worktree.diff("vllm")
    changed = worktree.changed("vllm")
    since = worktree.diff_from(root, "vllm")
    worktree.sync("c1", [patch.encode()])

    assert not ran.exists(), ran.read_text()
    assert "+max_num_seqs = 256" in patch and "+max_num_seqs = 256" in since
    assert changed.written == ("vllm/config.py",)
    assert worktree.head() == "c1"
    assert (worktree.path / "vllm" / "config.py").read_bytes() == b"max_num_seqs = 256\n"


def test_a_repository_made_a_partial_clone_by_its_config_fetches_nothing_behind_the_app_s_command(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    ran = tmp_path / "ran"
    transport = tmp_path / "transport"
    transport.write_text(f"#!/bin/sh\necho ran >> {ran}\nexit 1\n")
    transport.chmod(0o755)
    git = base.path / ".git"
    with (git / "config").open("a") as config:
        config.write(
            "[core]\n\trepositoryformatversion = 1\n"
            f"\tsshCommand = {transport}\n[extensions]\n\tpartialClone = promisor\n"
            '[remote "promisor"]\n\turl = ssh://example.invalid/repo\n\tpromisor = true\n'
        )
    blob = subprocess.run(
        ["git", "-C", str(base.path), "rev-parse", "installed:vllm/config.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (git / "objects" / blob[:2] / blob[2:]).unlink()
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")

    with pytest.raises(worktree_module.WorktreeError):
        worktree.diff("vllm")

    assert not ran.exists()


@pytest.mark.skipif(os.geteuid() != 0, reason="a repository of another owner needs root to chown")
def test_a_repository_trusted_by_the_command_s_own_safe_directory_has_its_filter_drivers_blanked_too(tmp_path):
    repo = tmp_path / "image-checkout"
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    ran = tmp_path / "ran"
    with (repo / ".git" / "config").open("a") as config:
        config.write(f'[filter "own"]\n\tclean = {_recording_filter(tmp_path, ran)}\n')
    (repo / ".gitattributes").write_text("* filter=own\n")
    (repo / "file").write_text("x\n")
    for path in (repo, *repo.rglob("*")):
        os.lchown(path, 65534, 65534)  # another owner's, as the image's checkout is: git trusts it only by -c

    worktree_module._git(repo, "-c", "safe.directory=*", "add", "--all")

    assert not ran.exists()


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


def _plain_git(cwd: Path, *argv: str, stdin: str | None = None) -> str:
    """git as a user runs it, with nothing but the repository's config: what proves a plant is live."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t")
    env.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    done = subprocess.run(["git", *argv], cwd=cwd, env=env, input=stdin, capture_output=True, text=True, check=False)
    return done.stdout


def test_a_signed_head_the_worktree_was_synced_to_starts_no_signature_program(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    ran = tmp_path / "ran"
    synced = _plain_git(worktree.path, "rev-parse", "refs/worktree/hyperloom/synced").strip()
    content = _plain_git(worktree.path, "rev-parse", f"{synced}^{{tree}}").strip()
    forged = (
        f"tree {content}\nparent {synced}\nauthor t <t@t> 1 +0000\ncommitter t <t@t> 1 +0000\n"
        "gpgsig -----BEGIN PGP SIGNATURE-----\n iQ==\n -----END PGP SIGNATURE-----\n\nhead c0\n"
    )
    signed = _plain_git(worktree.path, "hash-object", "-t", "commit", "-w", "--stdin", stdin=forged).strip()
    _plain_git(worktree.path, "update-ref", "refs/worktree/hyperloom/synced", signed)
    with (base.path / ".git" / "config").open("a") as config:
        config.write(f"[log]\n\tshowSignature = true\n[gpg]\n\tprogram = {_recording_filter(tmp_path, ran)}\n")

    assert worktree.head() == "c0"
    assert not ran.exists(), ran.read_text()
    _plain_git(worktree.path, "log", "-1", "--format=%s", signed)
    assert ran.exists(), "plain git runs the signature program, so the plant is live"


def test_a_submodule_the_installed_branch_holds_is_not_recursed_into_by_a_sync(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    ran = tmp_path / "ran"
    source = tmp_path / "subsource"
    _plain_git(tmp_path, "init", "-q", str(source))
    (source / "f.txt").write_text("a\n")
    (source / ".gitattributes").write_text("* filter=evil\n")
    _plain_git(source, "add", "-A")
    _plain_git(source, "commit", "-qm", "sub")
    # A sandbox moves the installed branch to a commit with a submodule, whose repository lies in the worktree's admin
    # directory with a filter of its own, and has every command recurse into submodules.
    _plain_git(worktree.path, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(source), "sub")
    _plain_git(worktree.path, "commit", "-qm", "with a submodule")
    _plain_git(worktree.path, "update-ref", "refs/heads/installed", "HEAD")
    sub = worktree.path / "sub"
    _plain_git(sub, "config", "filter.evil.smudge", str(_recording_filter(tmp_path, ran)))
    _plain_git(sub, "config", "filter.evil.clean", "cat")
    (sub / "f.txt").write_text("changed\n")
    _plain_git(base.path, "config", "submodule.recurse", "true")

    worktree.sync("c1", [])

    assert not ran.exists(), ran.read_text()
    _plain_git(worktree.path, "reset", "-q", "--hard", "installed")
    assert ran.exists(), "plain git recurses into the submodule and runs its smudge, so the plant is live"


def test_a_repository_nested_in_the_worktree_runs_nothing_as_the_change_is_read(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    worktree.sync("c0", [])
    root = worktree.commit()
    ran = tmp_path / "ran"
    # The agent commits a repository of its own inside the package, then gives it a filter and a local change.
    nested = worktree.path / "vllm" / "vendored"
    _plain_git(tmp_path, "init", "-q", str(nested))
    (nested / "f.txt").write_text("a\n")
    _plain_git(nested, "add", "-A")
    _plain_git(nested, "commit", "-qm", "nested")
    _plain_git(worktree.path, "add", "vllm/vendored")
    _plain_git(worktree.path, "commit", "-qm", "an agent's commit")
    _plain_git(nested, "config", "filter.inner.clean", str(_recording_filter(tmp_path, ran)))
    (nested / ".gitattributes").write_text("* filter=inner\n")
    (nested / "f.txt").write_text("b\n")
    (worktree.path / "vllm" / "config.py").write_bytes(b"max_num_seqs = 256\n")

    patch = worktree.diff("vllm")
    changed = worktree.changed("vllm")
    since = worktree.diff_from(root, "vllm")

    assert not ran.exists(), ran.read_text()
    assert "+max_num_seqs = 256" in patch and "+max_num_seqs = 256" in since
    assert "vllm/config.py" in changed.written
    _plain_git(worktree.path, "add", "--all", "--", "vllm")
    assert ran.exists(), "plain git's add checks the nested repository with its own filter, so the plant is live"


def test_a_filter_driver_git_cannot_be_told_to_blank_refuses_the_command(tmp_path, base):
    worktree = base.worktree("s1", tmp_path / "s1")
    with (base.path / ".git" / "config").open("a") as config:
        config.write('[filter "a=b"]\n\tclean = false\n')

    with pytest.raises(worktree_module.WorktreeError, match="'='"):
        worktree.sync("c0", [])


def test_a_framework_the_host_does_not_have_installed_has_no_source_change_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(_layout):
        raise UnsupportedComponentError("package vllm is not installed in this image")

    monkeypatch.setattr(worktree_module, "locate_installed_package", absent)

    assert "package vllm is not installed in this image" in source_unkept("vllm", replace(ROOT, components=()))


NIGHTLY = "dd4e40142a3280ef404493e5117bde0c60c6e389"


@pytest.fixture
def wheel(tmp_path, monkeypatch):
    """vLLM installed from a nightly wheel: no checkout, its version naming only an abbreviated commit."""
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "vllm" / "__init__.py").write_bytes(b"")
    version = "0.33.0rc1.dev711+gdd4e40142"
    info = site / f"vllm-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: vllm\nVersion: {version}\nProject-URL: Homepage, {VLLM}\n"
        "Project-URL: Slack, https://slack.vllm.ai/\n"
    )
    monkeypatch.setattr(worktree_module, "_RESOLVED", {})
    return PackageBase("vllm", site / "vllm", "vllm", version)


def _image(monkeypatch, ref):
    monkeypatch.setenv("HYPERLOOM_ACTIVE_IMAGE_REF", ref)
    monkeypatch.setenv("HYPERLOOM_ACTIVE_IMAGE_DIGEST", "sha256:" + "c" * 64)


def _fetched(monkeypatch, line):
    asked = []

    def head(url):
        asked.append(url)
        return line

    monkeypatch.setattr(worktree_module, "_patch_head", head)
    return asked


def test_a_wheel_s_upstream_is_the_full_commit_the_active_image_s_reference_names(monkeypatch, wheel):
    _image(monkeypatch, f"docker.io/vllm/vllm-openai-rocm:nightly-rocm100-{NIGHTLY}")
    asked = _fetched(monkeypatch, None)

    assert installed_upstream(wheel) == Upstream(VLLM, NIGHTLY)
    assert asked == []


def test_an_image_commit_the_wheel_s_version_does_not_name_is_not_taken(monkeypatch, wheel):
    _image(monkeypatch, "docker.io/vllm/vllm-openai-rocm:nightly-rocm100-" + "e" * 40)
    _fetched(monkeypatch, None)

    assert installed_upstream(wheel) is None


def test_a_wheel_s_upstream_falls_back_to_the_commit_its_repository_resolves_once(monkeypatch, wheel):
    _image(monkeypatch, "docker.io/vllm/vllm-openai-rocm:nightly")
    asked = _fetched(monkeypatch, f"From {NIGHTLY} Mon Sep 17 00:00:00 2001\n")

    assert installed_upstream(wheel) == installed_upstream(wheel) == Upstream(VLLM, NIGHTLY)
    assert asked == [f"{VLLM}/commit/dd4e40142.patch"]


def test_a_resolved_commit_the_version_does_not_name_is_not_taken(monkeypatch, wheel):
    monkeypatch.delenv("HYPERLOOM_ACTIVE_IMAGE_REF", raising=False)
    _fetched(monkeypatch, "From " + "e" * 40 + " Mon Sep 17 00:00:00 2001\n")

    assert installed_upstream(wheel) is None


class _Reply:
    def __init__(self, line: bytes | None = None) -> None:
        self._line = line

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def readline(self, _limit: int) -> bytes:
        if self._line is None:
            raise ConnectionResetError(104, "Connection reset by peer")
        return self._line


def test_a_wheel_s_upstream_github_failed_to_resolve_once_is_looked_up_again(monkeypatch, wheel):
    monkeypatch.delenv("HYPERLOOM_ACTIVE_IMAGE_REF", raising=False)
    asked = []

    def flaky(url, timeout):
        asked.append(url)
        if len(asked) == 1:
            raise worktree_module.urllib.error.URLError("temporary failure in name resolution")
        return _Reply(f"From {NIGHTLY} Mon Sep 17 00:00:00 2001\n".encode())

    monkeypatch.setattr(worktree_module.urllib.request, "urlopen", flaky)

    assert installed_upstream(wheel) is None
    assert installed_upstream(wheel) == installed_upstream(wheel) == Upstream(VLLM, NIGHTLY)
    assert len(asked) == 2


def test_a_patch_whose_connection_drops_while_its_body_is_read_is_unread(monkeypatch):
    monkeypatch.setattr(worktree_module.urllib.request, "urlopen", lambda url, timeout: _Reply())

    assert worktree_module._patch_head(f"{VLLM}/commit/dd4e40142.patch") is None


def test_a_wheel_neither_the_image_nor_its_repository_resolves_names_no_upstream(monkeypatch, wheel):
    monkeypatch.delenv("HYPERLOOM_ACTIVE_IMAGE_REF", raising=False)

    def unreachable(url, timeout):
        raise worktree_module.urllib.error.URLError("no route to host")

    monkeypatch.setattr(worktree_module.urllib.request, "urlopen", unreachable)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: wheel)

    assert installed_upstream(wheel) is None
    assert "names no upstream commit a checkout holds" in source_unkept("vllm", replace(ROOT, components=()))
