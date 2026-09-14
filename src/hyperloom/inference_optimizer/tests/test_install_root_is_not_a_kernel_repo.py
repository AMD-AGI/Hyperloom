# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``.git`` inside site-packages must not make the install a kernel repo.

Forge's fusion lane runs ``git init`` in the tree it edits so its iteration
loop can commit and revert. When that tree is the Python install, the ``.git``
it leaves behind covers every installed package at once, and the repo-root walk
-- which only looks for the nearest ``.git`` -- reports the whole install as
the kernel's repo ever after.

That is not a hypothetical. One fusion run on 2026-09-10 left such a repo, and
every kernel attempt in the four sessions that followed died the same way: the
untracked kernel file was refused a worktree, the non-git fallback honoured the
install root as an explicit repo, copied 14 GB, and spent the scaffold's whole
120s budget in ``git add -A``. All seven failures landed within 132.5-135.0s of
each other, and every one was reported as "not a clean git checkout or
source_file not tracked" -- a message about the *first* attempt that says
nothing about the timeout that actually stopped it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.common.git_safety import is_installed_packages_root, repo_root


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)


@pytest.mark.parametrize("name", ["site-packages", "dist-packages"])
def test_an_install_root_is_recognised_by_name(tmp_path, name):
    """Both spellings; the predicate reads the directory's own name."""
    assert is_installed_packages_root(tmp_path / name) is True


def test_a_project_checkout_is_not_an_install_root(tmp_path):
    """The guard must not swallow the ordinary case it sits in front of."""
    assert is_installed_packages_root(tmp_path / "Hyperloom") is False
    assert is_installed_packages_root("") is False
    assert is_installed_packages_root(None) is False


def test_an_install_root_holding_a_git_is_still_an_install_root(tmp_path):
    """Carrying a ``.git`` is exactly the case, not an exemption from it."""
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    _init_repo(site)
    assert repo_root(site) == str(site)  # it really is a git repo
    assert is_installed_packages_root(site) is True


def test_the_repo_walk_declines_an_install_root(tmp_path, monkeypatch):
    """A kernel inside a git-ified install resolves to no repo, not to the install.

    Returning "" is what lets the caller fall through to deriving the one
    package that owns the file, instead of treating 14 GB of unrelated
    packages as the repo to copy.
    """
    from hyperloom.orchestrator.kernel import request_handlers

    site = tmp_path / "lib" / "python3.12" / "site-packages"
    kernel = site / "aiter_meta" / "csrc" / "include" / "custom_all_reduce.cuh"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("__global__ void k() {}\n", encoding="utf-8")
    _init_repo(site)

    assert request_handlers._find_repo_root_for_source(str(kernel)) == ""


def test_the_repo_walk_still_finds_a_real_checkout(tmp_path):
    """The guard is scoped to install roots; a normal repo is unaffected."""
    from hyperloom.orchestrator.kernel import request_handlers

    repo = tmp_path / "vllm"
    src = repo / "csrc" / "attention.cu"
    src.parent.mkdir(parents=True)
    src.write_text("__global__ void a() {}\n", encoding="utf-8")
    _init_repo(repo)

    assert request_handlers._find_repo_root_for_source(str(src)) == str(repo)


def test_the_nogit_fallback_ignores_an_install_root_as_kernel_repo(tmp_path):
    """An explicit install-root ``kernel_repo`` must not become a whole-tree copy.

    The docstring of the fallback already promises never to copy the entire
    dist-packages directory, but that promise only covered the derived path;
    an explicit ``kernel_repo`` took the "copy the whole repo" branch and
    defeated it.
    """
    from hyperloom.agents.kernel.tools.backends import forge_submit

    site = tmp_path / "site-packages"
    pkg = site / "aiter_meta"
    kernel = pkg / "csrc" / "include" / "custom_all_reduce.cuh"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("__global__ void k() {}\n", encoding="utf-8")
    # A second, unrelated package: the thing a whole-tree copy would sweep up.
    bystander = site / "torch" / "big.bin"
    bystander.parent.mkdir(parents=True)
    bystander.write_bytes(b"x" * 4096)

    out = tmp_path / "out"
    out.mkdir()
    result = forge_submit._prepare_worktree_nogit(str(kernel), str(site), out, "forge/t")

    assert result is not None
    scratch_dir, scratch_kernel, _base_commit = result
    # The kernel is there to edit -- the layout under the scratch root is the
    # derived one and is not what this test pins.
    assert Path(scratch_kernel).read_text(encoding="utf-8") == "__global__ void k() {}\n"
    # ...and the unrelated package is not, which is the whole point: honouring
    # the install root as the repo is what made this a whole-install copy.
    assert not (Path(scratch_dir) / "torch").exists()
