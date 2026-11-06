# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch to sgl-kernel's C++ sources, which lie in the native build's checkout rather than beside the package."""

import difflib

from hyperloom.contracts.recipe import ComponentPatches
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

SGLANG = "https://github.com/sgl-project/sglang.git"
PIN = "2" * 40
AOT = "python/sglang/kernels/aot"
KERNEL = f"{AOT}/csrc/gemm/gemm.cu"
OPS = f"{AOT}/python/sgl_kernel/gemm.py"
LINES = [f"{name} = {value}\n" for value, name in enumerate("abcdefgh", 1)]


def _diff(path: str, line: int) -> str:
    after = [*LINES[:line], "changed\n", *LINES[line + 1 :]]
    return f"diff --git a/{path} b/{path}\n" + "".join(difflib.unified_diff(LINES, after, f"a/{path}", f"b/{path}"))


def _repository(tmp_path, *, native: bool) -> GitExperimentRepo:
    checkout = tmp_path / "sources" / "sgl_kernel" / PIN
    package = tmp_path / "site-packages" / "sgl_kernel"
    for path in (checkout / KERNEL, checkout / OPS, package / "gemm.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(LINES))
    installed = PackageBase("sgl_kernel", package, None, None)
    trees = {"sgl_kernel": NativeTree(SourceRoot(AOT, checkout / AOT), SGLANG, PIN)} if native else {}
    blobs = LocalBlobStore(tmp_path / "blobs")
    return GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _: installed, native=trees)


async def test_a_csrc_hunk_is_outside_every_root_without_a_native_tree(tmp_path):
    repo = _repository(tmp_path, native=False)

    error = await repo.apply_error(ComponentPatches("sgl_kernel", SGLANG, PIN), _diff(KERNEL, 2))

    assert error is not None and "outside" in error


async def test_a_patch_spanning_the_package_and_csrc_applies_over_the_native_tree(tmp_path):
    repo = _repository(tmp_path, native=True)
    stored = await repo.store_patch("sgl_kernel", _diff(KERNEL, 0) + _diff(OPS, 0))
    stack = ComponentPatches("sgl_kernel", SGLANG, PIN, (stored,))

    assert await repo.apply_error(stack, _diff(KERNEL, 7) + _diff(OPS, 7)) is None
    error = await repo.apply_error(stack, _diff(KERNEL, 0))
    assert error is not None and "does not apply" in error
