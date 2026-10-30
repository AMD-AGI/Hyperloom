# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Layouts of the components beyond the first three (triton, torch, sgl_kernel, flydsl): their sections place a patch's
upstream paths in an install modelled on the measured image, and a patch that names a native or binary file is refused
with its path, never laid beside the binary that runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    ComponentLayout,
    LocalBlobStore,
    PackageBase,
    PatchApplyError,
    compile_environment,
)
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with, write_tree

OLD, NEW = "x = 1\n", "x = 2\n"
SGL_KERNEL_PREFIX = "python/sglang/kernels/aot/python/sgl_kernel"
#: What the measured image installs of each component (names from its site-packages), only the files a test reads.
IMAGE_FILES = {
    "triton": {
        "__init__.py": "",
        "language/core.py": OLD,
        "backends/compiler.py": OLD,
        "backends/amd/compiler.py": OLD,
        "backends/amd/driver.py": OLD,
        "backends/amd/driver.c": "int main(void) { return 0; }\n",
        "language/extra/hip/libdevice.py": OLD,
        "_C/libtriton.so": "binary",
    },
    "torch": {
        "__init__.py": "",
        "_inductor/config.py": OLD,
        "lib/libtorch_hip.so": "binary",
        "lib/aotriton.images/a.aks2": "b",
    },
    "sgl_kernel": {"__init__.py": "", "gemm.py": OLD, "common_ops.cpython-310-x86_64-linux-gnu.so": "binary"},
    "flydsl": {
        "__init__.py": "",
        "compiler/jit_function.py": OLD,
        "_mlir/_mlir_libs/_mlir.cpython-310-x86_64-linux-gnu.so": "b",
    },
}


@pytest.fixture()
def blobs(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


def compile_patch(tmp_path: Path, blobs: LocalBlobStore, component: str, files: dict[str, str]):
    """Compile a one-patch stack that changes each upstream path of ``files`` from ``OLD`` to the text given."""
    patch = git_patch(tmp_path / "up", dict.fromkeys(files, OLD), files)
    installed = write_tree(tmp_path / "image" / COMPONENT_LAYOUTS[component].import_name, IMAGE_FILES[component])
    base = PackageBase(installed.name, installed, installed.name, "1.0")
    recipe = recipe_with((put_patch(blobs, "p.patch", patch),), component=component)
    return compile_environment(
        recipe, RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=lambda layout: base
    ).roots


@pytest.mark.parametrize(
    ("component", "upstream", "installed"),
    [
        ("triton", "python/triton/language/core.py", "language/core.py"),
        ("triton", "third_party/amd/backend/compiler.py", "backends/amd/compiler.py"),
        ("triton", "third_party/amd/language/hip/libdevice.py", "language/extra/hip/libdevice.py"),
        ("torch", "torch/_inductor/config.py", "_inductor/config.py"),
        ("sgl_kernel", f"{SGL_KERNEL_PREFIX}/gemm.py", "gemm.py"),
        ("flydsl", "python/flydsl/compiler/jit_function.py", "compiler/jit_function.py"),
    ],
)
def test_a_python_patch_is_laid_at_the_installed_path_its_section_names(
    tmp_path, blobs, component, upstream, installed
):
    (root,) = compile_patch(tmp_path, blobs, component, {upstream: NEW})

    (entry,) = root.entries
    assert (root.id, root.import_name, entry.path, entry.op) == (
        component,
        COMPONENT_LAYOUTS[component].import_name,
        installed,
        "upsert",
    )
    assert blobs.get(entry.blob) == NEW.encode()


def test_one_patch_may_change_files_of_two_sections_of_one_component(tmp_path, blobs):
    files = {"python/triton/language/core.py": NEW, "third_party/amd/backend/driver.py": NEW}

    (root,) = compile_patch(tmp_path, blobs, "triton", files)

    assert sorted(entry.path for entry in root.entries) == ["backends/amd/driver.py", "language/core.py"]


@pytest.mark.parametrize(
    ("component", "upstream", "pattern"),
    [
        ("triton", "third_party/amd/backend/driver.c", "*.c"),
        ("triton", "python/triton/_C/libtriton.so", "*.so"),
        ("triton", "python/triton/CMakeLists.txt", "CMakeLists.txt"),
        ("torch", "torch/csrc/autograd/engine.cpp", "*.cpp"),
        ("torch", "torch/lib/libtorch_hip.so.2", "*.so.*"),
        ("torch", "torch/include/ATen/Tensor.h", "*.h"),
        ("torch", "torch/lib/aotriton.images/a.aks2", "*.aks2"),
        ("sgl_kernel", f"{SGL_KERNEL_PREFIX}/common_ops.cpython-310-x86_64-linux-gnu.so", "*.so"),
        ("sgl_kernel", f"{SGL_KERNEL_PREFIX}/setup.py", "setup.py"),
        ("flydsl", "python/flydsl/_mlir/_mlir_libs/_mlir.cpython-310-x86_64-linux-gnu.so", "*.so"),
        ("flydsl", "python/flydsl/runtime/kernel.hsaco", "*.hsaco"),
    ],
)
def test_a_native_or_binary_file_is_refused_naming_its_path(tmp_path, blobs, component, upstream, pattern):
    with pytest.raises(PatchApplyError) as refused:
        compile_patch(tmp_path, blobs, component, {upstream: "text\n"})

    assert f"{upstream!r} is a native or binary file ({pattern})" in str(refused.value)


def test_a_file_outside_every_section_is_refused_naming_the_sections(tmp_path, blobs):
    with pytest.raises(PatchApplyError) as refused:
        compile_patch(tmp_path, blobs, "triton", {"docs/index.py": NEW})

    message = str(refused.value)
    assert "'docs/index.py' is outside the installed package" in message
    assert "python/triton/, third_party/amd/backend/, third_party/amd/language/hip/" in message


def test_two_upstream_paths_of_one_installed_file_are_refused(tmp_path, blobs):
    files = {"python/triton/backends/amd/compiler.py": NEW, "third_party/amd/backend/compiler.py": NEW}

    with pytest.raises(PatchApplyError, match="are one file of the image"):
        compile_patch(tmp_path, blobs, "triton", files)


def test_the_binary_names_of_aiter_are_refused_and_its_python_is_not(tmp_path):
    aiter = COMPONENT_LAYOUTS["aiter"]

    assert aiter.spec.refused_by("aiter/jit/module_activation.so") == "*.so"
    assert aiter.spec.refused_by("hsa/gfx942/pa/pa.co") == "*.co"
    assert aiter.spec.refused_by("aiter/ops/mha.py") is None
    assert aiter.spec.refused_by("csrc/kernels/activation_kernels.cu") is None


def test_a_layout_built_by_hand_is_one_package_section_that_refuses_nothing(tmp_path, blobs):
    layout = ComponentLayout("sglang", "python/sglang")

    assert layout.spec.place("python/sglang/srt/a.py") is not None and layout.spec.place("python/other/a.py") is None
    assert layout.spec.refused_by("python/sglang/srt/kernel.so") is None
