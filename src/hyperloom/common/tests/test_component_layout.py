# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A layout places an upstream path by its longest section prefix and refuses the files it cannot measure: a pattern
without a slash by the file's name wherever it lies, one with a slash by the upstream path below its directory."""

from __future__ import annotations

import pytest

from hyperloom.common.outside_facts import component_layouts
from hyperloom.common.outside_facts.component_layout import (
    META_BASE,
    PACKAGE_BASE,
    ComponentLayoutSpec,
    LayoutSection,
    LayoutSpecError,
    parse_layout,
)

SPEC = ComponentLayoutSpec(
    "lib",
    (
        LayoutSection("python/lib", PACKAGE_BASE),
        LayoutSection("python/lib/extra", PACKAGE_BASE, "plugins/extra"),
        LayoutSection("csrc", META_BASE, "csrc"),
    ),
    refuse=("*.so", "*.so.*", "CMakeLists.txt", "csrc/*.cu"),
)
#: The rule pack's layout of vLLM, as a session reads it.
VLLM = component_layouts()["vllm"]


@pytest.mark.parametrize(
    ("path", "base", "relative"),
    [
        ("python/lib/a/b.py", PACKAGE_BASE, "a/b.py"),
        ("python/lib/extra/c.py", PACKAGE_BASE, "plugins/extra/c.py"),
        ("python/lib/extras/c.py", PACKAGE_BASE, "extras/c.py"),
        ("csrc/kernels/k.cu", META_BASE, "csrc/kernels/k.cu"),
    ],
)
def test_the_longest_matching_prefix_places_a_path_at_the_directory_its_section_starts_at(path, base, relative):
    section, placed = SPEC.place(path)

    assert (section.base, placed) == (base, relative)


@pytest.mark.parametrize("path", ["python/lib", "python/other/a.py", "csrcs/k.cu", "docs/a.py"])
def test_a_path_outside_every_section_or_naming_a_section_itself_is_not_placed(path):
    assert SPEC.place(path) is None


@pytest.mark.parametrize(
    ("path", "pattern"),
    [("a/libx.so", "*.so"), ("a/libx.so.1.2", "*.so.*"), ("a/b/CMakeLists.txt", "CMakeLists.txt"), ("a/x.py", None)],
)
def test_a_file_is_refused_by_the_pattern_its_name_matches_not_by_its_directory(path, pattern):
    assert SPEC.refused_by(path) == pattern


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("csrc/k.cu", "csrc/*.cu"),
        ("csrc/kernels/rocm/k.cu", "csrc/*.cu"),
        ("./csrc/kernels/k.cu", "csrc/*.cu"),
        ("csrc//kernels/k.cu", "csrc/*.cu"),
        ("python/lib/k.cu", None),
        ("python/lib/csrc/k.cu", None),
        ("csrcs/k.cu", None),
        ("csrc/k.cuh", None),
    ],
)
def test_a_pattern_with_a_slash_refuses_its_files_below_its_directory_at_any_depth_and_nowhere_else(path, pattern):
    assert SPEC.refused_by(path) == pattern


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("vllm/_osk_gemm.hip", None),
        ("vllm/kernels/osk.cu", None),
        ("vllm/kernels/osk.cpp", None),
        ("vllm/kernels/osk.h", None),
        ("vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/hf3fs_utils.cpp", None),
        ("csrc/rocm/osk_gemm.hip", "csrc/*.hip"),
        ("csrc/cache_kernels.cu", "csrc/*.cu"),
        ("csrc/torch_bindings.cpp", "csrc/*.cpp"),
        ("csrc/ops.h", "csrc/*.h"),
        ("csrc/attention/dtype_fp8.cuh", "csrc/*.cuh"),
        ("vllm/_C.abi3.so", "*.so"),
        ("vllm/.osk_cache/osk_gemm.co", "*.co"),
        ("vllm/_osk_gemm.hsaco", "*.hsaco"),
        ("csrc/prebuilt.so", "*.so"),
        ("CMakeLists.txt", "CMakeLists.txt"),
        ("csrc/CMakeLists.txt", "CMakeLists.txt"),
        ("cmake/utils.cmake", "*.cmake"),
        ("setup.py", "setup.py"),
    ],
)
def test_vllms_layout_refuses_a_native_source_only_below_csrc_and_a_compiled_or_build_file_anywhere(path, pattern):
    """vLLM's CMake build compiles only csrc; a source inside the package is compiled by the package at run time."""
    assert VLLM.refused_by(path) == pattern
    if pattern is None:
        assert VLLM.place(path) is not None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"import_name": "not a name"}, "import_name"),
        ({"sections": []}, "at least one section"),
        ({"sections": [{"prefix": "csrc", "base": "meta"}]}, "must lie over the installed package"),
        ({"sections": [{"base": "package"}]}, "holds 'prefix' and 'base'"),
        ({"refuse": "*.so"}, "must be a list of strings"),
        ({"refuse": [""]}, "not a non-empty string"),
        ({"refuse": ["/csrc/*.cu"]}, "plain relative path"),
        ({"refuse": ["csrc/../*.cu"]}, "plain relative path"),
        ({"generated": ["/abs"]}, "plain relative path"),
    ],
)
def test_a_layout_entry_that_cannot_be_used_is_refused_with_what_is_wrong(change, message):
    body = {"import_name": "lib", "sections": [{"prefix": "lib", "base": "package"}], "refuse": [], "generated": []}

    with pytest.raises(LayoutSpecError, match=message):
        parse_layout("entry x", {**body, **change})
