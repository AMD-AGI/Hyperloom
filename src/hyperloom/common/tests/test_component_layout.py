# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A layout places an upstream path by its longest section prefix and names the files it cannot measure: those of
``refuse`` in any session, those of ``native`` without a native build. A pattern without a slash names a file by its name
wherever it lies, one with a slash by its upstream path below its directory."""

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
    refuse=("*.so", "*.so.*"),
    native=("CMakeLists.txt", "csrc/*.cu"),
)
#: The rule pack's layout of vLLM, as a session reads it.
VLLM = component_layouts()["vllm"]
#: vLLM's compiled files, refused in every session wherever they lie.
VLLM_COMPILED = ("*.so", "*.so.*", "*.a", "*.o", "*.bc", "*.co", "*.hsaco", "*.cubin", "*.pyd")
#: vLLM's native sources, native only below the top-level csrc, which its CMake build compiles.
VLLM_SOURCES = tuple(f"csrc/*.{ext}" for ext in ("c", "cc", "cpp", "cxx", "h", "hh", "hpp", "cu", "cuh", "hip", "inc"))
#: vLLM's build definition, native wherever it lies.
VLLM_BUILD = {
    "CMakeLists.txt": "CMakeLists.txt",
    "csrc/CMakeLists.txt": "CMakeLists.txt",
    "cmake/utils.cmake": "*.cmake",
    "setup.py": "setup.py",
    "vllm/sub/setup.py": "setup.py",
    "pyproject.toml": "pyproject.toml",
}


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
    ("path", "refused", "native"),
    [
        ("a/libx.so", "*.so", None),
        ("a/libx.so.1.2", "*.so.*", None),
        ("a/b/CMakeLists.txt", None, "CMakeLists.txt"),
        ("a/x.py", None, None),
    ],
)
def test_a_file_is_named_by_the_pattern_its_name_matches_not_by_its_directory(path, refused, native):
    assert (SPEC.refused_by(path), SPEC.native_by(path)) == (refused, native)


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
def test_a_pattern_with_a_slash_names_its_files_below_its_directory_at_any_depth_and_nowhere_else(path, pattern):
    assert (SPEC.refused_by(path), SPEC.native_by(path)) == (None, pattern)


def test_these_tests_pin_every_pattern_of_vllms_layout():
    assert (set(VLLM.refuse), set(VLLM.native)) == (set(VLLM_COMPILED), {*VLLM_SOURCES, *VLLM_BUILD.values()})


@pytest.mark.parametrize("pattern", VLLM_COMPILED)
@pytest.mark.parametrize("where", ["", "vllm/", "vllm/.osk_cache/", "csrc/", "csrc/rocm/"])
def test_vllms_layout_refuses_a_compiled_file_wherever_it_lies(where, pattern):
    """A compiled file is what a build makes: no session, with native builds or without, takes it from a patch."""
    path = where + pattern.replace("*", "x")

    assert (VLLM.refused_by(path), VLLM.native_by(path)) == (pattern, None)


@pytest.mark.parametrize("pattern", VLLM_SOURCES)
@pytest.mark.parametrize(
    ("where", "native"),
    [
        ("csrc/", True),
        ("csrc/rocm/attention/", True),
        ("", False),
        ("vllm/", False),
        ("vllm/csrc/", False),
        ("xcsrc/", False),
        ("csrcs/", False),
    ],
)
def test_vllms_layout_names_a_source_native_only_below_the_top_level_csrc(where, native, pattern):
    """vLLM's CMake build compiles only csrc; a source inside the package is compiled by the package at run time."""
    path = where + pattern.removeprefix("csrc/").replace("*", "k")

    assert (VLLM.refused_by(path), VLLM.native_by(path)) == (None, pattern if native else None)


@pytest.mark.parametrize(("path", "pattern"), VLLM_BUILD.items())
def test_vllms_build_definition_is_native_wherever_it_lies(path, pattern):
    assert (VLLM.refused_by(path), VLLM.native_by(path)) == (None, pattern)


@pytest.mark.parametrize(
    "path",
    [
        "vllm/_osk_gemm.hip",
        "vllm/kernels/osk.cu",
        "vllm/csrc/osk.hpp",
        "vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/hf3fs_utils.cpp",
        "vllm/Makefile",
    ],
)
def test_a_file_inside_vllms_package_that_no_build_of_it_compiles_or_reads_is_a_package_file(path):
    assert (VLLM.refused_by(path), VLLM.native_by(path)) == (None, None)
    assert VLLM.place(path) is not None


def test_a_makefile_is_like_any_other_file_outside_csrc_since_no_build_of_vllm_reads_one():
    """vLLM has no Makefile at f46a9dfe2c, and setup.py runs CMake in its build directory: outside the package, a
    Makefile is refused by the sections, not named by a pattern."""
    assert (VLLM.refused_by("Makefile"), VLLM.native_by("Makefile"), VLLM.place("Makefile")) == (None, None, None)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"import_name": "not a name"}, "import_name"),
        ({"sections": []}, "at least one section"),
        ({"sections": [{"prefix": "csrc", "base": "meta"}]}, "must lie over the installed package"),
        ({"sections": [{"base": "package"}]}, "holds 'prefix' and 'base'"),
        ({"refuse": "*.so"}, "must be a list of strings"),
        ({"refuse": [""]}, "not a non-empty string"),
        ({"refuse": ["/csrc/*.so"]}, "plain relative path"),
        ({"native": "*.cu"}, "must be a list of strings"),
        ({"native": [""]}, "not a non-empty string"),
        ({"native": ["/csrc/*.cu"]}, "plain relative path"),
        ({"native": ["csrc/../*.cu"]}, "plain relative path"),
        ({"generated": ["/abs"]}, "plain relative path"),
    ],
)
def test_a_layout_entry_that_cannot_be_used_is_refused_with_what_is_wrong(change, message):
    body = {
        "import_name": "lib",
        "sections": [{"prefix": "lib", "base": "package"}],
        "refuse": [],
        "native": [],
        "generated": [],
    }

    with pytest.raises(LayoutSpecError, match=message):
        parse_layout("entry x", {**body, **change})
