# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A layout places an upstream path by its longest section prefix and refuses the files it cannot measure."""

from __future__ import annotations

import pytest

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
    refuse=("*.so", "*.so.*", "CMakeLists.txt"),
)


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
    ("change", "message"),
    [
        ({"import_name": "not a name"}, "import_name"),
        ({"sections": []}, "at least one section"),
        ({"sections": [{"prefix": "csrc", "base": "meta"}]}, "must lie over the installed package"),
        ({"sections": [{"base": "package"}]}, "holds 'prefix' and 'base'"),
        ({"refuse": "*.so"}, "must be a list of strings"),
        ({"refuse": [""]}, "not a non-empty string"),
        ({"generated": ["/abs"]}, "plain relative path"),
    ],
)
def test_a_layout_entry_that_cannot_be_used_is_refused_with_what_is_wrong(change, message):
    body = {"import_name": "lib", "sections": [{"prefix": "lib", "base": "package"}], "refuse": [], "generated": []}

    with pytest.raises(LayoutSpecError, match=message):
        parse_layout("entry x", {**body, **change})
