# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runtime registry's two answers must stay two answers."""

from __future__ import annotations

from kernelforge.loop.path_ownership import (
    COMPILED_FILE_SUFFIXES,
    COPY_FILTER_DIRECTORY_NAMES,
    RUNTIME_DIRECTORY_GLOBS,
    RUNTIME_DIRECTORY_NAMES,
    RUNTIME_FILE_SUFFIXES,
    is_producer_owned_path,
    runtime_gitignore_globs,
)

# Directory names and suffixes a framework package uses for source. Dropping any
# of them from a scratch copy leaves an install that cannot be imported, and the
# copy shadows the real one, so there is nothing to fall back to.
IMPORTABLE_DIRECTORY_NAMES = ("jit", "dist", "ops", "kernels")


def test_copy_filter_keeps_directories_a_package_imports_from():
    for name in IMPORTABLE_DIRECTORY_NAMES:
        assert name not in RUNTIME_DIRECTORY_NAMES


def test_copy_filter_keeps_extension_modules():
    assert ".so" not in RUNTIME_FILE_SUFFIXES


def test_copy_filter_excludes_build_directories():
    assert "build" in COPY_FILTER_DIRECTORY_NAMES


def test_index_filter_does_not_cover_compiled_artefacts():
    # .so and build/ must stay in the index so git-revert can restore them.
    globs = runtime_gitignore_globs()
    for suffix in COMPILED_FILE_SUFFIXES:
        assert f"*{suffix}" not in globs
    for name in COPY_FILTER_DIRECTORY_NAMES:
        assert f"{name}/" not in globs


def test_index_filter_covers_cache_directories_and_bytecode():
    globs = runtime_gitignore_globs()
    for name in RUNTIME_DIRECTORY_NAMES:
        assert f"{name}/" in globs
    for glob in RUNTIME_DIRECTORY_GLOBS:
        assert f"{glob}/" in globs
    for suffix in RUNTIME_FILE_SUFFIXES:
        assert f"*{suffix}" in globs


def test_the_two_suffix_sets_do_not_overlap():
    assert not RUNTIME_FILE_SUFFIXES & COMPILED_FILE_SUFFIXES


def test_producer_ownership_covers_the_loop_ledger():
    assert is_producer_owned_path("forge_experiments/optimization_report.md")
    assert is_producer_owned_path("forge_experiments/best/manifest.json")
    assert is_producer_owned_path(".forge_rewrite/attempt_01/driver.py")
    assert is_producer_owned_path("pkg/.forge_driver_abc.py")


def test_producer_ownership_does_not_refuse_framework_sources():
    assert not is_producer_owned_path("aiter/ops/flydsl/moe.py")
    assert not is_producer_owned_path("aiter/jit/core.py")
    assert not is_producer_owned_path("docs/optimization_report.md")
    assert not is_producer_owned_path("pkg/optimized_versions/candidate.py")
