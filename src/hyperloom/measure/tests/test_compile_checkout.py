# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A component is found by what its layout lies over: its installed package, or the pinned checkout of its upstream
that ``FRAMEWORK_REPO_PATH`` names. A checkout's files are measured only by a native build, so a compile without one
refuses a patch to any of them, naming the path, and never looks for a package that is not there."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    ComponentLayout,
    EnvironmentCompiler,
    LocalBlobStore,
    PatchApplyError,
    UnsupportedComponentError,
    locate_installed_package,
    locate_source,
)
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with, write_tree

LLAMACPP = COMPONENT_LAYOUTS["llamacpp"]
CHECKOUT_FILES = {
    "ggml/src/ggml-cuda/mmq.cuh": "a\n",
    "src/llama.cpp": "a\n",
    "include/llama.h": "a\n",
    "common/common.cpp": "a\n",
    "tools/server/server.cpp": "a\n",
    "docs/build.md": "a\n",
}


@pytest.fixture()
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = write_tree(tmp_path / "llama.cpp", CHECKOUT_FILES)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(root))
    return root


def test_a_checkout_layout_is_found_at_the_framework_checkout(checkout):
    found = locate_source(LLAMACPP)

    assert (found.import_name, found.path, found.dist_name, found.dist_version) == (
        "llamacpp",
        checkout.resolve(),
        None,
        None,
    )


def test_a_package_layout_is_found_where_the_image_installs_it(checkout):
    layout = ComponentLayout("pytest", "pytest")

    assert locate_source(layout) == locate_installed_package(layout)


def test_a_checkout_is_not_found_when_no_framework_checkout_is_named(monkeypatch):
    monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)

    with pytest.raises(UnsupportedComponentError, match="FRAMEWORK_REPO_PATH names none"):
        locate_source(LLAMACPP)


def test_a_framework_checkout_of_another_project_is_not_the_components(tmp_path, monkeypatch):
    other = write_tree(tmp_path / "sglang", {"python/sglang/__init__.py": "", "src/x.c": ""})
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(other))

    with pytest.raises(
        UnsupportedComponentError, match=r"is not a checkout of llamacpp: it has no \['ggml', 'include'"
    ):
        locate_source(LLAMACPP)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("ggml/src/ggml-cuda/mmq.cuh", "is a native or binary file (ggml/*)"),
        ("docs/build.md", "is outside the installed package (ggml/, src/, include/, common/, tools/server/)"),
    ],
)
def test_a_compile_without_native_builds_refuses_every_change_to_a_checkout(tmp_path, checkout, path, message):
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = git_patch(tmp_path / "up", {path: "a\n"}, {path: "b\n"})
    recipe = recipe_with((put_patch(blobs, "p.patch", patch),), component="llamacpp")

    with pytest.raises(PatchApplyError) as refused:
        EnvironmentCompiler(blobs=blobs, active_image_digest=IMAGE).compile(recipe, RecipeDelta())

    assert f"{path!r} {message}" in str(refused.value)
