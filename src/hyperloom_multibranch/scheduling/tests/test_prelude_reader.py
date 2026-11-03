# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launched framework checkout reader's contract: the root recipe it reads, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import LaunchConfig
from hyperloom_multibranch.scheduling.prelude_reader import (
    MB_FRAMEWORK_REPO_ENV,
    MB_FRAMEWORK_VERSION_ENV,
    PreludeReadError,
    UpstreamPin,
    framework_pin,
    launched_framework_version,
    launched_root,
    root_recipe,
)
from hyperloom_multibranch.scheduling.tests.framework_checkout import REMOTE_URL, framework_checkout, head_of

#: The image the toolbox measures on, which the root recipe carries.
IMAGE = "sha256:" + "1" * 64
COMMIT = "ab" * 20


def test_the_framework_version_a_session_records_is_the_one_its_environment_names_else_the_installed_one() -> None:
    assert launched_framework_version("sglang", {MB_FRAMEWORK_VERSION_ENV: " 0.5.2 "}) == "0.5.2"
    assert launched_framework_version("no-such-framework", {}) == ""


def test_the_root_is_the_named_checkout_at_its_head_on_the_image(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path)

    recipe = launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(framework)}, IMAGE)

    assert recipe == root_recipe("sglang", UpstreamPin(REMOTE_URL, head_of(framework)), IMAGE)
    assert framework_pin(framework) == UpstreamPin(REMOTE_URL, head_of(framework))


def test_a_framework_without_remote_cannot_be_pinned(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path, remote="")
    with pytest.raises(PreludeReadError, match="no git remote"):
        launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(framework)}, IMAGE)


@pytest.mark.parametrize("named", [None, "site-packages"], ids=["none-named", "no-git-checkout"])
def test_a_framework_path_that_is_no_git_checkout_pins_nothing(tmp_path: Path, named: str | None) -> None:
    env = {}
    if named is not None:
        (tmp_path / named).mkdir()
        env[MB_FRAMEWORK_REPO_ENV] = str(tmp_path / named)

    assert launched_root("sglang", env, IMAGE).components == ()
    assert framework_pin(None) is None


def test_root_recipe_pins_the_framework_without_a_patch_on_the_main_flows_launch() -> None:
    recipe = root_recipe("sglang", UpstreamPin(REMOTE_URL, COMMIT), IMAGE)
    assert (recipe.base_image_digest, recipe.launch, recipe.unexpressed) == (IMAGE, LaunchConfig(), ())
    component = recipe.component("sglang")
    assert component is not None
    assert (component.upstream_repo, component.upstream_ref, component.patches) == (REMOTE_URL, COMMIT, ())


@pytest.mark.parametrize(
    "pin", [UpstreamPin(REMOTE_URL, "abc1234"), UpstreamPin("", COMMIT)], ids=["short-ref", "no-url"]
)
def test_a_pin_the_recipe_refuses_is_a_read_error(pin: UpstreamPin) -> None:
    with pytest.raises(PreludeReadError, match="not a valid root recipe"):
        root_recipe("sglang", pin, IMAGE)
