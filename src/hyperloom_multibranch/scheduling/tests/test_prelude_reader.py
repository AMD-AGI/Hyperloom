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


def test_the_root_pins_a_credentialed_remote_without_its_userinfo(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path, remote="https://user:ghp_secret@example.invalid/sglang.git")

    recipe = launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(framework)}, IMAGE)

    assert framework_pin(framework) == UpstreamPin(REMOTE_URL, head_of(framework))
    assert recipe == root_recipe("sglang", UpstreamPin(REMOTE_URL, head_of(framework)), IMAGE)


def test_a_framework_without_remote_cannot_be_pinned(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path, remote="")
    with pytest.raises(PreludeReadError, match="no git remote"):
        launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(framework)}, IMAGE)


def test_a_session_whose_environment_names_no_framework_checkout_pins_nothing() -> None:
    assert launched_root("sglang", {}, IMAGE).components == ()
    assert framework_pin(None) is None


def test_a_named_framework_path_that_is_no_git_checkout_is_refused(tmp_path: Path) -> None:
    (tmp_path / "site-packages").mkdir()
    with pytest.raises(PreludeReadError, match="not a git repository"):
        launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(tmp_path / "site-packages")}, IMAGE)


def test_a_named_framework_checkout_git_cannot_read_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    framework = framework_checkout(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-git"))
    with pytest.raises(PreludeReadError, match="rev-parse --show-toplevel"):
        launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(framework)}, IMAGE)


def test_a_named_framework_directory_inside_another_checkout_is_refused(tmp_path: Path) -> None:
    vendored = framework_checkout(tmp_path) / "vendor" / "sglang"
    vendored.mkdir(parents=True)
    with pytest.raises(PreludeReadError, match="not the top of its git checkout"):
        launched_root("sglang", {MB_FRAMEWORK_REPO_ENV: str(vendored)}, IMAGE)


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
