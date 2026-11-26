# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A single-branch session's native builds over the real store, planner and build runner child, with a stand-in
llama.cpp compile: the server every round runs is the build of the checkout's tree, never what ``PATH`` found."""

from __future__ import annotations

import os
import shutil
from functools import partial
from pathlib import Path

import pytest

from hyperloom.measure.runner.native import recipes as recipes_module
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.session import open_session
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuildStatus
from hyperloom.measure.runner.native import session_builds
from hyperloom.inference_optimizer.native_builds import NativeBuildsRefused
from hyperloom.measure.runner.native.session_builds import (
    open_native_builds,
    start_native_builds,
)
from hyperloom.measure.tests import _native_llamacpp as stand_in
from hyperloom.measure.tests._native_llamacpp import IMAGE, SOURCE, git, llama_checkout


@pytest.fixture()
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem(
        recipes_module.RECIPE_FACTORIES, "llamacpp", f"{stand_in.__name__}:{stand_in.stand_in_recipe.__name__}"
    )
    monkeypatch.setattr(session_builds, "native_recipes", lambda: {"llamacpp": stand_in.stand_in_recipe()})
    monkeypatch.setattr(session_builds, "open_session", partial(open_session, isolation=lambda: Isolation((), None)))
    return llama_checkout(tmp_path / "llama.cpp")


def _environ(checkout: Path) -> dict[str, str]:
    return {
        "FRAMEWORK_REPO_PATH": str(checkout),
        **IMAGE,
        "HYPERLOOM_GFX_ARCH": "gfx1100",
        "PATH": os.environ["PATH"],
    }


def _server(directory: Path) -> bytes:
    return (directory / "llama-server").read_bytes()


def test_the_session_runs_the_build_of_its_checkout_first_on_path(tmp_path: Path, checkout: Path) -> None:
    environ = _environ(checkout)

    native = start_native_builds(tmp_path / "session", "llamacpp", environ)

    assert native is not None
    assert environ["PATH"].split(os.pathsep)[0] == str(native.current)
    assert os.access(native.current / "llama-server", os.X_OK)
    assert native.base_commit == git(checkout, "rev-parse", "HEAD")


def test_a_candidates_tree_is_built_over_the_base_and_runs_only_once_accepted(tmp_path: Path, checkout: Path) -> None:
    native = start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    base = _server(native.current)
    (checkout / SOURCE).write_text("int a = 2;\n")

    candidate = native.build_tree([SOURCE])

    assert candidate.status is TreeBuildStatus.BUILT and candidate.key
    assert _server(candidate.bin) != base
    assert _server(native.current) == base
    native.accept(candidate)
    assert _server(native.current) == _server(candidate.bin)


def test_a_tracked_change_an_earlier_round_left_applied_is_part_of_every_later_build(
    tmp_path: Path, checkout: Path
) -> None:
    native = start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    base = _server(native.current)
    (checkout / SOURCE).write_text("int a = 4;\n")
    (checkout / "src" / "new.cpp").write_text("int added;\n")

    applied = native.build_tree(["src/new.cpp"])
    alone = native.build_tree()

    assert applied.status is alone.status is TreeBuildStatus.BUILT
    assert _server(applied.bin) == _server(alone.bin) != base
    assert applied.key != alone.key


def test_a_candidate_that_does_not_compile_is_its_compile_error_and_lays_nothing(
    tmp_path: Path, checkout: Path
) -> None:
    native = start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    (checkout / SOURCE).write_text("#error\n")

    failed = native.build_tree([SOURCE])

    assert (failed.status, failed.bin) == (TreeBuildStatus.COMPILE_ERROR, None)
    assert "the compile exited 1" in failed.detail


def test_a_candidate_that_changes_how_it_is_built_is_refused_as_its_own_error(tmp_path: Path, checkout: Path) -> None:
    native = start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    (checkout / "ggml" / "CMakeLists.txt").write_text("project(x)\n")

    refused = native.build_tree(["ggml/CMakeLists.txt"])

    assert (refused.status, refused.key) == (TreeBuildStatus.COMPILE_ERROR, "")
    assert "is a build definition" in refused.detail


def test_a_later_leg_keeps_the_base_and_runs_the_build_of_what_the_session_kept(tmp_path: Path, checkout: Path) -> None:
    session = tmp_path / "session"
    first = start_native_builds(session, "llamacpp", _environ(checkout))
    (checkout / SOURCE).write_text("int a = 3;\n")
    kept = first.build_tree([SOURCE])
    git(checkout, "commit", "-qam", "KEEP")
    first.accept(kept)

    later = start_native_builds(session, "llamacpp", _environ(checkout))

    assert later.base_commit == first.base_commit
    assert _server(later.current) == _server(kept.bin)


def test_a_resumed_session_does_not_run_a_candidate_it_left_pending(tmp_path: Path, checkout: Path) -> None:
    session = tmp_path / "session"
    first = start_native_builds(session, "llamacpp", _environ(checkout))
    accepted = _server(first.current)
    (checkout / SOURCE).write_text("int a = 6;\n")

    later = start_native_builds(session, "llamacpp", _environ(checkout))

    assert _server(later.current) == accepted


def test_a_framework_with_no_native_component_builds_nothing(tmp_path: Path) -> None:
    assert open_native_builds(tmp_path, "sglang", {}) is None


def test_a_checkout_with_changed_tracked_files_is_refused(tmp_path: Path, checkout: Path) -> None:
    (checkout / SOURCE).write_text("int dirty;\n")

    with pytest.raises(NativeBuildsRefused, match="changes to tracked files"):
        open_native_builds(tmp_path, "llamacpp", _environ(checkout))


def test_a_sweep_builds_patches_over_the_recorded_base_into_a_store_of_its_own(tmp_path: Path, checkout: Path) -> None:
    native = start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    base_source = git(checkout, "show", f"HEAD:{SOURCE}") + "\n"
    (checkout / SOURCE).write_text("int a = 9;\n")

    tunes = session_builds.tune_native_builds(tmp_path / "session")
    patch = native.build_tree([SOURCE])
    changed = tunes.build_patch(
        f"diff --git a/{SOURCE} b/{SOURCE}\n--- a/{SOURCE}\n+++ b/{SOURCE}\n@@ -1 +1 @@\n-{base_source}+int a = 7;\n".encode()
    )
    base = tunes.build_patch(b"")

    assert tunes.read_upstream(SOURCE) == base_source
    assert tunes.root == tmp_path / "session" / "native" / "tunes"
    assert base.status is changed.status is TreeBuildStatus.BUILT
    assert _server(base.bin) == _server(native.current)
    assert _server(changed.bin) not in (_server(base.bin), _server(patch.bin))


def test_a_build_another_caller_laid_while_this_one_staged_it_is_that_copy_and_the_staged_one_is_gone(
    tmp_path: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_native_builds(tmp_path / "session", "llamacpp", _environ(checkout))
    tunes = session_builds.tune_native_builds(tmp_path / "session")
    farms = tunes.root / "farms"
    key = tunes.build_patch(b"").key
    shutil.rmtree(farms / key)
    build_farm = session_builds.build_farm

    def staged_after_another_laid(roots, staged: Path, **how):
        farm = build_farm(roots, staged, **how)
        shutil.copytree(staged, farms / key)
        return farm

    monkeypatch.setattr(session_builds, "build_farm", staged_after_another_laid)

    base = tunes.build_patch(b"")

    assert base.status is TreeBuildStatus.BUILT
    assert base.bin == farms / key / "llamacpp" / "bin" and os.access(base.bin / "llama-server", os.X_OK)
    assert [entry.name for entry in farms.iterdir()] == [key]
