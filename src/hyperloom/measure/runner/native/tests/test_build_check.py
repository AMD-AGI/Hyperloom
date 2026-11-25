# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``build_check``: a specialist's worktree of a source-built framework compiled as the integrate gate builds it, with
nothing measured."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import pytest

from hyperloom.measure.runner.native import build_check, session_builds
from hyperloom.measure.runner.native import recipes as recipes_module
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.session import open_session
from hyperloom.measure.runner.native.session_builds import start_native_builds
from hyperloom.measure.tests import _native_llamacpp as stand_in
from hyperloom.measure.tests._native_llamacpp import IMAGE, SOURCE, git, llama_checkout


@pytest.fixture()
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem(
        recipes_module.RECIPE_FACTORIES, "llamacpp", f"{stand_in.__name__}:{stand_in.stand_in_recipe.__name__}"
    )
    monkeypatch.setattr(session_builds, "native_recipes", lambda: {"llamacpp": stand_in.stand_in_recipe()})
    monkeypatch.setattr(session_builds, "open_session", partial(open_session, isolation=lambda: Isolation((), None)))
    checkout = llama_checkout(tmp_path / "llama.cpp")
    for name, value in (
        ("FRAMEWORK_REPO_PATH", str(checkout)),
        *IMAGE.items(),
        ("HYPERLOOM_GFX_ARCH", "gfx1100"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    start_native_builds(tmp_path / "session", "llamacpp", os.environ)
    git(checkout, "worktree", "add", "-q", "--detach", str(tmp_path / "worktree"))
    return tmp_path / "session"


def _check(session: Path, tree: Path, capsys: pytest.CaptureFixture[str], *added: str) -> tuple[int, dict]:
    code = build_check.main(["--tree", str(tree), "--session", str(session), *(f"--add={path}" for path in added)])
    return code, json.loads(capsys.readouterr().out)


def test_a_worktree_that_builds_is_reported_built_with_its_key(session: Path, tmp_path: Path, capsys) -> None:
    (tmp_path / "worktree" / SOURCE).write_text("int a = 5;\n")
    (tmp_path / "worktree" / "src" / "new.cpp").write_text("int b;\n")

    code, found = _check(session, tmp_path / "worktree", capsys, "src/new.cpp")

    assert (code, found["status"], len(found["key"])) == (0, "built", 64)
    assert not (session / "native" / "checks" / "farms").exists(), "a check lays nothing"


def test_a_check_builds_on_the_site_the_session_recorded_whatever_its_own_environment(
    session: Path, tmp_path: Path, capsys, monkeypatch
) -> None:
    for name in (
        "HYPERLOOM_ACTIVE_IMAGE_REF",
        "HYPERLOOM_ACTIVE_IMAGE_DIGEST",
        "HYPERLOOM_GFX_ARCH",
        "FRAMEWORK_REPO_PATH",
    ):
        monkeypatch.delenv(name)
    (tmp_path / "worktree" / SOURCE).write_text("int a = 6;\n")

    code, found = _check(session, tmp_path / "worktree", capsys)

    assert (code, found["status"]) == (0, "built")


def test_a_worktree_that_does_not_compile_is_its_compile_error(session: Path, tmp_path: Path, capsys) -> None:
    (tmp_path / "worktree" / SOURCE).write_text("#error\n")

    code, found = _check(session, tmp_path / "worktree", capsys)

    assert (code, found["status"]) == (1, "compile_error")
    assert "error: stop" in found["detail"]


def test_a_session_that_builds_no_server_from_source_is_reported_as_such(tmp_path: Path, capsys) -> None:
    code, found = _check(tmp_path / "other-session", tmp_path, capsys)

    assert (code, found["status"]) == (2, "infra")
    assert "builds no framework server from source" in found["detail"]
