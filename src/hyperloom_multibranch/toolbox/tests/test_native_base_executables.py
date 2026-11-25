# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session whose framework is built from its pinned checkout measures its base on that build, as every candidate is
measured on its own, never on whatever ``PATH`` finds; built over the real store, planner and build runner child with
a stand-in llama.cpp compile."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest

from hyperloom.measure.runner.native import session_builds
from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.runner.native import recipes as recipes_module
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.session import NativeBuildSession, open_session
from hyperloom.measure.runner.native.session_builds import open_native_builds
from hyperloom.measure.tests import _native_llamacpp as stand_in
from hyperloom.measure.tests._native_llamacpp import IMAGE, SOURCE, llama_checkout
from hyperloom_multibranch.toolbox.native_builds import base_executables


@pytest.fixture()
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NativeBuildSession:
    monkeypatch.setitem(
        recipes_module.RECIPE_FACTORIES, "llamacpp", f"{stand_in.__name__}:{stand_in.stand_in_recipe.__name__}"
    )
    monkeypatch.setattr(session_builds, "native_recipes", lambda: {"llamacpp": stand_in.stand_in_recipe()})
    monkeypatch.setattr(session_builds, "open_session", partial(open_session, isolation=lambda: Isolation((), None)))
    checkout = llama_checkout(tmp_path / "llama.cpp")
    environ = {"FRAMEWORK_REPO_PATH": str(checkout), **IMAGE, "HYPERLOOM_GFX_ARCH": "gfx1100"}
    native = open_native_builds(tmp_path / "session", "llamacpp", environ)
    assert native is not None
    return native.session


def test_the_base_build_of_a_pinned_checkout_is_laid_for_every_test_to_run(
    tmp_path: Path, session: NativeBuildSession
) -> None:
    laid = base_executables(session, tmp_path / "farms")

    assert len(laid) == 1 and laid[0].is_relative_to(tmp_path / "farms")
    assert os.access(laid[0] / "llama-server", os.X_OK)
    assert base_executables(session, tmp_path / "farms") == laid


def test_a_base_that_does_not_build_stops_the_toolbox(tmp_path: Path, session: NativeBuildSession) -> None:
    sources = session.planner.site.sources / "llamacpp" / session.planner.site.commits["llamacpp"]
    (sources / SOURCE).write_text("#error\n")

    with pytest.raises(BuildStoreConfigError, match="llamacpp .* did not build"):
        base_executables(session, tmp_path / "farms")
