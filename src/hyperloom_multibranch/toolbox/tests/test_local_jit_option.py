# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache is an opt-in of the local toolbox: off unless a directory is named (design 4.21).

What is proved here is the wiring only: which directory the host publishes to and the runner reads. That a cache serves
a test and is published to is proved by the measure package's own tests, and on a GPU by the toolbox's measured runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import (
    JIT_STORE_ENV,
    LocalHostedToolbox,
    LocalOptions,
    jit_store_root,
)
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _toolbox_of

_RUNNER = "runner-1"


@pytest.fixture(autouse=True)
def _no_ambient_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the machine running the suite exports must not decide whether the cache is on."""
    monkeypatch.delenv(JIT_STORE_ENV, raising=False)


def _runner_env(toolbox: LocalHostedToolbox) -> dict[str, str]:
    return dict(toolbox.provider._templates[_RUNNER].envs)


async def _closing_build(tmp_path: Path, **extra: Any) -> tuple[LocalHostedToolbox, dict[str, str]]:
    """A toolbox built with ``extra`` options, its facts read, and closed: ``(toolbox, what a branch sandbox gets)``."""
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(notebook, tmp_path, tmp_path / "session", "a" * 40, **extra)
    try:
        branch_env = dict((await toolbox.lease_branch_sandbox("branch-1", 1)).env)
    finally:
        await toolbox.close()
    return toolbox, branch_env


def test_no_directory_named_means_no_cache(tmp_path: Path) -> None:
    assert jit_store_root(LocalOptions(), {}) is None
    assert jit_store_root(LocalOptions(), {JIT_STORE_ENV: "  "}) is None, "a blank variable is not a directory"


def test_the_host_environment_names_the_directory_when_the_option_does_not(tmp_path: Path) -> None:
    root = jit_store_root(LocalOptions(), {JIT_STORE_ENV: f" {tmp_path / 'from-env'} "})

    assert root == (tmp_path / "from-env").resolve()


def test_the_option_wins_over_the_environment(tmp_path: Path) -> None:
    options = LocalOptions(jit_store=tmp_path / "from-option")

    assert jit_store_root(options, {JIT_STORE_ENV: str(tmp_path / "from-env")}) == (tmp_path / "from-option").resolve()


def test_a_relative_directory_is_made_absolute_because_the_runner_starts_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert jit_store_root(LocalOptions(jit_store=Path("store")), {}) == (tmp_path / "store").resolve()
    assert jit_store_root(LocalOptions(), {JIT_STORE_ENV: "store"}) == (tmp_path / "store").resolve()


async def test_a_toolbox_built_without_a_directory_has_no_cache_anywhere(tmp_path: Path) -> None:
    toolbox, branch_env = await _closing_build(tmp_path)

    assert toolbox.backend._config.jit_store is None
    assert JIT_CACHE_ENV not in _runner_env(toolbox) and JIT_CACHE_ENV not in branch_env


async def test_a_directory_in_the_options_is_where_the_host_publishes_and_the_runner_reads(tmp_path: Path) -> None:
    store = tmp_path / "store"

    toolbox, branch_env = await _closing_build(tmp_path, jit_store=store)

    published = toolbox.backend._config.jit_store
    assert published is not None and published.root == store.resolve()
    assert _runner_env(toolbox)[JIT_CACHE_ENV] == str(store.resolve())
    assert JIT_CACHE_ENV not in branch_env, "the agent's own sandboxes are not told where the cache is"


async def test_a_directory_in_the_host_environment_turns_the_cache_on_for_a_caller_that_passes_no_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(JIT_STORE_ENV, str(tmp_path / "store"))

    toolbox, _ = await _closing_build(tmp_path)

    published = toolbox.backend._config.jit_store
    assert published is not None and published.root == (tmp_path / "store").resolve()
    assert _runner_env(toolbox)[JIT_CACHE_ENV] == str(published.root)


async def test_the_runner_reads_the_directory_the_host_publishes_to_even_when_the_runner_environment_names_another(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"

    toolbox, _ = await _closing_build(
        tmp_path, jit_store=store, runner_envs={JIT_CACHE_ENV: str(tmp_path / "elsewhere")}
    )

    assert _runner_env(toolbox)[JIT_CACHE_ENV] == str(store.resolve())
