# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache of the local toolbox: the directory named, else one below this host's temporary
directory (design 4.21).

What is proved here is the wiring only: which directory the host publishes to and the runner reads. That a cache serves
a test and is published to is proved by the measure package's own tests, and on a GPU by the toolbox's measured runs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom_multibranch.toolbox.local_hosted import (
    DEFAULT_JIT_STORE,
    JIT_STORE_ENV,
    LocalHostedToolbox,
    LocalOptions,
    jit_store_root,
)
from hyperloom_multibranch.toolbox.tests.helpers import lineage_blackboard
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _toolbox_of

_RUNNER = "runner-1"


@pytest.fixture(autouse=True)
def _no_ambient_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the machine running the suite exports must not decide where the cache is."""
    monkeypatch.delenv(JIT_STORE_ENV, raising=False)


@pytest.fixture
def host_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This host's temporary directory, the test's own: the default cache lies below it."""
    host_tmp = tmp_path / "host-tmp"
    host_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(host_tmp))
    return host_tmp


def _runner_env(toolbox: LocalHostedToolbox) -> dict[str, str]:
    return dict(toolbox.provider._templates[_RUNNER].envs)


async def _closing_build(tmp_path: Path, **extra: Any) -> tuple[LocalHostedToolbox, dict[str, str]]:
    """A toolbox built with ``extra`` options, its facts read, and closed: ``(toolbox, what a branch sandbox gets)``."""
    blackboard = lineage_blackboard(tmp_path / "blackboard")
    toolbox = _toolbox_of(blackboard, tmp_path, tmp_path / "session", "a" * 40, **extra)
    try:
        branch_env = dict((await toolbox.lease_branch_sandbox("branch-1", 1)).env)
    finally:
        await toolbox.close()
    return toolbox, branch_env


def test_no_directory_named_means_the_default_below_the_hosts_temporary_directory(host_temp_dir: Path) -> None:
    default = (host_temp_dir / DEFAULT_JIT_STORE).resolve()

    assert jit_store_root(LocalOptions(), {}) == default
    assert jit_store_root(LocalOptions(), {JIT_STORE_ENV: "  "}) == default, "a blank variable is not a directory"


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


async def test_a_toolbox_built_without_a_directory_publishes_to_the_default_and_its_runners_read_it(
    tmp_path: Path, host_temp_dir: Path
) -> None:
    toolbox, branch_env = await _closing_build(tmp_path)

    published = toolbox.backend._config.jit_store
    assert published is not None and published.root == (host_temp_dir / DEFAULT_JIT_STORE).resolve()
    assert _runner_env(toolbox)[JIT_CACHE_ENV] == str(published.root)
    assert JIT_CACHE_ENV not in branch_env, "the agent's own sandboxes are not told where the cache is"


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
