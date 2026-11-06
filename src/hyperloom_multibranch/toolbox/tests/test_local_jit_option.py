# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache of the local toolbox: the directory named, else this account's below this host's
temporary directory, and never one another account owns or can write.

What is proved here is the wiring only: which directory the host publishes to and the runner reads. That a cache serves
a test and is published to is proved by the measure package's own tests, and on a GPU by the toolbox's measured runs.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom_multibranch.toolbox.local_hosted import (
    DEFAULT_JIT_STORE,
    JIT_STORE_ENV,
    JitStoreNotPrivate,
    LocalHostedToolbox,
    LocalOptions,
    jit_store_root,
)
from hyperloom_multibranch.toolbox.tests.helpers import lineage_blackboard
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _toolbox_of

_RUNNER = "runner-1"
#: An account other than the one running the suite.
_NOBODY = 65534


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


def _default_store(host_temp_dir: Path) -> Path:
    return (host_temp_dir / f"{DEFAULT_JIT_STORE}-{os.getuid()}").resolve()


def test_no_directory_named_means_this_accounts_default_below_the_hosts_temporary_directory(
    host_temp_dir: Path,
) -> None:
    default = _default_store(host_temp_dir)

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
    assert published is not None and published.root == _default_store(host_temp_dir)
    made = published.root.stat()
    assert (made.st_uid, stat.S_IMODE(made.st_mode)) == (os.getuid(), 0o700), "made this account's, for it only"
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


def _another_accounts(directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``directory`` another account's: really, handed to user 65534, where the suite runs as root; elsewhere, as
    the ownership check sees it (this account's uid reads as another's), since no other account is at hand."""
    if os.geteuid() == 0:
        os.chown(directory, _NOBODY, _NOBODY)
        assert directory.stat().st_uid == _NOBODY != os.getuid()
        return
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)


def _refused_build(tmp_path: Path, **extra: Any) -> str:
    """Why building a toolbox with ``extra`` options was refused; nothing below the session was started."""
    blackboard = lineage_blackboard(tmp_path / "blackboard")
    with pytest.raises(JitStoreNotPrivate) as refused:
        _toolbox_of(blackboard, tmp_path, tmp_path / "session", "a" * 40, **extra)
    assert not (tmp_path / "session" / "sandboxes").exists(), "nothing was started"
    return str(refused.value)


async def test_a_store_another_account_owns_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = tmp_path / "store"
    store.mkdir(mode=0o755)
    _another_accounts(store, monkeypatch)

    assert re.search(
        f"{re.escape(str(store))} is not a directory of this account's", _refused_build(tmp_path, jit_store=store)
    )


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o1777])
async def test_a_store_its_group_or_others_can_write_is_refused(tmp_path: Path, mode: int) -> None:
    store = tmp_path / "store"
    store.mkdir()
    store.chmod(mode)

    assert f"{store} is not a directory of this account's that only it can write" in _refused_build(
        tmp_path, jit_store=store
    )
    assert stat.S_IMODE(store.stat().st_mode) == mode, "and it is left as it is"


async def test_a_default_store_planted_before_the_toolbox_made_it_is_refused_not_used(
    tmp_path: Path, host_temp_dir: Path
) -> None:
    planted = _default_store(host_temp_dir)
    planted.mkdir()
    planted.chmod(0o777)  # as another account on the host would leave it, to put its own snapshots in

    assert str(planted) in _refused_build(tmp_path)
