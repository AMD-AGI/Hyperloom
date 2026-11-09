# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session measures on: the runner cache of the run's commit, built readable by the sandbox's account
whatever umask the entry runs under from the session's own clone of the run's source, which a restart finds again,
failing the run when it cannot be had and pinning every runner sandbox to it otherwise, and the peers refused as on
one machine. The peers built on the service's sandboxes are pinned in ``test_local_peers``."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom.measure.runner.cache_build import RUNNER_TARGET_ENV, CacheBuildError, runner_wheel_target
from hyperloom.measure.runner.cache_layout import manifest_from_json
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig
from hyperloom_multibranch.scheduling.hosted_peers import (
    PinnedToolboxInputs,
    load_hosted_peers,
    runner_cache_pin,
    runner_source,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload

WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 2, 16, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
#: The runner's interpreter, which the rig's pure-Python wheels serve whatever the test runs on.
RUNNER_TAG = "cp310-manylinux_2_28_x86_64"
TARGET = runner_wheel_target({RUNNER_TARGET_ENV: RUNNER_TAG})
#: The umask of an entry that keeps everything it creates to its own account.
PRIVATE = 0o077


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    """A clone with a commit whose runner lock pins wheels a local wheelhouse holds."""
    return make_rig(tmp_path_factory.mktemp("rig") / "rig")


@pytest.fixture
def private_umask() -> Iterator[None]:
    previous = os.umask(PRIVATE)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def wheelhouse_index(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """pip is configured to take the wheels from the rig's wheelhouse and nowhere else."""
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(rig.wheelhouse))


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


@pytest.mark.usefixtures("private_umask", "wheelhouse_index")
def test_the_runner_cache_is_built_readable_by_the_sandbox_and_pinned_by_its_manifest(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "runner-cache"

    pin = runner_cache_pin(rig.repo, rig.cache.sha, cache, TARGET)

    entry = cache / rig.cache.sha
    assert pin == hashlib.sha256((entry / MANIFEST_NAME).read_bytes()).hexdigest()
    unreadable = [path for path in entry.rglob("*") if not path.stat().st_mode & stat.S_IROTH]
    assert unreadable == [], "the sandbox reads the cache as another account"
    assert _umask() == PRIVATE, "the entry's own umask is back once the cache is built"
    assert runner_cache_pin(rig.repo, rig.cache.sha, cache, TARGET) == pin, "a second run of the commit uses the entry"


@pytest.mark.usefixtures("private_umask")
def test_a_commit_the_clone_lacks_fails_the_run_and_builds_nothing(rig: Rig, tmp_path: Path) -> None:
    missing = "f" * 40

    with pytest.raises(CacheBuildError, match=missing):
        runner_cache_pin(rig.repo, missing, tmp_path / "runner-cache", TARGET)

    assert not (tmp_path / "runner-cache" / missing).exists()
    assert _umask() == PRIVATE


def test_hosted_peers_nobody_names_the_image_of_are_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST"):
        monkeypatch.delenv(name, raising=False)

    def inputs(layout: object, tp: int) -> object:
        raise AssertionError("no toolbox is built for peers that are refused")

    with pytest.raises(PeersUnavailable, match="HYPERLOOM_ACTIVE_IMAGE_REF"):
        load_hosted_peers(WORKLOAD, inputs)  # type: ignore[arg-type]


def _origin(path: Path) -> str:
    """A repository at ``path`` with one commit; its full hash."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)
        return done.stdout.strip()

    path.mkdir(parents=True)
    git("init", "--quiet")
    (path / "README").write_text("source\n", encoding="utf-8")
    git("add", "README")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "source")
    return git("rev-parse", "HEAD")


def test_the_runs_source_is_cloned_into_the_session_once_and_a_restart_finds_it(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    sha = _origin(origin)
    destination = tmp_path / "session" / "MB-runner-source"
    destination.parent.mkdir()

    assert runner_source(destination, str(origin), sha) == destination
    shutil.rmtree(origin)

    assert runner_source(destination, str(origin), sha) == destination
    found = subprocess.run(["git", "-C", str(destination), "cat-file", "-t", sha], capture_output=True, text=True)
    assert found.stdout.strip() == "commit"


def test_a_commit_the_source_lacks_fails_the_run_and_leaves_no_clone(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _origin(origin)
    destination = tmp_path / "MB-runner-source"

    with pytest.raises(CacheBuildError, match="f" * 40):
        runner_source(destination, str(origin), "f" * 40)

    assert not destination.exists()


def _host(tmp_path: Path, repo: Path, sha: str) -> HostInfo:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    env = {
        "HYPERLOOM_INPUT_FILE": str(spec),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_RUN_CREATED_AT": "2026-10-28T12:00:00Z",
        "HYPERLOOM_STATE_DIR": str(tmp_path),
        "E2B_API_URL": "http://service.invalid",
        "E2B_SANDBOX_URL": "http://sandboxes.invalid",
        "HYPERLOOM_SANDBOX_KEY_MEASUREMENT": "m-key-000000000000",
        "HYPERLOOM_SANDBOX_TEMPLATES": '{"measurement": {"2": "ar-01ja-measurement"}}',
        "HYPERLOOM_SOURCE_REPO": str(repo),
        "HYPERLOOM_SOURCE_SHA": sha,
    }
    return read_host_info(env, require=())


@pytest.mark.usefixtures("wheelhouse_index")
def test_every_runner_sandbox_is_pinned_to_the_runner_cache_built_from_the_sessions_clone(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    layout = SessionLayout(tmp_path / "MB-session")
    layout.root.mkdir()
    cache = tmp_path / "runner-cache"

    inputs = PinnedToolboxInputs(_host(tmp_path, rig.repo, rig.cache.sha), cache)(layout, 2)

    entry = cache / rig.cache.sha
    assert inputs.runner_cache_manifest_sha256 == hashlib.sha256((entry / MANIFEST_NAME).read_bytes()).hexdigest()
    assert (inputs.session_dir, inputs.tp, inputs.source_sha) == (layout.toolbox, 2, rig.cache.sha)
    assert layout.runner_source.is_dir()
    assert manifest_from_json((entry / MANIFEST_NAME).read_bytes()).target == TARGET.manifest_form(), (
        "the wheels are for the runner's interpreter the host names, not the application's"
    )


@pytest.mark.usefixtures("wheelhouse_index")
def test_a_host_that_does_not_name_the_runners_interpreter_fails_the_session_and_builds_nothing(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RUNNER_TARGET_ENV, raising=False)
    layout = SessionLayout(tmp_path / "MB-session")
    layout.root.mkdir()
    cache = tmp_path / "runner-cache"

    with pytest.raises(CacheBuildError, match=f"\\${RUNNER_TARGET_ENV} must name the runner's interpreter"):
        PinnedToolboxInputs(_host(tmp_path, rig.repo, rig.cache.sha), cache)(layout, 2)

    assert not (cache / rig.cache.sha).exists()


def test_a_runner_cache_without_the_commit_fails_the_session_before_it_measures(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    layout = SessionLayout(tmp_path / "MB-session")
    layout.root.mkdir()

    with pytest.raises(CacheBuildError, match="f" * 40):
        PinnedToolboxInputs(_host(tmp_path, rig.repo, "f" * 40), tmp_path / "runner-cache")(layout, 2)

    assert not (tmp_path / "runner-cache" / ("f" * 40)).exists()


@pytest.mark.parametrize(("run_id", "named"), [("run-1", "run-1"), (None, "MB-session")])
def test_a_grant_without_a_run_id_names_its_sandboxes_by_the_session(
    run_id: str | None, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hyperloom_multibranch.scheduling import hosted_peers

    monkeypatch.setattr(hosted_peers, "runner_source", lambda destination, repo, sha: destination)
    monkeypatch.setattr(hosted_peers, "runner_cache_pin", lambda source, sha, cache_root, target: "e" * 64)
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    host = _host(tmp_path, tmp_path / "repo", "c" * 40)
    host = dataclasses.replace(host, run_id=run_id)

    inputs = PinnedToolboxInputs(host, tmp_path / "runner-cache")(SessionLayout(tmp_path / "MB-session"), 2)

    assert inputs.run_id == named
