# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session measures on: the runner cache of the run's commit, built readable by the sandbox's account
whatever umask the entry runs under and failing the run when it cannot be had, and the peers refused as on one
machine. The peers built on the service's sandboxes are pinned in ``test_local_peers``."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.measure.runner.cache_build import CacheBuildError
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig
from hyperloom_multibranch.scheduling.hosted_peers import load_hosted_peers, runner_cache_pin
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload

WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 2, 16, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
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

    pin = runner_cache_pin(rig.repo, rig.cache.sha, cache)

    entry = cache / rig.cache.sha
    assert pin == hashlib.sha256((entry / MANIFEST_NAME).read_bytes()).hexdigest()
    unreadable = [path for path in entry.rglob("*") if not path.stat().st_mode & stat.S_IROTH]
    assert unreadable == [], "the sandbox reads the cache as another account"
    assert _umask() == PRIVATE, "the entry's own umask is back once the cache is built"
    assert runner_cache_pin(rig.repo, rig.cache.sha, cache) == pin, "a second run of the commit uses the entry"


@pytest.mark.usefixtures("private_umask")
def test_a_commit_the_clone_lacks_fails_the_run_and_builds_nothing(rig: Rig, tmp_path: Path) -> None:
    missing = "f" * 40

    with pytest.raises(CacheBuildError, match=missing):
        runner_cache_pin(rig.repo, missing, tmp_path / "runner-cache")

    assert not (tmp_path / "runner-cache" / missing).exists()
    assert _umask() == PRIVATE


def test_hosted_peers_nobody_names_the_image_of_are_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST"):
        monkeypatch.delenv(name, raising=False)

    def inputs(session_dir: Path, tp: int) -> object:
        raise AssertionError("no toolbox is built for peers that are refused")

    with pytest.raises(PeersUnavailable, match="HYPERLOOM_ACTIVE_IMAGE_REF"):
        load_hosted_peers(WORKLOAD, inputs)  # type: ignore[arg-type]
