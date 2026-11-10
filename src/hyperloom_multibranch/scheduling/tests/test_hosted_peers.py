# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted session measures on: the runner cache of the run's commit, built readable by the sandbox's account
whatever umask the entry runs under from the session's own clone of the run's source, which a restart finds again,
on another account of the group too, failing the run when it cannot be had and pinning every runner sandbox to it
otherwise, and the peers refused as on one machine. The peers built on the service's sandboxes are pinned in
``test_local_peers``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import hyperloom_multibranch
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import RUNNER_TARGET_ENV, CacheBuildError, CacheEntry, runner_wheel_target
from hyperloom.measure.runner.cache_layout import manifest_from_json
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME, SOURCE_DIR
from hyperloom.measure.runner.suite_publish import SuitePublishError
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig
from hyperloom_multibranch.scheduling import hosted_peers
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


class _FakeEntry:
    """``ensure_runner_cache`` finding the commit's entry built, and the staging publish it is followed by."""

    def __init__(self, cache: Path, refusal: str = "") -> None:
        self.entry = CacheEntry(cache / ("a" * 40), "b" * 64, reused=True)
        self.refusal = refusal
        self.published: list[tuple[Path, Path, int]] = []

    def ensure(self, repo: Path, sha: str, cache_root: Path, options: object) -> CacheEntry:
        return self.entry

    def publish(self, source: Path, cache_root: Path) -> Path:
        if self.refusal:
            raise SuitePublishError(self.refusal)
        self.published.append((source, cache_root, _umask()))
        return cache_root / "inference-suite" / "k"


@pytest.mark.usefixtures("private_umask")
def test_the_pin_publishes_the_staging_of_the_entrys_own_source_readable_by_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEntry(tmp_path / "runner-cache")
    monkeypatch.setattr(cache_build, "ensure_runner_cache", fake.ensure)
    monkeypatch.setattr(cache_build, "publish_inference_suite", fake.publish)

    pin = runner_cache_pin(tmp_path / "clone", "a" * 40, tmp_path / "runner-cache", TARGET)

    assert pin == "b" * 64
    assert fake.published == [
        (fake.entry.path / SOURCE_DIR, tmp_path / "runner-cache", hosted_peers.RUNNER_CACHE_UMASK)
    ]
    assert _umask() == PRIVATE


@pytest.mark.usefixtures("private_umask")
def test_a_staging_that_cannot_be_published_fails_the_run_as_a_cache_build_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEntry(tmp_path / "runner-cache", refusal="exited 2: staging_missing: offline")
    monkeypatch.setattr(cache_build, "ensure_runner_cache", fake.ensure)
    monkeypatch.setattr(cache_build, "publish_inference_suite", fake.publish)

    with pytest.raises(CacheBuildError, match="staging_missing: offline"):
        runner_cache_pin(tmp_path / "clone", "a" * 40, tmp_path / "runner-cache", TARGET)

    assert _umask() == PRIVATE


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

    left = [path.name for path in tmp_path.iterdir() if path.name.startswith(destination.name)]
    assert left == [], "no clone and no staging left"


def test_the_clone_is_made_where_no_other_account_can_plant_a_config_and_is_left_readable_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "origin"
    sha = _origin(origin)
    session = tmp_path / "session"
    session.mkdir()
    ran = tmp_path / "ran"
    # A repository another account of the group put where the clone used to be staged, under a name anyone could know,
    # after any cleanup of it (no cleanup removes it here): its config sends the fetch of the source to a program.
    planted = session / "MB-runner-source.partial"
    subprocess.run(["git", "init", "--quiet", str(planted)], check=True)
    with (planted / ".git" / "config").open("a") as config:
        config.write(
            f'[protocol "ext"]\n\tallow = always\n[url "ext::sh -c echo% ran>>{ran}"]\n\tinsteadOf = {origin}\n'
        )
    monkeypatch.setattr(hosted_peers.shutil, "rmtree", lambda *_args, **_kwargs: None)

    destination = runner_source(session / "MB-runner-source", str(origin), sha)

    assert not ran.exists()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755
    found = subprocess.run(["git", "-C", str(destination), "cat-file", "-t", sha], capture_output=True, text=True)
    assert found.stdout.strip() == "commit"


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


def test_a_clone_put_in_place_by_another_account_starts_no_program_of_its_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    layout = SessionLayout(tmp_path / "MB-session")
    layout.root.mkdir()
    sha = _origin(layout.runner_source)
    ran, transport = tmp_path / "ran", tmp_path / "transport"
    transport.write_text(f"#!/bin/sh\necho ran >> {ran}\nexit 1\n")
    transport.chmod(0o755)
    # A partial clone whose promisor remote's transport is a program, missing the commit, so the first read fetches.
    with (layout.runner_source / ".git" / "config").open("a") as config:
        config.write(
            f"[core]\n\trepositoryformatversion = 1\n\tsshCommand = {transport}\n"
            '[extensions]\n\tpartialClone = promisor\n[remote "promisor"]\n\turl = ssh://example.invalid/r\n'
            "\tpromisor = true\n"
        )
    (layout.runner_source / ".git" / "objects" / sha[:2] / sha[2:]).unlink()

    with pytest.raises(CacheBuildError, match=sha):
        PinnedToolboxInputs(_host(tmp_path, tmp_path / "origin", sha), tmp_path / "runner-cache")(layout, 2)

    assert not ran.exists()


@pytest.mark.usefixtures("wheelhouse_index")
def test_an_object_planted_in_the_clone_under_the_id_of_one_of_the_commits_fails_the_run_and_builds_nothing(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    layout = SessionLayout(tmp_path / "MB-session")
    layout.root.mkdir()
    sha = rig.cache.sha
    runner_source(layout.runner_source, str(rig.repo), sha)
    # Another account of the group rewrites the runner's entry point in the clone under the id git named it by.
    shown = subprocess.run(
        ["git", "-C", str(layout.runner_source), "rev-parse", f"{sha}:src/hyperloom/measure/runner/__main__.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    blob = shown.stdout.strip()
    loose = layout.runner_source / ".git" / "objects" / blob[:2] / blob[2:]
    assert loose.is_file(), "the clone keeps the commit's objects loose"
    planted = b"import os\nos.system('echo planted')\n"
    loose.chmod(0o644)
    loose.write_bytes(zlib.compress(b"blob %d\0" % len(planted) + planted))
    cache = tmp_path / "runner-cache"

    with pytest.raises(CacheBuildError, match=f"copying {sha}"):
        PinnedToolboxInputs(_host(tmp_path, rig.repo, sha), cache)(layout, 2)

    assert not (cache / sha).exists()


def _as_account(accounts: Accounts, uid: int, call: str) -> dict[str, object]:
    """``call``, an expression of this module's, evaluated by ``uid`` in a process of its own under umask ``022``: its
    JSON result."""

    source = Path(hyperloom_multibranch.__file__).parents[1]
    code = (
        f"import json, sys\nfrom pathlib import Path\nsys.path.insert(0, {str(source)!r})\n"
        "from hyperloom_multibranch.scheduling.tests import test_hosted_peers\n"
        f"print(json.dumps(test_hosted_peers.{call}))"
    )
    return json.loads(accounts.run(uid, code).splitlines()[-1])


def _clone_as_first_leg(session: str, origin: str, sha: str) -> dict[str, object]:
    """The session's clone of commit ``sha`` of ``origin``, made by the account this process is."""
    return {"clone": str(runner_source(SessionLayout(Path(session)).runner_source, origin, sha))}


def _pin_as_next_leg(session: str, state: str, origin: str, sha: str, cache: str) -> dict[str, object]:
    """The toolbox inputs of the session, had by the account this process is: their runner cache pin."""
    inputs = PinnedToolboxInputs(_host(Path(state), Path(origin), sha), Path(cache))(SessionLayout(Path(session)), 2)
    return {"pin": inputs.runner_cache_manifest_sha256}


def test_a_leg_on_another_account_of_the_group_builds_the_runner_cache_from_the_clone_an_earlier_leg_made(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The accounts' processes start in a directory they cannot read: no git command may look for a repository there.
    monkeypatch.chdir(tmp_path)
    with other_accounts() as accounts:
        source = Path(hyperloom_multibranch.__file__).parents[1]
        if any(not directory.stat().st_mode & stat.S_IXOTH for directory in (source, *source.parents)):
            pytest.skip(f"another account cannot reach this checkout's {source}")
        if accounts.python != sys.executable:
            pytest.skip(f"another account cannot run {sys.executable}, whose environment holds the project's packages")
        sha = rig.cache.sha
        # The run's source is the first account's own, and the wheels are where both accounts read them.
        origin, wheelhouse = accounts.root / "origin", accounts.root / "wheelhouse"
        shutil.copytree(rig.repo, origin, symlinks=True)
        for path in (origin, *origin.rglob("*")):
            os.lchown(path, accounts.first, accounts.group)
        shutil.copytree(rig.wheelhouse, wheelhouse)
        monkeypatch.setenv("PIP_NO_INDEX", "1")
        monkeypatch.setenv("PIP_FIND_LINKS", str(wheelhouse))
        monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
        session, state, cache = (accounts.shared_dir(name) for name in ("MB-session", "state", "runner-cache"))
        clone = _as_account(
            accounts, accounts.first, f"_clone_as_first_leg({str(session)!r}, {str(origin)!r}, {sha!r})"
        )

        pinned = _as_account(
            accounts,
            accounts.second,
            f"_pin_as_next_leg({str(session)!r}, {str(state)!r}, {str(origin)!r}, {sha!r}, {str(cache)!r})",
        )

        assert Path(str(clone["clone"])).stat().st_uid == accounts.first
        entry = cache / sha
        assert pinned["pin"] == hashlib.sha256((entry / MANIFEST_NAME).read_bytes()).hexdigest()
        assert entry.stat().st_uid == accounts.second, "the second account built the entry from its copy of the clone"


@pytest.mark.parametrize(("run_id", "named"), [("run-1", "run-1"), (None, "MB-session")])
def test_a_grant_without_a_run_id_names_its_sandboxes_by_the_session(
    run_id: str | None, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hyperloom_multibranch.scheduling import hosted_peers

    monkeypatch.setattr(hosted_peers, "runner_source", lambda destination, repo, sha: destination)
    monkeypatch.setattr(hosted_peers, "private_copy", lambda source, private, sha: source)
    monkeypatch.setattr(hosted_peers, "runner_cache_pin", lambda source, sha, cache_root, target: "e" * 64)
    monkeypatch.setenv(RUNNER_TARGET_ENV, RUNNER_TAG)
    host = _host(tmp_path, tmp_path / "repo", "c" * 40)
    host = dataclasses.replace(host, run_id=run_id)

    inputs = PinnedToolboxInputs(host, tmp_path / "runner-cache")(SessionLayout(tmp_path / "MB-session"), 2)

    assert inputs.run_id == named
    assert inputs.session_id == "MB-session", "a session the run starts again does not reuse a closed one's ids"
