# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache of the hosted toolbox: the directory named, the store it publishes accepted tests'
compiles into, and the prune at the session's start.

No hosted runner reads the directory. What is proved here is the host's side: which directory it publishes to, that it
uses none another account could change and goes on, that a verified result's compiles reach it through the hosted
executor and backend, and that its result records the cache states.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import JitCacheState, TestResult, Validity
from hyperloom.measure.jit.prune import RETIRED_DIR
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.tests.fake_runner import JIT_KEY, JIT_MODULE
from hyperloom.measure.tests.hosted_fixtures import make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions
from hyperloom_multibranch.toolbox.jit_store import JIT_STORE_ENV, JIT_STORE_NOTE, REFUSED_HOSTED
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import _toolbox

_DAY = 24 * 3600


def _options(**changed) -> HostedOptions:
    return HostedOptions(poll_interval_sec=0.05, timeout_sec=60, max_infra_retries=0, framework_configs="", **changed)


def test_a_host_that_names_no_store_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(JIT_STORE_ENV, raising=False)

    toolbox = _toolbox(None, tmp_path)

    assert toolbox.backend._config.jit_store is None, "a hosted host has no default: its tests could not read it"


def test_the_store_the_host_environment_names_is_made_this_accounts_and_published_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(JIT_STORE_ENV, str(tmp_path / "shared" / "jit-store"))

    toolbox = _toolbox(None, tmp_path)

    store = toolbox.backend._config.jit_store
    assert store is not None and store.root == tmp_path / "shared" / "jit-store"
    made = store.root.stat()
    assert (made.st_uid, stat.S_IMODE(made.st_mode)) == (os.getuid(), 0o700)


def test_the_option_wins_over_the_host_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JIT_STORE_ENV, str(tmp_path / "from-env"))

    toolbox = _toolbox(None, tmp_path, options=_options(jit_store=tmp_path / "from-option"))

    assert toolbox.backend._config.jit_store.root == tmp_path / "from-option"


@pytest.mark.parametrize("mode", [0o770, 0o707])
def test_a_store_its_group_or_others_can_write_is_not_published_to_and_the_session_starts_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: int
) -> None:
    """The runners load what it holds as compiled code: whoever else can write it could feed them. The toolbox is built
    inside the session, where a hosted run that raised would be started again into the same refusal: the session goes
    on without publishing, and says so once in its log and in its directory."""
    store = tmp_path / "jit-store"
    store.mkdir()
    store.chmod(mode)
    monkeypatch.setenv(JIT_STORE_ENV, str(store))

    toolbox = _toolbox(None, tmp_path)

    assert toolbox.backend._config.jit_store is None, "nothing is published"
    assert stat.S_IMODE(store.stat().st_mode) == mode and not any(store.iterdir()), "and it is left as it is"
    (warning,) = [record.getMessage() for record in caplog.records if "jit cache" in record.getMessage()]
    assert f"runs without the compiled-code cache {store}" in warning and "only it can write" in warning
    assert "this host publishes nothing to it" in warning
    note = json.loads((tmp_path / "session" / JIT_STORE_NOTE).read_text())
    assert note["store"] == str(store) and note["published"] is False and "only it can write" in note["reason"]
    assert note["without"] == REFUSED_HOSTED


@pytest.mark.parametrize("error", [NotADirectoryError, PermissionError])
def test_a_store_that_cannot_be_made_is_refused_like_one_the_check_refuses_and_the_session_starts_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: type[OSError]
) -> None:
    """A name below a file, or below a directory this account cannot write: raising would only start the hosted run
    again into the same failure."""
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    store = blocker / "jit-store"
    if error is PermissionError:
        store = tmp_path / "unwritable" / "jit-store"

        def unwritable(_root: Path) -> Path:
            raise PermissionError(13, "Permission denied", str(_root))

        monkeypatch.setattr("hyperloom_multibranch.toolbox.jit_store.private_jit_store", unwritable)
    monkeypatch.setenv(JIT_STORE_ENV, str(store))

    toolbox = _toolbox(None, tmp_path)

    assert toolbox.backend._config.jit_store is None, "nothing is published"
    (warning,) = [record.getMessage() for record in caplog.records if "jit cache" in record.getMessage()]
    assert f"cannot be made or checked ({error.__name__}" in warning and REFUSED_HOSTED in warning
    note = json.loads((tmp_path / "session" / JIT_STORE_NOTE).read_text())
    assert note["published"] is False and error.__name__ in note["reason"] and note["without"] == REFUSED_HOSTED


def test_building_the_toolbox_prunes_a_key_no_test_has_used_for_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jit-store"
    root.mkdir(mode=0o700)
    idle = root / "snapshots" / AITER_JIT.name / ("e" * 64)
    idle.mkdir(parents=True)
    long_ago = idle.stat().st_mtime - 30 * _DAY
    os.utime(idle, (long_ago, long_ago))
    monkeypatch.setenv(JIT_STORE_ENV, str(root))

    _toolbox(None, tmp_path)

    assert not idle.exists() and len(list((root / RETIRED_DIR).iterdir())) == 1


async def _measured(tmp_path: Path, store: DirJitStore, **fake: str) -> tuple[HostedExecutor, TestResult]:
    """One test of the hosted executor over the real hosted backend (local sandboxes, the fake runner) publishing to
    ``store``; the fake runner is told ``fake``."""
    rig = make_rig(tmp_path, jit_store=store)
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    tb = build(
        tmp_path,
        executor=executor,
        planner=PointPlanner(point_yaml(tmp_path, "planned.yaml", **fake)),
        policy=QueuePolicy(per_branch_k=1, global_concurrency=1),
        workload_kind=WorkloadKind.AGENTX,
    )
    try:
        await tb.toolbox.start()
        request_id = await tb.toolbox.submit_test(req())
        await asyncio.wait_for(tb.toolbox.drain(), 60)
        entry = tb.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None
        return executor, TestResult.from_document(entry.outcome["result"])
    finally:
        rig.backend.close()
        rig.provider.close()


async def test_a_verified_hosted_result_publishes_what_its_test_compiled_and_records_the_cache_state(
    tmp_path: Path,
) -> None:
    store = DirJitStore(tmp_path / "jit-store")

    executor, result = await _measured(tmp_path, store, fake_jit="true")

    assert result.validity is Validity.SUCCEEDED
    assert result.jit_cache == (JitCacheState(AITER_JIT.name, "miss"),), "the state the signed result states"
    got = store.fetch(AITER_JIT, JIT_KEY, tmp_path / "next-test")
    assert got.snapshot is not None and (tmp_path / "next-test" / "module_fake.so").read_bytes() == JIT_MODULE
    assert await asyncio.to_thread(executor.difference, result) is None, "the recorded states follow from the bundle"
    claimed = replace(result, jit_cache=(JitCacheState(AITER_JIT.name, "hit"),))
    assert "jit_cache: recorded" in str(await asyncio.to_thread(executor.difference, claimed))


async def test_a_hosted_result_given_the_cache_read_only_publishes_nothing(tmp_path: Path) -> None:
    store = DirJitStore(tmp_path / "jit-store")

    _, result = await _measured(tmp_path, store, fake_jit="read_only")

    assert result.validity is Validity.SUCCEEDED and result.jit_cache == (JitCacheState(AITER_JIT.name, "hit"),)
    assert not (store.root / "blobs").exists() and not (store.root / "snapshots").exists()


async def test_a_session_that_published_and_never_found_the_cache_warm_says_so_as_it_closes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A session whose runners never read what it published is publishing where they do not look."""
    store = DirJitStore(tmp_path / "jit-store")

    await _measured(tmp_path, store, fake_jit="true")

    (warning,) = [record.getMessage() for record in caplog.records if "found the cache warm" in record.getMessage()]
    assert f"published 1 contribution(s) to {store.root}" in warning


async def test_a_session_whose_tests_found_the_cache_warm_closes_without_that_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    await _measured(tmp_path, DirJitStore(tmp_path / "jit-store"), fake_jit="read_only")

    assert "found the cache warm" not in caplog.text
