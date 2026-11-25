# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The host's side of the shared compiled-code cache: what a signed result says its test compiled is published after the
result is accepted, and only then.

The runner is the fake runner, which stages a contribution below ``/out/jit-cache`` and signs it into its manifest; the host
is the real hosted backend over the local sandbox provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hyperloom.measure import hosted

from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.returncodes import HOSTED_RESULT_REJECTED_RC
from hyperloom.measure.tests.fake_runner import JIT_KEY, JIT_MODULE
from hyperloom.measure.tests.test_hosted_backend import _run, rig  # noqa: F401 - the hosted suite's fixture and runner


def _fetch(store: DirJitStore, destination: Path):
    return store.fetch(AITER_JIT, JIT_KEY, destination)


def test_what_an_accepted_result_says_its_test_compiled_is_published_and_the_next_test_can_read_it(
    rig, tmp_path
) -> None:
    store = DirJitStore(tmp_path / "jit-store")
    hosted = rig(jit_store=store)

    _, outcome = _run(hosted, tmp_path, fake_jit="true")

    assert outcome.returncode == 0
    got = _fetch(store, tmp_path / "private")
    assert got.snapshot is not None and got.problems == ()
    assert (tmp_path / "private" / "module_fake.so").read_bytes() == JIT_MODULE


def test_a_host_with_no_cache_configured_accepts_the_result_and_publishes_nothing(rig, tmp_path) -> None:
    hosted = rig()

    _, outcome = _run(hosted, tmp_path, fake_jit="true")

    assert outcome.returncode == 0 and not (tmp_path / "jit-store").exists()


def test_a_staged_list_that_is_not_the_one_the_result_signed_is_not_published_and_the_result_is_still_accepted(
    rig, tmp_path
) -> None:
    store = DirJitStore(tmp_path / "jit-store")
    hosted = rig(jit_store=store)

    _, outcome = _run(hosted, tmp_path, fake_jit="true", fake_jit_swap="true")

    assert outcome.returncode == 0
    assert _fetch(store, tmp_path / "private").snapshot is None and not (tmp_path / "jit-store" / "blobs").exists()


def test_a_result_that_is_not_accepted_publishes_nothing_even_though_the_files_are_staged(rig, tmp_path) -> None:
    """Nothing the runner staged is trusted until its signature has verified: a wrong-key result's cache is not read."""
    store = DirJitStore(tmp_path / "jit-store")
    hosted = rig(jit_store=store)

    _, outcome = _run(hosted, tmp_path, fake_jit="true", fake_mode="wrong_key")

    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert not (tmp_path / "jit-store").exists()


def test_the_archived_manifest_carries_the_cache_block_the_host_published_from(rig, tmp_path) -> None:
    hosted = rig(jit_store=DirJitStore(tmp_path / "jit-store"))
    _run(hosted, tmp_path, fake_jit="true")

    manifest = json.loads((Path(hosted.record_archive("req-1")) / "manifest.json").read_text())

    (kind,) = manifest["jit_cache"]["kinds"]
    assert kind["kind"] == "aiter_jit" and kind["contribution"]["files"] == 1


def test_a_cache_that_cannot_be_written_never_turns_an_accepted_result_into_a_failure(
    rig, tmp_path, monkeypatch
) -> None:
    def full_disk(*_args, **_kwargs) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(hosted, "publish_contributions", full_disk)
    hosted_backend = rig(jit_store=DirJitStore(tmp_path / "jit-store"))

    _, outcome = _run(hosted_backend, tmp_path, fake_jit="true")

    assert outcome.returncode == 0


def test_a_test_given_the_cache_read_only_publishes_nothing_though_files_lie_staged_and_the_entry_it_read_stays(
    rig, tmp_path
) -> None:
    """A test whose patched code the key does not cover is given the cache read only: the runner signs no contribution
    for it, and whatever its code left below ``/out/jit-cache`` never reaches the store another test's entry is in."""
    store = DirJitStore(tmp_path / "jit-store")
    hosted = rig(jit_store=store)
    _run(hosted, tmp_path, "req-1", fake_jit="true")
    stored = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    entry = store.root / "snapshots" / "aiter_jit" / JIT_KEY
    os.utime(entry, (1.0, 1.0))

    _, outcome = _run(hosted, tmp_path, "req-2", fake_jit="read_only")

    assert outcome.returncode == 0
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == stored, "nothing was added"
    assert _fetch(store, tmp_path / "private").snapshot is not None, "the entry the first test made is still served"
    assert entry.stat().st_mtime > 1.0, "the key the test read is noted as in use, so pruning keeps it"


def test_a_read_only_test_on_an_empty_store_leaves_it_empty(rig, tmp_path) -> None:
    store = DirJitStore(tmp_path / "jit-store")
    hosted = rig(jit_store=store)

    _, outcome = _run(hosted, tmp_path, fake_jit="read_only")

    assert outcome.returncode == 0
    assert _fetch(store, tmp_path / "private").snapshot is None and not (tmp_path / "jit-store" / "blobs").exists()
