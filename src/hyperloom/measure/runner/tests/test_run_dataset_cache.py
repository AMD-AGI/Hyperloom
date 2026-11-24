# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX client's tokenized dataset through the shared cache: a whole runner per test, as in ``test_run_jit``.

The grid plays the client the way aiperf's dataset cache behaves (``aiperf/dataset/mmap_cache.py``): it looks for
``<AIPERF_DATASET_MMAP_CACHE_DIR>/<key>/manifest.json``; on a hit it replays the cached ``dataset.dat``, on a miss it
tokenizes, takes a lock file in that directory and moves a complete entry into it. What reaches the store is what the host
publishes from the signed result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from hyperloom.measure import local, signing
from hyperloom.measure.jit.key import dataset_key
from hyperloom.measure.jit.publish import publish_contributions
from hyperloom.measure.runner.dataset_cache import dataset_plan, model_files
from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom.measure.runner.tests.test_run import _sheet, env, key  # noqa: F401 - fixtures
from hyperloom.measure.runner.tests.test_run_jit import Rig, _arm, _OutHandle
from hyperloom.measure.spec import MeasurementSpec
from hyperloom.measure.tests.test_local_backend import _Grid

KIND = "aiperf_dataset"
VARIABLE = "AIPERF_DATASET_MMAP_CACHE_DIR"
ENTRY = "0f" * 16
CORPUS = b"the corpus as token ids " * 64


class _ReplayingGrid(_Grid):
    """A ``run_grid`` whose AgentX client keeps its tokenized corpus as aiperf does."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.builds = 0
        self.replayed: list[bytes] = []
        self.cache_dirs: list[Path] = []

    async def __call__(self, **kwargs):
        if not kwargs["launch"].server_already_ready:
            self._client()
        return await super().__call__(**kwargs)

    def _client(self) -> None:
        named = os.environ.get(VARIABLE)
        cache = Path(named) if named else self.tmp_path / "home" / ".cache" / "aiperf" / "dataset_mmap"
        self.cache_dirs.append(cache)
        entry = cache / ENTRY
        if not (entry / "manifest.json").is_file():
            self.builds += 1
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"{ENTRY}.lock").write_text("")
            building = cache / f".{ENTRY}.tmp.{os.getpid()}"
            building.mkdir()
            (building / "dataset.dat").write_bytes(CORPUS)
            (building / "index.dat").write_bytes(b"offsets")
            (building / "manifest.json").write_bytes(b'{"version": 27}')
            os.replace(building, entry)
        self.replayed.append((entry / "dataset.dat").read_bytes())


class _DatasetRig(Rig):
    """The rig of ``test_run_jit`` whose grid is the replaying client and whose points run AgentX on a model of its own."""

    def __init__(self, tmp_path: Path, environ: dict, signing_key, monkeypatch) -> None:
        super().__init__(tmp_path, environ, signing_key, monkeypatch)
        self.model = tmp_path / "model"
        self.model.mkdir()
        (self.model / "tokenizer.json").write_text('{"model": {"vocab": {"a": 0}}}')
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"weights")
        for relative in ("aiperf/__init__.py", "aiperf/dataset/mmap_cache.py"):
            (self.bases / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.bases / relative).write_text("MANIFEST_VERSION = 27\n")
        self.client = _ReplayingGrid(tmp_path)

    def _start(self, name: str, *, cache: bool, fail: bool, extra_env: dict | None) -> tuple[Path, dict]:
        run = self.tmp_path / name
        run.mkdir()
        key_file = run / "key.pem"
        key_file.write_bytes(signing.private_key_pem(self.signing_key))
        environ = {**self.base_environ, "HYPERLOOM_SIGNING_KEY_FILE": str(key_file), **(extra_env or {})}
        if cache:
            environ[JIT_CACHE_ENV] = str(self.store.root)
        self.monkeypatch.setattr(local, "run_grid", self.client)
        return run, environ

    def agentx(self, **changes):
        benchmark = {"model": str(self.model), "agentx": True, **changes.pop("benchmark", {})}
        return _arm(self, benchmark=benchmark, **changes)

    def store_listing(self) -> dict[str, str]:
        """Every file of the store by its path, with the hash of its bytes."""
        root = self.store.root
        if not root.exists():
            return {}
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }


@pytest.fixture
def rig(tmp_path, env, key, monkeypatch) -> _DatasetRig:
    return _DatasetRig(tmp_path, env, key, monkeypatch)


def _kind(manifest, name: str = KIND):
    return next((kind for kind in manifest.jit_cache.kinds if kind.kind == name), None)


def test_a_hit_replays_the_bytes_the_first_test_built_and_tokenizes_nothing(rig) -> None:
    first, _, first_run = rig.run("first", rig.agentx())
    assert (_kind(first).state, rig.client.builds) == ("miss", 1)
    assert _kind(first).contribution is not None, "an unpatched test adds what its client built"

    rig.publish(first_run, first)
    second, status, _ = rig.run("second", rig.agentx())

    assert _kind(second).state == "hit" and _kind(second).key == _kind(first).key
    assert rig.client.builds == 1, "the second client found the corpus and tokenized nothing"
    assert rig.client.replayed == [CORPUS, CORPUS], "it replayed the same bytes the first one built"
    assert _kind(second).contribution is None and status["message"] == ""


def test_a_miss_builds_in_the_tests_own_directory_and_no_client_ever_writes_the_shared_store(rig) -> None:
    first, _, first_run = rig.run("first", rig.agentx())
    assert rig.store_listing() == {}, "the test built its corpus beside the store, never in it"
    rig.publish(first_run, first)
    published = rig.store_listing()
    rig.run("second", rig.agentx())

    assert rig.store_listing() == published, "a test that hit read the store and wrote nothing to it"
    store = rig.store.root.resolve()
    for cache in rig.client.cache_dirs:
        assert not cache.resolve().is_relative_to(store), "the client is never told the shared store"
        assert cache.resolve().is_relative_to(rig.host_tmp.resolve()), "it is a directory of the test's own"
    assert len(set(rig.client.cache_dirs)) == 2, "each test has its own"
    assert not any(name.endswith(".lock") for name in published), "aiperf's lock files are not stored"


def test_a_host_that_publishes_only_some_kinds_adds_nothing_of_the_others(rig) -> None:
    first, _, first_run = rig.run("first", rig.agentx())
    (first_run / "host").mkdir()
    out = _OutHandle(first_run / "out")

    assert publish_contributions(rig.store, out, first.jit_cache, first_run / "host", {"triton"}) == 0
    assert rig.store_listing() == {}
    assert publish_contributions(rig.store, out, first.jit_cache, first_run / "host", {KIND}) == 1
    assert {name.split("/")[1] for name in rig.store_listing() if name.startswith("snapshots/")} == {KIND}


def test_a_test_with_patched_code_reads_the_dataset_and_adds_nothing(rig) -> None:
    sglang = lambda run: [rig.root(run, "sglang", "srt/a.py", b"def schedule(batch):\n    return batch\n")]  # noqa: E731
    patched, _, patched_run = rig.run("patched", rig.agentx(roots=sglang))

    assert _kind(patched).state == "miss" and _kind(patched).contribution is None
    assert rig.publish(patched_run, patched) == 0 and rig.store_listing() == {}, "its server could have written it"

    base, _, base_run = rig.run("base", rig.agentx())
    rig.publish(base_run, base)
    again, _, _ = rig.run("again", rig.agentx(roots=sglang))
    assert _kind(again).state == "hit", "the patched test reads what an unpatched one built"


def test_a_test_that_patches_the_client_or_names_its_own_cache_does_not_use_the_shared_one(rig) -> None:
    aiperf = lambda run: [rig.root(run, "aiperf", "dataset/mmap_cache.py", b"MANIFEST_VERSION = 0\n")]  # noqa: E731
    patched, _, _ = rig.run("patched", rig.agentx(roots=aiperf))
    named, _, _ = rig.run("named", rig.agentx(envs={VARIABLE: "/elsewhere"}))

    assert _kind(patched).state == "disabled" and _kind(named).state == "disabled"
    assert any("aiperf_dataset is not used: the test patches aiperf" in note for note in patched.notes)
    assert any(f"the sheet sets {VARIABLE} itself" in note for note in named.notes)


def test_a_test_that_runs_no_agentx_client_has_no_dataset_kind(rig) -> None:
    fixed, _, _ = rig.run("fixed", _arm(rig))

    assert _kind(fixed) is None


def test_the_key_follows_the_client_and_the_tokenizer_and_not_the_weights(rig, tmp_path) -> None:
    files = model_files(rig.model)
    assert set(files) == {"tokenizer.json", "config.json"}, "the weights are not read"
    base = dataset_key("client-a", str(rig.model), files)

    (rig.model / "model.safetensors").write_bytes(b"other weights")
    assert dataset_key("client-a", str(rig.model), model_files(rig.model)) == base
    assert dataset_key("client-b", str(rig.model), files) != base, "another client is another aiperf"
    (rig.model / "tokenizer.json").write_text('{"model": {"vocab": {"b": 0}}}')
    assert dataset_key("client-a", str(rig.model), model_files(rig.model)) != base, "another tokenizer"


def test_a_plan_says_why_the_kind_is_off(tmp_path) -> None:
    spec = MeasurementSpec.from_bytes(json.dumps(_sheet()).encode())
    agentx = {"benchmark": {"model": str(tmp_path / "missing"), "framework": "sglang", "agentx": True, "envs": {}}}

    assert dataset_plan(spec, [agentx], None) == "the staging the AgentX client runs from is not known"
    assert "the model directory cannot be listed" in str(dataset_plan(spec, [agentx], "client"))
