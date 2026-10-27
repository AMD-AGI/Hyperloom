# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplementary coverage for LocalRecipeStore normalisation + edge paths."""

from __future__ import annotations

import json
import threading

import pytest

from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity, recipe_canonical_id
from hyperloom.orchestrator.knowledge.recipe_kb import cid_to_path_components
from hyperloom.orchestrator.knowledge.recipe_kb import local_store as ls


def _cid(model="m", hardware="mi300", framework_name="sglang", framework_version="v1", precision="fp8") -> str:
    return recipe_canonical_id(
        RecipeIdentity(
            model=model,
            hardware=hardware,
            framework_name=framework_name,
            framework_version=framework_version,
            precision=precision,
        )
    )


# ---- _CidLock ----


def test_cid_lock_mutex_is_shared_per_path(tmp_path):
    """Every lock instance on one path must contend on the same mutex."""
    path = tmp_path / ".lock"
    assert ls._cid_mutex(path) is ls._cid_mutex(tmp_path / "." / ".lock")
    assert ls._cid_mutex(path) is not ls._cid_mutex(tmp_path / "other.lock")


def test_cid_lock_releases_the_mutex_when_setup_fails(tmp_path, monkeypatch):
    """A failed open must not leave the shared mutex held for the process."""
    path = tmp_path / ".lock"
    real_open = ls.os.open
    calls = {"n": 0}

    def _flaky_open(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient NFS failure")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(ls.os, "open", _flaky_open)
    with pytest.raises(OSError):
        with ls._CidLock(path):
            pass
    assert ls._cid_mutex(path).locked() is False
    # The next acquisition still works rather than blocking forever.
    with ls._CidLock(path):
        assert ls._cid_mutex(path).locked() is True


def test_cid_lock_serialises_threads_on_the_same_path(tmp_path):
    path = tmp_path / ".lock"
    entered = threading.Event()

    def _worker():
        with ls._CidLock(path):
            entered.set()

    with ls._CidLock(path):
        worker = threading.Thread(target=_worker)
        worker.start()
        assert entered.wait(0.3) is False
    worker.join(5)
    assert entered.is_set()


# ---- normalisation ----


def test_put_recipe_normalises_list_fields(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(
        ls.RecipeUpsert(
            canonical_id=cid,
            model="m",
            what_worked=[{"name": "w", "gain_pct": 1.0}, "skip", None],
            what_failed=[["x"], {"name": "f"}],
            remaining_gaps=[{"description": "d", "metrics": "m", "extra": 1}, "x"],
            pitfalls=[{"description": "d", "severity": "high"}, None],
            lessons=[{"statement": "s", "measured_impact": {"x": 1}}, "x"],
            sessions=[{"date": "d", "throughput_before": 1.0}, "x"],
        )
    )
    row = store.get_recipe(canonical_id=cid)
    assert row["what_worked"] == [{"name": "w", "gain_pct": 1.0}]
    assert row["what_failed"] == [{"name": "f"}]
    assert row["remaining_gaps"] == [{"description": "d", "metrics": "m"}]
    assert row["pitfalls"] == [{"description": "d", "severity": "high"}]
    assert row["lessons"] == [{"statement": "s", "measured_impact": {"x": 1}}]
    assert [s["throughput_before"] for s in row["sessions"]] == [1.0]


# ---- put / get / history with full payload ----


def test_put_get_history_roundtrip(tmp_path):
    store = ls.LocalRecipeStore(root=str(tmp_path))
    cid = _cid()
    r1 = store.put_recipe(
        ls.RecipeUpsert(
            canonical_id=cid,
            model="m",
            best_throughput=100.0,
            what_worked=[{"description": "w", "measured_impact": "i"}],
            sessions=[{"date": "d", "throughput_before": 1.0}],
            provenance={"who": "test"},
            extras={"task": "pretrain"},
        )
    )
    assert r1["version"] == 1
    assert r1["created"] is True
    r2 = store.put_recipe(ls.RecipeUpsert(canonical_id=cid, model="m", best_throughput=200.0))
    assert r2["version"] == 2
    assert r2["created"] is False

    live = store.get_recipe(canonical_id=cid)
    assert live["version"] == 2
    assert live["best_throughput"] == 200.0

    history = tmp_path.joinpath(*cid_to_path_components(cid), "history")
    assert sorted(p.name for p in history.iterdir()) == ["v1.json"]
    archived = json.loads((history / "v1.json").read_text(encoding="utf-8"))["snapshot"]
    assert archived["version"] == 1
    assert archived["best_throughput"] == 100.0
    assert archived["sessions"][0]["throughput_before"] == 1.0


def test_get_recipe_missing(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    assert store.get_recipe(canonical_id=_cid()) is None


# ---- search ----


def test_search_filters_by_label_newest_first_and_truncates_to_limit(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    for model in ("a", "b", "c"):
        store.put_recipe(ls.RecipeUpsert(canonical_id=_cid(model=model), model=model, best_throughput=100.0))
    live = tmp_path.joinpath(*cid_to_path_components(_cid(model="c")), "recipe.json")
    row = json.loads(live.read_text(encoding="utf-8"))
    del row["updated_at"]
    live.write_text(json.dumps(row), encoding="utf-8")

    assert [r["model"] for r in store.search(label_match={"model": "a"})] == ["a"]
    assert [r["model"] for r in store.search(limit=10)] == ["b", "a", "c"]
    assert [r["model"] for r in store.search(limit=2)] == ["b", "a"]


# ---- low-level helpers ----


def test_read_json_bad(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(ls.LocalRecipeStoreError):
        ls._read_json(p)
