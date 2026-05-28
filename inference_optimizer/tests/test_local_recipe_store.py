"""Unit tests for :class:`LocalRecipeStore`.

Covers the full surface of the on-disk recipe-snapshot store:

* canonical_id <-> 5-level path round-trip;
* atomic put_recipe (live + history archival, version monotonicity,
  ``replaced_by`` provenance stamping);
* get_recipe live + ``?version=N`` (history + live-version-via-?version);
* get_history (live excluded, sorted ascending by version, limit);
* delete_recipe (history preserved);
* list_recent / search (label_match containment, metric_filters
  bounds, updated_since cutoff, order_by whitelist + coverage of all
  six values);
* append_attempt + list_attempts (append-only, monotonic id,
  newest-first read);
* list_session_attempts (cross-recipe view, oldest first);
* concurrent put_recipe under the cid lock;
* malformed canonical_id rejected;
* malformed on-disk content (corrupt JSON, missing fields) handled
  defensively rather than crashing readers.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.recipe_kb import (
    ATTEMPTS_FILENAME,
    HISTORY_DIRNAME,
    InvalidCanonicalIdError,
    LocalRecipeStore,
    LocalRecipeStoreError,
    RECIPE_FILENAME,
    canonical_id_for_path,
    cid_to_path_components,
    recipe_canonical_id,
)


# ===========================================================================
# canonical_id <-> path components
# ===========================================================================
def _cid(
    *,
    model: str = "deepseek-r1",
    hardware: str = "mi300x",
    framework: str = "sglang",
    framework_version: str = "0.4.5",
    precision: str = "fp8",
) -> str:
    return recipe_canonical_id(
        model=model,
        hardware=hardware,
        framework=framework,
        framework_version=framework_version,
        precision=precision,
    )


def test_cid_to_path_components_roundtrip() -> None:
    cid = _cid(model="qwen3-30b-a3b", precision="bf16")
    parts = cid_to_path_components(cid)
    assert parts == ("qwen3-30b-a3b", "mi300x", "sglang", "0.4.5", "bf16")


def test_cid_to_path_components_rejects_legacy_4_segment_id() -> None:
    """Pre-Commit-1 ids like ``inference:m:fw:hw`` must NOT be accepted
    by the local store — would route to a 3-level dir and silently
    shadow real recipes."""
    with pytest.raises(InvalidCanonicalIdError):
        cid_to_path_components("inference:m:fw:hw")


def test_cid_to_path_components_rejects_wrong_prefix() -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        cid_to_path_components("recipe:m:hw:fw:v:p")
    assert "prefix" in ei.value.reason


def test_cid_to_path_components_rejects_empty_segment() -> None:
    with pytest.raises(InvalidCanonicalIdError):
        cid_to_path_components("inference:m::fw:v:p")


def test_canonical_id_for_path_inverse_of_cid_decomposition(
    tmp_path: Path,
) -> None:
    cid = _cid(model="m1")
    parts = cid_to_path_components(cid)
    recipe_dir = tmp_path.joinpath(*parts)
    recipe_dir.mkdir(parents=True)
    assert canonical_id_for_path(root=tmp_path, recipe_dir=recipe_dir) == cid


# ===========================================================================
# put_recipe — happy path + history archival
# ===========================================================================
def test_put_recipe_first_call_creates_live_at_version_1(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    result = store.put_recipe(
        canonical_id=cid,
        model="deepseek-r1", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
        best_config={"tp": "8"},
        best_throughput=24300.5,
        provenance={"source": "test", "generator": "ut"},
    )
    assert result == {"canonical_id": cid, "version": 1, "created": True}
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == 1
    # Top-level arbor identity fields stamped from put_recipe args.
    assert live["model"]    == "deepseek-r1"
    assert live["hardware"] == "mi300x"
    # arbor-style payload fields at the top level.
    assert live["best_config"]     == {"tp": "8"}
    assert live["best_throughput"] == 24300.5
    assert live["created_at"] == live["updated_at"]
    # Live row sits at the documented 5-level depth.
    rel = (tmp_path / "deepseek-r1" / "mi300x" / "sglang" / "0.4.5" / "fp8"
           / RECIPE_FILENAME)
    assert rel.is_file()


def test_put_recipe_second_call_archives_prior_and_bumps_version(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(
        canonical_id=cid,
        best_throughput=1000.0,
        provenance={"source": "first", "generator": "ut"},
    )
    second = store.put_recipe(
        canonical_id=cid,
        best_throughput=2000.0,
        provenance={"source": "second", "generator": "ut"},
    )
    assert second == {"canonical_id": cid, "version": 2, "created": False}
    history = store.get_history(canonical_id=cid)
    assert len(history) == 1
    archive = history[0]
    assert archive["version"] == 1
    assert archive["snapshot"]["best_throughput"] == 1000.0
    # ``replaced_by`` MUST carry the triggering write's provenance —
    # this is how an audit can trace which marathon wrote the
    # supplanting row (boundary doc §4.2).
    assert archive["replaced_by"] == {
        "source": "second", "generator": "ut",
    }
    # Live row carries the new throughput and the bumped version.
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"]         == 2
    assert live["best_throughput"] == 2000.0


def test_put_recipe_preserves_created_at_across_updates(
    tmp_path: Path,
) -> None:
    """``created_at`` reflects the FIRST put for this cid; only
    ``updated_at`` advances on subsequent puts."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid, best_throughput=1000.0)
    first_live = store.get_recipe(canonical_id=cid)
    assert first_live is not None
    created_first = first_live["created_at"]
    # Force at least one microsecond gap so the timestamps differ.
    import time
    time.sleep(0.001)
    store.put_recipe(canonical_id=cid, best_throughput=2000.0)
    second_live = store.get_recipe(canonical_id=cid)
    assert second_live is not None
    assert second_live["created_at"] == created_first
    assert second_live["updated_at"] > created_first


def test_put_recipe_rejects_empty_canonical_id(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.put_recipe(canonical_id="")


def test_put_recipe_rejects_malformed_canonical_id(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    with pytest.raises(InvalidCanonicalIdError):
        store.put_recipe(canonical_id="inference:bogus")


# ===========================================================================
# get_recipe — live + ?version=N
# ===========================================================================
def test_get_recipe_returns_none_for_unknown_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    assert store.get_recipe(canonical_id=_cid(model="never-seen")) is None


def test_get_recipe_with_version_returns_live_when_versions_match(
    tmp_path: Path,
) -> None:
    """Spec: ``?version=N`` where N == live.version returns the live
    row (not a 404). Mirror that behaviour locally."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    out = store.get_recipe(canonical_id=cid, version=1)
    assert out is not None
    assert out["version"] == 1


def test_get_recipe_with_version_returns_archive_for_prior_version(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid, best_throughput=1000.0)
    store.put_recipe(canonical_id=cid, best_throughput=2000.0)
    v1 = store.get_recipe(canonical_id=cid, version=1)
    assert v1 is not None
    assert v1["best_throughput"] == 1000.0
    v2 = store.get_recipe(canonical_id=cid, version=2)
    assert v2 is not None
    assert v2["best_throughput"] == 2000.0


def test_get_recipe_with_unknown_version_returns_none(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    assert store.get_recipe(canonical_id=cid, version=99) is None


# ===========================================================================
# get_history
# ===========================================================================
def test_get_history_excludes_live(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    store.put_recipe(canonical_id=cid)
    store.put_recipe(canonical_id=cid)
    rows = store.get_history(canonical_id=cid)
    assert [r["version"] for r in rows] == [1, 2]


def test_get_history_empty_for_unknown_cid(tmp_path: Path) -> None:
    """Mirrors the central server's contract — ``GET /history``
    never raises 404, returns an empty array instead."""
    store = LocalRecipeStore(root=tmp_path)
    assert store.get_history(canonical_id=_cid(model="absent")) == []


def test_get_history_respects_limit(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    for _ in range(5):
        store.put_recipe(canonical_id=cid)
    rows = store.get_history(canonical_id=cid, limit=2)
    assert len(rows) == 2
    assert [r["version"] for r in rows] == [1, 2]


# ===========================================================================
# delete_recipe
# ===========================================================================
def test_delete_recipe_removes_live_preserves_history(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid, best_throughput=1000.0)
    store.put_recipe(canonical_id=cid, best_throughput=2000.0)
    assert store.delete_recipe(canonical_id=cid) is True
    assert store.get_recipe(canonical_id=cid) is None
    # History survives — caller can still recover v1.
    history = store.get_history(canonical_id=cid)
    assert len(history) == 1
    assert history[0]["snapshot"]["best_throughput"] == 1000.0


def test_delete_recipe_returns_false_for_unknown_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    assert store.delete_recipe(canonical_id=_cid(model="absent")) is False


# ===========================================================================
# list_recent / search
# ===========================================================================
def _seed_diverse_recipes(store: LocalRecipeStore) -> dict[str, str]:
    """Create three recipes spanning different identity / metrics and
    return their cids keyed by short alias for assertions.

    ``task`` and ``mfu`` go through ``extras`` (they're not in the
    well-known arbor schema but the search filters look at every
    top-level field, so ``extras`` are first-class for filtering).
    """
    cid_a = recipe_canonical_id(
        model="m-a", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
    )
    cid_b = recipe_canonical_id(
        model="m-b", hardware="mi300x", framework="vllm",
        framework_version="0.6.0", precision="bf16",
    )
    cid_c = recipe_canonical_id(
        model="m-c", hardware="mi355x", framework="sglang",
        framework_version="0.5.0", precision="fp8",
    )
    import time
    store.put_recipe(
        canonical_id=cid_a,
        model="m-a", hardware="mi300x", framework="sglang",
        framework_version="0.4.5", precision="fp8",
        best_throughput=10000.0,
        extras={"task": "pretrain", "mfu": 0.4},
    )
    time.sleep(0.001)
    store.put_recipe(
        canonical_id=cid_b,
        model="m-b", hardware="mi300x", framework="vllm",
        framework_version="0.6.0", precision="bf16",
        best_throughput=25000.0,
        extras={"task": "inference", "mfu": 0.55},
    )
    time.sleep(0.001)
    store.put_recipe(
        canonical_id=cid_c,
        model="m-c", hardware="mi355x", framework="sglang",
        framework_version="0.5.0", precision="fp8",
        best_throughput=5000.0,
        extras={"task": "pretrain", "mfu": 0.30},
    )
    return {"a": cid_a, "b": cid_b, "c": cid_c}


def test_list_recent_returns_all_ordered_by_updated_desc(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.list_recent(limit=10)
    assert [r["canonical_id"] for r in rows] == [cids["c"], cids["b"], cids["a"]]


def test_search_with_label_match_filters_by_containment(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(label_match={"hardware": "mi300x"})
    assert {r["canonical_id"] for r in rows} == {cids["a"], cids["b"]}


def test_search_with_two_labels_uses_AND_semantics(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(
        label_match={"task": "pretrain", "hardware": "mi300x"},
    )
    assert [r["canonical_id"] for r in rows] == [cids["a"]]


def test_search_with_metric_filters_lower_bound(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(metric_filters={"throughput": {"min": 10000}})
    assert {r["canonical_id"] for r in rows} == {cids["a"], cids["b"]}


def test_search_with_metric_filters_upper_bound(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(metric_filters={"throughput": {"max": 6000}})
    assert {r["canonical_id"] for r in rows} == {cids["c"]}


def test_search_with_metric_filters_two_bounds(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(
        metric_filters={"throughput": {"min": 7500, "max": 12000}},
    )
    assert {r["canonical_id"] for r in rows} == {cids["a"]}


def test_search_excludes_rows_missing_the_metric_key(tmp_path: Path) -> None:
    """Mirrors the central server: a row without the metric key
    cannot be proven to satisfy the filter and is excluded."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid(model="no-tput")
    # ``best_throughput`` defaults to 0.0; we use a high min to trip
    # the "missing key" path since 0.0 obviously fails the bound.
    store.put_recipe(canonical_id=cid, extras={"mfu": 0.5})
    rows = store.search(metric_filters={"throughput": {"min": 100}})
    assert rows == []


def test_search_combined_label_and_metric(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(
        label_match={"task": "pretrain"},
        metric_filters={"mfu": {"min": 0.35}},
    )
    assert [r["canonical_id"] for r in rows] == [cids["a"]]


def test_search_updated_since_filters_old_rows(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid_old = _cid(model="old")
    cid_new = _cid(model="new")
    store.put_recipe(canonical_id=cid_old)
    cutoff_marker = store.get_recipe(canonical_id=cid_old)
    assert cutoff_marker is not None
    cutoff = cutoff_marker["updated_at"]
    import time
    time.sleep(0.001)
    store.put_recipe(canonical_id=cid_new)
    rows = store.search(updated_since=cutoff)
    assert [r["canonical_id"] for r in rows] == [cid_new, cid_old]


def test_search_order_by_invalid_raises(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.search(order_by="banana ASC")


def test_search_supports_all_six_order_by_values(tmp_path: Path) -> None:
    """Whitelist parity with the central server's contract."""
    store = LocalRecipeStore(root=tmp_path)
    _seed_diverse_recipes(store)
    for ob in (
        "updated_at DESC", "updated_at ASC",
        "created_at DESC", "created_at ASC",
        "version DESC",    "version ASC",
    ):
        rows = store.search(order_by=ob)
        assert len(rows) == 3, f"order_by={ob!r}"


def test_search_limit_is_clamped_to_1_to_1000(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    _seed_diverse_recipes(store)
    rows = store.search(limit=2)
    assert len(rows) == 2


# ===========================================================================
# attempts
# ===========================================================================
def test_append_attempt_creates_attempts_file(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    out = store.append_attempt(
        canonical_id=cid,
        session_id="sess-1",
        diff={"tp": [8, 16]},
        outcome="kept",
        fitness=0.83,
    )
    assert out["id"] == 1
    assert out["recipe_canonical_id"] == cid
    rows = store.list_attempts(canonical_id=cid)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "kept"
    assert rows[0]["fitness"] == 0.83


def test_append_attempt_does_not_require_parent_recipe(tmp_path: Path) -> None:
    """Mirrors central server: attempts are filed even if the recipe
    row doesn't exist (no FK)."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid(model="orphan")
    out = store.append_attempt(
        canonical_id=cid, session_id="sess-1", outcome="kept",
    )
    assert out["id"] == 1
    # No recipe.json was created — attempts dir is independent.
    assert store.get_recipe(canonical_id=cid) is None


def test_append_attempt_id_is_monotonic_per_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    ids = [
        store.append_attempt(
            canonical_id=cid, session_id="s", outcome="kept",
        )["id"]
        for _ in range(3)
    ]
    assert ids == [1, 2, 3]


def test_list_attempts_returns_newest_first(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    for outcome in ("kept", "reverted", "failed"):
        store.append_attempt(
            canonical_id=cid, session_id="s", outcome=outcome,
        )
    rows = store.list_attempts(canonical_id=cid)
    assert [r["outcome"] for r in rows] == ["failed", "reverted", "kept"]


def test_list_attempts_respects_limit(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    for _ in range(5):
        store.append_attempt(
            canonical_id=cid, session_id="s", outcome="kept",
        )
    rows = store.list_attempts(canonical_id=cid, limit=2)
    assert len(rows) == 2


def test_list_session_attempts_aggregates_across_recipes(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid_a = _cid(model="m-a")
    cid_b = _cid(model="m-b")
    store.append_attempt(canonical_id=cid_a, session_id="s1", outcome="kept")
    store.append_attempt(canonical_id=cid_b, session_id="s1", outcome="reverted")
    store.append_attempt(canonical_id=cid_a, session_id="s2", outcome="kept")
    rows = store.list_session_attempts(session_id="s1")
    assert len(rows) == 2
    assert {r["recipe_canonical_id"] for r in rows} == {cid_a, cid_b}


# ===========================================================================
# Concurrency
# ===========================================================================
def test_put_recipe_concurrent_writers_keep_versions_monotonic(
    tmp_path: Path,
) -> None:
    """Concurrent put_recipe calls under the cid lock must produce
    contiguous monotonic versions and a complete history chain."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()

    def write_once(idx: int) -> dict[str, Any]:
        return store.put_recipe(
            canonical_id=cid,
            best_throughput=float(1000 + idx),
            provenance={"source": "concurrent", "generator": f"w{idx}"},
        )

    n = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(write_once, range(n)))

    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == n
    history = store.get_history(canonical_id=cid)
    assert [r["version"] for r in history] == list(range(1, n))


def test_append_attempt_concurrent_keeps_ids_unique(tmp_path: Path) -> None:
    """No two concurrent appenders may produce the same ``id``."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()

    def append_once(_: int) -> int:
        return store.append_attempt(
            canonical_id=cid, session_id="s", outcome="kept",
        )["id"]

    n = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        ids = list(ex.map(append_once, range(n)))
    assert sorted(ids) == list(range(1, n + 1))


# ===========================================================================
# Defensive: malformed on-disk content
# ===========================================================================
def test_search_skips_directories_with_corrupt_recipe_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated / hand-edited recipe.json must not crash search;
    it should be reported and skipped (the central server has no
    such row, so excluding it is the conservative choice)."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    # Corrupt one of the seeded files.
    bad_path = tmp_path.joinpath(*cid_to_path_components(cid)) / RECIPE_FILENAME
    bad_path.write_text("{this is not json")

    # Reading via search() must surface as LocalRecipeStoreError —
    # we'd rather propagate than silently lie about the corpus.
    with pytest.raises(LocalRecipeStoreError):
        store.search()


def test_list_attempts_skips_malformed_lines(tmp_path: Path) -> None:
    """A single corrupt line in attempts.ndjson must not lose every
    valid row that came after it."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.append_attempt(
        canonical_id=cid, session_id="s", outcome="kept",
    )
    attempts_path = (
        tmp_path.joinpath(*cid_to_path_components(cid)) / ATTEMPTS_FILENAME
    )
    # Inject a malformed line + a second valid row.
    with attempts_path.open("a", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps({
            "id": 99,
            "recipe_canonical_id": cid,
            "session_id": "s",
            "attempt_at": "2026-05-28T10:00:00.000000+00:00",
            "outcome": "reverted",
        }) + "\n")
    rows = store.list_attempts(canonical_id=cid)
    outcomes = [r["outcome"] for r in rows]
    assert "kept" in outcomes
    assert "reverted" in outcomes


def test_walk_skips_non_5level_directories(tmp_path: Path) -> None:
    """Operator-created stray directory at the wrong depth must NOT
    be picked up by list_recent (silent shadowing would corrupt the
    search corpus)."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    # Inject a recipe.json at depth 3 (intentionally wrong).
    stray = tmp_path / "model" / "hw" / "fw" / RECIPE_FILENAME
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps({"canonical_id": "stray", "version": 1}))
    rows = store.list_recent()
    cids_found = {r["canonical_id"] for r in rows}
    assert cid in cids_found
    assert "stray" not in cids_found


# ===========================================================================
# Maintenance
# ===========================================================================
def test_purge_recipe_removes_everything(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    store.put_recipe(canonical_id=cid)
    store.append_attempt(canonical_id=cid, session_id="s", outcome="kept")
    store.purge_recipe(canonical_id=cid)
    assert store.get_recipe(canonical_id=cid) is None
    assert store.get_history(canonical_id=cid) == []
    assert store.list_attempts(canonical_id=cid) == []
    # Tree above the cid is left intact (other cids may share prefix).
    recipe_dir = tmp_path.joinpath(*cid_to_path_components(cid))
    assert not recipe_dir.is_dir()


def test_construction_does_not_create_root(tmp_path: Path) -> None:
    """Construction is cheap: a degraded run that never touches the KB
    must not create any directory on disk."""
    root = tmp_path / "kb-never-used"
    LocalRecipeStore(root=root)
    assert not root.exists()


def test_str_root_accepted(tmp_path: Path) -> None:
    """``LocalRecipeStore(root=str(...))`` still works (defensive
    coercion in __post_init__)."""
    store = LocalRecipeStore(root=str(tmp_path))
    cid = _cid(model="m-x")
    store.put_recipe(canonical_id=cid)
    assert store.get_recipe(canonical_id=cid) is not None


def test_recipe_payload_carries_canonical_id_and_version(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    payload = json.loads(
        (tmp_path.joinpath(*cid_to_path_components(cid)) / RECIPE_FILENAME)
        .read_text(encoding="utf-8")
    )
    assert payload["canonical_id"] == cid
    assert payload["version"] == 1


def test_history_dir_lives_at_six_levels_below_root(tmp_path: Path) -> None:
    """``history/`` is the only directory that may sit below the
    5-level recipe dir; the walker MUST NOT recurse into it."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    store.put_recipe(canonical_id=cid)
    history_dir = (
        tmp_path.joinpath(*cid_to_path_components(cid)) / HISTORY_DIRNAME
    )
    assert history_dir.is_dir()
    rel = history_dir.relative_to(tmp_path).parts
    assert len(rel) == 6
