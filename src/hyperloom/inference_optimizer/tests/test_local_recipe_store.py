# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :class:`LocalRecipeStore` (on-disk recipe-snapshot store)."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb import (
    ATTEMPTS_FILENAME,
    HISTORY_DIRNAME,
    InvalidCanonicalIdError,
    LocalRecipeStore,
    LocalRecipeStoreError,
    RECIPE_FILENAME,
    Recipe,
    canonical_id_for_path,
    cid_to_path_components,
    recipe_canonical_id,
)
from hyperloom.orchestrator.knowledge.recipe_kb.local_store import _list_jsonl, _matches_labels


def test_from_dict_reads_legacy_framework_key() -> None:
    """``from_dict`` must hydrate ``framework_name`` from the legacy ``framework``
    key and must not leak the legacy key into ``extras``."""
    legacy = {
        "canonical_id": "inference:m:mi300x:sglang:unknown_model_type:unknown_arch:0.4.5:fp8",
        "model": "m",
        "hardware": "mi300x",
        "framework": "sglang",
        "framework_version": "0.4.5",
        "precision": "fp8",
    }
    recipe = Recipe.from_dict(legacy)
    assert recipe.framework_name == "sglang"
    assert "framework" not in recipe.extras


def test_from_dict_prefers_new_framework_name_over_legacy() -> None:
    """When both keys exist the current ``framework_name`` wins."""
    both = {"framework_name": "vllm", "framework": "sglang"}
    assert Recipe.from_dict(both).framework_name == "vllm"


def test_matches_labels_matches_legacy_framework_payload() -> None:
    """A search filtered by ``framework_name`` must still match recipe rows
    persisted with the legacy ``framework`` key."""
    legacy_payload = {"model": "m", "hardware": "mi300x", "framework": "sglang"}
    assert _matches_labels(legacy_payload, {"framework_name": "sglang"}) is True
    assert _matches_labels(legacy_payload, {"framework_name": "vllm"}) is False


def _cid(
    *,
    model: str = "deepseek-r1",
    hardware: str = "mi300x",
    framework_name: str = "sglang",
    framework_version: str = "0.4.5",
    precision: str = "fp8",
) -> str:
    return recipe_canonical_id(
        model=model,
        hardware=hardware,
        framework_name=framework_name,
        framework_version=framework_version,
        precision=precision,
    )


def _list_attempt_rows(store: LocalRecipeStore, canonical_id: str) -> list[dict[str, Any]]:
    return _list_jsonl(store._attempts_path(canonical_id))


def test_cid_to_path_components_roundtrip() -> None:
    cid = _cid(model="qwen3-30b-a3b", precision="bf16")
    parts = cid_to_path_components(cid)
    assert parts == ("qwen3-30b-a3b", "mi300x", "sglang", "unknown_model_type", "unknown_arch", "0.4.5", "bf16")


def test_cid_to_path_components_rejects_legacy_4_segment_id() -> None:
    """4-segment ids like ``inference:m:fw:hw`` must NOT be accepted."""
    with pytest.raises(InvalidCanonicalIdError):
        cid_to_path_components("inference:m:fw:hw")


def test_cid_to_path_components_rejects_empty_id() -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        cid_to_path_components("")
    assert "empty string" in ei.value.reason


def test_cid_to_path_components_rejects_legacy_6_segment() -> None:
    """Legacy 5-tuple (6-segment) ids are no longer accepted."""
    with pytest.raises(InvalidCanonicalIdError):
        cid_to_path_components("inference:model:hw:fw:ver:prec")


def test_cid_to_path_components_rejects_wrong_prefix() -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        cid_to_path_components("recipe:m:hw:fw:v:p:mt:arch")
    assert "prefix" in ei.value.reason


def test_cid_to_path_components_rejects_empty_segment() -> None:
    with pytest.raises(InvalidCanonicalIdError):
        cid_to_path_components("inference:m::fw:v:p:mt:arch")


def test_cid_to_path_components_rejects_traversal_segment() -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        cid_to_path_components("inference:..:hw:fw:mt:arch:fv:prec")
    assert "not a safe path component" in ei.value.reason


def test_cid_to_path_components_rejects_absolute_segment() -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        cid_to_path_components("inference:/abs:hw:fw:mt:arch:fv:prec")
    assert "not a safe path component" in ei.value.reason


def test_put_recipe_rejects_traversal_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    with pytest.raises(InvalidCanonicalIdError):
        store.put_recipe(canonical_id="inference:..:hw:fw:mt:arch:fv:prec")


def test_canonical_id_for_path_inverse_of_cid_decomposition(
    tmp_path: Path,
) -> None:
    cid = _cid(model="m1")
    parts = cid_to_path_components(cid)
    recipe_dir = tmp_path.joinpath(*parts)
    recipe_dir.mkdir(parents=True)
    assert canonical_id_for_path(root=tmp_path, recipe_dir=recipe_dir) == cid


def test_canonical_id_for_path_rejects_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside" / "recipe"
    with pytest.raises(InvalidCanonicalIdError) as ei:
        canonical_id_for_path(root=tmp_path, recipe_dir=outside)
    assert "not under store root" in ei.value.reason


def test_canonical_id_for_path_rejects_unexpected_depth(tmp_path: Path) -> None:
    with pytest.raises(InvalidCanonicalIdError) as ei:
        canonical_id_for_path(root=tmp_path, recipe_dir=tmp_path / "too" / "shallow")
    assert "expected" in ei.value.reason


def test_put_recipe_first_call_creates_live_at_version_1(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    result = store.put_recipe(
        canonical_id=cid,
        model="deepseek-r1",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        best_config={"tp": "8"},
        best_throughput=24300.5,
        provenance={"source": "test", "generator": "ut"},
    )
    assert result["canonical_id"] == cid
    assert result["version"] == 1
    assert result["created"] is True
    # A fresh row has no prior knowledge entries to diff against.
    assert set(result["prior_counts"].values()) == {0}
    assert set(result["counts"].values()) == {0}
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == 1
    assert live["model"] == "deepseek-r1"
    assert live["hardware"] == "mi300x"
    assert live["best_config"] == {"tp": "8"}
    assert live["best_throughput"] == 24300.5
    assert live["created_at"] == live["updated_at"]
    # Live row sits at the 7-level depth.
    rel = (
        tmp_path
        / "deepseek-r1"
        / "mi300x"
        / "sglang"
        / "unknown_model_type"
        / "unknown_arch"
        / "0.4.5"
        / "fp8"
        / RECIPE_FILENAME
    )
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
    assert second["canonical_id"] == cid
    assert second["version"] == 2
    assert second["created"] is False
    archived = store.get_recipe(canonical_id=cid, version=1)
    assert archived is not None
    assert archived["version"] == 1
    assert archived["best_throughput"] == 1000.0
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == 2
    assert live["best_throughput"] == 2000.0


def test_put_recipe_keeps_existing_history_envelope_when_snapshot_matches(
    tmp_path: Path,
) -> None:
    """A leftover ``history/v{live}`` from a crash between the two renames
    must not be overwritten by the next successful put."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(
        canonical_id=cid,
        best_throughput=1000.0,
        provenance={"source": "first", "generator": "ut"},
    )
    store.put_recipe(
        canonical_id=cid,
        best_throughput=2000.0,
        provenance={"source": "second", "generator": "ut"},
    )
    live = store.get_recipe(canonical_id=cid)
    assert live is not None and live["version"] == 2
    archive_path = store._history_version_path(cid, 2)
    crashed = {
        "canonical_id": cid,
        "version": 2,
        "archived_at": "2026-01-01T00:00:00.000000+00:00",
        "replaced_by": {"source": "crashed", "generator": "ut"},
        "snapshot": dict(live),
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(crashed, indent=2, sort_keys=True), encoding="utf-8")

    third = store.put_recipe(
        canonical_id=cid,
        best_throughput=3000.0,
        provenance={"source": "third", "generator": "ut"},
    )
    assert third["version"] == 3
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == 3
    assert live["best_throughput"] == 3000.0
    leftover = json.loads(archive_path.read_text(encoding="utf-8"))
    assert leftover["replaced_by"] == {"source": "crashed", "generator": "ut"}
    assert leftover["archived_at"] == "2026-01-01T00:00:00.000000+00:00"
    assert leftover["snapshot"]["best_throughput"] == 2000.0
    v1 = store.get_recipe(canonical_id=cid, version=1)
    assert v1 is not None and v1["best_throughput"] == 1000.0


@pytest.mark.parametrize("corrupt_body", ["{ truncated", "[]"])
def test_put_recipe_rewrites_unreadable_history_and_still_advances(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    corrupt_body: str,
) -> None:
    """A leftover history file that cannot be parsed must not block later puts."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(
        canonical_id=cid,
        best_throughput=1000.0,
        provenance={"source": "first", "generator": "ut"},
    )
    store.put_recipe(
        canonical_id=cid,
        best_throughput=2000.0,
        provenance={"source": "second", "generator": "ut"},
    )
    archive_path = store._history_version_path(cid, 2)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(corrupt_body, encoding="utf-8")

    with caplog.at_level("WARNING"):
        third = store.put_recipe(
            canonical_id=cid,
            best_throughput=3000.0,
            provenance={"source": "third", "generator": "ut"},
        )
    assert third["version"] == 3
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    assert live["version"] == 3
    assert live["best_throughput"] == 3000.0
    rewritten = json.loads(archive_path.read_text(encoding="utf-8"))
    assert rewritten["replaced_by"] == {"source": "third", "generator": "ut"}
    assert rewritten["snapshot"]["best_throughput"] == 2000.0
    assert any("unreadable history" in rec.message for rec in caplog.records)


def test_put_recipe_counts_report_pre_and_post_write_sizes(
    tmp_path: Path,
) -> None:
    """``prior_counts``/``counts`` let a caller derive what a write contributed.

    ``put_recipe`` rewrites the whole row, so absolute sizes alone cannot tell
    an amend that appended a lesson from a read-modify-write that round-trips
    the existing lists untouched (what the T0 anchor does).
    """
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(
        canonical_id=cid,
        lessons=[{"statement": "raise tp to 8", "measured_impact": "+12%"}],
    )

    # Amend: one more lesson, one new pitfall.
    amend = store.put_recipe(
        canonical_id=cid,
        lessons=[
            {"statement": "raise tp to 8", "measured_impact": "+12%"},
            {"statement": "enable chunked prefill", "measured_impact": "+4%"},
        ],
        pitfalls=[{"description": "ep=8 OOMs on 30B", "severity": "crash"}],
    )
    assert amend["prior_counts"]["lessons"] == 1
    assert amend["counts"]["lessons"] == 2
    assert amend["prior_counts"]["pitfalls"] == 0
    assert amend["counts"]["pitfalls"] == 1

    # Round-trip: same content written back; sizes unchanged (delta == 0).
    live = store.get_recipe(canonical_id=cid)
    assert live is not None
    roundtrip = store.put_recipe(
        canonical_id=cid,
        lessons=list(live["lessons"]),
        pitfalls=list(live["pitfalls"]),
    )
    assert roundtrip["prior_counts"] == roundtrip["counts"]


def test_put_recipe_preserves_created_at_across_updates(
    tmp_path: Path,
) -> None:
    """``created_at`` reflects the first put; only ``updated_at`` advances on subsequent puts."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid, best_throughput=1000.0)
    first_live = store.get_recipe(canonical_id=cid)
    assert first_live is not None
    created_first = first_live["created_at"]
    # Force a gap so the timestamps differ.
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


def test_get_recipe_returns_none_for_unknown_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    assert store.get_recipe(canonical_id=_cid(model="never-seen")) is None


def test_get_recipe_with_version_returns_live_when_versions_match(
    tmp_path: Path,
) -> None:
    """``?version=N`` where N == live.version returns the live row (not a 404)."""
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


def _seed_diverse_recipes(store: LocalRecipeStore) -> dict[str, str]:
    """Create three recipes spanning different identity/metrics, keyed by alias."""
    cid_a = recipe_canonical_id(
        model="m-a",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    cid_b = recipe_canonical_id(
        model="m-b",
        hardware="mi300x",
        framework_name="vllm",
        framework_version="0.6.0",
        precision="bf16",
    )
    cid_c = recipe_canonical_id(
        model="m-c",
        hardware="mi355x",
        framework_name="sglang",
        framework_version="0.5.0",
        precision="fp8",
    )
    import time

    store.put_recipe(
        canonical_id=cid_a,
        model="m-a",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
        best_throughput=10000.0,
        extras={"task": "pretrain", "mfu": 0.4},
    )
    time.sleep(0.001)
    store.put_recipe(
        canonical_id=cid_b,
        model="m-b",
        hardware="mi300x",
        framework_name="vllm",
        framework_version="0.6.0",
        precision="bf16",
        best_throughput=25000.0,
        extras={"task": "inference", "mfu": 0.55},
    )
    time.sleep(0.001)
    store.put_recipe(
        canonical_id=cid_c,
        model="m-c",
        hardware="mi355x",
        framework_name="sglang",
        framework_version="0.5.0",
        precision="fp8",
        best_throughput=5000.0,
        extras={"task": "pretrain", "mfu": 0.30},
    )
    return {"a": cid_a, "b": cid_b, "c": cid_c}


def test_search_default_order_is_updated_desc(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cids = _seed_diverse_recipes(store)
    rows = store.search(limit=10)
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
    """A row without the metric key cannot satisfy the filter and is excluded."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid(model="no-tput")
    # ``best_throughput`` defaults to 0.0; a high min trips the missing-key path.
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
        "updated_at DESC",
        "updated_at ASC",
        "created_at DESC",
        "created_at ASC",
        "version DESC",
        "version ASC",
    ):
        rows = store.search(order_by=ob)
        assert len(rows) == 3, f"order_by={ob!r}"


def test_search_limit_is_clamped_to_1_to_1000(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    _seed_diverse_recipes(store)
    rows = store.search(limit=2)
    assert len(rows) == 2


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
    rows = _list_attempt_rows(store, cid)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "kept"
    assert rows[0]["fitness"] == 0.83


def test_append_attempt_does_not_require_parent_recipe(tmp_path: Path) -> None:
    """Attempts are filed even if the recipe row doesn't exist (no FK)."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid(model="orphan")
    out = store.append_attempt(
        canonical_id=cid,
        session_id="sess-1",
        outcome="kept",
    )
    assert out["id"] == 1
    assert store.get_recipe(canonical_id=cid) is None


def test_append_attempt_id_is_monotonic_per_cid(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    ids = [
        store.append_attempt(
            canonical_id=cid,
            session_id="s",
            outcome="kept",
        )["id"]
        for _ in range(3)
    ]
    assert ids == [1, 2, 3]


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
    assert [store.get_recipe(canonical_id=cid, version=i)["version"] for i in range(1, n)] == list(range(1, n))


def test_append_attempt_concurrent_keeps_ids_unique(tmp_path: Path) -> None:
    """No two concurrent appenders may produce the same ``id``."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()

    def append_once(_: int) -> int:
        return store.append_attempt(
            canonical_id=cid,
            session_id="s",
            outcome="kept",
        )["id"]

    n = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        ids = list(ex.map(append_once, range(n)))
    assert sorted(ids) == list(range(1, n + 1))


def test_search_skips_directories_with_corrupt_recipe_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt recipe.json must surface as LocalRecipeStoreError, not silently lie about the corpus."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    bad_path = tmp_path.joinpath(*cid_to_path_components(cid)) / RECIPE_FILENAME
    bad_path.write_text("{this is not json")

    with pytest.raises(LocalRecipeStoreError):
        store.search()


def test_list_attempts_skips_malformed_lines(tmp_path: Path) -> None:
    """A single corrupt line in attempts.ndjson must not lose valid rows after it."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.append_attempt(
        canonical_id=cid,
        session_id="s",
        outcome="kept",
    )
    attempts_path = tmp_path.joinpath(*cid_to_path_components(cid)) / ATTEMPTS_FILENAME
    with attempts_path.open("a", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(
            json.dumps(
                {
                    "id": 99,
                    "recipe_canonical_id": cid,
                    "session_id": "s",
                    "attempt_at": "2026-05-28T10:00:00.000000+00:00",
                    "outcome": "reverted",
                }
            )
            + "\n"
        )
    rows = _list_attempt_rows(store, cid)
    outcomes = [r["outcome"] for r in rows]
    assert "kept" in outcomes
    assert "reverted" in outcomes


def test_walk_skips_non_5level_directories(tmp_path: Path) -> None:
    """A stray directory at the wrong depth must NOT be picked up by search."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    # Inject a recipe.json at depth 3 (wrong depth).
    stray = tmp_path / "model" / "hw" / "fw" / RECIPE_FILENAME
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps({"canonical_id": "stray", "version": 1}))
    rows = store.search()
    cids_found = {r["canonical_id"] for r in rows}
    assert cid in cids_found
    assert "stray" not in cids_found


def test_construction_does_not_create_root(tmp_path: Path) -> None:
    """A run that never touches the KB must not create any directory on disk."""
    root = tmp_path / "kb-never-used"
    LocalRecipeStore(root=root)
    assert not root.exists()


def test_str_root_accepted(tmp_path: Path) -> None:
    """``LocalRecipeStore(root=str(...))`` still works via coercion in __post_init__."""
    store = LocalRecipeStore(root=str(tmp_path))
    cid = _cid(model="m-x")
    store.put_recipe(canonical_id=cid)
    assert store.get_recipe(canonical_id=cid) is not None


def test_recipe_payload_carries_canonical_id_and_version(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    payload = json.loads(
        (tmp_path.joinpath(*cid_to_path_components(cid)) / RECIPE_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["canonical_id"] == cid
    assert payload["version"] == 1


def test_history_dir_lives_at_six_levels_below_root(tmp_path: Path) -> None:
    """``history/`` is the only directory that may sit below the 7-level recipe
    dir; the walker MUST NOT recurse into it."""
    store = LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid)
    store.put_recipe(canonical_id=cid)
    history_dir = tmp_path.joinpath(*cid_to_path_components(cid)) / HISTORY_DIRNAME
    assert history_dir.is_dir()
    rel = history_dir.relative_to(tmp_path).parts
    assert len(rel) == 8  # 7 identity levels + history dir
