# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplementary coverage for LocalRecipeStore normalisation + edge paths."""

from __future__ import annotations

import json

import pytest

from inference_optimizer.recipe_kb import local_store as ls
from inference_optimizer.recipe_kb.canonical_id import canonical_id_from_components


def _cid(model="m", hardware="mi300", framework="sglang",
         framework_version="v1", precision="fp8") -> str:
    return canonical_id_from_components(
        model=model, hardware=hardware, framework=framework,
        framework_version=framework_version, precision=precision,
    )


class _ToDictItem:
    """Item exposing to_dict() to exercise the coerce path."""

    def __init__(self, payload):
        self._p = payload

    def to_dict(self):
        return self._p


# ---- _coerce_dict ----

def test_coerce_dict_variants():
    assert ls._coerce_dict(None) is None
    assert ls._coerce_dict("x") is None
    assert ls._coerce_dict({"a": 1}) == {"a": 1}
    assert ls._coerce_dict(_ToDictItem({"k": 1})) == {"k": 1}
    assert ls._coerce_dict(_ToDictItem("not a dict")) is None


# ---- normalisation helpers ----

def test_normalise_findings_and_failures():
    assert ls._normalise_findings([{"description": "d", "measured_impact": "i"}, "skip"]) == [
        {"description": "d", "measured_impact": "i"},
    ]
    assert ls._normalise_failures([{"description": "d", "reason": "r"}]) == [
        {"description": "d", "reason": "r"},
    ]


def test_normalise_gaps_pitfalls_lessons():
    assert ls._normalise_gaps([{"description": "d", "metrics": "m"}]) == [
        {"description": "d", "metrics": "m"},
    ]
    assert ls._normalise_pitfalls([{"description": "d", "severity": "high"}]) == [
        {"description": "d", "severity": "high"},
    ]
    out = ls._normalise_lessons([{"statement": "s", "measured_impact": {"x": 1}}])
    assert out[0]["measured_impact"] == {"x": 1}


def test_normalise_prs_number_coercion():
    out = ls._normalise_prs([
        {"repo": "r", "number": "12", "outcome": "merged"},
        {"repo": "r2", "number": "bad"},
    ])
    assert out[0]["number"] == 12
    assert out[1]["number"] == 0


def test_normalise_sessions_coercion():
    out = ls._normalise_sessions([
        {"date": "d", "throughput_before": "10", "throughput_after": "bad",
         "gain_pct": "x", "stack_len": "5", "actions_taken": ["a"]},
    ])
    assert out[0]["throughput_before"] == 10.0
    assert out[0]["throughput_after"] == 0.0
    assert out[0]["gain_pct"] == 0.0
    assert out[0]["stack_len"] == 5


# ---- put / get / history with full payload ----

def test_put_get_history_roundtrip(tmp_path):
    store = ls.LocalRecipeStore(root=str(tmp_path))
    cid = _cid()
    r1 = store.put_recipe(
        canonical_id=cid, model="m", best_throughput=100.0,
        what_worked=[{"description": "w", "measured_impact": "i"}],
        prs_tested=[_ToDictItem({"repo": "r", "number": 1})],
        sessions=[{"date": "d", "throughput_before": 1.0}],
        extras={"task": "pretrain"},
        provenance={"who": "test"},
    )
    assert r1["version"] == 1
    assert r1["created"] is True
    # second put archives v1 and writes v2
    r2 = store.put_recipe(canonical_id=cid, model="m", best_throughput=200.0)
    assert r2["version"] == 2
    assert r2["created"] is False

    live = store.get_recipe(canonical_id=cid)
    assert live["version"] == 2

    # explicit live version
    assert store.get_recipe(canonical_id=cid, version=2)["version"] == 2
    # archived version snapshot
    archived = store.get_recipe(canonical_id=cid, version=1)
    assert archived["version"] == 1
    # unknown version
    assert store.get_recipe(canonical_id=cid, version=99) is None

    history = store.get_history(canonical_id=cid)
    assert len(history) == 1
    assert history[0]["version"] == 1


def test_get_recipe_missing(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    assert store.get_recipe(canonical_id=_cid()) is None
    assert store.get_history(canonical_id=_cid()) == []


def test_empty_cid_raises(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.put_recipe(canonical_id="")
    with pytest.raises(ValueError):
        store.get_recipe(canonical_id="")
    with pytest.raises(ValueError):
        store.get_history(canonical_id="")
    with pytest.raises(ValueError):
        store.delete_recipe(canonical_id="")
    with pytest.raises(ValueError):
        store.list_attempts(canonical_id="")
    with pytest.raises(ValueError):
        store.list_session_attempts(session_id="")


# ---- delete / purge ----

def test_delete_recipe(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    cid = _cid()
    assert store.delete_recipe(canonical_id=cid) is False  # nothing yet
    store.put_recipe(canonical_id=cid, model="m")
    assert store.delete_recipe(canonical_id=cid) is True
    assert store.get_recipe(canonical_id=cid) is None


def test_purge_recipe(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    cid = _cid()
    store.put_recipe(canonical_id=cid, model="m")
    store.purge_recipe(canonical_id=cid)
    assert store.get_recipe(canonical_id=cid) is None
    # purge of absent cid is a no-op
    store.purge_recipe(canonical_id=cid)
    with pytest.raises(ValueError):
        store.purge_recipe(canonical_id="")


# ---- attempts ----

def test_attempts_roundtrip(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    cid = _cid()
    a1 = store.append_attempt(canonical_id=cid, session_id="s1", fitness=1.0,
                              outcome="ok", diff={"x": 1})
    assert a1["id"] == 1
    store.append_attempt(canonical_id=cid, session_id="s2", fitness=None)
    listed = store.list_attempts(canonical_id=cid)
    assert listed[0]["id"] == 2  # newest first
    sess = store.list_session_attempts(session_id="s1")
    assert len(sess) == 1
    assert sess[0]["session_id"] == "s1"


def test_append_attempt_requires_session(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.append_attempt(canonical_id=_cid(), session_id="")


# ---- search ----

def test_search_filters(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    store.put_recipe(canonical_id=_cid(model="a"), model="a", best_throughput=100.0)
    store.put_recipe(canonical_id=_cid(model="b"), model="b", best_throughput=300.0)

    # label match
    res = store.search(label_match={"model": "a"})
    assert len(res) == 1
    # metric filter shorthand alias 'throughput'
    res = store.search(metric_filters={"throughput": {"min": 200.0}})
    assert len(res) == 1
    assert res[0]["model"] == "b"
    # metric scalar shorthand (equality-ish min/max)
    res = store.search(metric_filters={"best_throughput": {"max": 150.0}})
    assert len(res) == 1
    # updated_since far future -> none
    assert store.search(updated_since="9999-01-01T00:00:00") == []
    # list_recent
    assert len(store.list_recent(limit=10)) == 2


def test_search_bad_order_by(tmp_path):
    store = ls.LocalRecipeStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.search(order_by="bogus")


# ---- low-level helpers ----

def test_matches_metrics_missing_key():
    assert ls._matches_metrics({}, {"best_throughput": {"min": 1}}) is False
    assert ls._matches_metrics({"best_throughput": "x"}, {"best_throughput": {"min": 1}}) is False


def test_coerce_sort_value():
    assert ls._coerce_sort_value("3", "version") == 3
    assert ls._coerce_sort_value("bad", "version") == 0
    assert ls._coerce_sort_value(None, "updated_at") == ""


def test_read_json_bad(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(ls.LocalRecipeStoreError):
        ls._read_json(p)


def test_list_jsonl_skips_bad(tmp_path):
    p = tmp_path / "a.ndjson"
    p.write_text('{"a":1}\n\nnot json\n{"b":2}\n', encoding="utf-8")
    rows = ls._list_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_atomic_write_json_cleanup_on_error(tmp_path):
    # Non-serialisable payload makes json.dump raise -> tmp cleanup branch.
    with pytest.raises(TypeError):
        ls._atomic_write_json(tmp_path / "x.json", {"bad": object()})
    # tmp file should have been cleaned up
    assert list(tmp_path.glob("*.tmp")) == []


def test_coerce_dict_dataclass():
    from dataclasses import dataclass

    @dataclass
    class _D:
        a: int = 1

    assert ls._coerce_dict(_D()) == {"a": 1}


def test_normalisers_skip_uncoercible():
    assert ls._normalise_failures(["x", None]) == []
    assert ls._normalise_gaps(["x"]) == []
    assert ls._normalise_prs([None]) == []
    assert ls._normalise_pitfalls(["x"]) == []
    assert ls._normalise_lessons([None]) == []
    assert ls._normalise_sessions(["x"]) == []


def test_matches_metrics_scalar_shorthand_and_bad_bound():
    # scalar shorthand -> lo == hi == bounds (equality)
    assert ls._matches_metrics({"best_throughput": 100.0}, {"throughput": 100.0}) is True
    assert ls._matches_metrics({"best_throughput": 100.0}, {"throughput": 50.0}) is False
    # bad bound type -> False
    assert ls._matches_metrics(
        {"best_throughput": 100.0}, {"throughput": {"min": "bad"}},
    ) is False
    assert ls._matches_metrics(
        {"best_throughput": 100.0}, {"throughput": {"max": "bad"}},
    ) is False
