"""Tests for the unified KB-interface convergence (KB_query_design.MD).

Covers the cross-cutting behaviour that the cortex-KB and gbrain
adapters now share:

* the dispatcher runs a SINGLE ``_v2_to_arbor`` translation on any
  remote row (no per-backend ``returns_arbor_shape`` branch);
* ``prefer`` reranks candidate rows by workload similarity WITHOUT
  changing the ``required`` (5-tuple / metric) membership filter;
* the T0 anchor builds a model-facing ``WarmStartContext`` with an
  explicit ``hit`` / ``seed_only`` / ``miss`` status;
* warm-replay consumes ``recommended_replay`` (``extra_server_args`` +
  nested ``extra_envs``) from the WarmStartContext.

No network: remotes are fed in-memory nested rows / a fake MCP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.cortex_t0 import (
    _build_warm_prefer,
    _build_warm_start_context,
    _recipe_is_actionable,
    run_t0_anchor,
)
from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)
from inference_optimizer.recipe_kb.dispatcher import (
    _prefer_score,
    _rerank_by_prefer,
)


# ===========================================================================
# Fakes
# ===========================================================================
class _NestedRemote:
    """A remote that returns the unified nested KB-interface envelope.

    Stands in for BOTH cortex and gbrain: the dispatcher must run the
    same ``_v2_to_arbor`` translation regardless of which one served the
    row. ``search`` filters by ``label_match`` against ``labels``.
    """

    enabled = True

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.last_prefer: dict[str, Any] | None = None

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_prefer = prefer
        lm = label_match or {}
        matched = [
            r
            for r in self.rows
            if all(str((r.get("labels") or {}).get(k, "")) == str(v) for k, v in lm.items())
        ]
        return matched[: limit if limit and limit > 0 else None]

    def close(self) -> None:
        pass


def _nested_row(
    *,
    cid: str,
    model: str,
    hardware: str,
    framework: str = "sglang",
    framework_version: str = "0.5.11",
    precision: str = "fp8",
    args: str = "--x 1",
    envs: dict[str, str] | None = None,
    throughput: float = 1000.0,
    tp: int | None = None,
    findings: list[Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "best_config": {"extra_server_args": args, "extra_envs": dict(envs or {})},
        "best_throughput": throughput,
    }
    # tp rides as a top-level extra so it survives _v2_to_arbor and the
    # dispatcher rerank can read it.
    row: dict[str, Any] = {
        "canonical_id": cid,
        "version": 1,
        "labels": {
            "model": model,
            "hardware": hardware,
            "framework": framework,
            "framework_version": framework_version,
            "precision": precision,
        },
        "body": body,
        "metrics": {"throughput": throughput},
        "findings": findings or [],
        "failures": [],
        "gaps": [],
        "lessons": [],
        "pitfalls": [],
    }
    if tp is not None:
        row["tp"] = tp
    return row


# ===========================================================================
# Single-translation: dispatcher always runs _v2_to_arbor on remote rows
# ===========================================================================
def test_dispatcher_translates_nested_remote_row(tmp_path: Path) -> None:
    cid = recipe_canonical_id(
        model="qwen3-32b", hardware="mi300x", framework="sglang",
        framework_version="0.5.11", precision="fp8",
    )
    remote = _NestedRemote([
        _nested_row(cid=cid, model="qwen3-32b", hardware="mi300x",
                    args="--cuda-graph-max-bs 256", envs={"FOO": "1"},
                    throughput=5430.9),
    ])
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=remote)

    row = kb.get_recipe(canonical_id=cid)
    assert row is not None
    # _v2_to_arbor flattened the nested envelope:
    assert row["model"] == "qwen3-32b"
    assert row["best_config"]["extra_server_args"] == "--cuda-graph-max-bs 256"
    assert row["best_config"]["extra_envs"] == {"FOO": "1"}
    assert float(row["best_throughput"]) == pytest.approx(5430.9)


# ===========================================================================
# prefer rerank — reorders, never drops; required filter unchanged
# ===========================================================================
def test_prefer_score_counts_matches() -> None:
    row = {"framework_version": "0.5.11", "tp": 8, "conc": 32}
    assert _prefer_score(row, {"tp": 8}) == 1
    assert _prefer_score(row, {"tp": 8, "framework_version": "0.5.11"}) == 2
    # absent / mismatched fields contribute 0
    assert _prefer_score(row, {"tp": 4}) == 0
    assert _prefer_score(row, {"ep": 2}) == 0


def test_rerank_by_prefer_is_stable_and_lossless() -> None:
    rows = [
        {"canonical_id": "a", "tp": 4},
        {"canonical_id": "b", "tp": 8},
        {"canonical_id": "c", "tp": 8},
    ]
    ranked = _rerank_by_prefer(rows, {"tp": 8})
    # tp=8 rows float to the front; original relative order preserved.
    assert [r["canonical_id"] for r in ranked] == ["b", "c", "a"]
    # No rows dropped.
    assert len(ranked) == 3
    # Empty prefer is a no-op.
    assert _rerank_by_prefer(rows, None) == rows


def test_search_prefer_reorders_without_dropping(tmp_path: Path) -> None:
    # Two recipes share the same model label (the required filter); they
    # differ on tp. prefer={tp:8} must surface the tp=8 row first but keep
    # both.
    cid_a = recipe_canonical_id(
        model="qwen3-32b", hardware="mi300x", framework="sglang",
        framework_version="0.5.11", precision="fp8",
    )
    cid_b = recipe_canonical_id(
        model="qwen3-32b", hardware="mi355x", framework="sglang",
        framework_version="0.5.11", precision="fp8",
    )
    remote = _NestedRemote([
        _nested_row(cid=cid_a, model="qwen3-32b", hardware="mi300x", tp=4),
        _nested_row(cid=cid_b, model="qwen3-32b", hardware="mi355x", tp=8),
    ])
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=remote)

    rows = kb.search(label_match={"model": "qwen3-32b"}, prefer={"tp": 8})
    assert len(rows) == 2  # nothing dropped
    assert rows[0]["canonical_id"] == cid_b  # tp=8 reranked first
    # The dispatcher forwarded prefer to the adapter for parity.
    assert remote.last_prefer == {"tp": 8}


# ===========================================================================
# seed-only detection + WarmStartContext
# ===========================================================================
def test_recipe_is_actionable() -> None:
    assert _recipe_is_actionable(
        {"best_config": {"extra_server_args": "--x 1"}}
    )
    assert _recipe_is_actionable(
        {"best_config": {"extra_envs": {"A": "1"}}}
    )
    assert _recipe_is_actionable({"best_throughput": 1000.0})
    assert _recipe_is_actionable({"what_failed": [{"x": 1}]})
    # bare draft anchor — identity only, no champion / lists
    assert not _recipe_is_actionable(
        {"canonical_id": "inference:m:h:f:v:p", "best_config": {}}
    )
    assert not _recipe_is_actionable({})


def test_build_warm_start_context_hit() -> None:
    recipe = {
        "best_config": {
            "extra_server_args": "--cuda-graph-max-bs 256",
            "extra_envs": {"FOO": "1"},
        },
        "best_throughput": 5430.9,
        "validated_gain_pct": 7.8,
        "what_worked": [{"id": "w1"}],
        "what_failed": [{"id": "f1"}],
        "lessons": [{"attrs": {"statement": "x"}}],
        "pitfalls": [{"attrs": {"description": "y"}}],
    }
    ctx = _build_warm_start_context(
        status="hit", tier="exact", confidence=1.0,
        canonical_id="inference:m:h:f:v:p", source="gbrain", recipe=recipe,
    )
    assert ctx["status"] == "hit"
    assert ctx["match"]["source"] == "gbrain"
    assert ctx["recommended_replay"]["extra_server_args"] == "--cuda-graph-max-bs 256"
    assert ctx["recommended_replay"]["extra_envs"] == {"FOO": "1"}
    assert ctx["recommended_replay"]["expected_gain_pct"] == 7.8
    assert ctx["proven_prior"] == [{"id": "w1"}]
    assert ctx["do_not_repeat"] == [{"id": "f1"}]
    assert ctx["lessons"] == [{"attrs": {"statement": "x"}}]
    assert ctx["pitfalls"] == [{"attrs": {"description": "y"}}]


@pytest.mark.parametrize("status", ["seed_only", "miss", "error"])
def test_build_warm_start_context_non_hit_has_empty_replay(status: str) -> None:
    ctx = _build_warm_start_context(
        status=status, tier=status, confidence=0.0,
        canonical_id="inference:m:h:f:v:p", source="cortex-kb", recipe=None,
    )
    assert ctx["status"] == status
    assert ctx["recommended_replay"] == {}
    assert ctx["proven_prior"] == []
    assert ctx["do_not_repeat"] == []


# ===========================================================================
# T0 status matrix end-to-end (local-only dispatcher)
# ===========================================================================
@dataclass
class _FakeState:
    cortex_session_id: str = ""
    warm_start_ts: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    warm_start_context: dict[str, Any] = field(default_factory=dict)
    framework: str = "sglang"
    framework_version: str = "0.5.11"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    model_class: str = ""
    baseline_workload_extra: dict[str, Any] = field(default_factory=dict)

    def save(self, _path: Path) -> None:
        pass


def test_t0_status_miss_on_empty_corpus(tmp_path: Path) -> None:
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    sd = tmp_path / "session"
    sd.mkdir()
    state = _FakeState()
    run_t0_anchor(
        kb, state, workload="brand-new", hw="mi300x",
        extra_attrs={"framework": "sglang"}, session_dir=sd,
    )
    # The T0 put_recipe wrote a bare draft anchor, so the read-back is a
    # seed_only (not a miss) — there IS a row, it just isn't actionable.
    assert state.warm_start_context.get("status") == "seed_only"


def test_t0_status_hit_when_actionable_row_present(tmp_path: Path) -> None:
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)
    sd = tmp_path / "session"
    sd.mkdir()
    state = _FakeState()
    cid = recipe_canonical_id(
        model="M", hardware="mi300x", framework="sglang",
        framework_version="0.5.11", precision="fp8",
    )
    kb.put_recipe(
        canonical_id=cid, model="M", hardware="mi300x",
        framework="sglang", framework_version="0.5.11", precision="fp8",
        best_config={"extra_server_args": "--x 1", "extra_envs": {"A": "1"}},
        best_throughput=2000.0,
    )
    run_t0_anchor(
        kb, state, workload="M", hw="mi300x",
        extra_attrs={"framework": "sglang"}, session_dir=sd,
    )
    ctx = state.warm_start_context
    assert ctx.get("status") == "hit"
    assert ctx["recommended_replay"]["extra_server_args"] == "--x 1"
    assert ctx["recommended_replay"]["extra_envs"] == {"A": "1"}


def test_build_warm_prefer_skips_empties() -> None:
    state = _FakeState(tp=8, conc=0, baseline_workload_extra={"quant_scheme": "fp8"})
    prefer = _build_warm_prefer(state, "0.5.11")
    assert prefer["tp"] == 8
    assert prefer["framework_version"] == "0.5.11"
    assert prefer["quant_scheme"] == "fp8"
    # zero / empty values are skipped
    assert "conc" not in prefer
    assert "ep" not in prefer
