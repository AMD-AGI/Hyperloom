"""Roofline-v2 C4a: action registration + stub executor contract tests.

Pins three guarantees C4b/C4c/C5 depend on:

* ``MODEL_CLASS_ACTION_PRIORS`` has ``roofline`` in every model_class
  at the design-document-mandated prior (D1: 7.5). Without this the
  main LLM never sees the action in the rendered ``action_scores``
  table and would never propose it.
* The action_executors package exports the stub factory at the same
  surface C4b will export the real executor at
  (``make_roofline_stub_executor`` → swap to ``make_roofline_executor``
  in C4b), so cli wiring needs zero changes between C4a and C4b.
* The stub executor returns the **exact** schema
  ``SharedState.record_roofline_analysis`` accepts (asserted by
  round-tripping through the recorder), with ``degraded=True`` and
  ``primary_bottleneck="unknown"`` so the C5 prompt renderer surfaces
  "analysis unavailable" rather than blocking the loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import (
    RooflineStubExecutor,
    build_roofline_fallback_result,
    make_roofline_stub_executor,
)
from inference_optimizer.orchestrator.scoring import MODEL_CLASS_ACTION_PRIORS
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Scoring prior (D1: roofline at 7.5 in every model_class)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_class",
    ["dense", "moe_mla", "moe_swa", "moe_mla_nsa"],
)
def test_roofline_prior_present_in_every_model_class(model_class):
    """Without ``roofline`` in this table the main LLM never sees it
    in the rendered ``action_scores`` and would never propose the
    action — breaking the entire C4/C5/C7 chain. D1 anchored at 7.5."""
    assert "roofline" in MODEL_CLASS_ACTION_PRIORS[model_class]
    assert MODEL_CLASS_ACTION_PRIORS[model_class]["roofline"] == 7.5


def test_roofline_prior_sits_between_explore_and_kernel():
    """Sanity check the prior magnitude: roofline must rank below
    explicit-search actions (params/backends ≥8.4) so it doesn't
    crowd out actual optimisation, but above ``operator_tuning``
    (≤7.0) so the LLM willingly proposes it once available."""
    for mc, priors in MODEL_CLASS_ACTION_PRIORS.items():
        rp = priors["roofline"]
        assert rp <= max(priors.get("params", 0), priors.get("backends", 0)), (
            f"roofline prior {rp} too high for {mc}: must rank below "
            "params/backends so optimisation actions still win"
        )
        # operator_tuning is the conservative explore floor at ~7.0
        assert rp >= priors.get("operator_tuning", 0), (
            f"roofline prior {rp} too low for {mc}: must rank ≥ "
            "operator_tuning so the LLM proposes it willingly"
        )


# ---------------------------------------------------------------------------
# Stub executor: contract = build_roofline_fallback_result schema
# ---------------------------------------------------------------------------
def _make_ctx(task_kind: str = "roofline") -> RunnerContext:
    """Minimal RunnerContext for stub execution (no lease, no extras)."""
    task = Task(
        task_id="t-stub-1", kind=task_kind, state="running",
        params={}, idempotency_key=f"{task_kind}:t-stub-1",
    )
    return RunnerContext(task=task, lease=None, extra={})


@pytest.mark.asyncio
async def test_stub_returns_well_formed_fallback_without_shared_state():
    """Stub built with no SharedState reference still produces a
    schema-valid degraded result."""
    stub = make_roofline_stub_executor(shared_state=None)
    result = await stub(_make_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert result["primary_bottleneck"] == "unknown"
    assert result["bottleneck_distribution"] == {}
    assert result["suggested_prunes"] == []
    assert result["suggested_next_actions"] == []
    assert result["reprofile_recommended"] is False
    assert "stub" in result["error"]
    assert result["snapshot_id"] == 0
    assert result["analyzed_at_gain_pct"] == 0.0


@pytest.mark.asyncio
async def test_stub_surfaces_snapshot_id_and_path_from_shared_state():
    """When wired with a real SharedState, the stub mirrors the cached
    snapshot_id / analysis_md_path so C4c integration tests can already
    assert "roofline result for snapshot #N" before C4b lands."""
    state = SharedState()
    state.last_select_kernels = {
        "roofline_snapshot_id": 7,
        "analysis_md_path": "/sessions/abc/select_kernels/analysis.md",
    }
    state.cumulative_gain_validated = 3.2

    stub = make_roofline_stub_executor(shared_state=state)
    result = await stub(_make_ctx())

    assert result["snapshot_id"] == 7
    assert result["based_on_analysis_md"] == "/sessions/abc/select_kernels/analysis.md"
    assert result["analyzed_at_gain_pct"] == 3.2
    assert result["degraded"] is True


@pytest.mark.asyncio
async def test_stub_handles_missing_or_malformed_cache_safely():
    """Empty / partial ``last_select_kernels`` must not crash — the
    optimisation loop must keep running."""
    state = SharedState()
    state.last_select_kernels = {}
    stub = make_roofline_stub_executor(shared_state=state)
    result = await stub(_make_ctx())
    assert result["snapshot_id"] == 0
    assert result["based_on_analysis_md"] == ""

    state.last_select_kernels = {"roofline_snapshot_id": "not-an-int"}
    state.cumulative_gain_validated = "garbage"  # type: ignore[assignment]
    result = await stub(_make_ctx())
    assert result["snapshot_id"] == 0
    assert result["analyzed_at_gain_pct"] == 0.0


@pytest.mark.asyncio
async def test_stub_result_round_trips_through_record_roofline_analysis():
    """The C2 recorder must accept the stub output without dropping
    any documented field — this is the central C4a/C2 contract."""
    state = SharedState()
    state.last_select_kernels = {
        "roofline_snapshot_id": 1,
        "analysis_md_path": "/p/analysis.md",
    }
    state.cumulative_gain_validated = 0.0

    stub = make_roofline_stub_executor(shared_state=state)
    result = await stub(_make_ctx())

    state.record_roofline_analysis(result)
    cached = state.last_roofline_analysis
    assert cached["snapshot_id"] == 1
    assert cached["based_on_analysis_md"] == "/p/analysis.md"
    assert cached["primary_bottleneck"] == "unknown"
    assert cached["bottleneck_distribution"] == {}
    assert cached["suggested_prunes"] == []
    assert cached["suggested_next_actions"] == []
    assert cached["reprofile_recommended"] is False


# ---------------------------------------------------------------------------
# build_roofline_fallback_result — shared schema producer for C4b too
# ---------------------------------------------------------------------------
def test_fallback_builder_always_emits_required_schema_keys():
    """C4b's failure branches reuse this builder. The set of keys
    here is the canonical schema; if a key is renamed here, every
    consumer in C5 / C6 / record_roofline_analysis breaks loudly."""
    result = build_roofline_fallback_result(snapshot_id=3, error="x")
    required = {
        "status", "degraded", "snapshot_id", "analyzed_at_iso",
        "analyzed_at_gain_pct", "based_on_analysis_md",
        "primary_bottleneck", "bottleneck_distribution",
        "suggested_prunes", "suggested_next_actions",
        "reprofile_recommended", "reprofile_reason",
        "raw_llm_response", "error",
    }
    assert required.issubset(set(result.keys()))
    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert result["snapshot_id"] == 3
    assert result["error"] == "x"


def test_fallback_builder_error_field_optional():
    """Calling without ``error`` is valid (C4b uses it for explicit
    failure annotation; C4a stub always supplies one)."""
    result = build_roofline_fallback_result(snapshot_id=0)
    assert result["error"] == ""
    assert result["degraded"] is True


# ---------------------------------------------------------------------------
# Executor class instantiation surface (C4b will swap class but keep
# the same constructor signature)
# ---------------------------------------------------------------------------
def test_stub_executor_constructor_accepts_shared_state_kwarg():
    """C4b's RooflineExecutor will subclass / replace this with a
    constructor that ALSO accepts ``backend_factory=``. C4a's stub
    only needs ``shared_state=`` — the kwarg is named here so cli
    wiring stays stable across C4a → C4b."""
    stub = RooflineStubExecutor(shared_state=None)
    assert stub.shared_state is None

    sentinel = object()
    stub2 = RooflineStubExecutor(shared_state=sentinel)
    assert stub2.shared_state is sentinel
