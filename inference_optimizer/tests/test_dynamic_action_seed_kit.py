"""dynamic_action.MD P2 — seed kit assembler + dispatch artefact tests.

action_dynamic_plan/P2_session_artifact_seed_kit.md §9 lists 7
acceptance scenarios; each is mapped to a test below (TIDs match
``test_p2_scenario_N``).

The assembler is exercised against a thin SharedState double so the
deterministic selection rules can be tightly pinned without spinning
up a real Coordinator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.dynamic_action_seed_kit import (
    MAX_KB_PITFALLS,
    MAX_KEPT_PATCHES,
    MAX_PROFILE_KEYSLICES,
    MAX_REVERTED_PATCHES,
    MAX_SEED_KIT_TOKENS,
    SEED_KIT_FIELDS,
    SeedKitAssemblyError,
    assemble_seed_kit,
    estimate_tokens,
)
from inference_optimizer.orchestrator.policy import PolicyDenied
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_seed_kit_path,
    dynamic_action_spec_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
@dataclass
class _StateDouble:
    """Minimal SharedState double for the assembler.

    Only the four fields the assembler reads are exposed; everything
    else stays default-empty so the test surface is tight.
    """

    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    explore_search: dict[str, Any] = field(default_factory=dict)
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)


def _payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "motivation_gap_text": (
            "Combine kv cache layout + scheduler tweak; serving_specialist "
            "alone cannot reason across both."
        ),
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["framework_source"],
        "budget_hint": "medium",
    }
    base.update(overrides)
    return base


def _coordinator_double(tmp_path: Path, state: SharedState) -> Coordinator:
    """Coordinator stub wired with just enough surface for the
    dispatch hook to run end-to-end. Avoids the full constructor (db,
    bus, backends) since the prepare hook is a pure helper."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    return c


# ===========================================================================
# Seed kit invariants
# ===========================================================================
def test_seed_kit_fields_are_closed_set():
    """P2 §5.2.d — adding a top-level field outside the canonical set
    must be a design change, not a code change. The frozenset pins the
    surface so reviewers see drift immediately."""
    assert SEED_KIT_FIELDS == frozenset({
        "motivation_gap_text",
        "roofline_summary",
        "profile_keyslices",
        "kept_patches",
        "reverted_patches",
        "kb_pitfalls",
        "source_root_hints",
    })


def test_assemble_seed_kit_emits_exact_field_set():
    state = _StateDouble()
    result = assemble_seed_kit(state, _payload())
    assert set(result.payload.keys()) == SEED_KIT_FIELDS


def test_assemble_seed_kit_degraded_when_state_empty():
    """Empty state → all best-effort sources empty → degraded=True."""
    state = _StateDouble()
    result = assemble_seed_kit(state, _payload())
    assert result.degraded is True
    assert result.payload["roofline_summary"] == ""
    assert result.payload["profile_keyslices"] == []
    assert result.payload["kept_patches"] == []
    assert result.payload["kb_pitfalls"] == []


def test_assemble_seed_kit_non_degraded_when_all_sources_populated():
    state = _StateDouble(
        last_trace_analyze={
            "analysis_md_text": "roofline summary text",
            "hot_kernels_top15": [
                {"name": "rms_norm", "gpu_pct": 5.2, "bottleneck": "memory"},
                {"name": "rope", "gpu_pct": 3.1, "bottleneck": "compute"},
            ],
        },
        explore_search={
            "accepted": [
                {
                    "name": "fa_v2",
                    "action": "explore",
                    "gain_pct": 1.4,
                    "rationale": "fused attention",
                },
            ],
            "rejected": [
                {"name": "lossy_bs", "reason": "stack_unstable", "gain_pct": -0.5},
            ],
        },
        warm_start_pitfalls=[
            {
                "raw": "serving_specialist watch out for cuda graph capture",
            },
        ],
    )
    result = assemble_seed_kit(state, _payload())
    assert result.degraded is False
    assert result.payload["roofline_summary"].startswith("roofline summary")
    assert result.payload["profile_keyslices"]
    assert result.payload["kept_patches"][0]["name"] == "fa_v2"
    assert result.payload["reverted_patches"][0]["name"] == "lossy_bs"
    assert result.payload["kb_pitfalls"][0]["text"].startswith("serving")


def test_assemble_seed_kit_rejects_empty_motivation():
    """PolicyGate enforces non-empty motivation upstream; defense in
    depth still applies — the assembler raises if it is reached
    without a motivation."""
    state = _StateDouble()
    with pytest.raises(SeedKitAssemblyError):
        assemble_seed_kit(state, _payload(motivation_gap_text=""))


def test_assemble_seed_kit_token_cap_enforced(monkeypatch):
    """Make every per-item char cap a no-op, then push a multi-MB
    pitfall blob through to demonstrate the global token cap path
    rejects the assembly with :class:`SeedKitAssemblyError`."""
    # Pitfall content must contain the scope_domain keyword to survive
    # the relevance filter — prefix the blob with the canonical
    # serving_specialist tag so the substring matcher latches.
    huge_filler = "x" * (MAX_SEED_KIT_TOKENS * 5)
    big_blob = "serving_specialist " + huge_filler
    state = _StateDouble(
        warm_start_pitfalls=[{"raw": big_blob}] * MAX_KB_PITFALLS,
    )
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.dynamic_action_seed_kit."
        "_truncate",
        lambda text, max_chars: text,
    )
    payload = _payload()
    with pytest.raises(SeedKitAssemblyError):
        assemble_seed_kit(state, payload)


def test_kept_patches_capped_at_MAX_KEPT_PATCHES():
    accepted = [
        {"name": f"v{i}", "action": "explore", "gain_pct": 0.1}
        for i in range(MAX_KEPT_PATCHES + 5)
    ]
    state = _StateDouble(explore_search={"accepted": accepted})
    result = assemble_seed_kit(state, _payload())
    assert len(result.payload["kept_patches"]) == MAX_KEPT_PATCHES


def test_reverted_patches_capped_at_MAX_REVERTED_PATCHES():
    rejected = [
        {"name": f"r{i}", "reason": "regression", "gain_pct": -0.1}
        for i in range(MAX_REVERTED_PATCHES + 5)
    ]
    state = _StateDouble(explore_search={"rejected": rejected})
    result = assemble_seed_kit(state, _payload())
    assert len(result.payload["reverted_patches"]) == MAX_REVERTED_PATCHES


def test_profile_keyslices_capped_and_sorted_by_gpu_pct():
    state = _StateDouble(
        last_trace_analyze={
            "hot_kernels_top15": [
                {"name": f"op{i}", "gpu_pct": float(i)}
                for i in range(MAX_PROFILE_KEYSLICES + 3)
            ],
        },
    )
    result = assemble_seed_kit(state, _payload())
    rows = result.payload["profile_keyslices"]
    assert len(rows) == MAX_PROFILE_KEYSLICES
    gpu_pcts = [r["gpu_pct"] for r in rows]
    assert gpu_pcts == sorted(gpu_pcts, reverse=True)


def test_kept_patches_filter_drops_kernel_only_entries():
    """P2 §5.3 — kernel-owned action entries never leak into the kept
    patches summary (defense in depth even though PolicyGate rejects
    kernel-only scope_domains)."""
    state = _StateDouble(
        explore_search={
            "accepted": [
                {"name": "kfa", "action": "kernel_opt", "gain_pct": 2.0},
                {"name": "fwd", "action": "explore", "gain_pct": 1.0},
            ],
        },
    )
    result = assemble_seed_kit(state, _payload())
    names = {r["name"] for r in result.payload["kept_patches"]}
    assert "fwd" in names
    assert "kfa" not in names


def test_kb_pitfalls_filtered_by_scope_domain_substring():
    state = _StateDouble(
        warm_start_pitfalls=[
            {"raw": "serving_specialist: avoid cuda graph capture in long ctx"},
            {"raw": "compiler_specialist: torch.compile leaks on fp8"},
        ],
    )
    payload = _payload(
        scope_domains=["serving_specialist", "kernel_switch_specialist"],
    )
    result = assemble_seed_kit(state, payload)
    rows = result.payload["kb_pitfalls"]
    assert len(rows) == 1
    assert "serving_specialist" in rows[0]["text"]


# ===========================================================================
# Token estimator sanity
# ===========================================================================
@pytest.mark.parametrize("text,expected_min", [
    ("", 0),
    ("hello", 1),
    ("x" * 4_000, 1_000),
])
def test_estimate_tokens_monotonic(text: str, expected_min: int):
    assert estimate_tokens(text) >= expected_min


# ===========================================================================
# P2 §9 — dispatch-time integration scenarios
# ===========================================================================
def _state_with_round(round_idx: int = 0) -> SharedState:
    state = SharedState(session_id="test")
    state.explore_search = {"cursor": round_idx}
    return state


def test_p2_scenario_01_valid_dispatch_writes_spec_and_seed_kit(tmp_path: Path):
    """Happy path — both artefacts land + task params carry the
    artefact paths + round cap bumps + status is DISPATCHED."""
    state = _state_with_round(2)
    coord = _coordinator_double(tmp_path, state)
    params = dict(_payload()).copy()
    params.pop("motivation_gap_text", None)
    params["motivation_gap_text"] = (
        "real motivation across serving + kernel switch"
    )
    meta = coord._prepare_dynamic_action_dispatch(params)
    dyn_id = meta["dyn_id"]
    assert dyn_id == "dyn-2-1"
    spec_path = dynamic_action_spec_path(tmp_path, dyn_id)
    seed_path = dynamic_action_seed_kit_path(tmp_path, dyn_id)
    assert spec_path.is_file()
    assert seed_path.is_file()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert set(spec.keys()) >= {
        "dyn_id", "dispatched_at", "round_index", "payload",
        "policy_gate_decision", "resource_lane", "degraded_dispatch",
        "seed_kit_tokens",
    }
    assert spec["payload"]["scope_domains"] == params["scope_domains"]
    assert params["artifact_path"].endswith(dyn_id)
    assert params["spec_path"].endswith("spec.json")
    assert params["seed_kit_path"].endswith("seed_kit.json")
    # Counter is bumped only by the *finalize* hook (which we run inline
    # to simulate a successful task enqueue).
    coord._finalize_dynamic_action_dispatch(task_id="t-1", meta=meta)
    assert state.dynamic_action_round_count == 1
    assert dyn_id in state.dynamic_actions


def test_p2_scenario_02_assembly_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Force the assembler to fail — dispatch must raise PolicyDenied,
    the artefact dir must be empty/cleaned, and the round counter
    must NOT increment."""
    state = _state_with_round(3)
    coord = _coordinator_double(tmp_path, state)

    from inference_optimizer.orchestrator import dynamic_action_seed_kit as ds

    def _boom(*_args, **_kwargs):
        raise ds.SeedKitAssemblyError("forced for test")

    monkeypatch.setattr(ds, "assemble_seed_kit", _boom)
    with pytest.raises(PolicyDenied) as excinfo:
        coord._prepare_dynamic_action_dispatch(dict(_payload()))
    assert excinfo.value.rule == "dynamic_seed_kit_assembly_failed"
    artefact_dir = dynamic_action_artifact_dir(tmp_path, "dyn-3-1")
    # Either the dir was never created or it was cleaned up.
    assert not artefact_dir.exists() or not any(artefact_dir.iterdir())
    assert state.dynamic_action_round_count == 0
    assert "dyn-3-1" not in state.dynamic_actions


def test_p2_scenario_03_degraded_dispatch_marker(tmp_path: Path):
    """No roofline / profile / kept / pitfalls → spec.json marks
    degraded_dispatch=True; dispatch still proceeds."""
    state = _state_with_round(1)
    coord = _coordinator_double(tmp_path, state)
    meta = coord._prepare_dynamic_action_dispatch(dict(_payload()))
    spec = json.loads(
        dynamic_action_spec_path(tmp_path, meta["dyn_id"]).read_text(
            encoding="utf-8",
        ),
    )
    assert spec["degraded_dispatch"] is True
    assert spec["seed_kit_tokens"] > 0
    seed = json.loads(
        dynamic_action_seed_kit_path(tmp_path, meta["dyn_id"]).read_text(
            encoding="utf-8",
        ),
    )
    assert seed["roofline_summary"] == ""
    assert seed["profile_keyslices"] == []


def test_p2_scenario_04_missing_profile_and_kb_marker(tmp_path: Path):
    """A state that has roofline + kept patches but no profile + no
    KB pitfalls still produces a valid (degraded) seed kit."""
    state = _state_with_round(1)
    state.last_trace_analyze = {"analysis_md_text": "rooftext"}
    state.explore_search = {
        "cursor": 1,
        "accepted": [{"name": "v", "action": "explore", "gain_pct": 1.0}],
    }
    coord = _coordinator_double(tmp_path, state)
    meta = coord._prepare_dynamic_action_dispatch(dict(_payload()))
    spec = json.loads(
        dynamic_action_spec_path(tmp_path, meta["dyn_id"]).read_text(
            encoding="utf-8",
        ),
    )
    assert spec["degraded_dispatch"] is True


def test_p2_scenario_05_pitfall_zero_hits_not_error(tmp_path: Path):
    """KB-pitfall keyword filter returns zero hits → kb_pitfalls=[];
    spec.json still written; dispatch passes."""
    state = _state_with_round(0)
    state.warm_start_pitfalls = [
        {"raw": "completely unrelated content about gemini"},
    ]
    coord = _coordinator_double(tmp_path, state)
    meta = coord._prepare_dynamic_action_dispatch(dict(_payload()))
    seed = json.loads(
        dynamic_action_seed_kit_path(tmp_path, meta["dyn_id"]).read_text(
            encoding="utf-8",
        ),
    )
    assert seed["kb_pitfalls"] == []


def test_p2_scenario_06_dyn_id_collision_fail_fast(tmp_path: Path):
    """Pre-existing dyn_id in SharedState.dynamic_actions → dispatch
    raises PolicyDenied with ``dynamic_dyn_id_collision``."""
    state = _state_with_round(4)
    coord = _coordinator_double(tmp_path, state)
    state.record_dynamic_action_dispatch(
        "dyn-4-1", {"status": "DISPATCHED"},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        # round_count was bumped by the record_ call → next seq=2 → not
        # a collision. Force the collision by re-faking the counter.
        state.dynamic_action_round_count = 0
        coord._prepare_dynamic_action_dispatch(dict(_payload()))
    assert excinfo.value.rule == "dynamic_dyn_id_collision"


def test_p2_scenario_07_seed_kit_schema_static_invariants(tmp_path: Path):
    """seed_kit.json on disk must contain exactly the SEED_KIT_FIELDS
    keys; no extra / no missing."""
    state = _state_with_round(0)
    coord = _coordinator_double(tmp_path, state)
    meta = coord._prepare_dynamic_action_dispatch(dict(_payload()))
    seed = json.loads(
        dynamic_action_seed_kit_path(tmp_path, meta["dyn_id"]).read_text(
            encoding="utf-8",
        ),
    )
    assert set(seed.keys()) == SEED_KIT_FIELDS


# ===========================================================================
# Stub executor wiring — confirms artifact_path injection lands a row
# ===========================================================================
@pytest.mark.asyncio
async def test_stub_executor_writes_into_artifact_dir(tmp_path: Path):
    """The stub executor reads ``artifact_path`` from task params (P2
    §6 injection) and appends dispatch_history.jsonl + writes an
    empty proposal_set.json there — NOT into the legacy task workspace."""
    state = _state_with_round(0)
    coord = _coordinator_double(tmp_path, state)
    params = dict(_payload())
    meta = coord._prepare_dynamic_action_dispatch(params)
    dyn_id = meta["dyn_id"]
    from inference_optimizer.orchestrator.action_executors.dynamic_action import (
        dynamic_action_executor,
    )
    from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext

    @dataclass
    class _StubTask:
        task_id: str = "task-1"
        kind: str = "dynamic_action"
        params: dict[str, Any] = field(default_factory=dict)

    task = _StubTask(params=params)
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await dynamic_action_executor(ctx)
    assert result["proposal_set"] == []
    artefact = dynamic_action_artifact_dir(tmp_path, dyn_id)
    history = (artefact / "dispatch_history.jsonl").read_text(encoding="utf-8")
    parsed = [json.loads(l) for l in history.splitlines() if l.strip()]
    assert parsed[0]["dyn_id"] == dyn_id
    proposal = json.loads(
        (artefact / "proposal_set.json").read_text(encoding="utf-8"),
    )
    assert proposal["proposal_set"] == []
