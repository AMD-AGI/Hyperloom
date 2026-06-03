"""Tests for ``dynamic_action`` dispatch validation and round-cap accounting.

PolicyGate is exercised against the real ``default_role_registry``;
SharedState is replaced with a thin double that exposes only the
attributes the validator + Coordinator helpers touch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    DYNAMIC_ACTION_NAME,
    MAX_DYNAMIC_PER_ROUND,
    MAX_DYNAMIC_SOURCED_VARIANTS,
    PolicyDenied,
    PolicyGate,
)


# ===========================================================================
# Helpers
# ===========================================================================
@dataclass
class _State:
    """Minimal SharedState double for PolicyGate.

    PolicyGate reads ``phase`` (group A) and ``dynamic_action_round_count``
    (group D); every other attribute the dispatch validator touches is
    either absent or falsy on a fresh object.
    """

    phase: str = "EXPLORE"
    dynamic_action_round_count: int = 0
    tick: int = 0
    closing_phase: bool = False

    # noop denial bookkeeping so the Coordinator-side audit path can call
    # this stub without exploding.
    def record_policy_denial(self, **_kwargs):  # noqa: D401
        return 1


def _gate(state: _State | None = None, *, strict_phase: bool = True) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state or _State(),
        strict_phase=strict_phase,
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    """Build a baseline-valid dynamic_action delegate payload."""
    params = {
        "motivation_gap_text": (
            "Need to combine KV cache layout change with scheduler "
            "tweak — no single specialist domain covers both."
        ),
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["framework_source"],
        "budget_hint": "medium",
    }
    base_params = overrides.pop("params", None)
    if base_params is not None:
        params = base_params
    payload = {
        "action_name": DYNAMIC_ACTION_NAME,
        "params": params,
    }
    payload.update(overrides)
    return payload


def _intent(payload: dict[str, Any]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload=payload)


# ===========================================================================
# §9 #1 — happy path
# ===========================================================================
def test_p1_scenario_01_valid_dispatch_passes():
    """EXPLORE phase + orchestration source + complete payload + 2 valid
    scope domains → PolicyGate accepts."""
    gate = _gate()
    gate.validate_intent("orchestration", _intent(_payload()))


# ===========================================================================
# §9 #2 — wrong phase
# ===========================================================================
def test_p1_scenario_02_wrong_phase_denied():
    """PRELUDE phase → dynamic_phase_violation."""
    state = _State(phase="PRELUDE")
    gate = _gate(state)
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(_payload()))
    assert excinfo.value.rule == "dynamic_phase_violation"


# ===========================================================================
# §9 #3 — wrong source
# ===========================================================================
def test_p1_scenario_03_non_orchestration_source_denied():
    """Robustness emits a dynamic_action delegate → dynamic_source_violation.

    Robustness has ``can_delegate_side_effects=True`` but is not on
    DYNAMIC_ACTION_DISPATCH_SOURCE_ALLOWLIST.
    """
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("robustness", _intent(_payload()))
    assert excinfo.value.rule == "dynamic_source_violation"


# ===========================================================================
# §9 #4 — scope_domains too narrow
# ===========================================================================
def test_p1_scenario_04_scope_domains_too_narrow_denied():
    """scope_domains.length == 1 → dynamic_scope_too_narrow."""
    payload = _payload(params={
        "motivation_gap_text": "single-domain attempt",
        "scope_domains": ["serving_specialist"],
        "side_effects_declared": ["framework_source"],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_scope_too_narrow"


# ===========================================================================
# §9 #5 — unknown scope_domain
# ===========================================================================
def test_p1_scenario_05_unknown_scope_domain_denied():
    """scope_domains contains a not-registered specialist key →
    dynamic_scope_unknown_domain."""
    payload = _payload(params={
        "motivation_gap_text": "unknown-domain attempt",
        "scope_domains": ["serving_specialist", "lyrebird_specialist"],
        "side_effects_declared": ["framework_source"],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_scope_unknown_domain"


# ===========================================================================
# §9 #6 — kernel_owned side_effects
# ===========================================================================
@pytest.mark.parametrize("forbidden", [
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "metric", "accuracy_gate", "server", "magpie",
])
def test_p1_scenario_06_red_line_side_effects_denied(forbidden: str):
    """Any §1.2 red-line side-effect → dynamic_side_effects_red_line."""
    payload = _payload(params={
        "motivation_gap_text": "red-line probe",
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": [forbidden],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_side_effects_red_line"


# ===========================================================================
# §9 #7 — round-cap exhausted
# ===========================================================================
def test_p1_scenario_07_round_cap_exhausted_denied():
    """Once MAX_DYNAMIC_PER_ROUND dispatches have landed in
    SharedState the next dispatch is denied with
    dynamic_round_cap_exhausted (until the EXPLORE round resets)."""
    state = _State(dynamic_action_round_count=MAX_DYNAMIC_PER_ROUND)
    gate = _gate(state)
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(_payload()))
    assert excinfo.value.rule == "dynamic_round_cap_exhausted"


# ===========================================================================
# §9 #8 — kernel-only scope_domains
# ===========================================================================
def test_p1_scenario_08_kernel_only_scope_denied():
    """All scope_domains == 'kernel' → dynamic_kernel_only_disallowed."""
    payload = _payload(params={
        "motivation_gap_text": "kernel-only impersonation",
        "scope_domains": ["kernel", "kernel"],
        "side_effects_declared": ["framework_source"],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_kernel_only_disallowed"


# ===========================================================================
# §9 #9 — UPDATE_STATE on dynamic_actions blocked
# ===========================================================================
def test_p1_scenario_09_update_state_on_dynamic_actions_denied():
    """LLM cannot UPDATE_STATE the dynamic_actions aggregate view (D-C
    decision; CORE_STATE_FIELDS membership). The orchestration role
    has can_mutate_core_state=False, so the field check fires."""
    gate = _gate()
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"dynamic_actions": {"dyn-0-1": {"status": "OK"}}}},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "state_field"


def test_p1_scenario_09b_update_state_on_round_count_denied():
    """The round counter field is equally Coordinator-only."""
    gate = _gate()
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"dynamic_action_round_count": 0}},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "state_field"


# ===========================================================================
# Payload schema completeness
# ===========================================================================
def test_empty_motivation_gap_text_denied():
    payload = _payload(params={
        "motivation_gap_text": "   ",
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["framework_source"],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_payload_schema"


def test_empty_side_effects_denied():
    payload = _payload(params={
        "motivation_gap_text": "no side effects",
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": [],
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_payload_schema"


def test_bad_budget_hint_denied():
    payload = _payload(params={
        "motivation_gap_text": "budget probe",
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["framework_source"],
        "budget_hint": "unlimited",
    })
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", _intent(payload))
    assert excinfo.value.rule == "dynamic_payload_schema"


# ===========================================================================
# Rejected dispatches do not consume the round cap
# ===========================================================================
def test_rejected_dispatch_does_not_bump_round_counter():
    """SharedState round counter is only bumped by Coordinator after a
    successful PolicyGate pass; a rejected dispatch leaves the counter
    untouched. We assert PolicyGate does NOT mutate the counter (it is
    a read-only consumer)."""
    state = _State(dynamic_action_round_count=0)
    gate = _gate(state)
    # Trigger group-C deny
    payload = _payload(params={
        "motivation_gap_text": "red-line probe",
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["accuracy_gate"],
    })
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", _intent(payload))
    assert state.dynamic_action_round_count == 0


# ===========================================================================
# Explore-grid provenance is advisory-only
# ===========================================================================
def test_explore_grid_with_dynamic_provenance_passes_grid_gate():
    """Mixed-provenance explore grid with one ``dynamic`` variant is
    accepted; the explore-grid gate only caps specialist fan-out."""
    state = _State(phase="EXPLORE")
    gate = _gate(state)
    explore_payload = {
        "action_name": "explore",
        "params": {
            "grid": [
                {"name": "v1", "provenance": "dynamic"},
                {"name": "v2", "provenance": "specialist:serving_specialist"},
            ],
            "config_path": "/tmp/baseline.yaml",
        },
    }
    # _validate_explore_grid_size also runs; only one specialist-sourced
    # variant present, so the cap holds.
    gate.validate_intent(
        "orchestration",
        Intent(type=IntentType.DELEGATE, payload=explore_payload),
    )


def test_explore_grid_size_caps_independent_of_dynamic():
    """The dynamic literal does NOT count against
    MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS; that cap is only for
    ``specialist:`` prefixes. A round with one ``dynamic`` plus one
    ``specialist:*`` variant is legal."""
    assert MAX_DYNAMIC_SOURCED_VARIANTS == 1


# ===========================================================================
# Stub-executor integration
# ===========================================================================
@pytest.mark.asyncio
async def test_stub_executor_writes_dispatch_history_and_empty_proposal_set(
    tmp_path: Path,
):
    """End-to-end behavior of the P1 stub executor: given a workspace
    + a task carrying ``params.dyn_id``, the runner records one
    dispatch_history.jsonl line and writes ``proposal_set.json:[]``.
    """
    from inference_optimizer.orchestrator.action_executors.dynamic_action import (
        dynamic_action_executor,
    )
    from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext

    @dataclass
    class _StubTask:
        task_id: str = "task-1"
        kind: str = DYNAMIC_ACTION_NAME
        params: dict[str, Any] = field(
            default_factory=lambda: {"dyn_id": "dyn-0-1"},
        )

    from inference_optimizer.session_paths import (
        dynamic_action_artifact_dir,
    )
    artefact = dynamic_action_artifact_dir(tmp_path, "dyn-0-1")
    artefact.mkdir(parents=True, exist_ok=True)
    task = _StubTask()
    task.params["artifact_path"] = str(artefact)
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await dynamic_action_executor(ctx)
    assert result["empty"] is True
    assert result["proposal_set"] == []
    assert result["outcome"] == "stub_empty"
    assert result["dyn_id"] == "dyn-0-1"
    history = (artefact / "dispatch_history.jsonl").read_text(encoding="utf-8")
    parsed = [json.loads(line) for line in history.splitlines() if line.strip()]
    assert len(parsed) == 1
    row = parsed[0]
    assert row["event"] == "sub_agent_done"
    assert row["terminal_state"] == "COMPLETED_EMPTY"
    assert row["reason"] == "stub_empty"
    assert row["proposal_count"] == 0
    proposal = json.loads(
        (artefact / "proposal_set.json").read_text(encoding="utf-8"),
    )
    assert proposal["proposal_set"] == []
    assert proposal["empty"] is True
    assert proposal["dyn_id"] == "dyn-0-1"


# ===========================================================================
# SharedState helper — round counter idempotency
# ===========================================================================
def test_record_dynamic_action_dispatch_idempotent_on_dyn_id():
    """A re-record with the same dyn_id overwrites the summary but
    does NOT bump the round counter."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState(session_id="t")
    state.record_dynamic_action_dispatch(
        "dyn-0-1", {"status": "DISPATCHED"},
    )
    assert state.dynamic_action_round_count == 1
    state.record_dynamic_action_dispatch(
        "dyn-0-1", {"status": "COMPLETED_EMPTY"},
    )
    assert state.dynamic_action_round_count == 1
    assert state.dynamic_actions["dyn-0-1"]["status"] == "COMPLETED_EMPTY"
    state.record_dynamic_action_dispatch(
        "dyn-0-2", {"status": "DISPATCHED"},
    )
    assert state.dynamic_action_round_count == 2


def test_reset_dynamic_action_round_count_clears_only_counter():
    """``reset_dynamic_action_round_count`` zeros the counter but keeps
    the cumulative ``dynamic_actions`` ledger intact."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState(session_id="t")
    state.record_dynamic_action_dispatch(
        "dyn-0-1", {"status": "DISPATCHED"},
    )
    state.reset_dynamic_action_round_count()
    assert state.dynamic_action_round_count == 0
    assert "dyn-0-1" in state.dynamic_actions
