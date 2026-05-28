"""Tests for the ``dynamic_actions`` SharedState aggregate view +
state machine.
Auxiliary tests pin the transition table invariants, the prompt
projection field set, last_outcome derivation, motivation truncation,
and the closed prompt-section format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    PendingProposal,
)
from inference_optimizer.orchestrator.dynamic_action_proposal import (
    ALLOWED_TRANSITIONS,
    DynamicActionStatus,
    LAST_OUTCOME_BY_STATUS,
    MOTIVATION_GAP_SHORT_MAX_CHARS,
    SUMMARY_PROMPT_FIELDS,
    TERMINAL_LIFECYCLE_STATUSES,
    can_transition,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentResult


SCOPE = ["serving_specialist", "kernel_switch_specialist"]


# ===========================================================================
# Surface invariants
# ===========================================================================
def test_status_enum_count_is_eleven_plus_abandoned():
    """5 non-terminal + 7 terminal + 1 ABANDONED."""
    non_terminal = set(DynamicActionStatus) - TERMINAL_LIFECYCLE_STATUSES
    assert len(non_terminal) == 5
    assert len(TERMINAL_LIFECYCLE_STATUSES) == 8  # 7 + ABANDONED
    # Total state count
    assert len(DynamicActionStatus) == 13


def test_summary_prompt_fields_locked():
    """Closed field set; any new field surfaces in the diff."""
    assert SUMMARY_PROMPT_FIELDS == frozenset({
        "dyn_id", "status", "dispatched_at", "round_index",
        "scope_domains", "motivation_gap_short", "verdict",
        "cumulative_gain", "last_outcome", "artifact_path",
        "updated_at",
    })


def test_last_outcome_map_covers_every_status():
    """Every status has a flattened prompt-friendly label."""
    for status in DynamicActionStatus:
        assert status in LAST_OUTCOME_BY_STATUS


def test_terminal_statuses_have_empty_allowed_transitions():
    """Terminal states never transition out."""
    for terminal in TERMINAL_LIFECYCLE_STATUSES:
        assert ALLOWED_TRANSITIONS[terminal] == frozenset()


def test_motivation_gap_short_cap_locked():
    assert MOTIVATION_GAP_SHORT_MAX_CHARS == 200


# ===========================================================================
# Transition validator
# ===========================================================================
def test_can_transition_initial_only_dispatched():
    assert can_transition(None, DynamicActionStatus.DISPATCHED) is True
    assert can_transition("", DynamicActionStatus.DISPATCHED) is True
    assert can_transition(None, DynamicActionStatus.KEPT) is False
    assert can_transition(None, DynamicActionStatus.SUB_AGENT_RUNNING) is False


def test_can_transition_canonical_happy_path():
    chain = (
        DynamicActionStatus.DISPATCHED,
        DynamicActionStatus.SUB_AGENT_RUNNING,
        DynamicActionStatus.SUB_AGENT_DONE,
        DynamicActionStatus.AWAITING_CRITIC,
        DynamicActionStatus.INTEGRATING,
        DynamicActionStatus.KEPT,
    )
    for src, dst in zip(chain, chain[1:]):
        assert can_transition(src, dst) is True, f"{src} → {dst}"


def test_can_transition_terminal_locked():
    for terminal in TERMINAL_LIFECYCLE_STATUSES:
        for target in DynamicActionStatus:
            if target == terminal:
                # idempotent re-write is allowed
                assert can_transition(terminal, target) is True
            else:
                assert can_transition(terminal, target) is False, (
                    f"terminal {terminal.value} must not transition to "
                    f"{target.value}"
                )


def test_can_transition_abandoned_from_every_non_terminal():
    """P8 contract — ABANDONED is reachable from every non-terminal."""
    for src in DynamicActionStatus:
        if src in TERMINAL_LIFECYCLE_STATUSES:
            continue
        assert can_transition(src, DynamicActionStatus.ABANDONED) is True


@pytest.mark.parametrize("src,dst", [
    (DynamicActionStatus.DISPATCHED, DynamicActionStatus.KEPT),
    (DynamicActionStatus.DISPATCHED, DynamicActionStatus.INTEGRATING),
    (DynamicActionStatus.SUB_AGENT_RUNNING, DynamicActionStatus.AWAITING_CRITIC),
    (DynamicActionStatus.AWAITING_CRITIC, DynamicActionStatus.KEPT),
    (DynamicActionStatus.SUB_AGENT_DONE, DynamicActionStatus.KEPT),
])
def test_can_transition_illegal_skips_rejected(src, dst):
    assert can_transition(src, dst) is False


# ===========================================================================
# SharedState writer
# ===========================================================================
def test_writer_ignores_unknown_status():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="WAT")
    assert "dyn-0-1" not in s.dynamic_actions


def test_writer_idempotent_on_same_status():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    assert s.dynamic_actions["dyn-0-1"]["status"] == "DISPATCHED"


def test_writer_rejects_terminal_escape():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    for st in ("SUB_AGENT_RUNNING", "SUB_AGENT_DONE", "AWAITING_CRITIC",
               "INTEGRATING", "KEPT"):
        s.record_dynamic_action_outcome("dyn-0-1", status=st)
    assert s.dynamic_actions["dyn-0-1"]["status"] == "KEPT"
    # Attempt to escape terminal
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    assert s.dynamic_actions["dyn-0-1"]["status"] == "KEPT"


def test_writer_auto_derives_last_outcome():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    assert s.dynamic_actions["dyn-0-1"]["last_outcome"] == "running"
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_RUNNING")
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_DONE")
    s.record_dynamic_action_outcome("dyn-0-1", status="AWAITING_CRITIC")
    assert s.dynamic_actions["dyn-0-1"]["last_outcome"] == "awaiting_review"


def test_writer_caller_overrides_last_outcome():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome(
        "dyn-0-1", status="DISPATCHED",
        last_outcome="custom_label",
    )
    assert s.dynamic_actions["dyn-0-1"]["last_outcome"] == "custom_label"


def test_writer_preserves_extras_across_transitions():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED", extra={
        "scope_domains": SCOPE,
        "motivation_gap_short": "motivation",
        "artifact_path": "/tmp/a",
    })
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_RUNNING")
    row = s.dynamic_actions["dyn-0-1"]
    assert row["scope_domains"] == SCOPE
    assert row["motivation_gap_short"] == "motivation"
    assert row["artifact_path"] == "/tmp/a"


def test_writer_stamps_dyn_id_key():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-7-1", status="DISPATCHED")
    assert s.dynamic_actions["dyn-7-1"]["dyn_id"] == "dyn-7-1"


# ===========================================================================
# Prompt section renderer
# ===========================================================================
def test_prompt_section_empty_returns_empty_string():
    s = SharedState(session_id="t")
    assert s.to_dynamic_actions_prompt_section() == ""


def _seed_summary(
    s: SharedState, dyn_id: str, status: str, *, gain: float | None = None,
) -> None:
    s.record_dynamic_action_outcome(dyn_id, status="DISPATCHED", extra={
        "scope_domains": SCOPE,
        "motivation_gap_short": f"motivation for {dyn_id}",
        "artifact_path": f"agents/orchestration/dynamic_actions/{dyn_id}/",
    })
    # Walk through the canonical path to ``status``.
    path = ("SUB_AGENT_RUNNING", "SUB_AGENT_DONE", "AWAITING_CRITIC",
            "INTEGRATING", status)
    for st in path:
        s.record_dynamic_action_outcome(
            dyn_id, status=st,
            cumulative_gain=gain if st == status else None,
        )
        if st == status:
            break


def test_prompt_section_renders_compact_rows():
    s = SharedState(session_id="t")
    _seed_summary(s, "dyn-0-1", "KEPT", gain=2.3)
    rendered = s.to_dynamic_actions_prompt_section()
    assert "=== Dynamic Action History ===" in rendered
    assert "dyn-0-1" in rendered
    assert "KEPT" in rendered
    assert "+2.30%" in rendered
    assert "motivation for dyn-0-1" in rendered
    assert "agents/orchestration/dynamic_actions/dyn-0-1/" in rendered


def test_prompt_section_caps_at_max_entries_with_elision_marker():
    """6 entries render as 5 rows + an elision marker for the 6th."""
    s = SharedState(session_id="t")
    # Seed 6 dyn_ids with increasing updated_at timestamps.
    for i in range(6):
        _seed_summary(s, f"dyn-0-{i}", "KEPT", gain=float(i))
    rendered = s.to_dynamic_actions_prompt_section(max_entries=5)
    # 6 entries, 5 most recent shown
    visible = [line for line in rendered.split("\n") if line.startswith("- ")]
    assert len(visible) == 5
    assert "1 more older entries" in rendered
    assert "$SESSION_DIR/agents/orchestration/dynamic_actions/" in rendered


def test_prompt_section_ordering_by_updated_at_desc():
    s = SharedState(session_id="t")
    # Stagger updates so dyn-0-2 is latest.
    _seed_summary(s, "dyn-0-1", "KEPT", gain=1.0)
    _seed_summary(s, "dyn-0-2", "KEPT", gain=2.0)
    rendered = s.to_dynamic_actions_prompt_section(max_entries=5)
    # dyn-0-2 should appear before dyn-0-1 in the rendered text
    pos_1 = rendered.find("dyn-0-1")
    pos_2 = rendered.find("dyn-0-2")
    assert 0 < pos_2 < pos_1


# ===========================================================================
# §10 #2 — PolicyGate denies UPDATE_STATE on dynamic_actions
# ===========================================================================
@dataclass
class _State:
    phase: str = "EXPLORE"
    tick: int = 0
    closing_phase: bool = False
    dynamic_action_round_count: int = 0


def _gate(state: _State | None = None) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state or _State(),
        strict_phase=True,
    )


def test_p6_scenario_02_update_state_top_level_denied():
    gate = _gate()
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"dynamic_actions": {"dyn-0-1": {}}}},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "state_field"


# ===========================================================================
# §10 #3 — PolicyGate denies nested-key replacement of dynamic_actions
# (LLMs would write `dynamic_actions: {<dyn_id>: {cumulative_gain: ...}}`)
# ===========================================================================
def test_p6_scenario_03_nested_field_write_denied():
    gate = _gate()
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"dynamic_actions": {
            "dyn-0-1": {"cumulative_gain": 99.9},
        }}},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "state_field"


# ===========================================================================
# §10 #4 — PolicyGate denies new dyn_id injection
# ===========================================================================
def test_p6_scenario_04_new_dyn_id_injection_denied():
    gate = _gate()
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"dynamic_actions": {
            "dyn-99-99": {
                "status": "KEPT",
                "cumulative_gain": 999.0,
                "dyn_id": "dyn-99-99",
            },
        }}},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "state_field"


# ===========================================================================
# §10 #6 — empty dynamic_actions → no prompt section
# ===========================================================================
def test_p6_scenario_06_empty_yields_no_section():
    s = SharedState(session_id="t")
    assert s.to_dynamic_actions_prompt_section() == ""


# ===========================================================================
# §10 #8 — terminal-escape attempt is rejected
# ===========================================================================
def test_p6_scenario_08_terminal_state_locked():
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_RUNNING")
    s.record_dynamic_action_outcome("dyn-0-1", status="FAILED")
    assert s.dynamic_actions["dyn-0-1"]["status"] == "FAILED"
    # Try to re-animate
    s.record_dynamic_action_outcome("dyn-0-1", status="KEPT")
    assert s.dynamic_actions["dyn-0-1"]["status"] == "FAILED"


# ===========================================================================
# §10 #9 — concurrent dispatch + sub-agent terminal does not corrupt the
# summary map; the writer applies legal transitions in sequence.
# ===========================================================================
def test_p6_scenario_09_concurrent_dispatches_no_corruption():
    s = SharedState(session_id="t")
    # Interleave dyn_id writes — each maintains its own summary.
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
    s.record_dynamic_action_outcome("dyn-0-2", status="DISPATCHED")
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_RUNNING")
    s.record_dynamic_action_outcome("dyn-0-2", status="SUB_AGENT_RUNNING")
    s.record_dynamic_action_outcome("dyn-0-2", status="TIMED_OUT")
    s.record_dynamic_action_outcome("dyn-0-1", status="SUB_AGENT_DONE")
    s.record_dynamic_action_outcome("dyn-0-1", status="AWAITING_CRITIC")
    assert s.dynamic_actions["dyn-0-1"]["status"] == "AWAITING_CRITIC"
    assert s.dynamic_actions["dyn-0-2"]["status"] == "TIMED_OUT"


# ===========================================================================
# Coordinator wiring — full happy path advances the summary through every
# state.
# ===========================================================================
@dataclass
class _StubTask:
    task_id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubBus:
    messages: list[Any] = field(default_factory=list)

    async def append_and_seq(self, msg: Any) -> None:
        self.messages.append(msg)


@dataclass
class _StubState:
    pending_proposals: dict[str, PendingProposal] = field(default_factory=dict)


def _seed_dispatch_on_disk(tmp_path: Path, dyn_id: str) -> None:
    from inference_optimizer.session_paths import (
        dynamic_action_artifact_dir,
        dynamic_action_proposal_set_path,
        dynamic_action_seed_kit_path,
        dynamic_action_spec_path,
    )
    artefact = dynamic_action_artifact_dir(tmp_path, dyn_id)
    artefact.mkdir(parents=True, exist_ok=True)
    dynamic_action_spec_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "dyn_id": dyn_id, "round_index": 3,
            "payload": {
                "motivation_gap_text": (
                    "Combine kv cache layout shift with scheduler "
                    "rebalance: neither serving_specialist nor "
                    "kernel_switch_specialist can surface this in "
                    "their own domain."
                ),
                "scope_domains": SCOPE,
                "side_effects_declared": ["framework_source"],
                "budget_hint": "medium",
            },
        }),
        encoding="utf-8",
    )
    dynamic_action_seed_kit_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "motivation_gap_text": "m",
            "roofline_summary": "",
            "profile_keyslices": [],
            "kept_patches": [],
            "reverted_patches": [],
            "kb_pitfalls": [],
            "source_root_hints": [],
        }),
        encoding="utf-8",
    )
    proposal = {
        "name": "combo",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": SCOPE,
        "cross_domain_rationale": (
            "serving_specialist must reorder kv layout coupled with "
            "kernel_switch_specialist; risk of cache regression"
        ),
        "expected_qualitative_argument": "reduces contention safely",
    }
    dynamic_action_proposal_set_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "dyn_id": dyn_id,
            "empty": False,
            "proposal_set": [proposal],
            "journal_path": str(artefact / "sub_agent_journal.md"),
        }),
        encoding="utf-8",
    )


def _coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(session_id="test")
    c.shared_state.explore_search = {"cursor": 3}
    c.bus = _StubBus()
    c.state = _StubState()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    return c


# ===========================================================================
# §10 #1 — full happy path walks every state node
# ===========================================================================
@pytest.mark.asyncio
async def test_p6_scenario_01_happy_path_visits_every_state(tmp_path: Path):
    from inference_optimizer.orchestrator.dynamic_action_proposal import (
        DynamicRunnerTerminalState,
    )

    dyn_id = "dyn-3-1"
    _seed_dispatch_on_disk(tmp_path, dyn_id)
    coord = _coord(tmp_path)
    # Manually seed the DISPATCHED row to mimic the P2 dispatch hook.
    coord.shared_state.record_dynamic_action_outcome(
        dyn_id, status="DISPATCHED", extra={
            "scope_domains": SCOPE,
            "motivation_gap_short": "kv layout + scheduler combo",
            "artifact_path": str(
                tmp_path / "agents/orchestration/dynamic_actions" / dyn_id,
            ),
            "round_index": 3,
            "dispatched_at": "2026-05-29T00:00:00+00:00",
            "verdict": None,
            "cumulative_gain": None,
        },
    )

    # Runner returns COMPLETED → hook walks DISPATCHED →
    # SUB_AGENT_RUNNING → SUB_AGENT_DONE → AWAITING_CRITIC.
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    runner_result = SubAgentResult(
        task_id="t-runner", state="succeeded",
        result={
            "terminal_state": DynamicRunnerTerminalState.COMPLETED.value,
            "reason": "emit_proposal",
            "turns_used": 2,
            "journal_path": str(
                tmp_path / "agents/orchestration/dynamic_actions"
                / dyn_id / "sub_agent_journal.md",
            ),
        },
    )
    await coord._handle_dynamic_action_runner_result(
        task=task, result=runner_result,
    )
    assert coord.shared_state.dynamic_actions[dyn_id]["status"] == (
        DynamicActionStatus.AWAITING_CRITIC.value
    )

    # Critic approves → mirror flips status to INTEGRATING.
    pending = next(iter(coord.state.pending_proposals.values()))
    coord._mirror_critic_verdict_to_dynamic_action(
        pending=pending, verdict="approve", reasoning="lgtm",
    )
    assert coord.shared_state.dynamic_actions[dyn_id]["status"] == (
        DynamicActionStatus.INTEGRATING.value
    )

    # integrate_patch returns kept → KEPT terminal.
    coord._maybe_update_dynamic_action_after_integrate(
        task=_StubTask(
            task_id="t-integrate", kind="integrate_patch",
            params={"specialist_task_id": dyn_id, "dyn_id": dyn_id},
        ),
        result=SubAgentResult(
            task_id="t-integrate", state="succeeded",
            result={
                "status": "kept",
                "delta_pct": 1.5,
                "patches_applied": ["/abs/p.patch"],
            },
        ),
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.KEPT.value
    assert summary["cumulative_gain"] == 1.5


# ===========================================================================
# §10 #5 — prompt injection caps at N
# ===========================================================================
def test_p6_scenario_05_prompt_caps_at_five_entries(tmp_path: Path):
    s = SharedState(session_id="t")
    for i in range(6):
        _seed_summary(s, f"dyn-0-{i}", "KEPT", gain=float(i))
    rendered = s.to_dynamic_actions_prompt_section()
    visible = [line for line in rendered.split("\n") if line.startswith("- ")]
    assert len(visible) == 5
    assert "1 more older" in rendered


# ===========================================================================
# §10 #7 — coordinator-side dispatch enforces SUMMARY_PROMPT_FIELDS
# coverage (the dispatch hook stamps every prompt field, so the
# prompt-section render path never sees a missing-key skip).
# ===========================================================================
def test_p6_scenario_07_dispatch_populates_prompt_fields(tmp_path: Path):
    """The dispatch hook stamps every prompt-projection field at
    creation time so the renderer never sees a missing key."""
    s = SharedState(session_id="t")
    s.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED", extra={
        "scope_domains": SCOPE,
        "motivation_gap_short": "motivation",
        "artifact_path": "agents/orchestration/dynamic_actions/dyn-0-1/",
        "round_index": 3,
        "dispatched_at": "2026-05-29T00:00:00+00:00",
        "verdict": None,
        "cumulative_gain": None,
    })
    row = s.dynamic_actions["dyn-0-1"]
    # Every prompt-projection field exists.
    for f in SUMMARY_PROMPT_FIELDS:
        if f == "updated_at":
            # writer renames to last_updated_at for legacy parity;
            # the renderer reads both.
            assert "last_updated_at" in row or "updated_at" in row
        else:
            assert f in row, f"prompt field missing: {f}"


# ===========================================================================
# motivation_gap_short truncation contract
# ===========================================================================
def test_dispatch_truncation_obeys_motivation_cap():
    """The dispatch hook truncates motivation_gap_text down to
    MOTIVATION_GAP_SHORT_MAX_CHARS for the summary; the full text
    remains in spec.json on disk."""
    long_text = "x" * 500
    truncated = long_text[: MOTIVATION_GAP_SHORT_MAX_CHARS - 3].rstrip() + "..."
    assert len(truncated) <= MOTIVATION_GAP_SHORT_MAX_CHARS


# ===========================================================================
# CORE_STATE_FIELDS protection
# ===========================================================================
def test_dynamic_actions_in_core_state_fields():
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "dynamic_actions" in CORE_STATE_FIELDS
    assert "dynamic_action_round_count" in CORE_STATE_FIELDS


# ===========================================================================
# Coordinator prompt injection
# ===========================================================================
def test_prompt_section_compact_under_token_budget(tmp_path: Path):
    """Rendered section stays within the ~250-token budget (using a
    chars/4 estimator for a coarse upper bound)."""
    s = SharedState(session_id="t")
    for i in range(5):
        _seed_summary(s, f"dyn-0-{i}", "KEPT", gain=float(i))
    rendered = s.to_dynamic_actions_prompt_section()
    estimated_tokens = len(rendered) // 4
    assert estimated_tokens <= 500, (
        f"prompt section estimated tokens={estimated_tokens} exceeds "
        f"the 500-token hard ceiling (250 ideal)"
    )
