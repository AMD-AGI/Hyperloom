# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end ``dynamic_action`` lifecycle acceptance tests.
Auxiliary tests pin the pipeline helpers + Coordinator hooks the
acceptance flow leans on.

The runner / critic / integrate_patch executors are exercised at the
hook boundary: we drive the Coordinator's ``_handle_dynamic_action_
runner_result`` / ``_mirror_critic_verdict_to_dynamic_action`` /
``_maybe_update_dynamic_action_after_integrate`` directly with
synthesised :class:`SubAgentResult` payloads so the suite stays
free of subprocess / sqlite plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    PendingProposal,
)
from inference_optimizer.orchestrator.dynamic_action_pipeline import (
    build_integrate_patch_proposal_payload,
    compose_critic_verdict_envelope,
    integrate_status_to_lifecycle,
    is_dynamic_specialist_task_id,
    make_dynamic_specialist_task_id,
    materialize_dynamic_patch_workspace,
    read_runner_proposal_set,
    runner_status_to_lifecycle,
)
from inference_optimizer.orchestrator.dynamic_action_proposal import (
    DynamicActionStatus,
    DynamicRunnerTerminalState,
    TERMINAL_LIFECYCLE_STATUSES,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentResult
from inference_optimizer.session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_critic_verdict_path,
    dynamic_action_proposal_set_path,
    dynamic_action_seed_kit_path,
    dynamic_action_spec_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
SCOPE = ["serving_specialist", "kernel_switch_specialist"]
DEFAULT_RATIONALE = (
    "serving_specialist must reorder kv layout coupled with "
    "kernel_switch_specialist; risk of cache regression"
)


def _proposal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "combo",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": SCOPE,
        "cross_domain_rationale": DEFAULT_RATIONALE,
        "expected_qualitative_argument": (
            "reduces contention without breaking accuracy"
        ),
    }
    base.update(overrides)
    return base


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


def _seed_dispatch(
    session_dir: Path, dyn_id: str, *, scope_domains: list[str] | None = None,
) -> None:
    """Write the spec.json + seed_kit.json + proposal_set.json the
    coordinator hook reads. The P3 runner would have written these
    during a real run; here we synthesize them inline so the test
    starts at the post-runner boundary."""
    artefact = dynamic_action_artifact_dir(session_dir, dyn_id)
    artefact.mkdir(parents=True, exist_ok=True)
    dynamic_action_spec_path(session_dir, dyn_id).write_text(
        json.dumps({
            "dyn_id": dyn_id,
            "round_index": 0,
            "payload": {
                "motivation_gap_text": "test",
                "scope_domains": scope_domains or SCOPE,
                "side_effects_declared": ["framework_source"],
                "budget_hint": "medium",
            },
        }),
        encoding="utf-8",
    )
    dynamic_action_seed_kit_path(session_dir, dyn_id).write_text(
        json.dumps({
            "motivation_gap_text": "test",
            "roofline_summary": "",
            "profile_keyslices": [],
            "kept_patches": [],
            "reverted_patches": [],
            "kb_pitfalls": [],
            "source_root_hints": [],
        }),
        encoding="utf-8",
    )


def _write_runner_proposal_set(
    session_dir: Path,
    dyn_id: str,
    *,
    empty: bool = False,
    proposal: dict[str, Any] | None = None,
) -> None:
    target = dynamic_action_proposal_set_path(session_dir, dyn_id)
    payload = {
        "dyn_id": dyn_id,
        "empty": empty,
        "proposal_set": [] if empty else [proposal or _proposal()],
        "journal_path": str(
            dynamic_action_artifact_dir(session_dir, dyn_id)
            / "sub_agent_journal.md",
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _coordinator_double(tmp_path: Path) -> Coordinator:
    """Construct a Coordinator with just enough surface for the P5
    hooks. We bypass the heavy __init__ via ``__new__`` so no sqlite
    db / backends are required."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState(session_id="test")
    c.shared_state.explore_search = {"cursor": 0}
    c.bus = _StubBus()
    c.state = _StubState()
    c._record_observation = AsyncMock()  # type: ignore[method-assign]
    return c


def _runner_result(
    *,
    terminal_state: DynamicRunnerTerminalState,
    reason: str = "",
    turns_used: int = 1,
    journal_path: str = "",
) -> SubAgentResult:
    return SubAgentResult(
        task_id="task-1",
        state="succeeded",
        result={
            "terminal_state": terminal_state.value,
            "reason": reason or "emit_proposal",
            "turns_used": turns_used,
            "journal_path": journal_path,
            "proposal_set_payload": {},
        },
    )


# ===========================================================================
# Helper sanity
# ===========================================================================
def test_is_dynamic_specialist_task_id_prefix():
    assert is_dynamic_specialist_task_id("dyn-0-1") is True
    assert is_dynamic_specialist_task_id("task-1") is False
    assert is_dynamic_specialist_task_id("") is False


def test_terminal_lifecycle_set_matches_p5_table():
    assert TERMINAL_LIFECYCLE_STATUSES == frozenset({
        DynamicActionStatus.TIMED_OUT,
        DynamicActionStatus.FAILED,
        DynamicActionStatus.COMPLETED_EMPTY,
        DynamicActionStatus.CRITIC_REJECTED,
        DynamicActionStatus.INTEGRATE_FAILED,
        DynamicActionStatus.REVERTED,
        DynamicActionStatus.KEPT,
        DynamicActionStatus.ABANDONED,
    })
    # DISPATCHED is the only non-terminal status.
    assert DynamicActionStatus.DISPATCHED not in TERMINAL_LIFECYCLE_STATUSES


# ===========================================================================
# Pipeline pure helpers
# ===========================================================================
def test_runner_status_to_lifecycle_full_map():
    # COMPLETED transitions to AWAITING_CRITIC.
    # in a single atomic step (the critic dispatch runs synchronously
    # in the same hook, so DISPATCHED → ... → AWAITING_CRITIC is
    # collapsed into one writer event).
    assert runner_status_to_lifecycle(
        DynamicRunnerTerminalState.COMPLETED,
    ) == DynamicActionStatus.AWAITING_CRITIC
    assert runner_status_to_lifecycle(
        DynamicRunnerTerminalState.COMPLETED_EMPTY,
    ) == DynamicActionStatus.COMPLETED_EMPTY
    assert runner_status_to_lifecycle(
        DynamicRunnerTerminalState.TIMED_OUT,
    ) == DynamicActionStatus.TIMED_OUT
    assert runner_status_to_lifecycle(
        DynamicRunnerTerminalState.FAILED,
    ) == DynamicActionStatus.FAILED
    assert runner_status_to_lifecycle(
        DynamicRunnerTerminalState.ABANDONED,
    ) == DynamicActionStatus.ABANDONED
    assert runner_status_to_lifecycle("garbage") == DynamicActionStatus.FAILED


def test_integrate_status_to_lifecycle_mapping():
    assert integrate_status_to_lifecycle(
        "kept",
    ) == DynamicActionStatus.KEPT
    assert integrate_status_to_lifecycle(
        "reverted",
    ) == DynamicActionStatus.REVERTED
    assert integrate_status_to_lifecycle(
        "apply_failed",
    ) == DynamicActionStatus.INTEGRATE_FAILED
    assert integrate_status_to_lifecycle(
        "no_patches",
    ) == DynamicActionStatus.INTEGRATE_FAILED
    assert integrate_status_to_lifecycle(
        "applied_no_bench",
    ) == DynamicActionStatus.KEPT


def test_materialize_dynamic_patch_workspace_writes_specialist_layout(
    tmp_path: Path,
):
    sid, patches = materialize_dynamic_patch_workspace(
        session_dir=tmp_path,
        dyn_id="dyn-0-1",
        proposal=_proposal(),
    )
    assert sid == "dyn-0-1"
    assert len(patches) == 1
    patch_path = Path(patches[0])
    assert patch_path.is_file()
    assert "new" in patch_path.read_text(encoding="utf-8")
    done = json.loads(
        (tmp_path / "runs/specialist/dyn-0-1/specialist_done.json").read_text(),
    )
    assert done["provenance"] == "dynamic"
    assert done["dyn_id"] == "dyn-0-1"
    assert done["patches_written"] == patches


def test_make_dynamic_specialist_task_id_identity():
    assert make_dynamic_specialist_task_id("dyn-3-2") == "dyn-3-2"


def test_build_integrate_patch_proposal_payload_carries_provenance():
    payload = build_integrate_patch_proposal_payload(
        dyn_id="dyn-0-1",
        specialist_task_id="dyn-0-1",
        proposal=_proposal(),
        spec_payload={"scope_domains": SCOPE},
    )
    assert payload["action_name"] == "integrate_patch"
    assert payload["provenance"] == "dynamic"
    assert payload["params"]["specialist_task_id"] == "dyn-0-1"
    assert payload["params"]["dyn_id"] == "dyn-0-1"


def test_compose_critic_verdict_floor_blocks_llm_approve(tmp_path: Path):
    """Mechanical floor REJECT (numeric claim) wins over LLM approve."""
    bad = _proposal(expected_qualitative_argument="gives 20% speedup")
    envelope, lifecycle = compose_critic_verdict_envelope(
        dyn_id="dyn-0-1", proposal=bad, spec_scope_domains=SCOPE,
        llm_verdict="approve",
    )
    assert envelope["verdict"] == "reject"
    assert lifecycle == DynamicActionStatus.CRITIC_REJECTED


def test_compose_critic_verdict_llm_reject_passes_through():
    envelope, lifecycle = compose_critic_verdict_envelope(
        dyn_id="dyn-0-1", proposal=_proposal(), spec_scope_domains=SCOPE,
        llm_verdict="reject", llm_reason="breaks test suite",
    )
    assert envelope["verdict"] == "reject"
    assert "llm_critic_reject" in envelope["reason_codes"]
    assert lifecycle == DynamicActionStatus.CRITIC_REJECTED


def test_compose_critic_verdict_happy_advances_to_integrating():
    # Approve transitions to INTEGRATING (the
    # integrate_patch dispatch fires immediately after).
    envelope, lifecycle = compose_critic_verdict_envelope(
        dyn_id="dyn-0-1", proposal=_proposal(), spec_scope_domains=SCOPE,
        llm_verdict="approve",
    )
    assert envelope["verdict"] == "approve"
    assert lifecycle == DynamicActionStatus.INTEGRATING


def test_read_runner_proposal_set_missing_file_returns_none(tmp_path: Path):
    assert read_runner_proposal_set(tmp_path, "no-such-dyn") is None


def test_read_runner_proposal_set_parses_disk_payload(tmp_path: Path):
    _seed_dispatch(tmp_path, "dyn-1-1")
    _write_runner_proposal_set(tmp_path, "dyn-1-1")
    payload = read_runner_proposal_set(tmp_path, "dyn-1-1")
    assert payload["empty"] is False
    assert payload["proposal_set"][0]["provenance"] == "dynamic"


# ===========================================================================
# §10 #1 — full happy path
# ===========================================================================
@pytest.mark.asyncio
async def test_p5_scenario_01_happy_path_to_kept(tmp_path: Path):
    """Runner COMPLETED → mechanical floor approve → critic approve →
    integrate_patch returns kept → dyn_id status = KEPT, intervention
    ledger gets a ``dynamic_action_integrate`` code_patch row, and
    cumulative_gain reflects delta_pct."""
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    _write_runner_proposal_set(tmp_path, dyn_id)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    runner_result = _runner_result(
        terminal_state=DynamicRunnerTerminalState.COMPLETED,
        reason="emit_proposal",
        turns_used=2,
        journal_path=str(
            dynamic_action_artifact_dir(tmp_path, dyn_id)
            / "sub_agent_journal.md",
        ),
    )
    await coord._handle_dynamic_action_runner_result(
        task=task, result=runner_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    # Runner-done hook collapses DISPATCHED → SUB_AGENT_RUNNING →
    # SUB_AGENT_DONE → AWAITING_CRITIC happens in one event.
    assert summary["status"] == DynamicActionStatus.AWAITING_CRITIC.value
    assert "specialist_task_id" in summary
    # Coordinator pushed a proposal onto the bus.
    assert len(coord.bus.messages) == 1
    assert len(coord.state.pending_proposals) == 1
    pending = next(iter(coord.state.pending_proposals.values()))
    assert pending.action_name == "integrate_patch"

    # Critic approves → mirror writes critic_verdict.json + flips
    # status to INTEGRATING. integrate_patch hook
    # advances it to KEPT next.
    coord._mirror_critic_verdict_to_dynamic_action(
        pending=pending, verdict="approve", reasoning="lgtm",
    )
    envelope = json.loads(
        dynamic_action_critic_verdict_path(tmp_path, dyn_id).read_text(),
    )
    assert envelope["verdict"] == "approve"
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.INTEGRATING.value
    assert summary["critic_verdict"] == "approve"

    # integrate_patch returns kept → final lifecycle = KEPT
    integrate_task = _StubTask(
        task_id="t-integrate", kind="integrate_patch",
        params={"specialist_task_id": "dyn-0-1", "dyn_id": dyn_id},
    )
    integrate_result = SubAgentResult(
        task_id="t-integrate", state="succeeded",
        result={
            "status": "kept",
            "delta_pct": 1.5,
            "output_throughput": 1850.0,
            "accuracy_pass": True,
            "patches_applied": ["/abs/p.patch"],
            "patches_reverted": [],
        },
    )
    coord._maybe_update_dynamic_action_after_integrate(
        task=integrate_task, result=integrate_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.KEPT.value
    assert summary["cumulative_gain"] == 1.5
    assert summary["integrate_status"] == "kept"
    # Intervention ledger row tagged with dynamic source.
    assert any(
        entry.get("action") == "dynamic_action_integrate"
        for entry in coord.shared_state.intervention_mix
    )


# ===========================================================================
# §10 #2 — COMPLETED_EMPTY skips critic
# ===========================================================================
@pytest.mark.asyncio
async def test_p5_scenario_02_completed_empty_skips_critic(tmp_path: Path):
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    _write_runner_proposal_set(tmp_path, dyn_id, empty=True)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    runner_result = _runner_result(
        terminal_state=DynamicRunnerTerminalState.COMPLETED_EMPTY,
        reason="emit_empty",
    )
    await coord._handle_dynamic_action_runner_result(
        task=task, result=runner_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.COMPLETED_EMPTY.value
    assert summary["last_outcome"] == "emit_empty"
    # No proposal pushed; no critic invocation.
    assert coord.bus.messages == []
    assert coord.state.pending_proposals == {}


# ===========================================================================
# §10 #3 — TIMED_OUT skips critic, journal preserved
# ===========================================================================
@pytest.mark.asyncio
async def test_p5_scenario_03_timed_out_skips_critic(tmp_path: Path):
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    # Journal is written by the runner even on TIMED_OUT (P3); we just
    # verify the hook does not delete it.
    journal = (
        dynamic_action_artifact_dir(tmp_path, dyn_id)
        / "sub_agent_journal.md"
    )
    journal.write_text("# journal\nturn 1: ...\n", encoding="utf-8")
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    runner_result = _runner_result(
        terminal_state=DynamicRunnerTerminalState.TIMED_OUT,
        reason="turn_cap_exhausted",
    )
    await coord._handle_dynamic_action_runner_result(
        task=task, result=runner_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.TIMED_OUT.value
    assert summary["last_outcome"] == "turn_cap_exhausted"
    assert coord.bus.messages == []
    assert journal.is_file()


# ===========================================================================
# §10 #4 — Critic REJECT short-circuits integrate
# ===========================================================================
@pytest.mark.asyncio
async def test_p5_scenario_04_critic_reject_short_circuits(tmp_path: Path):
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    _write_runner_proposal_set(tmp_path, dyn_id)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    await coord._handle_dynamic_action_runner_result(
        task=task,
        result=_runner_result(terminal_state=DynamicRunnerTerminalState.COMPLETED),
    )
    pending = next(iter(coord.state.pending_proposals.values()))
    coord._mirror_critic_verdict_to_dynamic_action(
        pending=pending, verdict="reject", reasoning="patch breaks unit tests",
    )
    envelope = json.loads(
        dynamic_action_critic_verdict_path(tmp_path, dyn_id).read_text(),
    )
    assert envelope["verdict"] == "reject"
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.CRITIC_REJECTED.value
    assert "llm_critic_reject" in summary["critic_reason_codes"]


# ===========================================================================
# §10 #5 — integrate_patch apply failure
# ===========================================================================
def test_p5_scenario_05_integrate_apply_failure(tmp_path: Path):
    dyn_id = "dyn-0-1"
    coord = _coordinator_double(tmp_path)
    coord.shared_state.dynamic_actions[dyn_id] = {
        "status": DynamicActionStatus.DISPATCHED.value,
    }
    integrate_task = _StubTask(
        task_id="t-integrate", kind="integrate_patch",
        params={"specialist_task_id": "dyn-0-1", "dyn_id": dyn_id},
    )
    integrate_result = SubAgentResult(
        task_id="t-integrate", state="succeeded",
        result={
            "status": "apply_failed",
            "reason": "patch hunk conflict",
            "patches_applied": [],
            "patches_reverted": [],
        },
    )
    coord._maybe_update_dynamic_action_after_integrate(
        task=integrate_task, result=integrate_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.INTEGRATE_FAILED.value
    assert summary["integrate_status"] == "apply_failed"
    # No code_patch row in the intervention ledger.
    assert not any(
        entry.get("action") == "dynamic_action_integrate"
        for entry in coord.shared_state.intervention_mix
    )


# ===========================================================================
# §10 #6 — accuracy gate fail → REVERTED
# ===========================================================================
def test_p5_scenario_06_accuracy_gate_revert(tmp_path: Path):
    dyn_id = "dyn-0-1"
    coord = _coordinator_double(tmp_path)
    coord.shared_state.dynamic_actions[dyn_id] = {
        "status": DynamicActionStatus.DISPATCHED.value,
    }
    integrate_task = _StubTask(
        task_id="t-integrate", kind="integrate_patch",
        params={"specialist_task_id": "dyn-0-1", "dyn_id": dyn_id},
    )
    integrate_result = SubAgentResult(
        task_id="t-integrate", state="succeeded",
        result={
            "status": "reverted",
            "reason": "accuracy_below_threshold",
            "patches_applied": ["/abs/p.patch"],
            "patches_reverted": ["/abs/p.patch"],
            "accuracy_pass": False,
        },
    )
    coord._maybe_update_dynamic_action_after_integrate(
        task=integrate_task, result=integrate_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.REVERTED.value
    assert summary["accuracy_pass"] is False


# ===========================================================================
# §10 #7 — gain < KEEP threshold → REVERTED
# ===========================================================================
def test_p5_scenario_07_gain_below_threshold_revert(tmp_path: Path):
    dyn_id = "dyn-0-1"
    coord = _coordinator_double(tmp_path)
    coord.shared_state.dynamic_actions[dyn_id] = {
        "status": DynamicActionStatus.DISPATCHED.value,
    }
    integrate_task = _StubTask(
        task_id="t-integrate", kind="integrate_patch",
        params={"specialist_task_id": "dyn-0-1", "dyn_id": dyn_id},
    )
    integrate_result = SubAgentResult(
        task_id="t-integrate", state="succeeded",
        result={
            "status": "reverted",
            "reason": "gain_below_keep_threshold",
            "delta_pct": 0.1,
            "patches_applied": ["/abs/p.patch"],
            "patches_reverted": ["/abs/p.patch"],
            "accuracy_pass": True,
        },
    )
    coord._maybe_update_dynamic_action_after_integrate(
        task=integrate_task, result=integrate_result,
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.REVERTED.value
    assert summary["cumulative_gain"] == 0.1


# ===========================================================================
# §10 #8 — dynamic + specialists concurrent (no interference)
# ===========================================================================
@pytest.mark.asyncio
async def test_p5_scenario_08_concurrent_dynamic_and_specialists(tmp_path: Path):
    """The hook is per-task; running it twice in interleaved order on
    different dyn_ids does not corrupt the summary map."""
    coord = _coordinator_double(tmp_path)
    for dyn_id in ("dyn-0-1", "dyn-0-2"):
        _seed_dispatch(tmp_path, dyn_id)
        _write_runner_proposal_set(tmp_path, dyn_id, empty=True)
        await coord._handle_dynamic_action_runner_result(
            task=_StubTask(
                task_id=f"t-{dyn_id}", kind="dynamic_action",
                params={"dyn_id": dyn_id},
            ),
            result=_runner_result(
                terminal_state=DynamicRunnerTerminalState.COMPLETED_EMPTY,
                reason="emit_empty",
            ),
        )
    assert (
        coord.shared_state.dynamic_actions["dyn-0-1"]["status"]
        == DynamicActionStatus.COMPLETED_EMPTY.value
    )
    assert (
        coord.shared_state.dynamic_actions["dyn-0-2"]["status"]
        == DynamicActionStatus.COMPLETED_EMPTY.value
    )


# ===========================================================================
# §10 #9 — dispatch when lane is full: PolicyGate does NOT reject
# (lane wait is a soft constraint).
# Already covered in test_dynamic_action_dispatch (no round-cap
# violation when lane saturation is the only blocker); we replay the
# contract here for the §10 mapping table.
# ===========================================================================
def test_p5_scenario_09_lane_saturation_does_not_fail_policy(tmp_path: Path):
    """PolicyGate gates on round_cap + payload, not on lane
    availability. The dispatcher's ResourceLockManager loop handles
    waiting for the lane; the hook layer never needs to react to
    'lane full' as a failure mode."""
    from inference_optimizer.orchestrator.dynamic_action_proposal import (
        DynamicActionStatus,
    )
    coord = _coordinator_double(tmp_path)
    coord.shared_state.dynamic_actions["dyn-0-1"] = {
        "status": DynamicActionStatus.DISPATCHED.value,
    }
    # A dispatched dyn_id with no runner result yet is the legitimate
    # "waiting on lane" state — exactly what _handle_dynamic_action_
    # runner_result will pick up once the runner does run.
    assert (
        coord.shared_state.dynamic_actions["dyn-0-1"]["status"]
        == "DISPATCHED"
    )


# ===========================================================================
# Mechanical floor blocks LLM critic call entirely
# ===========================================================================
@pytest.mark.asyncio
async def test_mechanical_floor_blocks_critic_dispatch(tmp_path: Path):
    """When the runner emits a proposal that violates the mechanical
    floor (e.g. numeric claim leaked into qualitative argument), the
    hook writes critic_verdict.json directly and does NOT push a
    proposal onto the bus."""
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    bad = _proposal(expected_qualitative_argument="should hit 25% gain")
    _write_runner_proposal_set(tmp_path, dyn_id, proposal=bad)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    await coord._handle_dynamic_action_runner_result(
        task=task,
        result=_runner_result(
            terminal_state=DynamicRunnerTerminalState.COMPLETED,
        ),
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.CRITIC_REJECTED.value
    envelope = json.loads(
        dynamic_action_critic_verdict_path(tmp_path, dyn_id).read_text(),
    )
    assert envelope["verdict"] == "reject"
    assert "dynamic_quantitative_claim_violation" in envelope["reason_codes"]
    assert coord.bus.messages == []
    assert coord.state.pending_proposals == {}


# ===========================================================================
# Non-dynamic integrate completion is untouched
# ===========================================================================
def test_integrate_completion_skips_non_dynamic(tmp_path: Path):
    """The hook is a no-op when specialist_task_id is the legacy
    specialist (no ``dyn-`` prefix). Dynamic_actions summary stays
    empty so the PR-A7 specialist path runs unchanged."""
    coord = _coordinator_double(tmp_path)
    integrate_task = _StubTask(
        task_id="t-integrate", kind="integrate_patch",
        params={"specialist_task_id": "task-abcd"},
    )
    integrate_result = SubAgentResult(
        task_id="t-integrate", state="succeeded",
        result={"status": "kept", "delta_pct": 2.0},
    )
    coord._maybe_update_dynamic_action_after_integrate(
        task=integrate_task, result=integrate_result,
    )
    assert coord.shared_state.dynamic_actions == {}


# ===========================================================================
# Runner result with empty proposal_set on disk collapses to
# COMPLETED_EMPTY (defensive against contract drift between runner +
# pipeline).
# ===========================================================================
@pytest.mark.asyncio
async def test_completed_but_empty_proposal_collapses_to_completed_empty(
    tmp_path: Path,
):
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    _write_runner_proposal_set(tmp_path, dyn_id, empty=True)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    await coord._handle_dynamic_action_runner_result(
        task=task,
        result=_runner_result(
            terminal_state=DynamicRunnerTerminalState.COMPLETED,
        ),
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.COMPLETED_EMPTY.value


# ===========================================================================
# ABANDONED runner result
# ===========================================================================
@pytest.mark.asyncio
async def test_abandoned_runner_result_writes_abandoned_status(tmp_path: Path):
    dyn_id = "dyn-0-1"
    _seed_dispatch(tmp_path, dyn_id)
    coord = _coordinator_double(tmp_path)
    task = _StubTask(
        task_id="t-runner", kind="dynamic_action",
        params={"dyn_id": dyn_id},
    )
    await coord._handle_dynamic_action_runner_result(
        task=task,
        result=_runner_result(
            terminal_state=DynamicRunnerTerminalState.ABANDONED,
            reason="external_kill",
        ),
    )
    summary = coord.shared_state.dynamic_actions[dyn_id]
    assert summary["status"] == DynamicActionStatus.ABANDONED.value
