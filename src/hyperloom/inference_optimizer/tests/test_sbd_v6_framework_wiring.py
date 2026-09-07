# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FRAMEWORK_AGENT timeline event's wiring into the phase itself.

`test_sbd_v6_framework_timeline` covers the recorder against calls made
directly. These tests go through the real Coordinator seams instead, because
the failure they exist to catch is not in the recorder: it is an event that
never opens, a close that never fires, or a policy field read off an attribute
whose name drifted. None of those are visible to a test that calls the recorder
itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan


def _coordinator(session_dir: Path) -> Coordinator:
    """Build a real Coordinator with idle agents."""
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=session_dir,
        backends={
            "orchestration": MockBackend(idle),
            "critic": MockBackend(idle),
            "robustness": MockBackend(idle),
        },
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [e for e in read_timeline_events(session_dir) if e.get("type") == "framework_agent"]


@pytest.fixture
def session_dir(tmp_path: Path):
    """A bound session, the way startup binds one."""
    path = tmp_path / "session"
    path.mkdir()
    with session_scope(path):
        yield path


def test_open_records_the_resolved_policy(session_dir: Path):
    """The policy on the event is what the phase's own owners resolved.

    Each field is read from the function or attribute that owns it, so this
    also pins the attribute names: a rename that silently turned a threshold
    into ``None`` is exactly the regression the projection used to hide behind
    its fallback chains.
    """
    coord = _coordinator(session_dir)
    state = coord.shared_state
    state.phase = "FRAMEWORK_AGENT"
    state.macro_cycle = 0
    state.framework_agent_authoring_enabled = True
    state.explore_overtime_kill_ratio = 1.5
    state.explore_variant_timeout_sec_override = 1800
    state.plateau_overrides = {"explore_lookback": 7, "explore_keep_gain_pct": 1.25}

    coord._open_framework_timeline()
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    policy = _events(session_dir)[0]["ext"]["policy"]
    assert policy["keep_threshold_pct"] is not None
    assert policy["variant_timeout_sec"] == 1800
    assert policy["overtime_kill_ratio"] == 1.5
    assert policy["config"]["lookback"] == 7
    assert policy["config"]["keep_gain_threshold_pct"] == 1.25
    # Not overridden, so the library default the phase will actually apply.
    assert policy["config"]["empty_streak_threshold"] is not None
    assert policy["source"]["authoring_enabled"] is True
    assert policy["source"]["no_keep_streak_threshold"] is not None
    assert policy["source"]["discovery_retry_limit"] == 3


def test_force_exit_budget_pct_is_reported_unresolved(session_dir: Path):
    """No runtime path resolves it, so the event says so.

    Reporting a default here would claim the phase ran under a bar it never
    applied. The projection's fallback chain did exactly that.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    assert _events(session_dir)[0]["ext"]["policy"]["force_exit_budget_pct"] is None


@pytest.mark.asyncio
async def test_phase_transition_closes_the_event(session_dir: Path):
    """Leaving FRAMEWORK_AGENT closes the event through the machine's seam.

    The machine has entry hooks only, so a close that is not wired into the
    transition leaves every event open and the whole phase reads as
    interrupted.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.phase_history = [{"to_phase": "FRAMEWORK_AGENT", "reason": "prelude_done", "evidence": {}}]
    coord._open_framework_timeline()

    await coord._on_phase_entered(
        from_phase="FRAMEWORK_AGENT",
        to_phase="SWEEP",
        reason="optimize_no_more_leverage",
        evidence={"evidence": "both_arms_plateaued", "switch_bottleneck": True},
    )

    event = _events(session_dir)[0]
    assert event["status"] != "running"
    assert event["ext"]["exit"]["reason"] == "optimize_no_more_leverage"
    assert event["ext"]["exit"]["trigger"] == "both_arms_plateaued"
    assert event["ext"]["exit"]["switch_bottleneck"] is True


@pytest.mark.asyncio
async def test_exit_plateau_comes_from_the_deciding_evidence(session_dir: Path):
    """The exit plateau rows hold what the exit rule read, not a recount.

    The rule's evidence is the only record of the counts as they stood when it
    ruled; by close time more candidates may have resolved.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    await coord._on_phase_entered(
        from_phase="FRAMEWORK_AGENT",
        to_phase="KERNEL_AGENT",
        reason="optimize_no_more_leverage",
        evidence={
            "evidence": "both_arms_plateaued",
            "config_arm_plateaued": True,
            "recent_keep_gain_pct": 0.25,
            "keep_gain_threshold_pct": 0.5,
            "empty_streak": 5,
            "empty_streak_threshold": 5,
            "lookback": 5,
            "source_arm_plateaued": True,
            "source_consecutive_no_keep": 6,
            "source_threshold": 5,
            "source_candidates_exhausted": False,
        },
    )

    plateau = _events(session_dir)[0]["ext"]["plateau"]
    by_arm = {row["arm"]: row for row in plateau}
    assert {row["path"] for row in plateau} == {"exit"}
    assert by_arm["config"]["triggered"] is True
    assert by_arm["config"]["inputs"]["recent_keep_gain_pct"] == 0.25
    assert by_arm["config"]["thresholds"]["empty_streak_threshold"] == 5
    assert by_arm["source"]["inputs"]["consecutive_no_keep"] == 6
    assert by_arm["source"]["inputs"]["candidates_exhausted"] is False


@pytest.mark.asyncio
async def test_a_transition_without_a_plateau_reading_writes_no_rows(session_dir: Path):
    """A budget exit did not evaluate a plateau, so it claims none.

    Writing rows from an evidence map that never held a plateau reading would
    report an evaluation that did not happen.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    await coord._on_phase_entered(
        from_phase="FRAMEWORK_AGENT",
        to_phase="CLOSE",
        reason="session_budget_exhausted",
        evidence={"terminal": True},
    )

    assert _events(session_dir)[0]["ext"]["plateau"] == []


def test_advisory_plateau_snapshots_both_arms(session_dir: Path):
    """Composing the advisory records what it was composed from.

    Both arms land whether or not either fired: "evaluated and did not trip"
    is the reading that explains a phase staying open, and it is not
    recoverable from a history that kept growing afterwards.
    """
    coord = _coordinator(session_dir)
    state = coord.shared_state
    state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    coord._plateau_advisory_block()
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    advisory = [row for row in _events(session_dir)[0]["ext"]["plateau"] if row["path"] == "advisory"]
    assert {row["arm"] for row in advisory} == {"config", "source"}
    for row in advisory:
        assert row["triggered"] is not None
        assert row["thresholds"]


def test_discovery_round_records_its_run_and_both_outcomes(session_dir: Path):
    """A harvested round records the run, its candidates, and what it dropped.

    The audited-away entries are the round's most informative output: five PRs
    all judged already landed is a very different result from finding nothing,
    and the projection reported both as an empty round.
    """
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    task = SimpleNamespace(
        task_id="t-disc-1",
        params={"candidate_discovery": True, "domain": "candidate_discovery_specialist"},
    )
    coord._ingest_candidate_discovery(
        task=task,
        done_payload={
            "proposal_set": [
                {"pr_url": "https://x/pr/1", "title": "live one", "repo": "vllm", "verdict": "worth_a_bench"},
                {"pr_url": "https://x/pr/2", "title": "landed", "repo": "vllm", "verdict": "already_present"},
                {"pr_url": "https://x/pr/3", "title": "n/a", "repo": "vllm", "verdict": "not_applicable"},
            ]
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    ext = _events(session_dir)[0]["ext"]
    run = ext["runs"][0]
    assert run["run_id"] == "t-disc-1"
    assert run["role"] == "discovery"
    assert run["arm"] == "source"
    assert run["status"] == "succeeded"
    assert len(run["produced_ids"]) == 3

    by_id = {row["proposal_id"]: row for row in ext["proposals"]}
    assert len(by_id) == 3
    live = by_id["https://x/pr/1"]
    assert live["producer"] == "specialist"
    assert live["run_ref"] == "t-disc-1"
    assert live.get("terminal") in (None, {})
    assert [step["step"] for step in live["lifecycle"]] == ["proposed"]
    for dropped, why in (("https://x/pr/2", "already_present"), ("https://x/pr/3", "not_applicable")):
        assert by_id[dropped]["terminal"]["disposition"] == "dropped"
        assert by_id[dropped]["terminal"]["reason"] == why


def test_failed_discovery_round_records_the_failure(session_dir: Path):
    """A round that could not run reports nothing about what is out there.

    The projection had to identify this by scanning journal rows backwards for
    particular event names.
    """
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    coord._ingest_candidate_discovery(
        task=SimpleNamespace(task_id="t-disc-2", params={"candidate_discovery": True, "domain": "d"}),
        done_payload={},
        run_error="worktree checkout failed",
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    event = _events(session_dir)[0]
    run = event["ext"]["runs"][0]
    assert run["status"] == "failed"
    assert "worktree" in run["reason"]
    assert event["ext"]["proposals"] == []
    # The run failed, so the entry did not succeed at what it dispatched.
    assert event["status"] == "failed"


def test_terminal_row_settles_the_proposal(session_dir: Path):
    """Every dead end settles its proposal, through the one terminal writer.

    Hooked at that writer rather than at each of its dozen callers, so a newly
    added dead end cannot silently leave its proposal pending.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    coord._stamp_framework_progress(
        candidate_id="cand-1",
        batch_id="b1",
        status="apply_failed",
        rationale="patch did not apply",
        provenance="pump",
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    terminal = _events(session_dir)[0]["ext"]["proposals"][0]["terminal"]
    assert terminal["disposition"] == "dropped"
    assert terminal["reason"] == "apply_failed"


def test_critic_denial_records_the_review_and_the_drop(session_dir: Path):
    """A denial is a review on the proposal plus a terminal disposition."""
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    pending = SimpleNamespace(
        payload={"framework_agent_candidate_id": "cand-9", "batch_id": "b1"},
        action_name="integrate_patch",
    )
    asyncio.run(coord._record_framework_agent_critic_denied(pending, "touches the serving loop"))
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    proposal = _events(session_dir)[0]["ext"]["proposals"][0]
    # The Critic's own word for it, not the phase's past tense: one field
    # spelled two ways depending on which path wrote it cannot be selected on.
    assert proposal["critic_review"]["verdict"] == "reject"
    assert "serving loop" in proposal["critic_review"]["reason"]
    assert proposal["critic_review"]["outcome"]["denied"] is True
    assert proposal["terminal"]["disposition"] == "dropped"
    assert proposal["terminal"]["reason"] == "critic_denied"
    assert [step["step"] for step in proposal["lifecycle"]] == ["reviewed"]


def test_config_attempts_record_the_pair_and_the_verbatim_outcome(session_dir: Path):
    """Explore variants land as attempts with both ends of their pair.

    The anchor advances on every KEEP, so a gain without its own denominator
    cannot be added to anything -- and for a killed variant there is no gain to
    divide an anchor back out of. The outcome stays verbatim: the journal
    beside this collapses the non-plain-revert outcomes together.
    """
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    task = SimpleNamespace(task_id="t-exp-1", kind="explore", params={})
    result = {
        "round_id": "explore-001",
        "per_variant_outcomes": [
            {
                "variant_name": "v-keep",
                "outcome": "KEEP",
                "fingerprint": "fp1",
                "provenance": "llm_direct",
                "metrics": {"base_tput": 100.0, "tput": 112.0, "gain_pct": 12.0, "runtime_sec": 300.0},
                "variant": {"extra_server_args": "--foo 2", "extra_envs": {"BAR": "1"}},
            },
            {
                "variant_name": "v-killed",
                "outcome": "KILLED_OVERTIME",
                "fingerprint": "fp2",
                "provenance": "default_grid",
                "metrics": {"base_tput": 112.0, "estimated_output_throughput": 40.0},
                "variant": {},
            },
            {"variant_name": "v-dup", "outcome": "SKIPPED_DEDUP", "fingerprint": "fp3"},
        ],
    }
    asyncio.run(coord._fact_write_hook(task=task, result=result, kept=True))
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    attempts = {row["fingerprint"]: row for row in _events(session_dir)[0]["ext"]["attempts"]}
    # The deduped variant was never measured, so it is not in the funnel.
    assert set(attempts) == {"fp1", "fp2"}

    keep = attempts["fp1"]
    assert keep["arm"] == "config"
    assert keep["round_id"] == "explore-001"
    assert keep["measurement"]["before_tput"] == 100.0
    assert keep["measurement"]["after_tput"] == 112.0
    assert keep["adopted"] is True
    assert keep["attribution_eligible"] is True
    assert keep["config_delta"]["extra_server_args"] == "--foo 2"

    killed = attempts["fp2"]
    assert killed["outcome"] == "KILLED_OVERTIME"
    assert killed["adopted"] is False
    # An anchor with nothing measured against it: no gain to divide out, which
    # is exactly the row a back-solved anchor could not represent.
    assert killed["measurement"]["before_tput"] == 112.0
    assert killed["measurement"]["after_tput"] is None
    assert killed["attribution_eligible"] is False


def test_source_attempt_records_its_pair_gate_and_lifecycle_step(session_dir: Path):
    """An authored patch lands as the same uniform attempt row.

    Both arms are measured against whatever the session is serving, so one
    shape serves both and the ledger walks them with one reader.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.framework_agent_specialist_candidate_map = {"t-auth-1": "https://x/pr/1"}
    coord._open_framework_timeline()

    from types import SimpleNamespace

    task = SimpleNamespace(
        task_id="t-int-1",
        kind="integrate_patch",
        params={
            "framework_agent_authoring": True,
            "specialist_task_id": "t-auth-1",
            "framework_agent_candidate_id": "https://x/pr/1",
            "audit_step": "author_via_specialist",
            "lever_kind": "upstream_pr",
        },
    )
    coord._record_framework_agent_authored_outcome(
        task=task,
        result={
            "status": "kept",
            "base_tput": 100.0,
            "output_throughput": 108.0,
            "delta_pct": 8.0,
            "accuracy_pass": True,
            "accuracy_value": 0.83,
            "accuracy_reference": 0.80,
            "target_files": ["vllm/attention.py"],
            "reason": "above the floor",
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    ext = _events(session_dir)[0]["ext"]
    attempt = ext["attempts"][0]
    assert attempt["arm"] == "source"
    assert attempt["proposal_ref"] == "https://x/pr/1"
    assert attempt["measurement"] == {
        "before_tput": 100.0,
        "after_tput": 108.0,
        "gain_pct": 8.0,
        "runtime_sec": None,
        "estimated_output_throughput": None,
    }
    assert attempt["accuracy"]["passed"] is True
    assert attempt["adopted"] is True
    assert attempt["attribution_eligible"] is True
    assert attempt["target_files"] == ["vllm/attention.py"]
    assert [gate["gate"] for gate in attempt["gates"]] == ["accuracy"]

    proposal = ext["proposals"][0]
    assert proposal["attempt_refs"] == ["t-int-1"]
    assert ("attempted", "t-auth-1") in [(s["step"], s["run_ref"]) for s in proposal["lifecycle"]]
    assert proposal["terminal"]["disposition"] == "attempted"


def test_absent_accuracy_gate_writes_no_gate_row(session_dir: Path):
    """A gate that did not run is absent, not failed."""
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    coord._record_framework_agent_authored_outcome(
        task=SimpleNamespace(
            task_id="t-int-2",
            kind="integrate_patch",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": "cand-2"},
        ),
        result={"status": "reverted", "base_tput": 100.0, "output_throughput": 99.0, "delta_pct": -1.0},
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    attempt = _events(session_dir)[0]["ext"]["attempts"][0]
    assert attempt["gates"] == []
    assert attempt["blocked_by"] is None
    assert attempt["accuracy"]["passed"] is None
    assert attempt["accuracy"]["required"] is None


def _propose_grid(coord: Coordinator, grid: list[dict[str, Any]]) -> str:
    """Propose one explore grid through the real intent seam.

    Returns:
        The proposal's message id, which is also its row's id.
    """
    import asyncio

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    # The execution-order gate holds ``explore`` until a baseline exists, which
    # this phase always runs after; without it the proposal never mints.
    if coord.shared_state.baseline_tput <= 0:
        coord.shared_state.baseline_tput = 100.0
    before = set(coord.state.pending_proposals)
    asyncio.run(
        coord._handle_propose_action(
            "orchestration",
            Intent(type=IntentType.PROPOSE_ACTION, payload={"action_name": "explore", "params": {"grid": grid}}),
        )
    )
    minted = set(coord.state.pending_proposals) - before
    assert len(minted) == 1, "the propose seam did not mint exactly one pending proposal"
    return minted.pop()


def test_a_proposed_grid_lands_with_its_producer(session_dir: Path):
    """The config arm's attempts have an upstream row to point at.

    The attempts already carried a ``proposal_ref``; until this row existed it
    referred to nothing, so the event could say six variants were benched and
    not what was proposed or who proposed it.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(
        coord,
        [
            {"provenance": "llm_direct", "extra_server_args": "--foo 2"},
            {"provenance": "llm_direct", "extra_server_args": "--foo 4"},
        ],
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    proposal = _events(session_dir)[0]["ext"]["proposals"][0]
    assert proposal["proposal_id"] == msg_id
    assert proposal["arm"] == "config"
    assert proposal["producer"] == "orchestration_agent"
    assert proposal["lever_kind"] == "config"
    # No dispatch stands behind it, and absence is the load-bearing fact.
    assert not proposal.get("run_ref")
    assert [step["step"] for step in proposal["lifecycle"]] == ["proposed"]
    # Nothing has resolved it yet: the key is absent rather than holding an
    # empty disposition, so unresolved cannot be read as settled-on-nothing.
    assert "terminal" not in proposal


def test_a_specialist_labelled_grid_names_its_domain(session_dir: Path):
    """A specialist's config is its domain's, not the orchestrator's."""
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    _propose_grid(coord, [{"provenance": "specialist:attention", "scope": "domain"}])
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    proposal = _events(session_dir)[0]["ext"]["proposals"][0]
    assert proposal["producer"] == "specialist"
    assert proposal["producer_ref"] == "attention"
    assert proposal["scope"] == "domain"


def test_a_seeded_grid_is_not_the_agents_idea(session_dir: Path):
    """The seed grid is its own producer, so a default is never read as a proposal."""
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    _propose_grid(coord, [{"provenance": "default_grid"}])
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    assert _events(session_dir)[0]["ext"]["proposals"][0]["producer"] == "seed_grid"


def test_a_mixed_grid_is_the_assemblers(session_dir: Path):
    """Only the orchestration agent assembles a grid from several sources.

    The mix is not lost by naming the assembler: every variant still carries
    its own label on its own attempt.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    _propose_grid(coord, [{"provenance": "specialist:attention"}, {"provenance": "default_grid"}])
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    proposal = _events(session_dir)[0]["ext"]["proposals"][0]
    assert proposal["producer"] == "orchestration_agent"
    assert proposal["producer_ref"] == ""


def test_a_rejected_grid_is_reviewed_and_dropped(session_dir: Path):
    """A denied grid stays on record as something pursued and dropped.

    Recorded at proposal time for exactly this case: a grid that never reaches
    a bench is invisible to anything that reads only the attempts.
    """
    import asyncio

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    asyncio.run(
        coord._handle_single_verdict(
            source="critic",
            pending=coord.state.pending_proposals[msg_id],
            verdict="reject",
            authored_verdict="reject",
            reasoning="every variant raises the memory floor",
            advisory=None,
            approved_variant_names=None,
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    proposal = _events(session_dir)[0]["ext"]["proposals"][0]
    assert proposal["critic_review"]["verdict"] == "reject"
    assert "memory floor" in proposal["critic_review"]["reason"]
    assert proposal["terminal"]["disposition"] == "dropped"
    # A denial is a review plus a disposition, as it is on the source arm; the
    # drop is not a step of its own.
    assert [step["step"] for step in proposal["lifecycle"]] == ["proposed", "reviewed"]


def _review(coord: Coordinator, msg_id: str, payload: dict[str, Any]) -> None:
    """Rule on a pending proposal through the real review_verdict seam."""
    import asyncio

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    asyncio.run(
        coord._handle_review_verdict(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": msg_id, **payload},
            ),
        )
    )


def test_a_review_records_the_grounds_the_critic_stated(session_dir: Path):
    """The whole ruling lands on the proposal, not just its verdict.

    A bare verdict cannot be audited: an ``advise`` that let a grid through on
    a stated risk and one that let it through on nothing are the same word, and
    the risk was the reason the round is worth reading.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    _review(
        coord,
        msg_id,
        {
            "verdict": "advise",
            "reasoning": "the batch size is untested at this sequence length",
            "confidence": 0.4,
            "failure_reason_code": "insufficient_evidence",
            "required_evidence": ["a short run at conc=4"],
            "risks": [{"severity": "medium", "risk": "may raise the memory floor"}],
            "advice_text": "run the shortest variant first",
            "alternative_action": "shrink the grid",
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    review = _events(session_dir)[0]["ext"]["proposals"][0]["critic_review"]
    assert review["verdict"] == "advise"
    assert review["effective_verdict"] == "advise"
    assert review["reviewer"] == "critic"
    assert review["confidence"] == 0.4
    assert review["failure_reason_code"] == "insufficient_evidence"
    assert review["required_evidence"] == ["a short run at conc=4"]
    assert review["risks"] == [{"severity": "medium", "risk": "may raise the memory floor"}]
    assert review["advice_text"] == "run the shortest variant first"
    assert review["alternative_action"] == "shrink the grid"
    assert review["outcome"]["materialized"] is True
    assert review["outcome"]["denied"] is False


def test_a_verdict_held_to_its_rule_keeps_both_readings(session_dir: Path):
    """The ruling and what the loop acted on are different facts.

    Reporting only what the Critic wrote says a grid was refused that in fact
    ran; reporting only what the loop did says the Critic approved it.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    _review(
        coord,
        msg_id,
        {
            "verdict": "reject",
            "reasoning": "specialist_quantitative_claim_violation: the payload carries predicted_gain_pct.",
            "failure_reason_code": "specialist_quantitative_claim_violation",
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    review = _events(session_dir)[0]["ext"]["proposals"][0]["critic_review"]
    assert review["verdict"] == "reject"
    assert review["effective_verdict"] == "advise"
    assert review["held_to_rule"] == "specialist_quantitative_claim_violation"
    # A held reject still dispatches, which is the whole point of the hold.
    assert review["outcome"]["materialized"] is True


def test_a_per_variant_review_records_every_variants_ruling(session_dir: Path):
    """A rejected variant has no attempt row, so the map is its only record.

    The collapse is deliberately lossy -- the grid proceeds on its approved
    subset -- so the summary verdict cannot say which variants were refused.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(
        coord,
        [
            {"provenance": "llm_direct", "variant_name": "chunked_prefill"},
            {"provenance": "llm_direct", "variant_name": "cuda_graph"},
        ],
    )
    _review(
        coord,
        msg_id,
        {
            "verdict_map": {
                "chunked_prefill": {"verdict": "approve", "rationale": "cheap to test"},
                "cuda_graph": {"verdict": "reject", "rationale": "known to hang on this build"},
            }
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    review = _events(session_dir)[0]["ext"]["proposals"][0]["critic_review"]
    rows = {row["variant_name"]: row for row in review["variants"]}
    assert rows["chunked_prefill"]["verdict"] == "approve"
    assert rows["cuda_graph"]["verdict"] == "reject"
    assert "hang" in rows["cuda_graph"]["reason"]
    # The grid still ran, on the subset that survived.
    assert review["effective_verdict"] == "approve"
    assert review["outcome"]["materialized"] is True


def test_a_ruling_the_critic_could_not_ground_says_so(session_dir: Path):
    """A proposal blocked by a missing manifest was never actually examined."""
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    _review(
        coord,
        msg_id,
        {
            "verdict": "needs_review",
            "source": "critic_unavailable",
            "reasoning": "required_context missing: model manifest",
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    review = _events(session_dir)[0]["ext"]["proposals"][0]["critic_review"]
    assert review["reviewer"] == "critic_unavailable"
    assert review["outcome"]["materialized"] is False
    assert review["outcome"]["reauthored"] is True


def test_a_patch_review_lands_on_the_candidate_it_judged(session_dir: Path):
    """The source arm identifies a proposal by candidate id, not message id.

    A review keyed on the bus message would open a second, near-empty row
    beside the candidate's own, and the candidate would read as never reviewed.
    """
    import asyncio
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ARM_SOURCE, PRODUCER_SPECIALIST

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()
    coord._framework_timeline().record_proposal(
        "cand-42",
        arm=ARM_SOURCE,
        producer=PRODUCER_SPECIALIST,
        source_ref="https://example.invalid/pr/1",
    )

    pending = SimpleNamespace(
        proposal_msg_id="msg-1",
        payload={"framework_agent_candidate_id": "cand-42", "params": {"task_id": "sp-7"}},
        action_name="integrate_patch",
        from_agent="orchestration",
        decided=False,
        verdict="",
    )
    asyncio.run(
        coord._handle_single_verdict(
            source="critic",
            pending=pending,
            verdict="reject",
            authored_verdict="reject",
            reasoning="touches the serving loop",
            payload={"verdict": "reject", "failure_reason_code": "serving_loop_touched"},
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    proposals = _events(session_dir)[0]["ext"]["proposals"]
    assert [row["proposal_id"] for row in proposals] == ["cand-42"]
    review = proposals[0]["critic_review"]
    assert review["verdict"] == "reject"
    # Still the candidate's own row, with the facts recorded when it was raised.
    assert proposals[0]["source_ref"] == "https://example.invalid/pr/1"
    # The subject the patch gate will consult this ruling under.
    assert review["outcome"]["patch_verdict_key"]


def test_the_review_carries_what_it_was_grounded_in(session_dir: Path):
    """The artifacts and the KB write land on the ruling they belong to.

    Recorded from the Critic's turn rather than the phase's recorder, since
    that is the only place a KB write's result comes back -- and filed onto the
    proposal so there is no per-turn stream to join back on a turn index that
    resume reuses.
    """
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import record_review_evidence

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.macro_cycle = 0
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    _review(coord, msg_id, {"verdict": "reject", "reasoning": "raises the memory floor"})
    record_review_evidence(
        macro_cycle=0,
        proposal_id=msg_id,
        artifacts={"review_path": "critic-workdir/3/review.json"},
        kb={
            "persist_requested": True,
            "priors": {"prior_count": 2, "referenced_in_verdict": True},
            "write": {"trigger": "review_verdict", "status": "dead_lettered", "detail": "kb unreachable"},
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    review = _events(session_dir)[0]["ext"]["proposals"][0]["critic_review"]
    # Merged onto the ruling the verdict seam already wrote, not beside it.
    assert review["verdict"] == "reject"
    assert review["artifacts"]["review_path"].endswith("review.json")
    assert review["kb"]["persist_requested"] is True
    assert review["kb"]["priors"]["prior_count"] == 2
    assert review["kb"]["write"]["status"] == "dead_lettered"


def test_evidence_for_a_cycle_with_no_event_is_dropped_quietly(session_dir: Path):
    """The Critic runs on every tick, including ticks that record no event."""
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import record_review_evidence

    record_review_evidence(
        macro_cycle=7,
        proposal_id="msg-nobody",
        artifacts={"review_path": "critic-workdir/1/review.json"},
    )

    assert _events(session_dir) == []


def test_review_subjects_resolve_a_candidate_to_its_own_row():
    """A source-arm ruling is filed under the candidate id, not the message id."""
    from hyperloom.orchestrator.roles.critic_agent import _review_subjects

    bundle = {
        "proposals": [
            {"msg_id": "msg-1", "payload": {"framework_agent_candidate_id": "cand-42"}},
            {"msg_id": "msg-2", "payload": {"params": {"framework_agent_candidate_id": "cand-43"}}},
            {"msg_id": "msg-3", "payload": {"params": {"grid": []}}},
        ]
    }
    # A grid keeps its message id, so it is absent rather than mapped to itself.
    assert _review_subjects(bundle) == {"msg-1": "cand-42", "msg-2": "cand-43"}


def test_measured_variants_settle_their_grid(session_dir: Path):
    """The grid settles on being measured, and its attempts link back to it."""
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    task = SimpleNamespace(task_id="t-exp-9", kind="explore", params={"proposal_msg_id": msg_id})
    asyncio.run(
        coord._fact_write_hook(
            task=task,
            result={
                "round_id": "explore-009",
                "per_variant_outcomes": [
                    {
                        "variant_name": "v1",
                        "outcome": "KEEP",
                        "fingerprint": "fp9",
                        "provenance": "llm_direct",
                        "metrics": {"base_tput": 100.0, "tput": 110.0, "gain_pct": 10.0},
                        "variant": {},
                    }
                ],
            },
            kept=True,
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    ext = _events(session_dir)[0]["ext"]
    proposal = ext["proposals"][0]
    assert proposal["terminal"]["disposition"] == "attempted"
    assert proposal["attempt_refs"] == [ext["attempts"][0]["attempt_id"]]
    assert [step["step"] for step in proposal["lifecycle"]] == ["proposed", "attempted"]


def test_a_measured_variant_keeps_the_name_a_reader_knows_it_by(session_dir: Path):
    """The fingerprint is the join key; the name is what identifies it to a person.

    Recording only the fingerprint left every config attempt labelled by a
    12-char digest, so the arm's own history was unreadable without joining
    back through a state ledger that is going away.
    """
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    msg_id = _propose_grid(coord, [{"provenance": "llm_direct"}])
    asyncio.run(
        coord._fact_write_hook(
            task=SimpleNamespace(task_id="t-exp-7", kind="explore", params={"proposal_msg_id": msg_id}),
            result={
                "round_id": "explore-007",
                "per_variant_outcomes": [
                    {
                        "variant_name": "chunked-prefill",
                        "outcome": "KEEP",
                        "fingerprint": "fp7",
                        "provenance": "llm_direct",
                        "metrics": {"base_tput": 100.0, "tput": 105.0, "gain_pct": 5.0},
                        "variant": {},
                    }
                ],
            },
            kept=True,
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    attempt = _events(session_dir)[0]["ext"]["attempts"][0]
    assert attempt["variant_name"] == "chunked-prefill"
    assert attempt["fingerprint"] == "fp7"


def test_every_applied_patch_is_recorded_not_just_the_primary(session_dir: Path):
    """One attempt can land several patches, and the rest are not recoverable.

    ``patch_path`` names the one the attempt was dispatched for; a candidate
    that needed a fixup applied more, and which ones landed is what a revert
    or a re-apply has to walk.
    """
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    coord._record_framework_agent_authored_outcome(
        task=SimpleNamespace(
            task_id="t-int-7",
            kind="integrate_patch",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": "https://x/pr/7"},
        ),
        result={
            "status": "kept",
            "base_tput": 100.0,
            "output_throughput": 108.0,
            "delta_pct": 8.0,
            "patch_path": "patches/pr-7.patch",
            "patches_applied": ["patches/pr-7.patch", "patches/pr-7-fixup.patch"],
        },
    )
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")

    attempt = _events(session_dir)[0]["ext"]["attempts"][0]
    assert attempt["patches_applied"] == ["patches/pr-7.patch", "patches/pr-7-fixup.patch"]
    # The dispatched one stays singular, so the pair is not read as a list of two attempts.
    assert attempt["patch_path"] == "patches/pr-7.patch"


def test_config_gates_and_stack_come_from_the_round_that_ruled(session_dir: Path):
    """The config arm's verdicts are recorded, not re-derived downstream.

    ``REVERT`` on its own cannot say which gate ended the arc, and the stack a
    variant ran on top of advances on every KEEP -- so both have to travel from
    the round that decided them rather than be reconstructed at write-back from
    the session's current config or by matching on a reason string.
    """
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    task = SimpleNamespace(task_id="t-exp-2", kind="explore", params={})
    asyncio.run(
        coord._fact_write_hook(
            task=task,
            result={
                "round_id": "explore-002",
                "per_variant_outcomes": [
                    {
                        "variant_name": "v-blocked",
                        "outcome": "REVERT",
                        "fingerprint": "fpg",
                        "provenance": "llm_direct",
                        "metrics": {"base_tput": 100.0, "tput": 104.0, "gain_pct": 4.0},
                        "variant": {},
                        "measured_against": {
                            "throughput": 100.0,
                            "extra_server_args": "--already-won 1",
                            "extra_envs": {"KEPT": "1"},
                        },
                        "gates": [
                            {"gate": "keep_threshold", "passed": True, "observed": 4.0, "threshold": 3.0},
                            {
                                "gate": "accuracy",
                                "passed": False,
                                "observed": 0.71,
                                "threshold": 0.80,
                                "reason": "accuracy_drop",
                            },
                        ],
                        "validation_basis": "accuracy_pass",
                    }
                ],
            },
            kept=False,
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    attempt = _events(session_dir)[0]["ext"]["attempts"][0]
    # The stack it launched on, which already carries an earlier KEEP.
    assert attempt["measured_against"]["extra_server_args"] == "--already-won 1"
    assert attempt["measured_against"]["extra_envs"] == {"KEPT": "1"}
    assert attempt["validation_basis"] == "accuracy_pass"
    # Evaluation order is preserved, so the arc reads in the order it ruled.
    assert [(g["gate"], g["passed"]) for g in attempt["gates"]] == [
        ("keep_threshold", True),
        ("accuracy", False),
    ]
    # It cleared the gain bar and died on accuracy; the outcome alone could not
    # have said which.
    assert attempt["blocked_by"] == "accuracy"


def test_an_ungated_keep_does_not_claim_an_accuracy_pass(session_dir: Path):
    """A session with no baseline accuracy gates nothing, and says so.

    ``unscored_keep_count`` is computed from this field today, so a KEEP that
    rested on throughput alone must not read like one an accuracy gate ruled.
    """
    import asyncio
    from types import SimpleNamespace

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord._open_framework_timeline()

    asyncio.run(
        coord._fact_write_hook(
            task=SimpleNamespace(task_id="t-exp-3", kind="explore", params={}),
            result={
                "round_id": "explore-003",
                "per_variant_outcomes": [
                    {
                        "variant_name": "v-unscored",
                        "outcome": "KEEP",
                        "fingerprint": "fpu",
                        "metrics": {"base_tput": 100.0, "tput": 110.0, "gain_pct": 10.0},
                        "variant": {},
                        "gates": [{"gate": "keep_threshold", "passed": True, "observed": 10.0, "threshold": 3.0}],
                        "validation_basis": "keep_verdict_unscored",
                    }
                ],
            },
            kept=True,
        )
    )
    coord._close_framework_timeline(exit_reason="optimize_budget_cap")

    attempt = _events(session_dir)[0]["ext"]["attempts"][0]
    assert attempt["validation_basis"] == "keep_verdict_unscored"
    assert attempt["adopted"] is True
    # The accuracy gate never ran, so it is absent rather than reported failed.
    assert [g["gate"] for g in attempt["gates"]] == ["keep_threshold"]
    assert attempt["blocked_by"] is None


def test_no_recorder_leaves_the_phase_alone(session_dir: Path):
    """Every seam is a no-op when the event was never opened.

    The close seam fires on any transition out of the phase, including ones
    from a leg that never opened an event -- a resume, or a session recorded
    before this event existed.
    """
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"

    coord._close_framework_timeline(exit_reason="optimize_budget_cap")
    assert coord._plateau_advisory_block() is not None
    assert _events(session_dir) == []
