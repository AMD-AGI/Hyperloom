# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The untested-proposal queue renderer and its injection into the prompt."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._proposal_identity import (
    controls_of,
    effective_fingerprint,
    normalize_proposal,
)
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_FRAMEWORK_AGENT,
    PHASE_NAMES,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


@pytest.fixture
def coord(session_dir) -> Coordinator:
    plan = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    backends: dict[str, Backend] = {
        name: MockBackend(plan, name=name) for name in ("orchestration", "critic", "robustness")
    }
    return Coordinator(session_dir, backends=backends)


def _state(rounds, *, cycle: int = 0, tested=None, gaps=None, current_best=None) -> SharedState:
    state = SharedState()
    state.macro_cycle = cycle
    state.specialist_rounds = list(rounds)
    state.explore_search = {"tested": dict(tested or {})}
    state.gaps = list(gaps or [])
    state.current_best = dict(current_best or {})
    return state


def _ledger_row(proposal) -> dict:
    """A tested row as the executor writes it: variant-own fields, server-args key."""
    fields = normalize_proposal(proposal)
    return {
        "extra_server_args": fields["extra_args"],
        "extra_envs": fields["extra_envs"],
        **controls_of(fields),
    }


def _fingerprint(proposal, **base) -> str:
    fields = normalize_proposal(proposal)
    return effective_fingerprint(fields["extra_args"], fields["extra_envs"], controls=controls_of(fields), **base)


def _round(proposals, *, cycle: int = 0, domain: str = "serving_specialist", gap: str = "") -> dict:
    return {
        "cycle": cycle,
        "domain": domain,
        "gap_canonical_id": gap,
        "proposal_set": list(proposals),
    }


def test_empty_queue_renders_nothing():
    assert _state([]).to_untested_proposals_summary() == ""
    assert _state([_round([{"name": "research-only", "reason": "read it"}])]).to_untested_proposals_summary() == ""


def test_only_the_current_cycle_is_rendered():
    state = _state(
        [
            _round([{"name": "old", "extra_args": "--old"}], cycle=0),
            _round([{"name": "new", "extra_args": "--new"}], cycle=1),
        ],
        cycle=1,
    )
    out = state.to_untested_proposals_summary()
    assert "new" in out
    assert "old" not in out


def test_a_round_missing_its_cycle_field_reads_as_zero():
    entry = _round([{"name": "legacy", "extra_args": "--legacy"}])
    entry.pop("cycle")
    assert "legacy" in _state([entry], cycle=0).to_untested_proposals_summary()
    assert _state([entry], cycle=1).to_untested_proposals_summary() == ""


def test_benched_proposals_are_dropped():
    proposal = {"name": "benched", "extra_args": "--a 1"}
    tested = {"whatever-key": dict(_ledger_row(proposal), outcome="REVERT")}
    assert _state([_round([proposal])], tested=tested).to_untested_proposals_summary() == ""


def test_a_keep_that_changes_the_stack_does_not_resurrect_benched_proposals():
    """The ledger key is stack-relative; matching on it would miss after a KEEP."""
    proposal = {"name": "with-removal", "extra_args": "--a 1", "remove_args": ["--v"]}
    round_start = _fingerprint(proposal)
    after_keep = _fingerprint(proposal, base_remove_args=["--b"], base_args_mode="replace")
    assert round_start != after_keep

    state = _state(
        [_round([proposal])],
        tested={round_start: dict(_ledger_row(proposal), outcome="REVERT")},
        current_best={"remove_args": ["--b"], "args_mode": "replace"},
    )
    assert state.to_untested_proposals_summary() == ""


def test_a_nameless_proposal_gets_a_stable_name_the_grid_parser_accepts():
    from hyperloom.orchestrator.actions.executors.explore import _grid_variants_from_payload

    entry = _round([{"extra_args": "--z"}], domain="comm_specialist")
    entry["task_id"] = "deadbeef99"
    line = _state([entry]).to_untested_proposals_summary()
    assert "comm-deadbeef-0" in line
    assert _grid_variants_from_payload([{"name": "comm-deadbeef-0", "extra_args": "--z"}])


def test_a_non_numeric_cycle_does_not_break_prompt_assembly():
    entry = _round([{"name": "v", "extra_args": "--a"}])
    entry["cycle"] = "bad"
    assert "v" in _state([entry], cycle=0).to_untested_proposals_summary()


def test_duplicate_fingerprints_collapse_across_rounds():
    proposal = {"name": "dup", "extra_args": "--a 1"}
    out = _state([_round([proposal]), _round([dict(proposal, name="dup-again")])]).to_untested_proposals_summary()
    assert out.count("•") == 1


def test_every_control_field_reaches_the_line():
    proposal = {
        "name": "coupled",
        "extra_args": "--a 1",
        "extra_envs": {"E": "2"},
        "remove_args": ["--drop"],
        "unset_envs": ["DROP_ENV"],
        "args_mode": "replace",
        "atomic": True,
        "reason": "splitting it OOMs",
    }
    line = _state([_round([proposal])]).to_untested_proposals_summary()
    assert "ATOMIC" in line
    assert "+args=--a 1" in line
    assert "+envs=E=2" in line
    assert "-args=--drop" in line
    assert "-envs=DROP_ENV" in line
    assert "mode=replace" in line
    assert "why=splitting it OOMs" in line


def test_a_removal_only_proposal_does_not_render_as_a_no_op():
    proposal = {"name": "drop-prefix-caching", "remove_args": ["--enable-prefix-caching"]}
    line = _state([_round([proposal])]).to_untested_proposals_summary()
    assert "-args=--enable-prefix-caching" in line
    assert "+args=" not in line


def test_ranking_is_gap_severity_then_recency():
    state = _state(
        [
            _round([{"name": "low-old", "extra_args": "--1"}], gap="gap.low"),
            _round([{"name": "high", "extra_args": "--2"}], gap="gap.high"),
            _round([{"name": "low-new", "extra_args": "--3"}], gap="gap.low"),
        ],
        gaps=[
            {"canonical_id": "gap.high", "severity": "high"},
            {"canonical_id": "gap.low", "severity": "low"},
        ],
    )
    names = [line.split()[1] for line in state.to_untested_proposals_summary().splitlines() if line.startswith("•")]
    assert names == ["high", "low-new", "low-old"]


def test_a_pruned_gap_sorts_last_but_is_not_dropped():
    state = _state(
        [
            _round([{"name": "orphan", "extra_args": "--1"}], gap="gap.gone"),
            _round([{"name": "known", "extra_args": "--2"}], gap="gap.low"),
        ],
        gaps=[{"canonical_id": "gap.low", "severity": "low"}],
    )
    out = state.to_untested_proposals_summary()
    names = [line.split()[1] for line in out.splitlines() if line.startswith("•")]
    assert names == ["known", "orphan"]
    assert "sev?" in out


def test_overflow_is_truncated_and_counted():
    proposals = [{"name": f"v{i}", "extra_args": f"--flag {i}"} for i in range(20)]
    out = _state([_round(proposals)]).to_untested_proposals_summary(max_entries=12)
    assert out.count("•") == 12
    assert "(+8 more not shown)" in out


@pytest.mark.parametrize("phase", PHASE_NAMES)
@pytest.mark.asyncio
async def test_the_block_is_injected_only_in_the_optimisation_phase(coord, phase):
    """Untested proposals feed explore grids, so only the phase that runs them
    is shown the queue. Parameterised over the live chain: a phase added
    without a decision here fails rather than silently inheriting ``False``."""
    expected = phase == PHASE_FRAMEWORK_AGENT
    coord.shared_state.phase = phase
    coord.shared_state.macro_cycle = 0
    coord.shared_state.specialist_rounds = [_round([{"name": "queued", "extra_args": "--a 1"}])]
    out = await coord._compose_prompt("orchestration")
    assert ("=== Untested proposals (current cycle) ===" in out) is expected


@pytest.mark.asyncio
async def test_the_block_survives_a_delta_turn(coord, monkeypatch):
    coord.shared_state.phase = PHASE_FRAMEWORK_AGENT
    coord.shared_state.macro_cycle = 0
    coord.shared_state.specialist_rounds = [_round([{"name": "queued", "extra_args": "--a 1"}])]
    seed = await coord._compose_prompt("orchestration")

    monkeypatch.setattr(type(coord.conversation), "_orchestration_conversational", lambda self: True)
    coord._orchestration_seeded = True
    delta = await coord._compose_prompt("orchestration")

    assert "=== Shared session state ===" in seed
    assert "=== Shared session state ===" not in delta
    assert "queued" in seed
    assert "queued" in delta
