# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The untested-proposal queue renderer and its injection into the prompt."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._canonical_fingerprint import canonical_fingerprint
from hyperloom.orchestrator.actions.executors._proposal_identity import (
    controls_of,
    effective_fingerprint,
    normalize_proposal,
)
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_CLOSE,
    PHASE_EXPLORE,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    PHASE_PRELUDE,
    PHASE_SWEEP,
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


def test_benched_fingerprints_are_dropped():
    proposal = {"name": "benched", "extra_args": "--a 1"}
    tested = {_fingerprint(proposal): {"outcome": "REVERT"}}
    assert _state([_round([proposal])], tested=tested).to_untested_proposals_summary() == ""


def test_the_tested_lookup_folds_in_the_stack_base_controls():
    """A naive fingerprint misses the ledger whenever the stack removes a flag."""
    proposal = {"name": "with-removal", "extra_args": "--a 1", "remove_args": ["--v"]}
    current_best = {"remove_args": ["--b"], "unset_envs": ["BE"]}
    effective = _fingerprint(proposal, base_remove_args=["--b"], base_unset_envs=["BE"])
    naive = canonical_fingerprint("--a 1", {}, remove_args=["--v"])
    assert effective != naive

    hidden = _state([_round([proposal])], tested={effective: {}}, current_best=current_best)
    assert hidden.to_untested_proposals_summary() == ""

    still_shown = _state([_round([proposal])], tested={naive: {}}, current_best=current_best)
    assert "with-removal" in still_shown.to_untested_proposals_summary()


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


@pytest.mark.parametrize(
    "phase,expected",
    [
        (PHASE_PRELUDE, False),
        (PHASE_FRAMEWORK_AGENT, False),
        (PHASE_EXPLORE, True),
        (PHASE_KERNEL_AGENT, False),
        (PHASE_SWEEP, False),
        (PHASE_CLOSE, False),
    ],
)
@pytest.mark.asyncio
async def test_the_block_is_injected_only_in_explore(coord, phase, expected):
    coord.shared_state.phase = phase
    coord.shared_state.macro_cycle = 0
    coord.shared_state.specialist_rounds = [_round([{"name": "queued", "extra_args": "--a 1"}])]
    out = await coord._compose_prompt("orchestration")
    assert ("=== Untested proposals (current cycle) ===" in out) is expected


@pytest.mark.asyncio
async def test_the_block_survives_a_delta_turn(coord, monkeypatch):
    coord.shared_state.phase = PHASE_EXPLORE
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
