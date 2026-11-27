# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delegate idempotency, the denial's breakdown row and explore-grid admission."""

from __future__ import annotations

import re

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.task_registry import TaskSpec, task_dispatch_class
from hyperloom.orchestrator.tests.mock_backend import MockBackend, ScriptedPlan
from hyperloom.inference_optimizer.protocol.action_surfaces import FULL_ENABLED_ACTIONS
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
        },
    )


def _delegate(
    *,
    action: str = "target_analysis",  # proposable in the default PRELUDE phase
    params: dict | None = None,
    key: str | None = None,
) -> Intent:
    """A delegate; ``key`` is an ``idempotency_key`` the agent wrote, which names nothing."""
    payload: dict = {"action_name": action, "params": params or {"x": 1}}
    if key is not None:
        payload["idempotency_key"] = key
    return Intent(type=IntentType.DELEGATE, payload=payload)


async def _denied_rules(c) -> list[str]:
    obs = await c.bus.tail(topic="observation")
    return [m.payload.get("rule") for m in obs if m.payload.get("kind") == "policy_denied"]


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_a_repeated_delegate_whose_task_ended_starts_a_new_task_under_a_retry_key(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.router.handle_delegate("orchestration", _delegate())
        (first,) = await c.tasks.by_state("queued")
        await c.tasks.transition(first.task_id, "running", evidence={})
        await c.tasks.transition(first.task_id, "succeeded", evidence={})

        await c.router.handle_delegate("orchestration", _delegate())

        (second,) = await c.tasks.by_state("queued")
        assert second.idempotency_key == f"{first.idempotency_key}-retry1"
        assert await _denied_rules(c) == []
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_repeated_delegate_whose_task_is_live_is_denied_without_a_new_task(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.router.handle_delegate("orchestration", _delegate())
        (first,) = await c.tasks.by_state("queued")

        await c.router.handle_delegate("orchestration", _delegate())

        assert [t.task_id for t in await c.tasks.by_state("queued")] == [first.task_id]
        assert await _denied_rules(c) == ["duplicate_idempotency_key_running"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_onto_a_live_coordinator_row_is_answered_as_a_duplicate(session_dir):
    """A key that names a live task is a duplicate whichever author created the row, and the row keeps its class."""
    c = _silent_coordinator(session_dir)
    try:
        held, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="target_analysis",
                params={"x": 1},
                idempotency_key=c.proposals.action_key("target_analysis", {"x": 1}),
                dispatch_class="coordinator",
            ),
        )
        await c.router.handle_delegate("orchestration", _delegate())

        assert [t.task_id for t in await c.tasks.by_state("queued")] == [held.task_id]
        assert task_dispatch_class(await c.tasks.get(held.task_id)) == "coordinator"
        assert await _denied_rules(c) == ["duplicate_idempotency_key_running"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_source_patch_without_git_root_is_refused_and_recorded_once(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.framework = "sglang"
        c.shared_state.framework_repo_path = ""
        c.shared_state.phase = "FRAMEWORK_AGENT"
        for _ in range(2):
            await c.router.handle_delegate(
                "orchestration",
                _delegate(action="specialist", params={"domain": "serving_specialist"}),
            )

        assert await c.tasks.by_state("queued") == []
        failures = [(row["action"], row["task_id"], row["error_class"]) for row in c.shared_state.last_action_failures]
        assert failures == [("specialist", "", "no_git_framework_source_root")]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_delegate_is_keyed_by_its_work_not_by_the_tick_or_a_key_the_agent_wrote(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.tick = 42
        await c.router.handle_delegate("orchestration", _delegate(params={"depth": 2}))
        (task,) = await c.tasks.by_state("queued")
        assert re.match(r"^target_analysis:[0-9a-f]{16}$", task.idempotency_key)

        c.shared_state.tick = 43
        await c.router.handle_delegate("orchestration", _delegate(params={"depth": 2}, key="analysis-take-2"))
        await c.router.handle_delegate(
            "orchestration", _delegate(params={"depth": 2, "idempotency_key": "analysis-take-3"})
        )

        assert [t.task_id for t in await c.tasks.by_state("queued")] == [task.task_id]
        assert "idempotency_key" not in task.params
        assert await _denied_rules(c) == ["duplicate_idempotency_key_running"] * 2

        await c.router.handle_delegate("orchestration", _delegate(params={"depth": 3}))
        assert len(await c.tasks.by_state("queued")) == 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_an_identical_specialist_delegate_is_the_task_already_in_flight(session_dir):
    """A specialist is keyed by what the agent asked for, not by the context its dispatch is warmed with."""
    c = _silent_coordinator(session_dir)
    try:
        state = c.shared_state
        state.phase, state.framework, state.baseline_tput = "FRAMEWORK_AGENT", "sglang", 800.0
        ask = {"scope": "freeform", "task_description": "find why prefill blocks decode"}
        await c.router.handle_intent("orchestration", _delegate(action="specialist", params=dict(ask)))
        (first,) = await c.tasks.by_state("queued")
        assert re.match(r"^specialist:[0-9a-f]{16}$", first.idempotency_key)

        # The repeat is warmed on a later tick, against a stack that has moved.
        state.tick = int(state.tick or 0) + 5
        state.current_best = {"tput": 900.0}
        await c.router.handle_intent("orchestration", _delegate(action="specialist", params=dict(ask)))

        assert [t.task_id for t in await c.tasks.by_state("queued")] == [first.task_id]
        assert await _denied_rules(c) == ["duplicate_idempotency_key_running"]

        await c.tasks.transition(first.task_id, "running", evidence={})
        await c.tasks.transition(first.task_id, "failed", evidence={})
        await c.router.handle_intent("orchestration", _delegate(action="specialist", params=dict(ask)))
        other = {**ask, "task_description": "find why decode stalls under load"}
        await c.router.handle_intent("orchestration", _delegate(action="specialist", params=other))

        retried, distinct = await c.tasks.by_state("queued")
        assert retried.idempotency_key == f"{first.idempotency_key}-retry1"
        assert distinct.idempotency_key not in {first.idempotency_key, retried.idempotency_key}
        assert retried.params["current_tput"] == 900.0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_policy_denial_is_a_row_of_the_open_phase_event(session_dir):
    from hyperloom.inference_optimizer.breakdown.exporter import build
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseEntry, PhaseExit
    from hyperloom.inference_optimizer.session.session_binding import session_scope
    from hyperloom.orchestrator.phases.phase_predicates import workflow_predicate_inputs
    from hyperloom.orchestrator.policy.denial import PolicyDenied

    coordinator = _silent_coordinator(session_dir)
    coordinator.shared_state.phase = "PRELUDE"
    coordinator.shared_state.macro_cycle = 0
    try:
        with session_scope(session_dir):
            phase_event.record_entry(PhaseEntry(at=PhaseCycle("PRELUDE", 0), sequence=1))
            intent = Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "baseline"},
            )
            await coordinator.router.record_policy_denied(
                "orchestration",
                intent,
                PolicyDenied("denied", rule="phase_action_not_allowed", hint="wait"),
            )
            phase_event.record_exit(
                PhaseExit(
                    at=PhaseCycle("PRELUDE", 0),
                    to_phase="CLOSE",
                    reason="prelude_baseline_failed",
                    evidence={"predicate_inputs": workflow_predicate_inputs(coordinator.shared_state)},
                )
            )
        breakdown = build(session_dir)
    finally:
        await coordinator.stop()

    event = next(row for row in breakdown["timeline"] if row["type"] == "phase")
    denial = event["ext"]["denials"]["rows"][0]
    assert (denial["actor"], denial["action"], denial["rule"]) == (
        "orchestration",
        "baseline",
        "phase_action_not_allowed",
    )
    assert denial["proposal_msg_id"] is None


# Mission summary: the stack-changed warning
def test_the_stack_changed_hint_names_an_action_orchestration_is_offered():
    """Orchestration echoes the named action back as ``action_name``; a retired name would only earn a policy denial."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate", "kernel_id": "k1"}],
    )
    hint = s.to_mission_summary().split("stack changed", 1)[1].splitlines()[0]
    named = re.findall(r"`(\w+)`", hint)
    assert named
    assert set(named) <= set(FULL_ENABLED_ACTIONS)


def test_mission_summary_flags_a_stack_that_outgrew_its_validation():
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate_patch", "variant_name": "p1"}],
        cumulative_gain_validated_stack_len=0,
    )
    text = s.to_mission_summary()
    assert "stack changed since last validation" in text


def _explore(*variants: tuple[str, str], key: str | None = None) -> Intent:
    grid = [{"name": name, "extra_args": args} for name, args in variants]
    return _delegate(action="explore", key=key, params={"grid": grid})


async def _explore_denials(c) -> list[dict]:
    obs = await c.bus.tail(topic="observation")
    return [
        m.payload
        for m in obs
        if m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == "explore_grid_already_covered"
    ]


async def _queued_grids(c) -> list[list[str]]:
    """The variant names of each queued explore round, oldest first."""
    return [[v["name"] for v in t.params["grid"]] for t in await c.tasks.by_state("queued") if t.kind == "explore"]


@pytest.mark.asyncio
async def test_delegate_explore_grid_already_queued_under_other_names_is_denied(session_dir):
    """A second grid with the same launches under different variant names is not queued again."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2"), ("k", "--k 0")))
        await c.router.handle_delegate("orchestration", _explore(("folded_again", "--folded 2"), ("k2", "--k 0")))
        assert await _queued_grids(c) == [["folded", "k"]]
        assert len(await _explore_denials(c)) == 1

        # One genuinely new variant admits the grid.
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2"), ("new", "--new 1")))
        assert await _queued_grids(c) == [["folded", "k"], ["folded", "new"]]

        # A grid queued on an older stack does not cover the same launches on the current one.
        c.shared_state.baseline_tput = 820.0
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2"), ("k", "--k 0")))
        assert await _queued_grids(c) == [["folded", "k"], ["folded", "new"], ["folded", "k"]]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_kernel_overlay_grid_is_never_counted_as_queued(session_dir):
    """Two overlay variants share args and envs but not the kernel they load, so neither covers the other."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        for overlay in ("/overlays/a", "/overlays/b"):
            grid = [{"name": "patched", "extra_args": "--k 0", "overlay_pythonpath": overlay}]
            await c.router.handle_delegate("orchestration", _delegate(action="explore", params={"grid": grid}))
        queued = [t for t in await c.tasks.by_state("queued") if t.kind == "explore"]
        assert [t.params["grid"][0]["overlay_pythonpath"] for t in queued] == ["/overlays/a", "/overlays/b"]
        assert await _explore_denials(c) == []
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_grid_already_measured_on_this_stack_is_denied(session_dir):
    """A grid whose every launch already has a verdict on the current stack is not queued; on a moved stack it is."""
    from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint, workload_signature

    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        fp = canonical_fingerprint("--max-running-requests 48", {})
        c.shared_state.explore_search = {
            "tested": {
                fp: {
                    "fingerprint": fp,
                    "name": "max_running_48",
                    "extra_server_args": "--max-running-requests 48",
                    "extra_envs": {},
                    "outcome": "REVERT",
                    "workload_signature": workload_signature(),
                    "measured_against": {
                        "throughput": 800.0,
                        "extra_server_args": "",
                        "extra_envs": {},
                        "remove_args": [],
                        "unset_envs": [],
                        "args_mode": "append",
                    },
                }
            }
        }
        await c.router.handle_delegate("orchestration", _explore(("mrr_48", "--max-running-requests 48")))
        assert [t for t in await c.tasks.by_state("queued") if t.kind == "explore"] == []
        assert len(await _explore_denials(c)) == 1

        c.shared_state.baseline_tput = 820.0  # the anchor moved: the old verdict no longer speaks for this stack
        await c.router.handle_delegate("orchestration", _explore(("mrr_48", "--max-running-requests 48")))
        assert await _queued_grids(c) == [["mrr_48"]]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_grid_refused_on_throughput_is_admitted_once_the_bar_decays(session_dir):
    """A +0.7% REVERT answers a grid under cycle 0's 1.0% bar but not under cycle 1's lower one."""
    from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint, workload_signature

    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        fp = canonical_fingerprint("--max-running-requests 48", {})
        c.shared_state.explore_search = {
            "tested": {
                fp: {
                    "fingerprint": fp,
                    "name": "max_running_48",
                    "extra_server_args": "--max-running-requests 48",
                    "extra_envs": {},
                    "outcome": "REVERT",
                    "reason": "gain_below_threshold",
                    "graded_gain_pct": 0.7,
                    "workload_signature": workload_signature(),
                    "measured_against": {
                        "throughput": 800.0,
                        "extra_server_args": "",
                        "extra_envs": {},
                        "remove_args": [],
                        "unset_envs": [],
                        "args_mode": "append",
                    },
                }
            }
        }
        c.shared_state.macro_cycle = 0
        await c.router.handle_delegate("orchestration", _explore(("mrr_48", "--max-running-requests 48")))
        assert [t for t in await c.tasks.by_state("queued") if t.kind == "explore"] == []
        assert len(await _explore_denials(c)) == 1

        c.shared_state.macro_cycle = 1
        await c.router.handle_delegate("orchestration", _explore(("mrr_48", "--max-running-requests 48")))
        assert await _queued_grids(c) == [["mrr_48"]]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_round_is_judged_by_its_grid_not_by_the_key_it_names(session_dir):
    """The name the model gives a round decides nothing: a new grid under a queued round's name is a new round."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2"), key="explore-round-2"))
        await c.router.handle_delegate("orchestration", _explore(("mem_090", "--mem 0.90"), key="explore-round-2"))
        assert await _queued_grids(c) == [["folded"], ["mem_090"]]

        # The same grid again, under any name, adds nothing and is refused naming its variants.
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2"), key="explore-round-3"))
        assert await _queued_grids(c) == [["folded"], ["mem_090"]]
        assert [d["reason"] for d in await _explore_denials(c)] == [
            "delegate{action_name='explore'} adds nothing new: folded already measured, queued or running on the "
            "current stack"
        ]
        obs = await c.bus.tail(topic="observation")
        assert [m.payload.get("rule") for m in obs if m.payload.get("kind") == "policy_denied"] == [
            "explore_grid_already_covered"
        ]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_grid_is_admitted_while_rewrite_levers_await_attribution(session_dir, monkeypatch):
    """The executor adds pending rewrite levers to a grid, so a grid of measured variants still has work to run."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore(("folded", "--folded 2")))
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.explore_prefilter.grid_payload.framework_lever_grid",
            lambda _state: [{"name": "lever_off", "extra_envs": {"HL_X": "0"}}],
        )
        await c.router.handle_delegate("orchestration", _explore(("folded_again", "--folded 2")))
        assert await _queued_grids(c) == [["folded"], ["folded_again"]]
        assert await _explore_denials(c) == []

        # The identical request is still the round already queued.
        await c.router.handle_delegate("orchestration", _explore(("folded_again", "--folded 2")))
        assert await _queued_grids(c) == [["folded"], ["folded_again"]]
        obs = await c.bus.tail(topic="observation")
        assert [m.payload.get("rule") for m in obs if m.payload.get("kind") == "policy_denied"] == [
            "duplicate_idempotency_key_running"
        ]
    finally:
        await c.stop()


def test_explore_grid_coverage_tells_a_variant_removal_from_the_base_one():
    """On a replacing base the runner applies only the variant's removal, so a removal the base also names is new."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.actions.executors.explore_prefilter.measured_variants import (
        explore_grid_already_covered,
    )

    base = {"base_tput": 800.0, "base_extra_args": "--x 1", "base_remove_args": ["--foo"], "base_args_mode": "replace"}
    kept = {"name": "keeps_foo", "extra_args": "--foo 1 --b 1", "args_mode": "replace"}
    removes = {**kept, "name": "removes_foo", "remove_args": ["--foo"]}
    queued = [SimpleNamespace(kind="explore", params={**base, "grid": [kept]})]
    assert explore_grid_already_covered({**base, "grid": [removes]}, queued, {}) is None
    assert explore_grid_already_covered({**base, "grid": [dict(kept, name="again")]}, queued, {}) is not None


def test_explore_grid_coverage_refuses_a_grid_made_only_of_known_dead_changes():
    """A grid whose every variant carries a change the framework refused at start launches nothing; it is refused."""
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
    from hyperloom.orchestrator.actions.executors.explore_prefilter.measured_variants import (
        explore_grid_already_covered,
    )

    aiter_on = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "1"}
    death = {
        "name": "v00",
        "outcome": "FAILED",
        "workload_signature": workload_signature(),
        "extra_server_args": "",
        "extra_envs": dict(aiter_on),
        "error_class": "server_init_dead",
        "error_excerpt": "ValueError: Unquantized MoE backend 'ROCm AITER' is unsupported",
    }
    ledger = {"tested": {"fp-v00": death}}
    base = {"base_tput": 800.0}
    sibling = {"name": "v02", "extra_envs": {**aiter_on, "HL_KNOB": "1"}}
    new = {"name": "v01", "extra_envs": {"HL_KNOB": "1"}}

    denied = explore_grid_already_covered({**base, "grid": [sibling]}, [], ledger)
    assert denied is not None and denied.rule == "explore_grid_already_covered"
    assert str(denied) == (
        "delegate{action_name='explore'} adds nothing new: v02 carrying a change the framework refused at start"
    )
    queued = [SimpleNamespace(kind="explore", params={**base, "grid": [new]})]
    mixed = explore_grid_already_covered({**base, "grid": [new, sibling]}, queued, ledger)
    assert str(mixed) == (
        "delegate{action_name='explore'} adds nothing new: v01 already measured, queued or running on the current "
        "stack; v02 carrying a change the framework refused at start"
    )
    assert explore_grid_already_covered({**base, "grid": [sibling, new]}, [], ledger) is None
    assert explore_grid_already_covered({**base, "source": "stack_revalidate", "grid": [sibling]}, [], ledger) is None
