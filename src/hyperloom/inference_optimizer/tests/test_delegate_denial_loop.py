# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delegate idempotency + policy-denial ladder tests."""

from __future__ import annotations

import re

import pytest

from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "critic": MockBackend(silent, name="c"),
        },
    )


def _delegate(
    *,
    action: str = "target_analysis",  # proposable in the default PRELUDE phase
    params: dict | None = None,
    key: str | None = "dup-key-1",
) -> Intent:
    payload: dict = {"action_name": action, "params": params or {"x": 1}}
    if key is not None:
        payload["idempotency_key"] = key
    return Intent(type=IntentType.DELEGATE, payload=payload)


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_delegate_terminal_collision_appends_retry_suffix(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        first, _ = await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="target_analysis",
                params={"x": 1},
                idempotency_key="dup-key-1",
                dispatch_class="llm",
            ),
        )
        await c.tasks.transition(first.task_id, "running", evidence={})
        await c.tasks.transition(first.task_id, "succeeded", evidence={})

        await c.router.handle_delegate("orchestration", _delegate(key="dup-key-1"))
        queued = await c.tasks.by_state("queued")
        keys = {t.idempotency_key for t in queued}
        assert "dup-key-1-retry1" in keys
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_running_collision_denies_without_new_task(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.tasks.create_or_return_existing(
            TaskSpec(
                kind="target_analysis",
                params={"x": 1},
                idempotency_key="dup-key-run",
                dispatch_class="llm",
            ),
        )
        before = len(await c.tasks.by_state("queued"))
        await c.router.handle_delegate("orchestration", _delegate(key="dup-key-run"))
        after = len(await c.tasks.by_state("queued"))
        assert after == before
        obs = await c.bus.tail(topic="observation")
        denied = [
            m
            for m in obs
            if m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == "duplicate_idempotency_key_running"
        ]
        assert denied
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_source_patch_without_git_root_is_refused_and_recorded_once(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.framework = "sglang"
        c.shared_state.framework_repo_path = ""
        c.shared_state.phase = "FRAMEWORK_AGENT"
        for key in ("patch-no-root", "patch-no-root-again"):
            await c.router.handle_delegate(
                "orchestration",
                _delegate(action="specialist", key=key, params={"domain": "serving_specialist"}),
            )

        assert await c.tasks.by_state("queued") == []
        failures = [(row["action"], row["task_id"], row["error_class"]) for row in c.shared_state.last_action_failures]
        assert failures == [("specialist", "patch-no-root", "no_git_framework_source_root")]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_fallback_key_uses_tick_and_content_fingerprint(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.tick = 42
        await c.router.handle_delegate(
            "orchestration",
            _delegate(key=None, params={"grid": [{"name": "a"}]}),
        )
        queued = await c.tasks.by_state("queued")
        assert len(queued) == 1
        key = queued[0].idempotency_key
        assert re.match(
            r"^orchestration:target_analysis:t42:[0-9a-f]{10}$",
            key,
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_records_target_proposal_message_id(session_dir):
    from hyperloom.inference_optimizer.breakdown.exporter import build
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import PhaseCycle, PhaseEntry, PhaseExit
    from hyperloom.inference_optimizer.session.session_binding import session_scope
    from hyperloom.orchestrator.phases.machine_state import workflow_predicate_inputs
    from hyperloom.orchestrator.policy.gate import PolicyDenied

    coordinator = _silent_coordinator(session_dir)
    coordinator.shared_state.phase = "PRELUDE"
    coordinator.shared_state.macro_cycle = 0
    try:
        with session_scope(session_dir):
            phase_event.record_entry(PhaseEntry(at=PhaseCycle("PRELUDE", 0), sequence=1))
            intent = Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "baseline", "target_proposal_msg_id": "proposal-42"},
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
    assert denial["proposal_msg_id"] == "proposal-42"
    assert denial["rule"] == "phase_action_not_allowed"


@pytest.mark.asyncio
async def test_policy_denial_streak_records_streak_at_two(session_dir):
    """The denial streak is tracked via ``SharedState.policy_denial_streak`` as a count, not a priority lock."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c.router.record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="backends",
        )
        await c.router.record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="backends",
        )
        streak = c.shared_state.policy_denial_streak.get(
            "backends:duplicate_idempotency_key",
            0,
        )
        assert streak >= 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_no_longer_prunes_family_at_five(session_dir):
    """Long-run continuity: the streak >= 5 auto-prune_family reaction was removed; the family is NOT pruned."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="params", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(5):
            await c.router.record_policy_denied(
                "orchestration",
                intent,
                pd,
                action_name="params",
            )
        assert "params" not in c.shared_state.pruned_families
        assert (
            c.shared_state.policy_denial_streak.get(
                "params:duplicate_idempotency_key",
                0,
            )
            == 5
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_no_longer_stops_run_at_ten(session_dir):
    """Long-run continuity: the streak >= 10 ``policy_loop`` stop was removed; stop_reason stays unset."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(10):
            await c.router.record_policy_denied(
                "orchestration",
                intent,
                pd,
                action_name="backends",
            )
        assert not (c.shared_state.stop_reason or "").strip()
        assert (
            c.shared_state.policy_denial_streak.get(
                "backends:duplicate_idempotency_key",
                0,
            )
            == 10
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_successful_delegate_resets_policy_denial_streak(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(key="k-reset")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c.router.record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="target_analysis",
        )
        assert c.shared_state.policy_denial_streak.get("target_analysis:duplicate_idempotency_key") == 1
        await c.router.handle_delegate("orchestration", _delegate(key="fresh-key"))
        assert not any(k.startswith("target_analysis:") for k in c.shared_state.policy_denial_streak)
    finally:
        await c.stop()


# PolicyGate denial paths (analysis actions are never LLM-proposable)

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_FRAMEWORK_AGENT,
)
from hyperloom.orchestrator.policy.gate import (
    PolicyGate,
)
from hyperloom.inference_optimizer.protocol.action_surfaces import FULL_ENABLED_ACTIONS
from hyperloom.orchestrator.state.task_registry import TaskSpec


@pytest.fixture
def gate() -> PolicyGate:
    """Plain gate without an action catalogue / shared_state."""
    return PolicyGate(role_registry=default_role_registry())


# ``roofline`` and ``profile`` are Coordinator-enqueued; PolicyGate denies any propose/delegate/request that names
# either action.
_INTERNAL_ANALYSIS_ACTIONS = ("roofline", "profile")


# Supporting infrastructure parity
def test_phase_explore_allowlist_drops_legacy_actions():
    """The EXPLORE allowlist contains only the canonical action set."""
    assert PHASE_ALLOWED_ACTIONS[PHASE_FRAMEWORK_AGENT] == frozenset(
        {
            "explore",
            "specialist",
            "integrate_patch",
            "roofline",
            "profile",
        }
    )


def test_full_enabled_actions_still_contains_explore():
    """Sanity: ``explore`` / ``baseline`` stay enabled; ``recover`` is intentionally NOT enabled."""
    assert "explore" in FULL_ENABLED_ACTIONS
    assert "sweep" not in FULL_ENABLED_ACTIONS
    assert "recover" not in FULL_ENABLED_ACTIONS
    assert "baseline" in FULL_ENABLED_ACTIONS


def test_cli_real_executors_still_contains_explore():
    """Sanity: the canonical EXPLORE-phase executors stay registered."""
    from hyperloom.inference_optimizer.cli.executors import _REAL_EXECUTORS_FULL

    assert "explore" in _REAL_EXECUTORS_FULL
    assert "conc_sweep" in _REAL_EXECUTORS_FULL
    assert "sweep" not in _REAL_EXECUTORS_FULL
    assert "baseline" in _REAL_EXECUTORS_FULL


# Mission-summary + robustness prompt point at explore, not retired names
def test_dead_c_mission_summary_tag_points_at_explore():
    """The ``stack changed`` warning points at ``explore``, not the retired ``validate_stack``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate", "kernel_id": "k1"}],
    )
    text = s.to_mission_summary()
    assert "stack changed" in text
    assert "RUN `explore`" in text
    assert "validate_stack" not in text


def test_mission_summary_flags_a_stack_that_outgrew_its_validation():
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate_patch", "variant_name": "p1"}],
        cumulative_gain_validated_stack_len=0,
    )
    text = s.to_mission_summary()
    assert "stack changed since last validation" in text


# The Robustness prune_branch family list used to live in robustness.md, which was loaded every tick and discarded by
# the backend.


def _explore(key: str, *variants: tuple[str, str]) -> Intent:
    grid = [{"name": name, "extra_args": args} for name, args in variants]
    return _delegate(action="explore", key=key, params={"grid": grid})


async def _explore_denials(c) -> list[dict]:
    obs = await c.bus.tail(topic="observation")
    return [
        m.payload
        for m in obs
        if m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == "explore_grid_already_covered"
    ]


@pytest.mark.asyncio
async def test_delegate_explore_grid_already_queued_under_another_key_is_denied(session_dir):
    """A second grid with the same launches under a different key is not queued again."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore("round-a", ("folded", "--folded 2"), ("k", "--k 0")))
        await c.router.handle_delegate(
            "orchestration", _explore("round-b", ("folded_again", "--folded 2"), ("k2", "--k 0"))
        )
        queued = [t for t in await c.tasks.by_state("queued") if t.kind == "explore"]
        assert [t.idempotency_key for t in queued] == ["round-a"]
        assert len(await _explore_denials(c)) == 1

        # One genuinely new variant admits the grid.
        await c.router.handle_delegate(
            "orchestration", _explore("round-c", ("folded", "--folded 2"), ("new", "--new 1"))
        )
        queued = [t for t in await c.tasks.by_state("queued") if t.kind == "explore"]
        assert [t.idempotency_key for t in queued] == ["round-a", "round-c"]

        # A grid queued on an older stack does not cover the same launches on the current one.
        c.shared_state.baseline_tput = 820.0
        await c.router.handle_delegate("orchestration", _explore("round-d", ("folded", "--folded 2"), ("k", "--k 0")))
        queued = [t for t in await c.tasks.by_state("queued") if t.kind == "explore"]
        assert [t.idempotency_key for t in queued] == ["round-a", "round-c", "round-d"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_kernel_overlay_grid_is_never_counted_as_queued(session_dir):
    """Two overlay variants share args and envs but not the kernel they load, so neither covers the other."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        for key, overlay in (("overlay-a", "/overlays/a"), ("overlay-b", "/overlays/b")):
            grid = [{"name": "patched", "extra_args": "--k 0", "overlay_pythonpath": overlay}]
            await c.router.handle_delegate("orchestration", _delegate(action="explore", key=key, params={"grid": grid}))
        queued = [t for t in await c.tasks.by_state("queued") if t.kind == "explore"]
        assert [t.idempotency_key for t in queued] == ["overlay-a", "overlay-b"]
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
        await c.router.handle_delegate("orchestration", _explore("again", ("mrr_48", "--max-running-requests 48")))
        assert [t for t in await c.tasks.by_state("queued") if t.kind == "explore"] == []
        assert len(await _explore_denials(c)) == 1

        c.shared_state.baseline_tput = 820.0  # the anchor moved: the old verdict no longer speaks for this stack
        await c.router.handle_delegate("orchestration", _explore("moved", ("mrr_48", "--max-running-requests 48")))
        assert [t.idempotency_key for t in await c.tasks.by_state("queued") if t.kind == "explore"] == ["moved"]
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
        await c.router.handle_delegate("orchestration", _explore("cycle-0", ("mrr_48", "--max-running-requests 48")))
        assert [t for t in await c.tasks.by_state("queued") if t.kind == "explore"] == []
        assert len(await _explore_denials(c)) == 1

        c.shared_state.macro_cycle = 1
        await c.router.handle_delegate("orchestration", _explore("cycle-1", ("mrr_48", "--max-running-requests 48")))
        assert [t.idempotency_key for t in await c.tasks.by_state("queued") if t.kind == "explore"] == ["cycle-1"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_same_key_resend_is_a_duplicate_not_a_covered_grid(session_dir):
    """Re-sending a key whose grid is still queued is answered by the key rule, which tells the model to wait."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore("round-a", ("folded", "--folded 2")))
        await c.router.handle_delegate("orchestration", _explore("round-a", ("folded", "--folded 2")))
        assert [t.idempotency_key for t in await c.tasks.by_state("queued") if t.kind == "explore"] == ["round-a"]
        assert await _explore_denials(c) == []
        obs = await c.bus.tail(topic="observation")
        rules = [m.payload.get("rule") for m in obs if m.payload.get("kind") == "policy_denied"]
        assert rules == ["duplicate_idempotency_key_running"]

        # A different key for the same launches is still refused as covered.
        await c.router.handle_delegate("orchestration", _explore("round-b", ("folded", "--folded 2")))
        assert len(await _explore_denials(c)) == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_explore_grid_is_admitted_while_rewrite_levers_await_attribution(session_dir, monkeypatch):
    """The executor adds pending rewrite levers to a grid, so a grid of measured variants still has work to run."""
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.phase = "FRAMEWORK_AGENT"  # the phase that proposes explore
        c.shared_state.baseline_tput = 800.0
        await c.router.handle_delegate("orchestration", _explore("round-a", ("folded", "--folded 2")))
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.explore.framework_lever_grid",
            lambda _state: [{"name": "lever_off", "extra_envs": {"HL_X": "0"}}],
        )
        await c.router.handle_delegate("orchestration", _explore("round-b", ("folded", "--folded 2")))
        assert [t.idempotency_key for t in await c.tasks.by_state("queued") if t.kind == "explore"] == [
            "round-a",
            "round-b",
        ]
        assert await _explore_denials(c) == []
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
