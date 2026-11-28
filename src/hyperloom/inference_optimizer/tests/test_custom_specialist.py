# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator-defined ``custom_specialist`` type: prompts, gate, CLI, and its Coordinator-owned dispatch and bench."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.cli import (
    OrchestrationPromptContext,
    _build_orchestration_prompt,
    _custom_specialist_conflict,
    _load_custom_specialist_args,
    main,
)
from hyperloom.inference_optimizer.cli.custom_specialist import _apply_custom_specialist_resume
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.phases.dispatch_order import in_dispatch_order
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.dispatch import SpecialistDispatchCollaborator
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.specialists.runner import SpecialistRunner, SpecialistSubprocessConfig
from hyperloom.orchestrator.state.objective import build_objective
from hyperloom.orchestrator.state.shared_state import SharedState

_PROMPT = "Focus on RCCL all-reduce.\n<keep>```verbatim```</keep>"
_DESCRIPTION = "Tunes RCCL all-reduce for TP communication."


def _custom_domain():
    return replace(get_domain("custom_specialist"), description=_DESCRIPTION)


def _inputs(domain, **kw: Any) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        domain=domain,
        gap_canonical_id="gap.x",
        gap_symptom="s",
        framework="sglang",
        workspace_path="/tmp/w",
        **kw,
    )


def _headers(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


# Specialist prompt
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "research",
            ["## 0. MANDATE", "## 2. HARDWARE CONTEXT", "## 3. GAP STATEMENT", "## 10. NOTES FROM ORCHESTRATION"],
        ),
        (
            "patch",
            [
                "## 0. MANDATE",
                "## 2. HARDWARE CONTEXT",
                "## 3. GAP STATEMENT",
                "## 7. LOCAL SOURCE NAVIGATION HINT",
                "## 10. NOTES FROM ORCHESTRATION",
            ],
        ),
    ],
)
def test_custom_user_prompt_is_minimal(mode, expected):
    _, user = build_specialist_prompts(
        _inputs(_custom_domain(), custom_specialist_prompt=_PROMPT, mode=mode, notes="n")
    )
    assert _headers(user) == expected


def test_builtin_tagged_custom_specialist_renders_custom_focus_only_when_configured():
    serving = get_domain("serving_specialist")
    configured, _ = build_specialist_prompts(
        _inputs(serving, extra_focus_tags=("custom_specialist",), custom_specialist_prompt=_PROMPT)
    )
    unconfigured, _ = build_specialist_prompts(_inputs(serving, extra_focus_tags=("custom_specialist",)))
    generic_tag, _ = build_specialist_prompts(
        _inputs(serving, extra_focus_tags=("custom",), custom_specialist_prompt=_PROMPT)
    )
    untagged, _ = build_specialist_prompts(_inputs(serving))
    untagged_configured, _ = build_specialist_prompts(_inputs(serving, custom_specialist_prompt=_PROMPT))

    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in configured
    assert unconfigured == untagged
    assert generic_tag == untagged_configured


@dataclass
class _StubTask:
    task_id: str
    params: dict[str, Any]
    kind: str = "specialist"


async def test_runner_prepare_reads_the_custom_definition_from_shared_state(tmp_path):
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=tmp_path)
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    task = _StubTask(
        task_id="task-custom",
        params={"domain": "custom_specialist", "gap_canonical_id": "gap.x", "mode": "research"},
    )

    prep = await runner._prepare(RunnerContext(task=task, lease=None, extra={"shared_state": state}))

    assert prep.domain.description == _DESCRIPTION
    assert f"Description: {_DESCRIPTION}\n" in prep.system_prompt
    assert f"### Domain focus — custom_specialist\n\n{_PROMPT}\n" in prep.system_prompt


def _coordinator(session_dir: Path, state: SharedState):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.tests.mock_backend import MockBackend, MockTurn, ScriptedPlan

    state.save(session_dir)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=session_dir,
        backends={"orchestration": MockBackend(idle)},
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    return coord


async def test_custom_specialist_prompt_shows_the_experience_kb_block_its_warmup_records(tmp_path):
    block = "Experience exp-1: raise chunked prefill"
    evidence = SimpleNamespace(
        read_id="read-1",
        rendered_refs=({"id": "exp-1"},),
        status="completed",
        prompt_block=block,
        experiences=({"id": "exp-1"},),
    )
    coord = _coordinator(
        tmp_path, SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    )
    coord.__dict__["_experience_kb"] = SimpleNamespace(read_for_specialist=lambda state, params: evidence)
    params = {"domain": "custom_specialist", "gap_canonical_id": "gap.x", "mode": "research"}

    await coord.specialist_dispatch.warm_specialist_params(params)
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=tmp_path)
    prep = await runner._prepare(
        RunnerContext(task=_StubTask("task-custom", params), lease=None, extra={"shared_state": coord.shared_state}),
    )

    assert [r["read_id"] for r in coord.shared_state.experience_kb_injections] == ["read-1"]
    assert block in prep.user_prompt


# Orchestration prompt
@pytest.mark.parametrize("phase", ["ENABLEMENT", "FRAMEWORK_AGENT"])
def test_orchestration_prompt_never_offers_custom_specialist(phase):
    prompt = _build_orchestration_prompt(
        OrchestrationPromptContext(
            no_kernel=False, framework="sglang", objective=build_objective({"MAX_HOURS": 1}), max_minutes=60
        ),
        phase=phase,
    )
    emit = next(line for line in prompt.splitlines() if "delegate{action_name='specialist'" in line)
    domains = emit.split("domain=<one of ", 1)[1].split(">", 1)[0]

    assert domains == (
        "serving_specialist|kernel_switch_specialist|comm_specialist|compiler_specialist|system_specialist"
        "|speculative_decoding_specialist"
        "|candidate_discovery_specialist|research_scout_specialist|static_recon_specialist"
        "|framework_rewrite_specialist"
    )
    assert prompt.count("custom_specialist") == 0


# PolicyGate
def _delegate(params: dict[str, Any]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={"action_name": "specialist", "params": params})


@pytest.mark.parametrize(
    "params",
    [
        {"domain": "custom_specialist", "gap_canonical_id": "gap.x"},
        {"domain": "serving_specialist", "tags": ["custom_specialist"], "gap_canonical_id": "gap.x"},
        {"tags": ["custom_specialist"], "gap_canonical_id": "gap.x"},
        {"scope": "freeform", "domain": "custom_specialist", "task_description": "do it"},
    ],
)
def test_gate_denies_every_custom_specialist_delegate(params):
    with pytest.raises(PolicyDenied) as exc:
        PolicyGate().validate_intent("orchestration", _delegate(params))
    assert exc.value.rule == "specialist_custom_coordinator_owned"


def test_gate_allows_a_builtin_dispatch_with_the_generic_custom_tag():
    PolicyGate().validate_intent(
        "orchestration",
        _delegate({"domain": "serving_specialist", "tags": ["framework", "custom"], "gap_canonical_id": "gap.x"}),
    )


# CLI
def _ns(**kw: Any) -> argparse.Namespace:
    base = {
        "custom_specialist_prompt_file": None,
        "custom_specialist_description": None,
        "custom_specialist_only": False,
        "orch_prompt": None,
        "no_framework_agent": False,
        "research_lane_capacity": 4,
        "reset_state": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_load_custom_specialist_args_strips_and_sets_values(tmp_path):
    path = tmp_path / "p.md"
    path.write_text(f"\n{_PROMPT}\n\n", encoding="utf-8")
    args = _ns(custom_specialist_prompt_file=str(path), custom_specialist_description=f"  {_DESCRIPTION} ")

    _load_custom_specialist_args(args)

    assert (args.custom_specialist_prompt, args.custom_specialist_description) == (_PROMPT, _DESCRIPTION)


@pytest.mark.parametrize(
    ("content", "description", "message"),
    [
        (None, _DESCRIPTION, "must be passed together"),
        ("x", None, "must be passed together"),
        ("missing", _DESCRIPTION, "cannot read"),
        ("  \n", _DESCRIPTION, "is empty"),
        ("a" * (16 * 1024 + 1), _DESCRIPTION, "exceeds 16384 bytes"),
        ("x", "   ", "single non-empty line"),
        ("x", "two\nlines", "single non-empty line"),
        ("x", "d" * 201, "exceeds 200 characters"),
    ],
)
def test_load_custom_specialist_args_rejects_bad_input(tmp_path, content, description, message):
    path = None
    if content == "missing":
        path = str(tmp_path / "absent.md")
    elif content is not None:
        (tmp_path / "p.md").write_text(content, encoding="utf-8")
        path = str(tmp_path / "p.md")
    with pytest.raises(ValueError, match=message):
        _load_custom_specialist_args(_ns(custom_specialist_prompt_file=path, custom_specialist_description=description))


def test_resume_without_flags_keeps_the_stored_definition():
    state = SharedState(custom_specialist_prompt="old", custom_specialist_description="old desc")
    _apply_custom_specialist_resume(_ns(custom_specialist_prompt=""), state)
    assert (state.custom_specialist_prompt, state.custom_specialist_description) == ("old", "old desc")


def test_resume_with_a_new_definition_replaces_the_stored_one():
    state = SharedState(custom_specialist_prompt="old", custom_specialist_description="old desc")
    _apply_custom_specialist_resume(
        _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION), state
    )
    assert (state.custom_specialist_prompt, state.custom_specialist_description) == (_PROMPT, _DESCRIPTION)


@pytest.mark.parametrize(
    ("overrides", "state_kw", "expected"),
    [
        ({}, {}, ""),
        ({"orch_prompt": "x"}, {}, ""),
        (
            {"no_framework_agent": True},
            {},
            "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run",
        ),
        (
            {},
            {"framework_agent_phase_enabled": False},
            "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run",
        ),
        ({"research_lane_capacity": 0}, {}, "--research-lane-capacity 0 builds no specialist executor"),
        ({"reset_state": True}, {}, ""),
    ],
)
def test_custom_specialist_conflicts(overrides, state_kw, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, **state_kw)
    assert _custom_specialist_conflict(_ns(**overrides), state) == expected


@pytest.mark.parametrize("stored", [True, False])
def test_reset_state_conflicts_with_a_definition_passed_on_this_launch(stored):
    args = _ns(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, reset_state=True)
    state = SharedState(custom_specialist_prompt=_PROMPT if stored else "")
    assert _custom_specialist_conflict(args, state) == (
        "--reset-state wipes the custom specialist definition; reset first, then launch with the flags"
    )


def test_conflicts_are_ignored_when_no_custom_specialist_is_configured():
    args = _ns(orch_prompt="x", no_framework_agent=True, research_lane_capacity=0, reset_state=True)
    assert _custom_specialist_conflict(args, SharedState()) == ""


def test_custom_specialist_fields_roundtrip_and_default_on_old_state():
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    restored = SharedState.from_dict(state.to_dict())
    old = SharedState.from_dict({"session_id": "old", "custom_specialist_dispatched": True})

    assert (restored.custom_specialist_prompt, restored.custom_specialist_description) == (_PROMPT, _DESCRIPTION)
    assert (old.custom_specialist_prompt, old.custom_specialist_description) == ("", "")


# Coordinator-owned dispatch
def _custom_key(cycle: int = 0, prompt: str = _PROMPT) -> str:
    return f"custom-c{cycle}-{sha1(prompt.encode(), usedforsecurity=False).hexdigest()[:8]}"


@pytest.fixture
def custom_coord(tmp_path: Path, monkeypatch):
    """A Coordinator with a configured custom specialist, a git framework tree and a warm-up that names the framework."""
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 600
    source_root = tmp_path / "framework"
    (source_root / ".git").mkdir(parents=True)
    state.framework_repo_path = str(source_root)
    monkeypatch.setattr(
        SpecialistDispatchCollaborator,
        "warm_specialist_params",
        AsyncMock(side_effect=lambda params: params.setdefault("framework", "sglang")),
    )
    return _coordinator(tmp_path, state)


async def _dispatched(coord) -> tuple[dict[str, Any], str]:
    """The params and key of the one task the dispatch queued, which the Coordinator created itself."""
    from hyperloom.orchestrator.state.task_registry import task_dispatch_class

    (task,) = await coord.tasks.queued()
    assert (task.kind, task_dispatch_class(task)) == ("specialist", "coordinator")
    return task.params, task.idempotency_key


async def _custom_observations(coord) -> list[dict[str, Any]]:
    tail = await coord.bus.tail(topic="observation", n=50)
    return [m.payload for m in reversed(tail) if str(m.payload.get("kind") or "").startswith("custom_specialist")]


_BASE_PARAMS = {
    "domain": "custom_specialist",
    "scope": "domain",
    "source": "coordinator_internal",
    "reason": "custom_specialist",
    "framework": "sglang",
    # The owner a specialist that names a domain is frozen with when its task is created.
    "source_phase": "EXPLORE",
}


async def test_dispatch_targets_the_most_actionable_gap(custom_coord):
    state = custom_coord.shared_state
    state.upsert_gap({"canonical_id": "gap.tried"})
    state.upsert_gap({"canonical_id": "gap.untried"})
    state.append_gap_attempt("gap.tried", {"action": "explore", "variant_name": "v1", "outcome": "REVERT"})

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, key = await _dispatched(custom_coord)
    assert params == {**_BASE_PARAMS, "gap_canonical_id": "gap.untried"}
    assert key == _custom_key()
    assert await _custom_observations(custom_coord) == []
    # Orchestration sent nothing, so nothing is announced to it or charged to it.
    assert await custom_coord.bus.tail(topic="event") == []


async def test_dispatch_without_gaps_uses_the_session_gap(custom_coord):
    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, _ = await _dispatched(custom_coord)
    assert params == {
        **_BASE_PARAMS,
        "gap_canonical_id": "gap.custom.session",
        "gap_symptom": "No profiled gap; follow the operator-defined focus.",
        "gap_layer": "operator-defined",
    }


async def test_dispatch_key_names_the_cycle_and_the_definition(custom_coord):
    custom_coord.shared_state.macro_cycle = 2
    custom_coord.shared_state.custom_specialist_prompt = "other focus"

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    _, key = await _dispatched(custom_coord)
    assert key == _custom_key(2, "other focus")


@pytest.mark.parametrize(("prompt", "phase"), [("", "FRAMEWORK_AGENT"), (_PROMPT, "ENABLEMENT")])
async def test_dispatch_is_a_no_op(tmp_path, prompt, phase):
    state = SharedState(custom_specialist_prompt=prompt, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    coord.shared_state.phase = phase

    await coord.specialist_dispatch.ensure_custom_specialist()

    assert await coord.tasks.queued() == []
    assert coord.specialist_dispatch.refused_custom_key == ""


async def test_dispatch_falls_back_to_research_when_patches_are_impossible(custom_coord, monkeypatch):
    import hyperloom.orchestrator.specialists.runner as runner_mod
    from hyperloom.orchestrator.specialists.runner import NO_GIT_FRAMEWORK_SOURCE_ROOT

    # An installed or env-pointed framework tree on the host must not satisfy the preflight.
    monkeypatch.setattr(runner_mod, "resolve_framework_tree", lambda _framework: "")
    custom_coord.shared_state.framework_repo_path = ""

    await custom_coord.specialist_dispatch.ensure_custom_specialist()

    params, _ = await _dispatched(custom_coord)
    assert params["mode"] == "research"
    assert await _custom_observations(custom_coord) == [
        {"kind": "custom_specialist_research_only", "reason": NO_GIT_FRAMEWORK_SOURCE_ROOT}
    ]


async def test_dispatch_enqueues_one_task_per_cycle(tmp_path):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    dispatch = coord.specialist_dispatch

    await dispatch.ensure_custom_specialist()
    await dispatch.ensure_custom_specialist()
    first = [t.idempotency_key for t in await coord.tasks.queued()]
    coord.shared_state.macro_cycle = 1
    await dispatch.ensure_custom_specialist()

    queued = await coord.tasks.queued()
    assert first == [_custom_key()]
    assert [(t.idempotency_key, t.params["domain"], t.params["mode"]) for t in queued] == [
        (_custom_key(0), "custom_specialist", "research"),
        (_custom_key(1), "custom_specialist", "research"),
    ]


@pytest.mark.parametrize("only", [False, True])
async def test_the_coordinator_custom_task_passes_the_dispatch_time_gate_and_runs(tmp_path, only):
    state = SharedState(
        custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION, custom_specialist_only=only
    )
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    ran: list[str] = []

    async def _specialist(ctx) -> dict:
        ran.append(str(ctx.task.params.get("domain")))
        return {"runner_status": "succeeded"}

    coord.sub.register_executor("specialist", _specialist)
    await coord.specialist_dispatch.ensure_custom_specialist()
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await asyncio.wait_for(coord.dispatcher.wait_for_running_work(timeout=5), timeout=5)

    assert ran == ["custom_specialist"]


async def test_a_refused_custom_dispatch_is_recorded_once_per_cycle_and_not_as_orchestrations_denial(tmp_path):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    # Below the research-mode base, so admission refuses it.
    state.max_minutes = 5
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    for cycle in (0, 0, 0, 1, 1):
        coord.shared_state.macro_cycle = cycle
        await coord.specialist_dispatch.ensure_custom_specialist()

    refusals = [p for p in await _custom_observations(coord) if p["kind"] == "custom_specialist_refused"]
    assert [p["rule"] for p in refusals] == ["time_budget", "time_budget"]
    assert all("specialist" in p["reason"] for p in refusals)
    tail = await coord.bus.tail(topic="observation", n=50)
    assert [m.payload for m in tail if m.payload.get("kind") == "policy_denied"] == []
    assert await coord.tasks.queued() == []
    assert coord.specialist_dispatch.refused_custom_key == _custom_key(1)


# Dispatch order
def _task(task_id: str, kind: str, **params: Any) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, kind=kind, params=params)


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("FRAMEWORK_AGENT", ["custom", "operator-explore", "patch", "serving", "tag-only", "explore"]),
        ("CYCLE_END", ["custom", "operator-explore", "serving", "tag-only", "explore", "patch"]),
    ],
)
def test_operator_work_dispatches_first_in_every_phase(phase, expected):
    queued = [
        _task("serving", "specialist", domain="serving_specialist"),
        _task("tag-only", "specialist", domain="serving_specialist", tags=["custom_specialist"]),
        _task("explore", "explore", grid=[]),
        _task("patch", "integrate_patch"),
        _task("custom", "specialist", domain="custom_specialist"),
        _task("operator-explore", "explore", grid=[], operator_priority=True),
    ]

    assert [t.task_id for t in in_dispatch_order(queued, phase)] == expected


# Bench of custom proposals
class _BenchTasks:
    def __init__(self, queued=(), running=()) -> None:
        self._queued, self._running = list(queued), list(running)
        self.created: list[dict[str, Any]] = []
        self.known: dict[str, Any] = {}

    async def queued(self):
        return self._queued

    async def running(self):
        return self._running

    def queued_context_sync(self):
        return self._queued

    def running_context_sync(self):
        return self._running

    def with_key_prefix_sync(self, kind, key_prefix):
        return [
            SimpleNamespace(kind=c["kind"], params=c["params"], state="succeeded")
            for c in self.created
            if c["kind"] == kind and c["idempotency_key"].startswith(key_prefix)
        ]

    async def create_or_return_existing(self, spec):
        existing = any(c["idempotency_key"] == spec.idempotency_key for c in self.created)
        self.created.append(vars(spec))
        return SimpleNamespace(task_id=f"t-{len(self.created)}"), existing

    async def find_by_idempotency_key(self, key):
        if key in self.known:
            return self.known[key]
        return next((SimpleNamespace(state="succeeded") for c in self.created if c["idempotency_key"] == key), None)


def _bench_round(task_id: str, domain: str, *names: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": domain,
        "gap_canonical_id": "gap.s1",
        "cycle": 0,
        "proposal_set": [{"name": name, "extra_args": f"--{name}", "reason": "why"} for name in names],
    }


def _framework(*, rounds, queued=(), running=(), phase_done=False):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.phase = "FRAMEWORK_AGENT"
    state.baseline_tput = 100.0
    state.baseline_config_path = "/cfg.yaml"
    state.specialist_rounds = list(rounds)
    state.framework_agent_phase_done = phase_done
    dispatcher = SimpleNamespace(
        dispatch_paused_for_phase_budget=lambda: False,
        registry_lanes_ttl=lambda kind: (["benchmark_lane"], 1800),
    )
    coord = SimpleNamespace(
        shared_state=state,
        tasks=_BenchTasks(queued, running),
        dispatcher=dispatcher,
        specialist_dispatch=SimpleNamespace(refused_custom_key=""),
    )
    coord.phase_framework = FrameworkPhase(coord)
    return coord.phase_framework


_ROUNDS = [
    _bench_round("spec-serving", "serving_specialist", "serving-a"),
    _bench_round("spec-custom", "custom_specialist", "idx-4", "idx-2"),
]


@pytest.mark.parametrize(
    ("queued", "phase_done"),
    [([], False), ([_task("orch", "explore", grid=[])], False), ([], True)],
)
async def test_custom_proposals_are_benched_alone_ahead_of_every_other_gate(queued, phase_done):
    fp = _framework(rounds=_ROUNDS, queued=queued, phase_done=phase_done)

    await fp._maybe_bench_untested_proposals()

    (call,) = fp._coord.tasks.created
    assert [v["name"] for v in call["params"]["grid"]] == ["idx-4", "idx-2"]
    assert call["params"]["operator_priority"] is True
    assert call["idempotency_key"].startswith("auto-explore-custom-c0-")


async def test_a_queued_custom_bench_holds_back_another_one_and_the_general_bench():
    fp = _framework(rounds=_ROUNDS, queued=[_task("bench", "explore", grid=[], operator_priority=True)])

    await fp._maybe_bench_untested_proposals()

    assert fp._coord.tasks.created == []


async def test_the_general_bench_leaves_custom_proposals_out():
    rounds = [_bench_round("spec-serving", "serving_specialist", "serving-a")]
    fp = _framework(rounds=rounds)

    await fp._maybe_bench_untested_proposals()

    (call,) = fp._coord.tasks.created
    assert [v["name"] for v in call["params"]["grid"]] == ["serving-a"]
    assert "operator_priority" not in call["params"]


async def test_a_closed_candidate_track_still_benches_what_specialists_proposed():
    """The track closing ends the source arm's supply; a proposal already returned is still measured."""
    fp = _framework(rounds=[_bench_round("spec-serving", "serving_specialist", "serving-a")], phase_done=True)

    await fp._maybe_bench_untested_proposals()

    (call,) = fp._coord.tasks.created
    assert [v["name"] for v in call["params"]["grid"]] == ["serving-a"]


@pytest.mark.parametrize(
    ("rounds", "queued", "expected"),
    [
        (_ROUNDS, [], True),
        ([], [_task("custom", "specialist", domain="custom_specialist")], True),
        ([], [_task("bench", "explore", grid=[], operator_priority=True)], True),
        ([_ROUNDS[0]], [_task("tag-only", "specialist", tags=["custom_specialist"])], False),
    ],
)
async def test_operator_work_pending(rounds, queued, expected):
    fp = _framework(rounds=rounds, queued=queued)

    assert await fp.operator_work_pending() is expected


# FRAMEWORK exit waits for operator work
@pytest.mark.parametrize(
    ("next_phase", "held"),
    [
        (("CYCLE_END", "optimize_no_more_leverage", {"kernel_skipped": True}), True),
        (("KERNEL_AGENT", "optimize_no_more_leverage", {}), True),
        (("CYCLE_END", "optimize_phase_budget_exhausted", {"kernel_skipped": True}), False),
        (("CLOSE", "time_exhausted", {"terminal": True}), False),
    ],
)
def test_only_a_plateau_exit_from_framework_waits_for_operator_work(next_phase, held):
    from hyperloom.orchestrator.phases.machine import is_leverage_exit

    assert is_leverage_exit(next_phase) is held


@pytest.mark.parametrize(
    ("hint", "elapsed_min", "expected"),
    [("skip_to_sweep", 0, ("FRAMEWORK_AGENT", ["custom_specialist"])), ("", 600, ("KERNEL_AGENT", ["kernel_agent"]))],
)
async def test_framework_holds_a_leverage_exit_for_queued_custom_work_but_not_a_budget_exit(
    tmp_path, hint, elapsed_min, expected
):
    import time

    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_description=_DESCRIPTION)
    state.max_minutes = 120
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)
    state = coord.shared_state
    await coord.specialist_dispatch.ensure_custom_specialist()
    state.pending_escalate_hint = hint
    state.phase_started_unix = time.time() - elapsed_min * 60

    await coord.phase_machine.advance_phase_if_needed()

    assert (state.phase, [t.params.get("domain", t.kind) for t in await coord.tasks.queued()]) == expected
    if hint:
        coord.dispatcher.admission_frozen = True
        await coord.phase_machine.refresh_admission_freeze()
        assert not coord.dispatcher.admission_frozen


# HYPERLOOM_CUSTOM_SPECIALIST_ONLY
@pytest.fixture
def launch(tmp_path, monkeypatch):
    """Run ``main(["optimize", ...])`` up to ``_preflight``; return its exit code and the only mode it settled."""
    import os

    reached: list[bool] = []

    def _preflight(args):
        reached.append(args.custom_specialist_only)
        raise SystemExit(99)

    monkeypatch.setattr("hyperloom.inference_optimizer.cli._preflight", _preflight)
    prompt_file = tmp_path / "p.md"
    prompt_file.write_text(_PROMPT, encoding="utf-8")
    pair = ["--custom-specialist-prompt-file", str(prompt_file), "--custom-specialist-description", _DESCRIPTION]
    saved = dict(os.environ)
    os.environ.pop("HYPERLOOM_CUSTOM_SPECIALIST_ONLY", None)

    def _run(*argv: str, with_pair: bool = True) -> tuple[int, list[bool]]:
        with pytest.raises(SystemExit) as exc:
            main(["optimize", *(pair if with_pair else []), *argv])
        return exc.value.code, reached

    yield _run
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize(
    ("argv", "exported", "expected"),
    [
        (["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1"], None, (99, [True])),
        (["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=0"], "1", (99, [False])),
        ([], "1", (99, [True])),
        ([], None, (99, [False])),
    ],
)
def test_only_mode_reaches_the_launch_by_export_or_by_extra_env_pin(launch, argv, exported, expected):
    import os

    if exported is not None:
        os.environ["HYPERLOOM_CUSTOM_SPECIALIST_ONLY"] = exported

    assert launch(*argv) == expected


@pytest.mark.parametrize(
    ("argv", "with_pair", "message"),
    [
        (["--custom-specialist-only"], True, "unrecognized arguments: --custom-specialist-only"),
        (
            ["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1"],
            False,
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY requires --custom-specialist-prompt-file and "
            "--custom-specialist-description",
        ),
        (
            ["--extra-env", "HYPERLOOM_CUSTOM_SPECIALIST_ONLY=maybe"],
            True,
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY='maybe' is not a boolean",
        ),
        (["--research-lane-capacity", "0"], True, "--research-lane-capacity 0 builds no specialist executor"),
    ],
)
def test_a_bad_custom_specialist_launch_exits_before_preflight(launch, capsys, argv, with_pair, message):
    assert launch(*argv, with_pair=with_pair) == (2, [])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args_kw", "state_kw", "expected"),
    [
        (
            {"custom_specialist_only": True},
            {},
            "HYPERLOOM_CUSTOM_SPECIALIST_ONLY requires --custom-specialist-prompt-file and "
            "--custom-specialist-description",
        ),
        ({"custom_specialist_only": True}, {"custom_specialist_prompt": _PROMPT}, ""),
        (
            {"custom_specialist_only": True, "custom_specialist_prompt": _PROMPT},
            {},
            "",
        ),
    ],
)
def test_only_mode_needs_a_custom_specialist(args_kw, state_kw, expected):
    assert _custom_specialist_conflict(_ns(**args_kw), SharedState(**state_kw)) == expected


@pytest.mark.parametrize(
    ("passed", "stored", "expected"), [(True, False, True), (False, True, True), (False, False, False)]
)
def test_resume_turns_only_mode_on_and_never_off(passed, stored, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=stored)
    _apply_custom_specialist_resume(_ns(custom_specialist_prompt="", custom_specialist_only=passed), state)
    assert state.custom_specialist_only is expected


def _recorder(calls: list[str]):
    def _rec(name: str):
        async def _call(*args, **kwargs):
            calls.append(name)

        return _call

    return _rec


@pytest.mark.parametrize(
    ("only", "expected"),
    [
        (True, ["custom", "bench", "finish"]),
        (False, ["framework", "bench", "scout", "recon", "custom", "predictor"]),
    ],
)
async def test_only_mode_tick_runs_the_custom_work_alone(monkeypatch, only, expected):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    calls: list[str] = []
    _rec = _recorder(calls)
    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _rec("framework"))
    monkeypatch.setattr(FrameworkPhase, "_maybe_bench_untested_proposals", _rec("bench"))
    monkeypatch.setattr(FrameworkPhase, "_maybe_finish_custom_only_cycle", _rec("finish"))
    monkeypatch.setattr(FrameworkPhase, "_predictor_pump", lambda self: SimpleNamespace(step=_rec("predictor")))
    monkeypatch.setattr(FrameworkPhase, "_record_advisory_plateau", lambda self: None)
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    coord = SimpleNamespace(
        shared_state=state,
        phase_internal=SimpleNamespace(
            maybe_enqueue_explore_research_scout=_rec("scout"),
            maybe_enqueue_config_plateau_static_recon=_rec("recon"),
        ),
        specialist_dispatch=SimpleNamespace(ensure_custom_specialist=_rec("custom")),
        record_exception=lambda **kwargs: pytest.fail(f"the tick raised: {kwargs}"),
    )

    await FrameworkPhase(coord).pump()

    assert calls == expected


@pytest.mark.parametrize(
    ("only", "expected"),
    [(True, ["reprofile", "custom"]), (False, ["reprofile", "custom", "framework", "predictor"])],
)
async def test_framework_entry_enqueues_the_custom_specialist_before_any_other_work(monkeypatch, only, expected):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    calls: list[str] = []
    _rec = _recorder(calls)
    monkeypatch.setattr(FrameworkPhase, "_open_framework_timeline", lambda self: None)
    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _rec("framework"))
    monkeypatch.setattr(FrameworkPhase, "_predictor_pump", lambda self: SimpleNamespace(settle=_rec("predictor")))
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    coord = SimpleNamespace(
        shared_state=state,
        phase_macro_cycle=SimpleNamespace(on_cycle_start_reprofile=_rec("reprofile")),
        specialist_dispatch=SimpleNamespace(ensure_custom_specialist=_rec("custom")),
    )

    await FrameworkPhase(coord).on_enter_framework(SimpleNamespace(from_phase="PRELUDE"))

    assert calls == expected


async def test_only_mode_benches_no_proposal_but_the_custom_ones():
    fp = _framework(rounds=[_bench_round("spec-serving", "serving_specialist", "serving-a")])
    fp.shared_state.custom_specialist_only = True

    await fp._maybe_bench_untested_proposals()

    assert fp._coord.tasks.created == []


def _only_cycle(*, custom_state, rounds=(), queued=(), refused=""):
    from hyperloom.orchestrator.specialists.domains import custom_specialist_key

    fp = _framework(rounds=rounds, queued=queued)
    fp.shared_state.custom_specialist_only = True
    fp._coord.specialist_dispatch.refused_custom_key = refused
    if custom_state:
        fp._coord.tasks.known[custom_specialist_key(0, _PROMPT)] = SimpleNamespace(
            task_id="custom-task", state=custom_state
        )
    return fp


_SILENT_CUSTOM_ROUND = {**_bench_round("custom-task", "custom_specialist"), "proposal_set": []}


@pytest.mark.parametrize(
    ("kw", "hint"),
    [
        ({"custom_state": "succeeded", "rounds": [_SILENT_CUSTOM_ROUND]}, "skip_to_sweep"),
        ({"custom_state": "failed"}, "skip_to_sweep"),
        ({"custom_state": None}, ""),
        ({"custom_state": None, "refused": _custom_key(0)}, "skip_to_sweep"),
        ({"custom_state": None, "refused": _custom_key(1)}, ""),
        ({"custom_state": "running"}, ""),
        ({"custom_state": "succeeded"}, ""),
        ({"custom_state": "succeeded", "rounds": [_bench_round("custom-task", "custom_specialist", "idx-4")]}, ""),
        (
            {
                "custom_state": "succeeded",
                "rounds": [_SILENT_CUSTOM_ROUND],
                "queued": [_task("patch", "integrate_patch")],
            },
            "",
        ),
    ],
)
async def test_only_mode_leaves_framework_once_the_custom_cycle_is_done(kw, hint):
    fp = _only_cycle(**kw)

    await fp._maybe_finish_custom_only_cycle()

    assert fp.shared_state.pending_escalate_hint == hint


@pytest.mark.parametrize(
    ("phase", "only", "expected"),
    [
        ("FRAMEWORK_AGENT", True, []),
        ("FRAMEWORK_AGENT", False, ["orchestration"]),
        ("CYCLE_END", True, ["orchestration"]),
    ],
)
async def test_only_mode_framework_ticks_without_an_orchestration_turn(tmp_path, monkeypatch, phase, only, expected):
    import time

    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only, cycle_end_settled_cycle=0)
    coord = _coordinator(tmp_path, state)
    coord.shared_state.phase = phase
    turns: list[str] = []

    async def _noop(*args, **kwargs):
        return None

    async def _turn(**kwargs):
        turns.append("orchestration")

    monkeypatch.setattr(coord, "reactor_pass", _turn)
    monkeypatch.setattr(coord.reconciler, "run", _noop)
    monkeypatch.setattr(coord.phase_machine, "advance_phase_if_needed", _noop)
    monkeypatch.setattr(coord.dispatcher, "pump_dispatcher_once", _noop)
    monkeypatch.setattr(coord.enablement_lane, "pump_enablement_safely", _noop)
    coord.phase_machine.pump_table = {"CLOSE": _noop}
    coord._last_maintenance_ts = time.monotonic()

    await coord._tick_once()

    assert turns == expected


@pytest.mark.parametrize(("only", "expected"), [(True, 240 * 60), (False, 10 * 60)])
async def test_only_mode_custom_specialist_runs_to_the_phase_end_without_an_extension(tmp_path, only, expected):
    state = SharedState(custom_specialist_prompt=_PROMPT, custom_specialist_only=only)
    state.max_minutes = 30
    state.framework = "sglang"
    coord = _coordinator(tmp_path, state)

    await coord.specialist_dispatch.ensure_custom_specialist()

    (task,) = await coord.tasks.queued()
    assert coord.dispatcher._specialist_wall_budget_sec(params=task.params) == expected
