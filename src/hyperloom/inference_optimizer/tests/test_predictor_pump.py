# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the predictor pump: when it asks, what reaches the queue, and how it is attributed."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.predictor import pump as pump_mod
from hyperloom.orchestrator.predictor.client import Action, Prediction
from hyperloom.orchestrator.predictor.prompt import queue_block
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def service(monkeypatch):
    """Stand in for ``predict``: record each request and return ``service.answer``."""
    box = SimpleNamespace(requests=[], answer=Prediction(), gate=None)

    def fake_predict(request, *, endpoint, timeout_sec):
        if box.gate is not None:
            box.gate.wait(5)
        box.requests.append(request)
        return box.answer

    monkeypatch.setattr(pump_mod, "predict", fake_predict)
    monkeypatch.setenv("HYPERLOOM_PREDICTOR_ENDPOINT", "http://p:8973")
    monkeypatch.setenv("HYPERLOOM_PREDICTOR_MODE", "active")
    return box


def _state(**fields: Any) -> SharedState:
    state = SharedState()
    state.phase, state.framework, state.baseline_tput = "FRAMEWORK_AGENT", "vllm", 1000.0
    state.current_best = {"tput": 1000.0, "extra_server_args": "--max-num-seqs 256", "extra_envs": {"VLLM_A": "1"}}
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def _answer(*actions: Action, votes: dict[int, int] | None = None) -> Prediction:
    candidates = [
        {"server_args": a.server_args, "envs": a.envs, "source_change": a.source_change}
        for i, a in enumerate(actions)
        for _ in range((votes or {}).get(i, 0))
    ]
    return Prediction(parsed=True, actions=actions, meta={"samples": 8, "candidates": candidates})


def _pump() -> pump_mod.PredictorPump:
    from hyperloom.orchestrator.phases.framework import predictor_budget

    return pump_mod.PredictorPump(budget=predictor_budget)


async def _ask_and_file(pump: pump_mod.PredictorPump, state: SharedState) -> None:
    await pump.step(state)
    while pump._inflight is not None:
        await asyncio.sleep(0.01)
        if pump._inflight.done():
            await pump.step(state)


def _predictor_rounds(state: SharedState) -> list[dict[str, Any]]:
    return [r for r in state.specialist_rounds if r.get("domain") == "primatune"]


async def _auto_bench_grid(state: SharedState) -> list[dict[str, Any]]:
    """The grid FRAMEWORK's auto-bench builds from the head of the queue."""
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    created: list[dict[str, Any]] = []

    async def _none():
        return []

    async def _create(spec):
        created.append(vars(spec))
        return SimpleNamespace(task_id="t1"), False

    tasks = SimpleNamespace(queued=_none, running=_none, create_or_return_existing=_create)
    dispatcher = SimpleNamespace(
        admission_frozen=False,
        dispatch_paused_for_phase_budget=lambda: False,
        registry_lanes_ttl=lambda kind: (["benchmark_lane"], 1800),
    )
    coord = SimpleNamespace(shared_state=state, tasks=tasks, dispatcher=dispatcher)
    coord.phase_framework = FrameworkPhase(coord)
    await coord.phase_framework._maybe_bench_untested_proposals()
    return created[0]["params"]["grid"]


async def test_an_active_answer_is_queued_ahead_of_specialists_and_benched_as_primatune(service):
    specialist = {"task_id": "s1", "domain": "serving_specialist", "gap_canonical_id": "g1", "cycle": 0,
                  "proposal_set": [{"name": "spec", "extra_args": "--enable-chunked-prefill"}]}  # fmt: skip
    state = _state(specialist_rounds=[specialist], gaps=[{"canonical_id": "g1", "severity": "high"}])
    service.answer = _answer(
        Action(server_args={"--kv-cache-dtype": "fp8", "--max-num-seqs": 256}, envs={"VLLM_A": "1", "SGLANG_B": "1"})
    )
    await _ask_and_file(_pump(), state)

    (round_,) = _predictor_rounds(state)
    assert (round_["round_id"], round_["priority"]) == ("c0-s0-r0", 1)
    # The champion's echo and the other stack's env are gone; only the change is left.
    assert [(r["extra_args"], r["extra_envs"]) for r in round_["proposal_set"]] == [("--kv-cache-dtype fp8", {})]
    assert state.predictor_asked_keys == ["c0-s0-r0"]
    assert [r["provenance"] for r in state.untested_proposal_rows()] == ["primatune", "specialist:serving"]
    assert [v["provenance"] for v in await _auto_bench_grid(state)] == ["primatune", "specialist:serving"]


async def test_a_specialist_proposal_cannot_choose_its_provenance(service):
    claimed = [
        {"name": "as-primatune", "extra_args": "--max-num-seqs 512", "provenance": "primatune"},
        {"name": "as-replay", "extra_args": "--block-size 32", "provenance": "geak_revalidate"},
    ]
    specialist = {"task_id": "s1", "domain": "serving_specialist", "cycle": 0, "proposal_set": claimed}
    state = _state(specialist_rounds=[specialist])
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    await _ask_and_file(_pump(), state)

    labels = {row["name"]: row["provenance"] for row in state.untested_proposal_rows()}
    assert labels == {
        "primatune-c0-s0-r0-0": "primatune",
        "as-primatune": "specialist:serving",
        "as-replay": "specialist:serving",
    }
    assert [v["provenance"] for v in await _auto_bench_grid(state)] == ["primatune"] + ["specialist:serving"] * 2
    offered = queue_block(state)
    assert "primatune-c0-s0-r0-0" in offered and "as-primatune" not in offered


async def test_votes_rank_rows_a_knob_family_takes_one_slot_and_six_is_the_cap(service):
    state = _state()
    actions = [
        Action(server_args={"--block-size": 16}),
        Action(server_args={"--block-size": 32}),
        Action(server_args={"--max-num-batched-tokens": 8192}),
        *(Action(server_args={f"--knob-{i}": True}) for i in range(6)),
    ]
    service.answer = _answer(*actions, votes={0: 1, 1: 3, 2: 2})
    await _ask_and_file(_pump(), state)

    rows = _predictor_rounds(state)[0]["proposal_set"]
    assert len(rows) == 6
    assert [r["extra_args"] for r in rows[:3]] == ["--block-size 32", "--max-num-batched-tokens 8192", "--knob-0"]
    assert [r["votes"] for r in rows[:2]] == [3, 2] and rows[0]["samples"] == 8
    assert rows[0]["reason"] == "predictor: --block-size"


async def test_proposals_that_restate_more_of_the_champion_vote_for_one_row(service):
    state = _state()
    fp8 = {"--kv-cache-dtype": "fp8"}
    service.answer = _answer(
        Action(server_args=fp8, rationale="Halves the bytes every decode step reads."),
        Action(server_args={"--block-size": 32}),
        Action(server_args={**fp8, "--max-num-seqs": 256}, envs={"VLLM_A": "1"}),
        Action(server_args=fp8, envs={"VLLM_A": "1"}),
        votes={0: 1, 1: 4, 2: 2, 3: 2},
    )
    await _ask_and_file(_pump(), state)

    rows = _predictor_rounds(state)[0]["proposal_set"]
    assert [(r["name"], r["extra_args"], r["votes"]) for r in rows] == [
        ("primatune-c0-s0-r0-0", "--kv-cache-dtype fp8", 5),
        ("primatune-c0-s0-r0-1", "--block-size 32", 4),
    ]
    assert rows[0]["reason"] == "PrimaTune 5/8: Halves the bytes every decode step reads."


async def test_a_rationale_is_the_rows_reason_and_the_queue_shows_it_at_length(service):
    long_reason = "Attention is 38% of GPU time and every one of its rows is memory-bound at single-digit efficiency."
    specialist = {
        "task_id": "s1",
        "domain": "serving_specialist",
        "cycle": 0,
        "proposal_set": [{"name": "spec", "extra_args": "--enable-chunked-prefill", "reason": long_reason}],
    }
    rationale = ("So the change to make is to set --kv-cache-dtype fp8, which halves the bytes every decode step "
                 "reads from the cache. The argument stands only if accuracy holds at fp8.")  # fmt: skip
    state = _state(specialist_rounds=[specialist])
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}, rationale=rationale), votes={0: 4})
    await _ask_and_file(_pump(), state)

    (row,) = _predictor_rounds(state)[0]["proposal_set"]
    assert row["reason"] == f"PrimaTune 4/8: {rationale}"
    predictor_line = queue_block(state).splitlines()[-1]
    specialist_line = state.to_untested_proposals_summary(skip_provenance="primatune").splitlines()[-1]
    assert predictor_line.endswith(f"why=PrimaTune 4/8: {rationale}")
    assert specialist_line.endswith(f"why={long_reason[:80].rstrip()}")


async def test_the_queue_offers_predictor_rows_to_orchestrations_own_grid(service):
    specialist = {"task_id": "s1", "domain": "serving_specialist", "cycle": 0,
                  "proposal_set": [{"name": "spec", "extra_args": "--enable-chunked-prefill"}]}  # fmt: skip
    specialists_only = _state(specialist_rounds=[specialist]).to_untested_proposals_summary()
    state = _state(specialist_rounds=[specialist])
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    await _ask_and_file(_pump(), state)

    header = " ".join(queue_block(state).split())
    assert "These rows are the exception" in header
    assert "into your next `explore` grid verbatim" in header and "`provenance: primatune`" in header
    assert "grid verbatim" not in specialists_only


@pytest.mark.parametrize(("accuracy", "stated"), [(0.938, True), (0.0, False)])
async def test_the_predictor_block_states_the_accuracy_gate_a_keep_passes(service, accuracy, stated):
    state = _state(baseline_accuracy=accuracy)
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    await _ask_and_file(_pump(), state)

    header = " ".join(queue_block(state).split())
    assert ("a KEEP needs accuracy no more than 0.05 below the baseline's 0.938" in header) is stated


async def test_a_predictor_row_already_in_a_grid_is_named_not_offered_again(service):
    state = _state()
    service.answer = _answer(
        Action(server_args={"--kv-cache-dtype": "fp8"}), Action(server_args={"--max-num-seqs": 128}), votes={0: 2, 1: 1}
    )
    await _ask_and_file(_pump(), state)
    held, offered = state.untested_proposal_rows()

    lines = queue_block(state, in_grids={held["fingerprint"]}).splitlines()
    assert [line.split()[1] for line in lines if line.startswith("• ")] == [offered["name"]]
    assert (
        lines[-1]
        == f"Predictor rows already in a queued or running explore grid, not to be copied again: {held['name']}"
    )


@pytest.fixture
async def coordinator(service, tmp_path, monkeypatch):
    """A real Coordinator in FRAMEWORK whose predictor has queued one row, ``--kv-cache-dtype fp8``."""
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    silent = ScriptedPlan(turns=[])
    coord = Coordinator(
        make_session_dir(),
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    try:
        state = coord.shared_state
        state.phase, state.framework, state.baseline_tput = "FRAMEWORK_AGENT", "vllm", 1000.0
        state.current_best = {"tput": 1000.0}
        service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 3})
        await _ask_and_file(_pump(), state)
        yield coord
    finally:
        await coord.stop()


async def _delegate_explore(coord: Any, grid: list[dict[str, Any]]) -> None:
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "explore", "idempotency_key": "e1", "params": {"grid": grid}},
    )
    await coord.router.handle_delegate("orchestration", intent)


async def test_an_explore_orchestration_copies_off_the_queue_is_credited_to_the_predictor(coordinator):
    grid = [
        {"name": "kv-fp8", "extra_args": "--kv-cache-dtype fp8", "provenance": "llm_direct"},
        {"name": "kv-fp8-async", "extra_args": "--kv-cache-dtype fp8 --async-scheduling", "provenance": "llm_direct"},
        {"name": "spec", "extra_args": "--kv-cache-dtype fp8 --block-size 32", "provenance": "specialist:serving"},
        {"name": "mine", "extra_args": "--max-num-seqs 128", "provenance": "llm_direct"},
    ]
    await _delegate_explore(coordinator, grid)

    (task,) = await coordinator.tasks.by_state("queued")
    variants = {v["name"]: v for v in task.params["grid"]}
    assert variants["kv-fp8"]["provenance"] == "primatune"
    # A variant that only contains a predictor row's delta keeps its author.
    assert variants["kv-fp8-async"]["provenance"] == "llm_direct"
    assert variants["spec"]["provenance"] == "specialist:serving"
    assert variants["mine"]["provenance"] == "llm_direct"


async def test_orchestrations_prompt_names_a_predictor_row_its_queued_grid_holds(coordinator):
    assert "• primatune-c0-s0-r0-0 " in (await coordinator.conversation.compose_prompt("orchestration")).text
    await _delegate_explore(coordinator, [{"name": "kv-fp8", "extra_args": "--kv-cache-dtype fp8"}])

    prompt = (await coordinator.conversation.compose_prompt("orchestration")).text
    assert "• primatune-c0-s0-r0-0 " not in prompt
    assert "not to be copied again: primatune-c0-s0-r0-0" in prompt


async def test_benched_queued_and_on_stack_proposals_are_not_queued_again(service):
    queued = {"task_id": "s1", "domain": "serving_specialist", "cycle": 0,
              "proposal_set": [{"name": "q", "extra_args": "--enable-chunked-prefill"}]}  # fmt: skip
    state = _state(
        specialist_rounds=[queued], explore_search={"tested": {"fp": {"extra_args": "--kv-cache-dtype fp8"}}}
    )
    service.answer = _answer(
        Action(server_args={"--kv-cache-dtype": "fp8"}),
        Action(server_args={"--enable-chunked-prefill": True}),
        Action(server_args={"--max-num-seqs": 256}, envs={"VLLM_A": "1"}),
    )
    await _ask_and_file(_pump(), state)

    assert _predictor_rounds(state) == []
    assert state.predictor_asked_keys == ["c0-s0-r0"]


async def test_shadow_logs_queues_nothing_and_does_not_ask_again(service, monkeypatch, caplog):
    monkeypatch.setenv("HYPERLOOM_PREDICTOR_MODE", "shadow")
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}))
    pump = _pump()
    with caplog.at_level("INFO", logger=pump_mod.__name__):
        await _ask_and_file(pump, state)
        await _ask_and_file(pump, state)

    assert len(service.requests) == 1 and state.specialist_rounds == []
    assert "predictor (shadow): key=c0-s0-r0 parsed=True" in caplog.text


@pytest.mark.parametrize(
    "change",
    [
        {"env": None},
        {"framework": "atom"},
        {"predictor_asked_keys": ["c0-s0-r0"]},
        {"phase_budget_pct": {"FRAMEWORK_AGENT": 0.0}},
    ],
)
async def test_nothing_is_asked_without_an_endpoint_or_an_open_framework_phase(service, monkeypatch, change):
    if "env" in change:
        monkeypatch.delenv("HYPERLOOM_PREDICTOR_ENDPOINT")
        change = {}
    state = _state(**change)
    asked_before = list(state.predictor_asked_keys)
    await _ask_and_file(_pump(), state)
    assert service.requests == [] and state.specialist_rounds == []
    assert state.predictor_asked_keys == asked_before


async def test_the_predictor_is_still_asked_once_the_source_arm_runs_dry(service):
    """The flag marks the source arm exhausted; FRAMEWORK stays open on the config arm the predictor feeds."""
    state = _state(framework_agent_phase_done=True)
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    await _ask_and_file(_pump(), state)
    assert len(service.requests) == 1 and len(_predictor_rounds(state)) == 1


async def test_one_request_in_flight_and_a_keep_opens_a_new_decision_point(service):
    state = _state()
    service.gate = threading.Event()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}))
    pump = _pump()
    await pump.step(state)
    await pump.step(state)
    service.gate.set()
    while not pump._inflight.done():
        await asyncio.sleep(0.01)
    await pump.step(state)
    assert len(service.requests) == 1 and state.predictor_asked_keys == ["c0-s0-r0"]

    state.optimization_stack = [{"candidate_extra_server_args": "--kv-cache-dtype fp8", "tput": 1100.0}]
    await _ask_and_file(pump, state)
    assert len(service.requests) == 2 and state.predictor_asked_keys == ["c0-s0-r0", "c0-s1-r0"]


async def test_a_decision_point_waits_for_the_reprofile_in_flight(service, monkeypatch):
    landed = _state(auto_roofline_pending_task_id="roofline-1")
    pump = _pump()
    await pump.step(landed)
    assert service.requests == [] and landed.predictor_asked_keys == []
    landed.auto_roofline_pending_task_id = ""
    landed.roofline_snapshots = [{}]
    await _ask_and_file(pump, landed)
    assert landed.predictor_asked_keys == ["c0-s0-r1"]

    stuck = _state(auto_roofline_pending_task_id="roofline-2")
    pump = _pump()
    await pump.step(stuck)
    monkeypatch.setattr(pump_mod, "MAX_PROFILE_WAIT_SEC", 0.0)
    await _ask_and_file(pump, stuck)
    assert stuck.predictor_asked_keys == ["c0-s0-r0"]


async def test_a_request_that_raises_is_not_retried_at_the_same_decision_point(service, monkeypatch):
    def _raising(request, *, endpoint, timeout_sec):
        service.requests.append(request)
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pump_mod, "predict", _raising)
    state = _state()
    pump = _pump()
    await pump.step(state)
    while not pump._inflight.done():
        await asyncio.sleep(0.01)
    with pytest.raises(RuntimeError):
        await pump.step(state)
    await pump.step(state)
    assert len(service.requests) == 1 and state.predictor_asked_keys == ["c0-s0-r0"]


async def test_an_answer_that_arrives_in_the_next_cycle_stays_out_of_its_queue(service):
    state = _state()
    service.gate = threading.Event()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}, source_change="fuse the norm"))
    pump = _pump()
    await pump.step(state)
    state.macro_cycle = 1
    service.gate.set()
    while not pump._inflight.done():
        await asyncio.sleep(0.01)
    await pump.step(state)

    (round_,) = _predictor_rounds(state)
    assert round_["cycle"] == 0
    assert state.untested_proposal_rows() == [] and queue_block(state) == ""


async def test_a_source_change_is_filed_as_a_mandate_on_its_round(service):
    state = _state()
    service.answer = _answer(Action(source_change="Fuse rmsnorm into the QKV GEMM\nin tuned_gemm.py"))
    await _ask_and_file(_pump(), state)

    (round_,) = _predictor_rounds(state)
    assert round_["proposal_set"] == []
    assert round_["mandate_id"] == "primatune-patch-c0-s0-r0"
    assert round_["mandate"].startswith("Fuse rmsnorm") and "\n" not in round_["mandate"]


async def test_the_framework_tick_steps_the_predictor(service, monkeypatch):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _noop)
    monkeypatch.setattr(FrameworkPhase, "_maybe_bench_untested_proposals", _noop)
    monkeypatch.setattr(FrameworkPhase, "_record_advisory_plateau", lambda self: None)
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}))
    coord = SimpleNamespace(
        shared_state=state,
        phase_internal=SimpleNamespace(
            maybe_enqueue_explore_research_scout=_noop,
            maybe_enqueue_config_plateau_static_recon=_noop,
        ),
        specialist_dispatch=SimpleNamespace(
            maybe_force_stalled_domain_specialist=_noop, ensure_custom_specialist=_noop
        ),
        record_exception=lambda **kwargs: pytest.fail(f"the tick raised: {kwargs}"),
    )
    phase = FrameworkPhase(coord)
    await phase.pump()
    while not phase._predictor._inflight.done():
        await asyncio.sleep(0.01)
    await phase.pump()

    assert len(service.requests) == 1
    assert [r["extra_args"] for r in _predictor_rounds(state)[0]["proposal_set"]] == ["--kv-cache-dtype fp8"]


def _entering_phase(monkeypatch, state: SharedState) -> Any:
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(FrameworkPhase, "_pump_framework_agent_phase", _noop)
    monkeypatch.setattr(FrameworkPhase, "_open_framework_timeline", lambda self: None)
    coord = SimpleNamespace(
        shared_state=state,
        phase_macro_cycle=SimpleNamespace(on_cycle_start_reprofile=_noop),
        specialist_dispatch=SimpleNamespace(ensure_custom_specialist=_noop),
    )
    return FrameworkPhase(coord)


async def test_entering_framework_files_the_answer_before_orchestrations_first_turn(service, monkeypatch):
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    await _entering_phase(monkeypatch, state).on_enter_framework(SimpleNamespace(from_phase="ENABLEMENT"))

    assert len(service.requests) == 1 and service.requests[0]["phase"]["phase"] == "EXPLORE"
    (round_,) = _predictor_rounds(state)
    assert [row["extra_args"] for row in round_["proposal_set"]] == ["--kv-cache-dtype fp8"]


@pytest.mark.parametrize(
    ("mode", "endpoint", "asks"),
    [("shadow", "http://p:8973", 1), ("off", "http://p:8973", 0), ("active", None, 0)],
)
async def test_framework_entry_waits_only_when_the_answer_would_be_queued(service, monkeypatch, mode, endpoint, asks):
    monkeypatch.setenv("HYPERLOOM_PREDICTOR_MODE", mode)
    if endpoint is None:
        monkeypatch.delenv("HYPERLOOM_PREDICTOR_ENDPOINT")
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    service.gate = threading.Event()
    phase = _entering_phase(monkeypatch, state)
    await asyncio.wait_for(phase.on_enter_framework(SimpleNamespace(from_phase="PRELUDE")), timeout=1.0)

    service.gate.set()
    if phase._predictor._inflight is not None:
        await phase._predictor._inflight
    assert (len(service.requests), _predictor_rounds(state)) == (asks, [])


async def test_entry_asks_at_its_own_decision_point_after_filing_an_earlier_answer(service, monkeypatch):
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    phase = _entering_phase(monkeypatch, state)
    await phase._predictor_pump().step(phase.shared_state)
    # Answered while FRAMEWORK was away, so nothing filed it.
    await phase._predictor._inflight
    state.macro_cycle = 1
    await phase.on_enter_framework(SimpleNamespace(from_phase="SWEEP"))

    assert len(service.requests) == 2
    assert [(r["round_id"], r["cycle"]) for r in _predictor_rounds(state)] == [("c0-s0-r0", 0), ("c1-s0-r0", 1)]


async def test_a_filed_answer_is_the_phases_work_and_a_run_on_its_event(service, monkeypatch):
    from hyperloom.orchestrator.phases.machine_state import _framework_predicate_inputs
    from hyperloom.orchestrator.phases.exit_predicates import framework_exit
    from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_KERNEL

    state = _state(phase_started_unix=time.time() - 60)
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    phase = _entering_phase(monkeypatch, state)
    runs: list[tuple[str, dict[str, Any]]] = []
    phase._framework_timeline_recorder = SimpleNamespace(record_run=lambda run_id, **f: runs.append((run_id, f)))
    await phase.on_enter_framework(SimpleNamespace(from_phase="ENABLEMENT"))

    (round_,) = _predictor_rounds(state)
    assert "task_id" not in round_ and round_["completed_at"] >= round_["asked_at"]
    plateau, budget = _framework_predicate_inputs(state, now_unix=time.time())
    assert plateau["specialist_round_count"] == 1
    # The skip hint is honoured only once the phase did work.
    exit_ = framework_exit({"plateau": plateau, "budget": budget, "hint": ESCALATE_HINT_SKIP_TO_KERNEL})
    assert exit_ is not None and exit_[0] == "optimize_no_more_leverage"
    assert [(run_id, f["role"], f["status"], f["domain"], f["proposals_total"]) for run_id, f in runs] == [
        ("c0-s0-r0", "config", "succeeded", "primatune", 1)
    ]
    assert runs[0][1]["dispatched_at"] == round_["asked_at"]


async def test_a_slow_answer_does_not_hold_framework_entry_past_its_bound(service, monkeypatch):
    state = _state()
    service.answer = _answer(Action(server_args={"--kv-cache-dtype": "fp8"}), votes={0: 1})
    service.gate = threading.Event()
    monkeypatch.setattr(pump_mod, "MAX_ENTRY_WAIT_SEC", 0.05)
    phase = _entering_phase(monkeypatch, state)
    await phase.on_enter_framework(SimpleNamespace(from_phase="ENABLEMENT"))
    assert _predictor_rounds(state) == []

    service.gate.set()
    await phase._predictor._inflight
    await phase._predictor_pump().step(phase.shared_state)
    assert len(_predictor_rounds(state)) == 1


def test_primatune_is_its_own_producer():
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import producer_for_provenance

    assert producer_for_provenance("primatune") == ("primatune", "")
    assert producer_for_provenance("specialist:serving") == ("specialist", "serving")
