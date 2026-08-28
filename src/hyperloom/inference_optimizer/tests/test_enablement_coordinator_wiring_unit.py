# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the Coordinator enablement wiring.

Covers the pure launch-log extractor, the ``build_mandate``-backed specialist
param builder, and the one-shot ``_maybe_enqueue_enablement_specialist`` gate.
"""

from __future__ import annotations

import types

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

from hyperloom.orchestrator.loop.coordinator import (
    Coordinator,
    PendingProposal,
    _extract_enablement_launch_log,
)


_MISSING_ARCH_LOG = (
    "Traceback (most recent call last):\n"
    '  File "/opt/sglang/server.py", line 42, in load\n'
    "ValueError: Model architecture 'Glm5ForCausalLM' is not supported"
)


# ---- _extract_enablement_launch_log (pure) ----


def test_extract_launch_log_joins_error_fields():
    payload = {"error": "boom", "stderr": "trace here", "status": "failed"}
    out = _extract_enablement_launch_log(payload)
    assert "boom" in out
    assert "trace here" in out


def test_extract_launch_log_handles_list_fields():
    payload = {"log_tail": ["line1", "", "line2"]}
    out = _extract_enablement_launch_log(payload)
    assert "line1" in out and "line2" in out


def test_extract_launch_log_empty_on_none_or_blank():
    assert _extract_enablement_launch_log(None) == ""
    assert _extract_enablement_launch_log({}) == ""
    assert _extract_enablement_launch_log({"error": "   "}) == ""


# ---- _build_enablement_specialist_params (uses build_mandate) ----


def _fake_self(**state_kw):
    state = types.SimpleNamespace(
        framework=state_kw.get("framework", "sglang"),
        model_name=state_kw.get("model_name", "zai-org/GLM-5"),
        gpu_type=state_kw.get("gpu_type", "mi300x"),
        enablement=EnablementRound(),
    )
    fake = types.SimpleNamespace(shared_state=state)
    # Bind the real discovery method so the builder path is exercised.
    fake._discover_enablement_candidate_refs = types.MethodType(Coordinator._discover_enablement_candidate_refs, fake)
    # Stub source-context read to empty so the builder path stays pure.
    fake._read_enablement_source_context = lambda _sig: ""
    # Stub weight-facts derivation to empty so the builder path stays pure.
    fake._derive_checkpoint_weight_facts = lambda _log: ""
    # No GPU pool, so dispatch degrades to the research-lane-only path.
    fake._framework_gpu_params = lambda: {}
    return fake


def _stub_enumerate(monkeypatch, cands):
    """Patch ``sources.enumerate_candidates`` to return ``cands`` (or raise)."""
    import hyperloom.agents.framework.sources as src

    def _fake_enum(_req):
        if isinstance(cands, Exception):
            raise cands
        return list(cands)

    monkeypatch.setattr(src, "enumerate_candidates", _fake_enum)


def test_build_params_actionable_failure_tags_enablement(monkeypatch):
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self()
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    assert params["domain"] == "enablement_specialist"
    assert params["source_phase"] == "ENABLEMENT"
    # Reuses FRAMEWORK authoring machinery + tags the objective.
    assert params["framework_agent_authoring"] is True
    assert params["enablement"] is True
    assert params["enablement_failure_kind"] == "missing_model_arch"
    assert params["enablement_before_signature"]["kind"] == "missing_model_arch"
    # Mandate body is rendered by _section_enablement_playbook; notes only carries per-dispatch context.
    assert params.get("notes", "") == ""  # fresh attempt with no stacked patches or build failure
    # Boot-origin: no eval carriers.
    assert "enablement_origin" not in params


def test_build_params_threads_eval_origin_carriers(monkeypatch):
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self()
    fake.shared_state.enablement.origin = "eval"
    fake.shared_state.enablement.accuracy_floor = 0.3
    fake.shared_state.enablement.probe_config_path = "/runs/baseline/materialized.yaml"
    fake.shared_state.enablement.eval_contract_fingerprint = "abc123"
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    assert params["enablement_origin"] == "eval"
    assert params["enablement_accuracy_floor"] == 0.3
    assert params["enablement_probe_config_path"] == "/runs/baseline/materialized.yaml"
    # The eval-contract fingerprint is no longer carried: nothing downstream
    # reads it. Correctness is judged from the candidate's own measurement.
    assert "enablement_eval_contract_fingerprint" not in params


_TRANSFORMERS_UNRECOGNIZED_LOG = (
    "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig\n"
    "  Value error, The checkpoint you are trying to load has model type "
    "`deepseek_v4` but Transformers does not recognize this architecture."
)


def test_build_params_seeds_no_deterministic_shared_venv_mutation(monkeypatch):
    """An arch-miss round must NOT auto-seed ANY shared-venv mutation.

    Both ``pip install -U vllm`` and ``pip install -U transformers`` were removed
    from the deterministic seed: an unpinned upgrade of the shared serving venv
    bypasses the LLM/Critic/ROCm guard and can brick serving (CUDA-wheel clobber
    of the ROCm vLLM/torch; transformers-major skew breaking vLLM's pin + the
    compiled tokenizers/hf-hub/numpy ABI). Environment/build acquisition is owned
    by the isolated, ROCm-safe targeted-build path and the specialist's own
    recorded setup_commands — never a blind Coordinator seed."""
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self(model_name="deepseek-ai/DeepSeek-V4-Flash")
    params = Coordinator._build_enablement_specialist_params(fake, _TRANSFORMERS_UNRECOGNIZED_LOG)
    assert params is not None
    assert params["enablement_failure_kind"] == "missing_model_arch"
    setup = params.get("enablement_setup_commands") or []
    # Regression guard: no unpinned serving-framework / transformers upgrade seeded.
    assert not any("vllm" in c for c in setup)
    assert not any("transformers" in c for c in setup)


def test_build_params_feeds_ranked_candidate_refs_into_mandate(monkeypatch):
    from hyperloom.agents.framework.models import Candidate

    cands = [
        Candidate(ref="PR:1", repo="r", title="minor perf tweak", html_url="http://x/1"),
        Candidate(
            ref="PR:2",
            repo="r",
            title="Add support to enable GLM architecture on ROCm",
            html_url="http://x/2",
        ),
    ]
    _stub_enumerate(monkeypatch, cands)
    fake = _fake_self()
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    # Enablement-intent PR ranks first and its html_url is threaded through.
    assert params["enablement_candidate_refs"][0] == "http://x/2"


def test_build_params_degrades_gracefully_when_discovery_raises(monkeypatch):
    _stub_enumerate(monkeypatch, RuntimeError("network down"))
    fake = _fake_self()
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    # Discovery failure -> repos-only mandate, no candidate refs.
    assert params["enablement_candidate_refs"] == []


def test_build_params_dispatches_even_for_unknown_failure(monkeypatch):
    """Q1: a non-blank UNKNOWN log still dispatches (kind is advisory, not a gate)."""
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self()
    params = Coordinator._build_enablement_specialist_params(
        fake, "some brand-new failure the rule table has never seen xyz"
    )
    assert params is not None
    assert params["enablement"] is True
    # Classified as unknown, but still handed to the specialist.
    assert params["enablement_failure_kind"] == "unknown"


def test_build_params_none_for_blank_log():
    fake = _fake_self()
    assert Coordinator._build_enablement_specialist_params(fake, "   ") is None


# ---- _maybe_enqueue_enablement_specialist (one-shot gate) ----


class _FakeTasks:
    def __init__(self, queued=None, running=None):
        self.created = []
        self._queued = list(queued or [])
        self._running = list(running or [])

    async def create_or_return_existing(self, **kwargs):
        self.created.append(kwargs)
        task = types.SimpleNamespace(task_id=f"spec-{len(self.created)}", state="queued")
        return task, False

    async def get(self, task_id: str):
        from hyperloom.orchestrator.state.task_registry import TaskNotFound

        for t in self._running:
            if getattr(t, "task_id", None) == task_id:
                return types.SimpleNamespace(task_id=task_id, state="running")
        for t in self._queued:
            if getattr(t, "task_id", None) == task_id:
                return types.SimpleNamespace(task_id=task_id, state="queued")
        raise TaskNotFound(task_id)

    async def queued(self):
        return list(self._queued)

    async def running(self):
        return list(self._running)


def _enqueue_self(**state_kw):
    state = types.SimpleNamespace(
        framework=state_kw.get("framework", "sglang"),
        model_name=state_kw.get("model_name", "zai-org/GLM-5"),
        gpu_type=state_kw.get("gpu_type", "mi300x"),
        # Admission is exercised separately; these cases target the dispatch machinery.
        enablement_mode=state_kw.get("enablement_mode", "all"),
        enablement=EnablementRound(
            succeeded=state_kw.get("enablement_succeeded", False),
            attempts=state_kw.get("enablement_attempts", 0),
            human_review_logged=state_kw.get("enablement_human_review_logged", []),
            kept_patches=state_kw.get("enablement_kept_patches", []),
            stall_streak=state_kw.get("enablement_stall_streak", 0),
            launch_log=state_kw.get("enablement_launch_log", _MISSING_ARCH_LOG),
            inflight_task_id=state_kw.get("enablement_inflight_task_id", ""),
            origin=state_kw.get("enablement_origin", ""),
            validation_pending=state_kw.get("enablement_validation_pending", False),
            probe_config_path=state_kw.get("enablement_probe_config_path", ""),
            accepted_config_path=state_kw.get("enablement_accepted_config_path", ""),
            accepted_config=state_kw.get("enablement_accepted_config", {}),
            accuracy_floor=state_kw.get("enablement_accuracy_floor", 0.0),
            eval_contract_fingerprint=state_kw.get("enablement_eval_contract_fingerprint", ""),
            active_runtime=state_kw.get("enablement_active_runtime", {}),
            revalidation_task_id=state_kw.get("enablement_revalidation_task_id", ""),
            revalidation_generation=state_kw.get("enablement_revalidation_generation", 0),
        ),
        baseline_tput=state_kw.get("baseline_tput", 0.0),
        baseline_failure_streak=state_kw.get("baseline_failure_streak", 1),
        tick=state_kw.get("tick", 0),
        stop_reason=state_kw.get("stop_reason", ""),
        save=lambda *a, **k: None,
    )
    # Minimal set_stop_reason shim.
    state.set_stop_reason = lambda v, **k: setattr(state, "stop_reason", str(v or ""))

    async def _warm(_params):
        return None

    observations: list = []

    async def _record_obs(_source, _topic, payload):
        observations.append(payload)

    fake = types.SimpleNamespace(
        shared_state=state,
        tasks=_FakeTasks(),
        session_dir="/tmp/session",
        _run_deadline=state_kw.get("run_deadline", None),
        _warm_specialist_params=_warm,
        _record_observation=_record_obs,
        observations=observations,
    )
    # Bind the real param builder + discovery so the gate exercises the full path.
    fake._build_enablement_specialist_params = types.MethodType(Coordinator._build_enablement_specialist_params, fake)
    fake._discover_enablement_candidate_refs = types.MethodType(Coordinator._discover_enablement_candidate_refs, fake)
    fake._read_enablement_source_context = lambda _sig: ""
    fake._derive_checkpoint_weight_facts = lambda _log: ""
    # No GPU pool, so dispatch stays on research_lane only.
    fake._framework_gpu_params = lambda: {}
    fake._framework_authoring_lanes_ttl = lambda params, *, base_ttl_sec: (
        ["research_lane"],
        base_ttl_sec,
    )
    # Real catalogue resolution, so a row enqueued without a TTL is visible here.
    fake.action_registry = ACTION_CATALOGUE
    fake._registry_lanes_ttl = types.MethodType(Coordinator._registry_lanes_ttl, fake)
    fake._maybe_record_enablement_human_review = types.MethodType(
        Coordinator._maybe_record_enablement_human_review, fake
    )
    fake._maybe_rearm_enablement = types.MethodType(Coordinator._maybe_rearm_enablement, fake)
    fake._maybe_enqueue_enablement_baseline_revalidation = types.MethodType(
        Coordinator._maybe_enqueue_enablement_baseline_revalidation, fake
    )
    fake._open_revalidation_row = types.MethodType(Coordinator._open_revalidation_row, fake)
    fake._open_row_past_spent_generations = types.MethodType(Coordinator._open_row_past_spent_generations, fake)
    # Admission on the session wall-clock is exercised in test_coordinator_runtime
    # against a real coordinator; here nothing is ever denied for want of budget.
    fake._time_budget_denial_for_action = lambda _action: None
    from hyperloom.orchestrator.enablement.lane import EnablementLane

    fake._enablement_in_flight = types.MethodType(EnablementLane._enablement_in_flight, fake)
    # _enablement_in_flight reads the coordinator's ephemeral pending_proposals.
    fake.state = types.SimpleNamespace(pending_proposals={})
    return fake


@pytest.mark.asyncio
async def test_enqueue_dispatches_when_baseline_unrunnable(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self()
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == "spec-1"
    # In-flight guard set; attempt counter advanced for candidate rotation.
    assert bool(fake.shared_state.enablement.inflight_task_id)
    assert fake.shared_state.enablement.attempts == 1
    assert len(fake.tasks.created) == 1
    assert fake.tasks.created[0]["params"]["enablement"] is True
    assert fake.tasks.created[0]["params"]["enablement_attempt"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,origin,dispatched",
    [
        ("off", "", False),
        ("off", "eval", False),
        ("launch", "", True),
        ("launch", "eval", False),
        ("eval", "", False),
        ("eval", "eval", True),
        ("all", "", True),
        ("all", "eval", True),
    ],
)
async def test_enqueue_admission_follows_mode_and_origin(monkeypatch, mode, origin, dispatched):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self(enablement_mode=mode, enablement_origin=origin)
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert bool(tid) is dispatched
    assert bool(fake.shared_state.enablement.inflight_task_id) is dispatched


@pytest.mark.asyncio
async def test_enqueue_noop_when_already_succeeded(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(enablement_succeeded=True)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_noop_when_run_deadline_passed(monkeypatch):
    import time as _time

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    # Deadline already in the past -> no new enablement work is opened.
    fake = _enqueue_self(run_deadline=_time.monotonic() - 1.0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_retries_with_next_attempt_after_revert(monkeypatch):
    from hyperloom.agents.framework.models import Candidate
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    cands = [
        Candidate(ref="PR:1", repo="r", title="enable GLM arch on ROCm", html_url="http://x/1"),
        Candidate(ref="PR:2", repo="r", title="add GLM support fix", html_url="http://x/2"),
    ]
    _stub_enumerate(monkeypatch, cands)
    fake = _enqueue_self()
    # First dispatch.
    tid1 = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid1 == "spec-1"
    assert fake.shared_state.enablement.attempts == 1
    first_idem = fake.tasks.created[0]["idempotency_key"]

    # Simulate the authored patch being REVERTED -> re-arm clears in-flight.
    fake._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert not fake.shared_state.enablement.inflight_task_id
    assert fake.shared_state.enablement.succeeded is False

    # Next tick re-dispatches with a new attempt index + distinct idempotency.
    tid2 = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid2 == "spec-2"
    assert fake.shared_state.enablement.attempts == 2
    second_idem = fake.tasks.created[1]["idempotency_key"]
    assert first_idem != second_idem
    assert fake.tasks.created[1]["params"]["enablement_attempt"] == 1
    # Retry mandate flags the prior revert.
    assert "RETRY" in fake.tasks.created[1]["params"]["notes"]


@pytest.mark.asyncio
async def test_watchdog_rearms_silently_finished_round(monkeypatch):
    """A round whose inflight_task_id maps to a terminal task counts as a stall
    and clears the guard so a fresh round can dispatch."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self(
        enablement_stall_streak=1,
        enablement_inflight_task_id="spec-stuck",
    )

    async def _in_flight_false(self=fake):
        return False

    fake._enablement_in_flight = _in_flight_false
    await Coordinator._maybe_enqueue_enablement_specialist(fake)
    # Stall streak should have advanced.
    assert fake.shared_state.enablement.stall_streak == 2


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_when_task_running(monkeypatch):
    """When the inflight_task_id maps to a running task, the round is not
    counted as a stall — the specialist is still working."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self(
        enablement_stall_streak=1,
        enablement_inflight_task_id="spec-running",
    )

    async def _in_flight_true(self=fake):
        return True

    fake._enablement_in_flight = _in_flight_true
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    # Streak unchanged; still blocked on the running task.
    assert fake.shared_state.enablement.stall_streak == 1


def _integrate_proposal(specialist_task_id: str, *, decided: bool = False):
    return PendingProposal(
        proposal_msg_id=f"m-{specialist_task_id}",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"specialist_task_id": specialist_task_id, "enablement": True}},
        decided=decided,
    )


@pytest.mark.asyncio
async def test_in_flight_defers_on_undecided_integrate_proposal():
    """The specialist goes terminal a tick before the Critic sees the integrate
    proposal; the round must still count as in flight."""
    # No task rows, so the specialist lookup raises TaskNotFound (terminal).
    fake = _enqueue_self(enablement_inflight_task_id="spec-done")
    fake.state.pending_proposals["m-spec-done"] = _integrate_proposal("spec-done")
    assert await fake._enablement_in_flight() is True


@pytest.mark.asyncio
async def test_in_flight_ignores_decided_integrate_proposal():
    """Once the Critic has ruled, the proposal stops deferring so a dropped
    proposal cannot hold the round open forever."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-done")
    fake.state.pending_proposals["m-spec-done"] = _integrate_proposal("spec-done", decided=True)
    assert await fake._enablement_in_flight() is False


@pytest.mark.asyncio
async def test_in_flight_defers_on_queued_integrate_task():
    """An approved integrate_patch task for this specialist keeps the round open."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-done")
    fake.tasks = _FakeTasks(
        queued=[
            types.SimpleNamespace(
                task_id="ip-1",
                kind="integrate_patch",
                params={"specialist_task_id": "spec-done"},
            )
        ]
    )
    assert await fake._enablement_in_flight() is True


@pytest.mark.asyncio
async def test_in_flight_ignores_integrate_task_for_other_specialist():
    fake = _enqueue_self(enablement_inflight_task_id="spec-done")
    fake.tasks = _FakeTasks(
        queued=[
            types.SimpleNamespace(
                task_id="ip-1",
                kind="integrate_patch",
                params={"specialist_task_id": "spec-other"},
            )
        ]
    )
    assert await fake._enablement_in_flight() is False


@pytest.mark.asyncio
async def test_no_false_stall_while_integrate_proposal_pending(monkeypatch):
    """Regression: a terminal specialist with an unreviewed integrate proposal
    must not bump the stall streak nor dispatch a second concurrent round."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self(
        enablement_stall_streak=1,
        enablement_inflight_task_id="spec-done",
    )
    fake.state.pending_proposals["m-spec-done"] = _integrate_proposal("spec-done")
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert fake.shared_state.enablement.stall_streak == 1
    assert fake.shared_state.enablement.inflight_task_id == "spec-done"
    assert fake.tasks.created == []


@pytest.mark.asyncio
async def test_rearm_kept_is_terminal(monkeypatch):
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake._maybe_rearm_enablement({"enablement": True, "status": "kept"})
    assert fake.shared_state.enablement.succeeded is True
    # A subsequent enqueue attempt is a no-op.
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_rearm_kept_eval_origin_holds_for_revalidation(monkeypatch):
    fake = _enqueue_self(
        enablement_inflight_task_id="spec-1", enablement_origin="eval", enablement_revalidation_generation=1
    )
    fake._maybe_rearm_enablement({"enablement": True, "status": "kept"})
    # eval-origin KEEP is NOT terminal: hold succeeded, open validation window.
    assert fake.shared_state.enablement.validation_pending is True
    assert fake.shared_state.enablement.succeeded is False
    assert not fake.shared_state.enablement.inflight_task_id
    # Generation is incremented to get a fresh idempotency key.
    assert fake.shared_state.enablement.revalidation_generation == 2
    # Tracked task_id is cleared for the new window.
    assert fake.shared_state.enablement.revalidation_task_id == ""
    # The specialist pump is blocked while validation is pending.
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_revalidation_enqueues_genuine_baseline():
    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_probe_config_path="/runs/baseline/materialized.yaml",
        enablement_accuracy_floor=0.3,
    )
    tid = await fake._maybe_enqueue_enablement_baseline_revalidation()
    assert tid
    created = fake.tasks.created[-1]
    assert created["kind"] == "baseline"
    assert created["params"]["config_path"] == "/runs/baseline/materialized.yaml"
    assert created["params"]["disable_run_eval"] is False
    assert created["params"]["enablement_origin"] == "eval"


@pytest.mark.asyncio
async def test_revalidation_prefers_accepted_config_over_probe():
    """accepted_config_path (from KEEP'd bench) takes precedence over probe config."""
    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_probe_config_path="/runs/baseline/probe.yaml",
        enablement_accepted_config_path="/runs/specialist/accepted.yaml",
        enablement_accuracy_floor=0.3,
    )
    tid = await fake._maybe_enqueue_enablement_baseline_revalidation()
    assert tid
    created = fake.tasks.created[-1]
    assert created["params"]["config_path"] == "/runs/specialist/accepted.yaml"


@pytest.mark.asyncio
async def test_revalidation_carries_active_runtime():
    """When an active runtime is recorded, its override is included in params."""
    from hyperloom.orchestrator.framework.stack_actions import FrameworkRuntime

    rt = FrameworkRuntime(bin_path="/attempt/bin", python_path="/attempt/bin/python", venv_root="/attempt/venv")
    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_probe_config_path="/runs/baseline/probe.yaml",
        enablement_accepted_config_path="/runs/specialist/accepted.yaml",
        enablement_active_runtime=rt.to_state(),
    )
    tid = await fake._maybe_enqueue_enablement_baseline_revalidation()
    assert tid
    created = fake.tasks.created[-1]
    rt_override = created["params"].get("runtime_override")
    assert isinstance(rt_override, dict) and rt_override
    assert rt_override.get("framework_bin") == "/attempt/bin"


@pytest.mark.asyncio
async def test_revalidation_skips_when_already_tracked():
    """If a tracked revalidation task is already alive, no new task is created."""
    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_revalidation_task_id="existing-spec-1",
    )
    # Put the tracked task in the running list so it appears alive.
    import types as _types

    fake.tasks = _FakeTasks(running=[_types.SimpleNamespace(kind="baseline", task_id="existing-spec-1")])
    result = await fake._maybe_enqueue_enablement_baseline_revalidation()
    assert result == "existing-spec-1"
    assert fake.tasks.created == []


@pytest.mark.asyncio
async def test_revalidation_skips_when_tracked_task_in_flight():
    """Skip enqueue when the tracked revalidation task is still running."""
    import types as _types

    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_revalidation_task_id="reval-in-flight",
    )
    fake.tasks = _FakeTasks(running=[_types.SimpleNamespace(kind="baseline", task_id="reval-in-flight")])
    result = await fake._maybe_enqueue_enablement_baseline_revalidation()
    assert result == "reval-in-flight"
    assert fake.tasks.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accepted_config",
    [
        {"extra_envs": {"VLLM_ROCM_USE_AITER_MOE": "0"}},
        {"extra_server_args": "--kv-cache-dtype fp8_e4m3"},
        {"remove_args": ["--block-size"], "unset_envs": ["VLLM_X"], "args_mode": "replace"},
        {},
    ],
)
async def test_revalidation_forwards_accepted_config(accepted_config):
    """The KEEP's env/arg layers reach the revalidation baseline, and nothing else does."""
    fake = _enqueue_self(
        enablement_validation_pending=True,
        enablement_origin="eval",
        enablement_accepted_config_path="/runs/specialist/accepted.yaml",
        enablement_accepted_config=accepted_config,
    )
    await fake._maybe_enqueue_enablement_baseline_revalidation()
    params = fake.tasks.created[-1]["params"]
    for key in ("extra_envs", "extra_server_args", "remove_args", "unset_envs", "args_mode"):
        assert params.get(key) == accepted_config.get(key), key


@pytest.mark.asyncio
async def test_rearm_kept_stores_accepted_config():
    """A KEEP's effective config is persisted on the round for the revalidation to replay."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1", enablement_origin="eval")
    effective = {"extra_envs": {"VLLM_ROCM_USE_AITER_FP4BMM": "0"}, "args_mode": "append"}
    fake._maybe_rearm_enablement(
        {
            "status": "kept",
            "enablement": True,
            "enablement_accepted_config_path": "/runs/specialist/accepted.yaml",
            "enablement_effective_config": effective,
        }
    )
    assert fake.shared_state.enablement.accepted_config == effective


@pytest.mark.asyncio
async def test_rearm_kept_records_patches_in_stack():
    """KEEP patches are added to kept_patches so a revalidation-rearmed round inherits them."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1", enablement_origin="eval")
    fake.shared_state.enablement.kept_rounds = [{"patches": ["/prior/advance.patch"], "artifacts": []}]
    fake._maybe_rearm_enablement(
        {
            "status": "kept",
            "enablement": True,
            "patches_applied": ["/this/round/fix.patch"],
            "enablement_effective_config": {"extra_envs": {}},
        }
    )
    assert "/prior/advance.patch" in fake.shared_state.enablement.kept_patches
    assert "/this/round/fix.patch" in fake.shared_state.enablement.kept_patches


@pytest.mark.asyncio
async def test_rearm_ignores_non_enablement(monkeypatch):
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake._maybe_rearm_enablement({"status": "reverted"})
    # No enablement marker -> state untouched.
    assert bool(fake.shared_state.enablement.inflight_task_id)
    assert fake.shared_state.enablement.succeeded is False


@pytest.mark.asyncio
async def test_rearm_advanced_stacks_patch_and_reclassifies(monkeypatch):
    """A patch that clears one gap and reveals a new gap is STACKED, not reverted."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1", enablement_stall_streak=2)
    new_gap_log = (
        "ValueError: Following weights were not initialized from checkpoint: "
        "{'model.layers.19.self_attn.indexer.k_norm.weight'}"
    )
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "patches_applied": ["/s/runs/specialist/t1/patches/001_qk_rope.patch"],
            "enablement_launch_log": new_gap_log,
        }
    )
    st = fake.shared_state
    # Not terminal; not stalled.
    assert st.enablement.succeeded is False
    assert st.stop_reason == ""
    # Progressing patch is recorded for stacking; stall streak reset; guard cleared.
    assert st.enablement.kept_patches == ["/s/runs/specialist/t1/patches/001_qk_rope.patch"]
    assert st.enablement.stall_streak == 0
    assert not st.enablement.inflight_task_id
    # Launch log now points at the NEW (deeper) gap so the next round targets it.
    assert "not initialized from checkpoint" in st.enablement.launch_log


@pytest.mark.asyncio
async def test_rearm_advanced_dedups_stacked_patches(monkeypatch):
    """Re-applied base patches are not double-recorded; only the new one is added."""
    fake = _enqueue_self(
        enablement_inflight_task_id="spec-1",
        enablement_kept_patches=["/s/runs/specialist/t1/patches/001_qk_rope.patch"],
    )
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            # integrate re-applies the base + the new patch this round.
            "patches_applied": [
                "/s/runs/specialist/t1/patches/001_qk_rope.patch",
                "/s/runs/specialist/t2/patches/002_indexer_share.patch",
            ],
            "enablement_launch_log": "RuntimeError: some third gap",
        }
    )
    assert fake.shared_state.enablement.kept_patches == [
        "/s/runs/specialist/t1/patches/001_qk_rope.patch",
        "/s/runs/specialist/t2/patches/002_indexer_share.patch",
    ]


@pytest.mark.asyncio
async def test_rearm_stall_cap_stops_run(monkeypatch):
    """N consecutive no-progress reverts stop the run with enablement_stalled."""
    from hyperloom.orchestrator.loop.coordinator import _ENABLEMENT_MAX_STALL

    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    st = fake.shared_state
    # First _ENABLEMENT_MAX_STALL-1 reverts: bump streak, keep retrying.
    for i in range(_ENABLEMENT_MAX_STALL - 1):
        fake._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
        assert st.stop_reason == ""
        assert not st.enablement.inflight_task_id
        assert st.enablement.stall_streak == i + 1
    # The final revert trips the cap -> terminal stop_reason.
    fake._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert st.enablement.stall_streak == _ENABLEMENT_MAX_STALL
    assert st.stop_reason == "enablement_stalled"


@pytest.mark.asyncio
async def test_rearm_advanced_then_revert_streak_resets(monkeypatch):
    """A progressing round resets the stall streak so serial gaps aren't capped early."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    st = fake.shared_state
    fake._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert st.enablement.stall_streak == 1
    # Progress resets the streak.
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "patches_applied": ["/p/a.patch"],
            "enablement_launch_log": "ValueError: not initialized from checkpoint",
        }
    )
    assert st.enablement.stall_streak == 0
    assert st.stop_reason == ""


@pytest.mark.asyncio
async def test_rearm_advanced_stacks_setup_commands(monkeypatch):
    """Q3: applied setup commands are stacked on advance for durable replay next round."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake.shared_state.enablement.setup_commands = []
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "patches_applied": ["/p/a.patch"],
            "setup_commands_applied": ["pip install -U transformers"],
            "enablement_launch_log": "ValueError: not initialized from checkpoint",
        }
    )
    assert fake.shared_state.enablement.setup_commands == ["pip install -U transformers"]


@pytest.mark.asyncio
async def test_rearm_kept_stacks_setup_commands(monkeypatch):
    """Q3: a runnable KEEP also records the setup commands it relied on."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake.shared_state.enablement.setup_commands = ["apt-get install -y gh"]
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "kept",
            "setup_commands_applied": ["apt-get install -y gh", "pip install vllm==0.24"],
        }
    )
    assert fake.shared_state.enablement.succeeded is True
    assert fake.shared_state.enablement.setup_commands == [
        "apt-get install -y gh",
        "pip install vllm==0.24",
    ]


def test_enablement_close_guard_blocks_premature_skip_to_close():
    """Q2: pre-enablement (baseline never established) the close guard is active."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState()
    s.phase = "PRELUDE"
    s.baseline_tput = 0.0
    s.enablement.succeeded = False
    assert s.enablement_close_guard_active() is True
    # Once a baseline exists, or enablement succeeded, the guard lifts.
    s.baseline_tput = 100.0
    assert s.enablement_close_guard_active() is False
    s.baseline_tput = 0.0
    s.enablement.succeeded = True
    assert s.enablement_close_guard_active() is False


def test_enablement_close_guard_active_during_validation_pending():
    """An eval-origin KEEP awaiting revalidation must keep the guard active even
    if a stale positive tput is present."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState()
    s.phase = "SWEEP"
    s.baseline_tput = 100.0
    s.enablement.succeeded = False
    s.enablement.validation_pending = True
    assert s.enablement_close_guard_active() is True
    s.enablement.validation_pending = False
    assert s.enablement_close_guard_active() is False


def test_build_params_threads_base_setup_commands_when_stacked(monkeypatch):
    """Q3: stacked base setup commands are passed to the next round + noted."""
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self()
    fake.shared_state.enablement.setup_commands = ["pip install -U transformers"]
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    assert params["enablement_setup_commands"] == ["pip install -U transformers"]
    assert "STACKED ENABLEMENT" in params["notes"]
    assert "setup command" in params["notes"]


def test_build_params_threads_base_patches_when_stacked(monkeypatch):
    """Stacked kept-patches are passed to the next round + noted in the mandate."""
    _stub_enumerate(monkeypatch, [])
    fake = _fake_self()
    fake.shared_state.enablement.kept_patches = [
        "/s/runs/specialist/t1/patches/001_qk_rope.patch",
    ]
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    assert params["enablement_base_patches"] == [
        "/s/runs/specialist/t1/patches/001_qk_rope.patch",
    ]
    # The mandate tells the specialist the prior patch is already applied.
    assert "STACKED ENABLEMENT" in params["notes"]
    assert "001_qk_rope.patch" in params["notes"]


@pytest.mark.asyncio
async def test_enqueue_dispatches_for_unknown_nonblank_log(monkeypatch):
    """Q1: a non-blank UNKNOWN failure DISPATCHES (never wedges in human_review)."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self(enablement_launch_log="some totally unrelated noise line")
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == "spec-1"
    assert len(fake.tasks.created) == 1
    assert fake.tasks.created[0]["params"]["enablement"] is True
    # No human-review dead-end for a non-blank log.
    reviews = [o for o in fake.observations if o.get("kind") == "enablement_needs_human_review"]
    assert reviews == []


@pytest.mark.asyncio
async def test_enqueue_noop_when_already_dispatched(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    # Simulate the task being actively running in the registry.
    fake.tasks = _FakeTasks(running=[types.SimpleNamespace(task_id="spec-1")])
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_noop_when_baseline_runnable(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(baseline_tput=1234.0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_enqueue_noop_when_no_failure_streak(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(baseline_failure_streak=0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_enqueue_noop_on_multi_node(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    fake = _enqueue_self()
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0


# ---- _derive_checkpoint_weight_facts (auto-feedback structural grounding) ----


def _facts_self(model_path=""):
    """Minimal fake carrying a shared_state.model_path for the facts deriver."""
    state = types.SimpleNamespace(model_path=model_path)
    return types.SimpleNamespace(shared_state=state)


_WEIGHT_INIT_LOG = (
    "Some weights were not initialized from checkpoint and are newly initialized: "
    "['model.layers.3.self_attn.indexer.k_norm.weight', "
    "'model.layers.5.self_attn.indexer.k_norm.weight']"
)


def _write_index(model_dir, weight_map):
    import json

    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_derive_weight_facts_blank_log_returns_empty():
    fake = _facts_self()
    assert Coordinator._derive_checkpoint_weight_facts(fake, "") == ""
    assert Coordinator._derive_checkpoint_weight_facts(fake, None) == ""


def test_derive_weight_facts_no_trigger_returns_empty():
    """A log with no weight-init phrase and no weighty names does not fire."""
    fake = _facts_self()
    out = Coordinator._derive_checkpoint_weight_facts(fake, "ValueError: Model architecture 'Foo' is not supported")
    assert out == ""


def test_derive_weight_facts_no_model_path_returns_empty():
    fake = _facts_self(model_path="")
    assert Coordinator._derive_checkpoint_weight_facts(fake, _WEIGHT_INIT_LOG) == ""


def test_derive_weight_facts_missing_index_returns_empty(tmp_path):
    # Directory exists but has no *.index.json → degrades to empty.
    fake = _facts_self(model_path=str(tmp_path))
    assert Coordinator._derive_checkpoint_weight_facts(fake, _WEIGHT_INIT_LOG) == ""


def test_derive_weight_facts_reports_present_and_missing_layers(tmp_path):
    # The checkpoint carries the k_norm weight only for layers 0 and 3, so the
    # offending layer 5 is reported MISSING while 3 is PRESENT.
    _write_index(
        tmp_path,
        {
            "model.layers.0.self_attn.indexer.k_norm.weight": "a.safetensors",
            "model.layers.3.self_attn.indexer.k_norm.weight": "a.safetensors",
        },
    )
    fake = _facts_self(model_path=str(tmp_path))
    out = Coordinator._derive_checkpoint_weight_facts(fake, _WEIGHT_INIT_LOG)
    assert "CHECKPOINT WEIGHT FACTS" in out
    assert "PRESENT in checkpoint for layers [0, 3]" in out
    assert "MISSING" in out
    # The offending layers (3, 5) both appear in the missing-layers set.
    assert "[3, 5]" in out


def test_derive_weight_facts_family_absent_from_checkpoint(tmp_path):
    # The checkpoint has NO k_norm weight for any layer → the "NOT present for
    # ANY layer" branch fires (model should guard/skip instantiation).
    _write_index(
        tmp_path,
        {"model.layers.0.self_attn.q_proj.weight": "a.safetensors"},
    )
    fake = _facts_self(model_path=str(tmp_path))
    out = Coordinator._derive_checkpoint_weight_facts(fake, _WEIGHT_INIT_LOG)
    assert "NOT present in the checkpoint for ANY layer" in out


def test_derive_weight_facts_fires_on_missing_key_phrase(tmp_path):
    # A state_dict "Missing key(s)" phrasing also triggers derivation.
    _write_index(
        tmp_path,
        {"model.layers.0.self_attn.indexer.k_norm.weight": "a.safetensors"},
    )
    fake = _facts_self(model_path=str(tmp_path))
    log = (
        "RuntimeError: Error(s) in loading state_dict for Model:\n"
        "Missing key(s) in state_dict: "
        "'model.layers.5.self_attn.indexer.k_norm.weight'."
    )
    out = Coordinator._derive_checkpoint_weight_facts(fake, log)
    assert "CHECKPOINT WEIGHT FACTS" in out


def test_derive_weight_facts_exception_guarded(monkeypatch, tmp_path):
    # An unreadable index (invalid JSON) must degrade to "" rather than raise.
    (tmp_path / "model.safetensors.index.json").write_text("{not valid json")
    fake = _facts_self(model_path=str(tmp_path))
    assert Coordinator._derive_checkpoint_weight_facts(fake, _WEIGHT_INIT_LOG) == ""


# ---- _read_enablement_source_context (best-effort grounding snippet) ----


def _sig(offending_file="", offending_symbol=""):
    return types.SimpleNamespace(offending_file=offending_file, offending_symbol=offending_symbol)


def test_read_source_context_empty_when_no_file():
    fake = types.SimpleNamespace(shared_state=types.SimpleNamespace())
    assert Coordinator._read_enablement_source_context(fake, _sig()) == ""


def test_read_source_context_empty_when_file_absent(tmp_path):
    fake = types.SimpleNamespace(shared_state=types.SimpleNamespace())
    missing = str(tmp_path / "nope.py")
    assert Coordinator._read_enablement_source_context(fake, _sig(missing)) == ""


def test_read_source_context_returns_window_around_symbol(tmp_path):
    src = tmp_path / "model.py"
    body = "\n".join(f"line{i}" for i in range(20))
    src.write_text(body.replace("line10", "def NEEDLE(): pass"))
    fake = types.SimpleNamespace(shared_state=types.SimpleNamespace())
    out = Coordinator._read_enablement_source_context(fake, _sig(str(src), "NEEDLE"), window=6)
    assert str(src) in out
    assert "NEEDLE" in out
    # The header carries the resolved line window.
    assert "lines " in out


def test_read_source_context_head_when_symbol_absent(tmp_path):
    src = tmp_path / "model.py"
    src.write_text("\n".join(f"line{i}" for i in range(20)))
    fake = types.SimpleNamespace(shared_state=types.SimpleNamespace())
    out = Coordinator._read_enablement_source_context(fake, _sig(str(src), "not_there"), window=4)
    # Symbol absent → snippet starts at file head.
    assert "line0" in out


def test_read_source_context_empty_on_blank_file(tmp_path):
    src = tmp_path / "empty.py"
    src.write_text("")
    fake = types.SimpleNamespace(shared_state=types.SimpleNamespace())
    assert Coordinator._read_enablement_source_context(fake, _sig(str(src))) == ""


# ---------------------------------------------------------------------------
# _maybe_rearm_authored_lane
# ---------------------------------------------------------------------------


def _make_coord_with_phase(session_dir) -> "Coordinator":
    """Build a minimal Coordinator for authored-lane tests."""
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    plan = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    backends = {name: MockBackend(plan, name=name) for name in ("orchestration", "critic", "robustness")}
    return Coordinator(session_dir, backends=backends)


def test_rearm_authored_lane_delegates_enablement(session_dir):
    """_maybe_rearm_authored_lane with lane=enablement calls _maybe_rearm_enablement."""
    coord = _make_coord_with_phase(session_dir)
    called = []

    def _fake_rearm(res):
        called.append(res)

    coord.phase_framework._maybe_rearm_enablement = _fake_rearm  # type: ignore[method-assign]

    res = {"status": "apply_failed", "lane": "enablement", "enablement": True}
    coord._maybe_rearm_authored_lane(res)
    assert len(called) == 1 and called[0] is res


def test_rearm_authored_lane_perf_framework_increments_counter(session_dir):
    """apply_failed on perf_framework lane increments apply_fail_reauthor_attempts."""
    coord = _make_coord_with_phase(session_dir)
    cand_id = "https://github.com/example/repo/pull/99"
    res = {
        "status": "apply_failed",
        "lane": "perf_framework",
        "candidate": {"candidate_id": cand_id, "pr_url": cand_id},
        "specialist_task_id": "spec-99",
        "retry_feedback": [],
        "prior_patches": [],
    }
    coord._maybe_rearm_authored_lane(res)
    attempts = getattr(coord.shared_state, "apply_fail_reauthor_attempts", {})
    assert attempts.get(cand_id) == 1
    # A pending retry context should be queued.
    pending = getattr(coord.shared_state, "apply_fail_retry_pending", [])
    assert len(pending) == 1
    assert pending[0]["lane"] == "perf_framework"
    assert pending[0]["attempt"] == 1


def test_rearm_authored_lane_perf_framework_stamps_terminal_at_cap(session_dir):
    """After _AUTHORED_LANE_MAX_ATTEMPTS, the next call stamps a terminal row."""
    from hyperloom.orchestrator.loop.coordinator import _AUTHORED_LANE_MAX_ATTEMPTS

    coord = _make_coord_with_phase(session_dir)
    cand_id = "https://github.com/example/repo/pull/100"
    res = {
        "status": "apply_failed",
        "lane": "perf_framework",
        "candidate": {"candidate_id": cand_id, "pr_url": cand_id},
        "specialist_task_id": "spec-100",
        "retry_feedback": [],
        "prior_patches": [],
    }
    # Exhaust cap.
    coord.shared_state.apply_fail_reauthor_attempts = {cand_id: _AUTHORED_LANE_MAX_ATTEMPTS}
    # Clear any pending from prior.
    coord.shared_state.apply_fail_retry_pending = []

    coord._maybe_rearm_authored_lane(res)

    progress = getattr(coord.shared_state, "framework_agent_phase_progress", [])
    cap_rows = [p for p in progress if p.get("status") == "apply_fail_cap"]
    assert len(cap_rows) == 1, f"expected terminal row; got {progress}"
    # No new pending retry.
    pending = getattr(coord.shared_state, "apply_fail_retry_pending", [])
    assert pending == []


def test_rearm_authored_lane_enablement_apply_failed_is_not_counted_as_perf(session_dir):
    """Enablement apply_failed (with enablement:True) does NOT increment apply_fail counter."""
    coord = _make_coord_with_phase(session_dir)
    rearm_called = []

    def _fake_rearm(res):
        rearm_called.append(res)

    coord.phase_framework._maybe_rearm_enablement = _fake_rearm  # type: ignore[method-assign]

    # Even when lane=enablement is absent but enablement=True is present, should delegate.
    res = {"status": "apply_failed", "enablement": True}
    coord._maybe_rearm_authored_lane(res)
    assert len(rearm_called) == 1
    # apply_fail_reauthor_attempts not touched.
    assert not getattr(coord.shared_state, "apply_fail_reauthor_attempts", {})


@pytest.mark.asyncio
async def test_rearm_advanced_accumulates_config_only_envs(monkeypatch):
    """An advanced round that carries only env changes (no patch) is recorded."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "patches_applied": [],
            "extra_envs_applied": {"VLLM_ROCM_USE_AITER": "1"},
            "extra_server_args_applied": "--kv-cache-dtype fp8",
            "setup_commands_applied": [],
            "enablement_launch_log": "some boot log",
        }
    )
    cfg = fake.shared_state.enablement.accepted_config
    assert cfg.get("extra_envs", {}).get("VLLM_ROCM_USE_AITER") == "1"
    assert "--kv-cache-dtype fp8" in (cfg.get("extra_server_args") or "")


@pytest.mark.asyncio
async def test_rearm_advanced_merges_repeated_config_rounds(monkeypatch):
    """Successive advanced rounds accumulate envs without overwriting prior ones."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "patches_applied": [],
            "extra_envs_applied": {"A": "1"},
            "extra_server_args_applied": "--flag-a",
            "setup_commands_applied": [],
        }
    )
    fake.shared_state.enablement.inflight_task_id = "spec-2"
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "patches_applied": [],
            "extra_envs_applied": {"B": "2"},
            "extra_server_args_applied": "--flag-b",
            "setup_commands_applied": [],
        }
    )
    cfg = fake.shared_state.enablement.accepted_config
    assert cfg["extra_envs"].get("A") == "1"
    assert cfg["extra_envs"].get("B") == "2"
    assert "--flag-a" in cfg["extra_server_args"]
    assert "--flag-b" in cfg["extra_server_args"]


@pytest.mark.asyncio
async def test_rearm_advanced_merges_args_by_flag_not_substring():
    """A prefix flag survives, and a restated flag overrides instead of duplicating."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1")
    for tid, args in (
        ("spec-1", "--enable-chunked-prefill --tp 4"),
        ("spec-2", "--enable-chunked"),
        ("spec-3", "--tp 8"),
    ):
        fake.shared_state.enablement.inflight_task_id = tid
        fake._maybe_rearm_enablement(
            {
                "enablement": True,
                "status": "advanced",
                "patches_applied": [],
                "extra_envs_applied": {},
                "extra_server_args_applied": args,
                "setup_commands_applied": [],
            }
        )
    merged = fake.shared_state.enablement.accepted_config["extra_server_args"].split()
    assert "--enable-chunked-prefill" in merged
    assert "--enable-chunked" in merged
    assert merged[merged.index("--tp") + 1] == "8"


# ---- advanced: artifact stacking (issue 2) ----------------------------------


@pytest.mark.asyncio
async def test_rearm_advanced_stacks_artifacts(monkeypatch):
    """Artifacts applied in an advanced round must be recorded in kept_artifacts
    so that _replay_base_artifacts re-installs them at the start of the next round.
    Before the fix, the advanced result had no 'artifacts_applied' key, so
    _stack_kept_artifacts() always saw an empty list."""
    fake = _enqueue_self(enablement_inflight_task_id="spec-1", enablement_stall_streak=0)
    art = {
        "target": "/sgl-workspace/sglang/srt/server_args.py",
        "rel_target": "srt/server_args.py",
        "kind": "python_source",
        "existed": True,
        "backup": "/s/runs/integrate_patch/t1/artifact_backups/000_server_args.py.bak",
        "source": "/s/runs/specialist/spec-1/worktree/artifacts/server_args.py",
    }
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "patches_applied": [],
            "artifacts_applied": [art],
            "enablement_launch_log": "new gap after artifact fix",
        }
    )
    st = fake.shared_state
    assert len(st.enablement.kept_artifacts) == 1
    assert st.enablement.kept_artifacts[0]["target"] == "/sgl-workspace/sglang/srt/server_args.py"
    assert st.enablement.stall_streak == 0


@pytest.mark.asyncio
async def test_rearm_advanced_deduplicates_artifacts(monkeypatch):
    """A base artifact present from a prior round is not double-counted."""
    art = {
        "target": "/sgl-workspace/sglang/srt/server_args.py",
        "rel_target": "srt/server_args.py",
        "kind": "python_source",
        "source": "/s/runs/specialist/spec-1/worktree/artifacts/server_args.py",
    }
    fake = _enqueue_self(enablement_inflight_task_id="spec-2")
    # Pre-load the prior round's artifact into state.
    fake.shared_state.enablement.kept_artifacts = [art]
    # Second advanced round returns the same artifact (the base was re-applied).
    fake._maybe_rearm_enablement(
        {
            "enablement": True,
            "status": "advanced",
            "advanced": True,
            "patches_applied": [],
            "artifacts_applied": [art],
            "enablement_launch_log": "yet another gap",
        }
    )
    assert len(fake.shared_state.enablement.kept_artifacts) == 1
