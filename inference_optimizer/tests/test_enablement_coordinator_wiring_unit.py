# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Coordinator enablement wiring (framework-ref1).

Covers the pure launch-log extractor, the ``build_mandate``-backed specialist
param builder, and the one-shot ``_maybe_enqueue_enablement_specialist`` gate.
These exercise the seams without spinning up the full orchestrator.
"""

from __future__ import annotations

import types

import pytest

from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
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
    )
    fake = types.SimpleNamespace(shared_state=state)
    # Bind the real discovery method so the builder path is exercised; tests
    # that don't want the network stub the low-level enumerate_candidates.
    fake._discover_enablement_candidate_refs = types.MethodType(
        Coordinator._discover_enablement_candidate_refs, fake
    )
    # Source-context read is best-effort grounding; stub to empty so the builder
    # path stays pure (no filesystem dependency in these unit tests).
    fake._read_enablement_source_context = lambda _sig: ""
    # Item J whole-machine GPU request is exercised in its own suite; here the
    # fake has no GPU pool, so it degrades to the research-lane-only path.
    fake._framework_gpu_params = lambda: {}
    return fake


def _stub_enumerate(monkeypatch, cands):
    """Patch ``sources.enumerate_candidates`` to return ``cands`` (or raise)."""
    import framework_agent.sources as src

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
    # Reuses FRAMEWORK authoring machinery + tags the objective.
    assert params["framework_agent_authoring"] is True
    assert params["enablement"] is True
    assert params["enablement_failure_kind"] == "missing_model_arch"
    # The pre-patch signature is serialized for the runnable-gate replay.
    assert params["enablement_before_signature"]["kind"] == "missing_model_arch"
    # The notes body is the single-source mandate rendered by build_mandate.
    assert "RUNNABILITY" in params["notes"]
    assert "git apply --check" in params["notes"]
    assert "GLM-5" in params["notes"]


def test_build_params_feeds_ranked_candidate_refs_into_mandate(monkeypatch):
    from framework_agent.models import Candidate

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
    assert "http://x/2" in params["notes"]


def test_build_params_degrades_gracefully_when_discovery_raises(monkeypatch):
    _stub_enumerate(monkeypatch, RuntimeError("network down"))
    fake = _fake_self()
    params = Coordinator._build_enablement_specialist_params(fake, _MISSING_ARCH_LOG)
    assert params is not None
    # Discovery failure -> repos-only mandate, no candidate refs.
    assert params["enablement_candidate_refs"] == []


def test_build_params_none_for_unknown_failure():
    fake = _fake_self()
    assert (
        Coordinator._build_enablement_specialist_params(fake, "totally unrelated noise")
        is None
    )


def test_build_params_none_for_blank_log():
    fake = _fake_self()
    assert Coordinator._build_enablement_specialist_params(fake, "   ") is None


# ---- _maybe_enqueue_enablement_specialist (one-shot gate) ----


class _FakeTasks:
    def __init__(self):
        self.created = []

    async def create_or_return_existing(self, **kwargs):
        self.created.append(kwargs)
        task = types.SimpleNamespace(task_id=f"spec-{len(self.created)}", state="queued")
        return task, False


def _enqueue_self(**state_kw):
    state = types.SimpleNamespace(
        framework=state_kw.get("framework", "sglang"),
        model_name=state_kw.get("model_name", "zai-org/GLM-5"),
        gpu_type=state_kw.get("gpu_type", "mi300x"),
        enablement_dispatched=state_kw.get("enablement_dispatched", False),
        enablement_succeeded=state_kw.get("enablement_succeeded", False),
        enablement_attempts=state_kw.get("enablement_attempts", 0),
        enablement_human_review_logged=state_kw.get("enablement_human_review_logged", []),
        baseline_tput=state_kw.get("baseline_tput", 0.0),
        baseline_failure_streak=state_kw.get("baseline_failure_streak", 1),
        enablement_launch_log=state_kw.get("enablement_launch_log", _MISSING_ARCH_LOG),
        save=lambda *a, **k: None,
    )

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
    fake._build_enablement_specialist_params = types.MethodType(
        Coordinator._build_enablement_specialist_params, fake
    )
    fake._discover_enablement_candidate_refs = types.MethodType(
        Coordinator._discover_enablement_candidate_refs, fake
    )
    fake._read_enablement_source_context = lambda _sig: ""
    # Item J helpers (whole-machine GPU) are covered in their own suite; the fake
    # has no GPU pool → no needs_gpu, so dispatch stays on research_lane only.
    fake._framework_gpu_params = lambda: {}
    fake._framework_authoring_lanes_ttl = lambda params, *, base_ttl_sec: (
        ["research_lane"],
        base_ttl_sec,
    )
    fake._maybe_record_enablement_human_review = types.MethodType(
        Coordinator._maybe_record_enablement_human_review, fake
    )
    fake._maybe_rearm_enablement = types.MethodType(
        Coordinator._maybe_rearm_enablement, fake
    )
    return fake


@pytest.mark.asyncio
async def test_enqueue_dispatches_when_baseline_unrunnable(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self()
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == "spec-1"
    # In-flight guard set; attempt counter advanced for candidate rotation.
    assert fake.shared_state.enablement_dispatched is True
    assert fake.shared_state.enablement_attempts == 1
    assert len(fake.tasks.created) == 1
    assert fake.tasks.created[0]["params"]["enablement"] is True
    assert fake.tasks.created[0]["params"]["enablement_attempt"] == 0


@pytest.mark.asyncio
async def test_enqueue_noop_when_already_succeeded(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(enablement_succeeded=True)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_noop_when_run_deadline_passed(monkeypatch):
    import time as _time

    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    # Deadline already in the past -> no new enablement work is opened.
    fake = _enqueue_self(run_deadline=_time.monotonic() - 1.0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_retries_with_next_attempt_after_revert(monkeypatch):
    from framework_agent.models import Candidate
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

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
    assert fake.shared_state.enablement_attempts == 1
    first_idem = fake.tasks.created[0]["idempotency_key"]

    # Simulate the authored patch being REVERTED -> re-arm clears in-flight.
    fake._maybe_rearm_enablement({"enablement": True, "status": "reverted"})
    assert fake.shared_state.enablement_dispatched is False
    assert fake.shared_state.enablement_succeeded is False

    # Next tick re-dispatches with a new attempt index + distinct idempotency.
    tid2 = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid2 == "spec-2"
    assert fake.shared_state.enablement_attempts == 2
    second_idem = fake.tasks.created[1]["idempotency_key"]
    assert first_idem != second_idem
    assert fake.tasks.created[1]["params"]["enablement_attempt"] == 1
    # Retry mandate flags the prior revert.
    assert "RETRY" in fake.tasks.created[1]["params"]["notes"]


@pytest.mark.asyncio
async def test_rearm_kept_is_terminal(monkeypatch):
    fake = _enqueue_self(enablement_dispatched=True)
    fake._maybe_rearm_enablement({"enablement": True, "status": "kept"})
    assert fake.shared_state.enablement_succeeded is True
    # A subsequent enqueue attempt is a no-op.
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_rearm_ignores_non_enablement(monkeypatch):
    fake = _enqueue_self(enablement_dispatched=True)
    fake._maybe_rearm_enablement({"status": "reverted"})
    # No enablement marker -> state untouched.
    assert fake.shared_state.enablement_dispatched is True
    assert fake.shared_state.enablement_succeeded is False


@pytest.mark.asyncio
async def test_enqueue_records_human_review_for_unknown(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(enablement_launch_log="some totally unrelated noise line")
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == ""
    # No authoring dispatched, but a needs_human_review observation emitted once.
    assert len(fake.tasks.created) == 0
    reviews = [o for o in fake.observations if o.get("kind") == "enablement_needs_human_review"]
    assert len(reviews) == 1
    assert reviews[0]["applicability"] == "needs_human_review"

    # Same log again -> deduped (no second record).
    await Coordinator._maybe_enqueue_enablement_specialist(fake)
    reviews = [o for o in fake.observations if o.get("kind") == "enablement_needs_human_review"]
    assert len(reviews) == 1


@pytest.mark.asyncio
async def test_enqueue_noop_when_already_dispatched(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(enablement_dispatched=True)
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == ""
    assert len(fake.tasks.created) == 0


@pytest.mark.asyncio
async def test_enqueue_noop_when_baseline_runnable(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(baseline_tput=1234.0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_enqueue_noop_when_no_failure_streak(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    fake = _enqueue_self(baseline_failure_streak=0)
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""


@pytest.mark.asyncio
async def test_enqueue_noop_on_multi_node(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    fake = _enqueue_self()
    assert await Coordinator._maybe_enqueue_enablement_specialist(fake) == ""
    assert len(fake.tasks.created) == 0
