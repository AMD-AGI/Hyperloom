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
        enablement_dispatched=state_kw.get("enablement_dispatched", False),
        baseline_tput=state_kw.get("baseline_tput", 0.0),
        baseline_failure_streak=state_kw.get("baseline_failure_streak", 1),
        enablement_launch_log=state_kw.get("enablement_launch_log", _MISSING_ARCH_LOG),
        save=lambda *a, **k: None,
    )

    async def _warm(_params):
        return None

    fake = types.SimpleNamespace(
        shared_state=state,
        tasks=_FakeTasks(),
        session_dir="/tmp/session",
        _warm_specialist_params=_warm,
    )
    # Bind the real param builder + discovery so the gate exercises the full path.
    fake._build_enablement_specialist_params = types.MethodType(
        Coordinator._build_enablement_specialist_params, fake
    )
    fake._discover_enablement_candidate_refs = types.MethodType(
        Coordinator._discover_enablement_candidate_refs, fake
    )
    return fake


@pytest.mark.asyncio
async def test_enqueue_dispatches_once_when_baseline_unrunnable(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    _stub_enumerate(monkeypatch, [])
    fake = _enqueue_self()
    tid = await Coordinator._maybe_enqueue_enablement_specialist(fake)
    assert tid == "spec-1"
    assert fake.shared_state.enablement_dispatched is True
    assert len(fake.tasks.created) == 1
    assert fake.tasks.created[0]["params"]["enablement"] is True


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
