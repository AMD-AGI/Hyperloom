"""Profile trace promotion + last_profile_status tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.profile import ProfileExecutor
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "kernel": MockBackend(silent, name="k"),
            "critic": MockBackend(silent, name="c"),
            "robustness": MockBackend(silent, name="r"),
        },
    )


@pytest.mark.asyncio
async def test_profile_executor_fails_when_no_trace_files(tmp_path, monkeypatch):
    # ProfileExecutor runs ``harvest_leaked_artifacts`` against the
    # configured leak roots (defaults to ``/workspace`` which is not
    # writable inside the CI runner). Pin the env to an isolated
    # sandbox so the harvest never tries to stat a privileged path.
    sandbox = tmp_path / "leak_sandbox"
    sandbox.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "capture_traces").mkdir()
    (workspace / "torch_trace").mkdir()

    async def fake_call(self, ctx):
        return {
            "status": "succeeded",
            "workspace": str(workspace),
            "trace_dir": str(workspace / "torch_trace"),
            "trace_files": [],
        }

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.profile."
        "BaselineExecutor.__call__",
        fake_call,
    )
    task = Task(
        task_id="prof-empty",
        kind="profile",
        state="queued",
        params={},
        idempotency_key="prof-empty",
    )
    result = await ProfileExecutor()(RunnerContext(task=task, lease=None))
    assert result["status"] == "failed"
    assert result["error_class"] == "no_trace_files"


@pytest.mark.asyncio
async def test_profile_promotion_sets_failed_status_on_empty_trace(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.last_profile_trace = "/old/trace.json.gz"
        await c._promote_to_shared_state(
            "profile",
            {
                "status": "failed",
                "error_class": "no_trace_files",
                "trace_dir": "/tmp/empty",
                "trace_files": [],
            },
        )
        assert c.shared_state.last_profile_status == "failed"
        assert c.shared_state.last_profile_trace == ""
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_promotion_does_not_use_trace_dir_fallback(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c._promote_to_shared_state(
            "profile",
            {
                "status": "succeeded",
                "trace_dir": "/tmp/ws/torch_trace",
                "trace_files": [],
            },
        )
        assert c.shared_state.last_profile_trace == ""
        assert c.shared_state.last_profile_status != "succeeded"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_applicable_when_after_failed_status():
    reg = ActionRegistry().load()
    meta = reg.get("profile")
    assert meta is not None
    assert any("last_profile_status" in pred for pred in meta.applicable_when)


def test_last_profile_status_round_trips(tmp_path):
    sd = tmp_path / "sess"
    sd.mkdir()
    state = SharedState.load_or_init(sd)
    state.last_profile_status = "failed"
    state.policy_denial_history = [{"tick": 1, "action_name": "backends"}]
    state.policy_denial_streak = {"backends:dup": 2}
    state.discovered_flags_error = "no flags"
    state.save(sd)
    loaded = SharedState.load_or_init(sd)
    assert loaded.last_profile_status == "failed"
    assert loaded.policy_denial_history
    assert loaded.policy_denial_streak == {"backends:dup": 2}
    assert loaded.discovered_flags_error == "no flags"
