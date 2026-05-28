"""Profile trace promotion + last_profile_status tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.profile import ProfileExecutor
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
    # ProfileExecutor's BaselineExecutor base falls back to
    # ``_resolve_session_dir`` (defaults to ``/workspace/hyperloom``)
    # and ``harvest_leaked_artifacts`` defaults to ``/workspace`` —
    # neither is writable inside CI. Pin both to a tmp_path sandbox so
    # the executor stays inside the fixture tree.
    user_data = tmp_path / "user_data"
    user_data.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(user_data))
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
    # ``profile`` is no longer a registered action; the only production
    # caller is RooflineExecutor's `_wrap_profile_ctx`, which inherits
    # the parent roofline's workspace via ``ctx.extra['workspace']``.
    # Mirror that here so ProfileExecutor's `_resolve_workspace` takes
    # the extra-supplied path instead of trying to call ``runs_dir(sd,
    # 'profile', ...)`` (which now rejects 'profile' as unknown).
    task = Task(
        task_id="prof-empty",
        kind="profile",
        state="queued",
        params={},
        idempotency_key="prof-empty",
    )
    ctx_extra = {"workspace": str(workspace)}
    result = await ProfileExecutor()(
        RunnerContext(task=task, lease=None, extra=ctx_extra),
    )
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
