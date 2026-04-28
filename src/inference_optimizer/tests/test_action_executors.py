"""Tests for the Python↔shell ``ActionExecutor`` bridge.

We mock the shared :func:`run_subprocess` so no real GPU / sglang /
GEAK is required. Each executor is exercised end-to-end:

1. Required env vars validated → raises :class:`ExecutorEnvError`
   when missing.
2. Subprocess invoked with the right script path + env block.
3. Result file (metrics.json / results.tsv / filtered trace) parsed
   into the expected fields on :class:`ExecutorResult`.
4. Emitted intents include the right ``update_state`` /
   ``send_message`` payloads.

The :class:`SubAgentRunner` integration test verifies the executor
path wins over the LLM path when an executor is registered, and that
``intent_sink`` actually publishes intents.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import (
    EXECUTOR_REGISTRY,
    ExecutorContext,
    ExecutorEnvError,
    ExecutorResult,
    get_executor,
)
from inference_optimizer.orchestrator.action_executors import base as base_mod
from inference_optimizer.orchestrator.action_executors import (
    baseline as baseline_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    bench_runner as bench_runner_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    kernel_opt as kernel_opt_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    param_sweep_run as param_sweep_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    profile as profile_mod,
)
from inference_optimizer.orchestrator.intent_parser import IntentType
from inference_optimizer.orchestrator.action_registry import ActionMetadata
from inference_optimizer.orchestrator.execution_mode import ExecutionMode


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _FakeTask:
    """Mirrors task_registry.Task shape just enough for ExecutorContext."""

    task_id: str
    params: dict | None = None
    requires_lanes: list = None
    lease_ttl_sec: int = 0


def _make_action(name: str, **overrides) -> ActionMetadata:
    base = dict(
        name=name,
        family="prep",
        cost_minutes_p50=1.0,
        cost_minutes_p75=2.0,
        expected_gain_pct=(0.0, 0.0),
        accuracy_risk=0.0,
        crash_risk=0.0,
        prerequisites=(),
        requires_lanes=("benchmark_lane",),
        allowed_tools=("emit_intent",),
        side_effects=("reads_server",),
        allowed_modes=(ExecutionMode.QUICK_PARAM_SWEEP,
                       ExecutionMode.GUIDED_KERNEL_OPT,
                       ExecutionMode.MARATHON_MULTI_AGENT),
        preferred_backend="claude",
        preferred_model="claude-opus-4-7",
        max_turns=20,
        lease_ttl_sec=900,
        applicable_when=("any",),
    )
    base.update(overrides)
    return ActionMetadata(**base)


def _make_ctx(
    *,
    name: str,
    session_dir: Path,
    env: dict[str, str],
    params: dict | None = None,
) -> ExecutorContext:
    action = _make_action(name)
    task = _FakeTask(
        task_id=f"t-{name}",
        params={"action_name": name, "params": params or {}},
    )
    return ExecutorContext(
        task=task,
        action_meta=action,
        lanes_held=list(action.requires_lanes),
        session_dir=session_dir,
        env=dict(env),
    )


def _patch_run_subprocess(monkeypatch, side_effect):
    """Replace base.run_subprocess in every module that imports it."""
    monkeypatch.setattr(base_mod, "run_subprocess", side_effect)
    monkeypatch.setattr(baseline_mod, "run_subprocess", side_effect)
    monkeypatch.setattr(bench_runner_mod, "run_subprocess", side_effect)
    monkeypatch.setattr(profile_mod, "run_subprocess", side_effect)
    monkeypatch.setattr(param_sweep_mod, "run_subprocess", side_effect)
    monkeypatch.setattr(kernel_opt_mod, "run_subprocess", side_effect)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_has_five_executors():
    expected = {"baseline", "bench_runner", "profile",
                "param_sweep_run", "kernel_opt"}
    assert expected.issubset(set(EXECUTOR_REGISTRY))


def test_get_executor_normalizes_dashes():
    assert get_executor("kernel-opt") is get_executor("kernel_opt")
    assert get_executor("bench-runner") is get_executor("bench_runner")


def test_get_executor_returns_none_for_unknown():
    assert get_executor("definitely_not_an_action") is None


# ---------------------------------------------------------------------------
# BaselineExecutor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_baseline_missing_env_raises_executor_env_error(session_dir):
    ex = get_executor("baseline")
    ctx = _make_ctx(name="baseline", session_dir=session_dir, env={})
    with pytest.raises(ExecutorEnvError) as exc:
        await ex.run(ctx)
    msg = str(exc.value)
    for k in ("MODEL", "TP", "INFERENCEX_PATH"):
        assert k in msg


@pytest.mark.asyncio
async def test_baseline_writes_metrics_and_emits_update_state(
    monkeypatch, session_dir,
):
    """Mock subprocess that pretends run_baseline.sh dropped a metrics
    JSON, then verify the executor parses + emits the right intents."""
    ex = get_executor("baseline")
    ctx = _make_ctx(
        name="baseline", session_dir=session_dir,
        env=dict(MODEL="m", TP="2", CONC="16", ISL="1024", OSL="1024",
                 INFERENCEX_PATH="/fake/InferenceX"),
    )

    async def fake_subprocess(cmd, *, env, cwd, timeout_s, log_path):
        # Pretend the script ran and wrote one baseline_*.json
        results_dir = Path(env["RESULT_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / "baseline_sglang_tp2_conc16_isl1024_osl1024.json"
        out.write_text(json.dumps({
            "output_throughput": 8000.0,
            "total_token_throughput": 9000.0,
            "mean_tpot_ms": 12.5,
            "mean_ttft_ms": 30.0,
            "completed": 16,
            "num_prompts": 16,
        }))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    _patch_run_subprocess(monkeypatch, fake_subprocess)

    result = await ex.run(ctx)
    assert result.status == "succeeded"
    assert result.rc == 0
    assert pytest.approx(result.metrics["tput_per_gpu"], rel=1e-3) == 4000.0
    assert any(i.type == IntentType.UPDATE_STATE for i in result.intents)
    upd = next(i for i in result.intents if i.type == IntentType.UPDATE_STATE)
    assert upd.payload["changes"]["baseline_tput"] == pytest.approx(4000.0)
    assert upd.payload["changes"]["current_tput"] == pytest.approx(4000.0)


@pytest.mark.asyncio
async def test_baseline_failed_subprocess_returns_failed(monkeypatch, session_dir):
    ex = get_executor("baseline")
    ctx = _make_ctx(
        name="baseline", session_dir=session_dir,
        env=dict(MODEL="m", TP="1", CONC="1", ISL="1", OSL="1",
                 INFERENCEX_PATH="/x"),
    )

    async def fake_fail(cmd, *, env, cwd, timeout_s, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("boom\n")
        return 17

    _patch_run_subprocess(monkeypatch, fake_fail)
    result = await ex.run(ctx)
    assert result.status == "failed"
    assert result.rc == 17
    assert "rc=17" in result.notes


# ---------------------------------------------------------------------------
# BenchRunnerExecutor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bench_runner_emits_current_tput_only(monkeypatch, session_dir):
    ex = get_executor("bench_runner")
    ctx = _make_ctx(
        name="bench_runner", session_dir=session_dir,
        env=dict(MODEL="m", TP="4", CONC="8", ISL="1024", OSL="1024",
                 INFERENCEX_PATH="/x"),
    )

    async def fake_run(cmd, *, env, cwd, timeout_s, log_path):
        # Verify KEEP_SERVER=1 was passed (bench reuses live server).
        assert env.get("KEEP_SERVER") == "1"
        results_dir = Path(env["RESULT_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "baseline_x.json").write_text(json.dumps({
            "output_throughput": 12000.0, "mean_tpot_ms": 8.0,
        }))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    _patch_run_subprocess(monkeypatch, fake_run)
    result = await ex.run(ctx)
    assert result.status == "succeeded"
    assert pytest.approx(result.metrics["tput_per_gpu"], rel=1e-3) == 3000.0
    upd = next(i for i in result.intents if i.type == IntentType.UPDATE_STATE)
    # bench_runner must NOT touch baseline_tput
    assert "baseline_tput" not in upd.payload["changes"]
    assert "current_tput" in upd.payload["changes"]


# ---------------------------------------------------------------------------
# ProfileExecutor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_profile_finds_filtered_trace(monkeypatch, session_dir):
    ex = get_executor("profile")
    ctx = _make_ctx(
        name="profile", session_dir=session_dir,
        env=dict(MODEL="m", CONC="8", ISL="1024", OSL="1024",
                 INFERENCEX_PATH="/x"),
    )

    async def fake_run(cmd, *, env, cwd, timeout_s, log_path):
        traces = Path(env["TRACE_DIR"])
        traces.mkdir(parents=True, exist_ok=True)
        (traces / "filtered-TP-0.trace.json.gz").write_bytes(b"x" * 4096)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    _patch_run_subprocess(monkeypatch, fake_run)
    result = await ex.run(ctx)
    assert result.status == "succeeded"
    assert result.metrics["trace_size_bytes"] == 4096
    assert any("filtered-TP-0.trace.json.gz" in a for a in result.artifacts)


@pytest.mark.asyncio
async def test_profile_no_trace_means_failed(monkeypatch, session_dir):
    ex = get_executor("profile")
    ctx = _make_ctx(
        name="profile", session_dir=session_dir,
        env=dict(MODEL="m", CONC="1", ISL="1", OSL="1", INFERENCEX_PATH="/x"),
    )

    async def fake_run(cmd, *, env, cwd, timeout_s, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok but no trace\n")
        return 0  # rc 0 but no .json.gz produced

    _patch_run_subprocess(monkeypatch, fake_run)
    result = await ex.run(ctx)
    assert result.status == "failed"
    assert "filtered TP-0" in result.notes


# ---------------------------------------------------------------------------
# ParamSweepRunExecutor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_param_sweep_picks_best_row(monkeypatch, session_dir):
    ex = get_executor("param_sweep_run")
    ctx = _make_ctx(
        name="param_sweep_run", session_dir=session_dir,
        env=dict(MODEL="m", TP="2", INFERENCEX_PATH="/x"),
    )

    async def fake_run(cmd, *, env, cwd, timeout_s, log_path):
        results_dir = Path(env["RESULT_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        # 3 rows; best output_tput = 6500
        tsv = results_dir / "results.tsv"
        tsv.write_text(
            "framework\tconc\tisl\tosl\toutput_tput\ttotal_tput\tttft_ms\t"
            "tpot_ms\titl_ms\te2el_ms\tstatus\tdescription\n"
            "sglang\t4\t1024\t1024\t4000.0\t5000.0\t30\t10\t8\t40\tswept\tx\n"
            "sglang\t16\t1024\t1024\t6500.0\t7000.0\t35\t12\t9\t45\tswept\tx\n"
            "sglang\t64\t8192\t1024\t3000.0\t-\t-\t-\t-\t-\tskipped\tlimit\n"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    _patch_run_subprocess(monkeypatch, fake_run)
    result = await ex.run(ctx)
    assert result.status == "succeeded"
    assert pytest.approx(result.metrics["tput_per_gpu"], rel=1e-3) == 3250.0
    assert result.metrics["best_conc"] == "16"
    upd = next(i for i in result.intents if i.type == IntentType.UPDATE_STATE)
    assert upd.payload["changes"]["current_tput"] == pytest.approx(3250.0)


# ---------------------------------------------------------------------------
# KernelOptExecutor
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kernel_opt_requires_kernel_candidates(session_dir):
    ex = get_executor("kernel_opt")
    ctx = _make_ctx(
        name="kernel_opt", session_dir=session_dir,
        env=dict(INFERENCEX_PATH="/x", KERNEL_OPT_BACKENDS="geak"),
        params={},  # missing kernel_candidates
    )
    with pytest.raises(ExecutorEnvError) as exc:
        await ex.run(ctx)
    assert "kernel_candidates" in str(exc.value)


@pytest.mark.asyncio
async def test_kernel_opt_geak_only_succeeds(monkeypatch, session_dir, tmp_path):
    ex = get_executor("kernel_opt")
    candidate = tmp_path / "kernel_a.py"
    candidate.write_text("# fake kernel")
    ctx = _make_ctx(
        name="kernel_opt", session_dir=session_dir,
        env=dict(INFERENCEX_PATH="/x", KERNEL_OPT_BACKENDS="geak"),
        params={"kernel_candidates": [str(candidate)]},
    )

    seen_cmds: list[list[str]] = []

    # Note: kernel_opt's _submit_geak / _submit_oob don't pass cwd, so
    # accept it as optional to mirror the real run_subprocess signature.
    async def fake_run(cmd, *, env, log_path, timeout_s=None, cwd=None):
        seen_cmds.append(list(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("geak ok\n")
        return 0

    _patch_run_subprocess(monkeypatch, fake_run)
    result = await ex.run(ctx)
    assert result.status == "succeeded", f"unexpected: {result.notes}"
    assert result.metrics["n_succeeded"] == 1
    assert result.metrics["n_failed"] == 0
    # GEAK invocation includes the candidate path + --yolo.
    assert any("geak_ray_submit.py" in c for cmd in seen_cmds for c in cmd)
    assert any(str(candidate) in c for cmd in seen_cmds for c in cmd)
    # Should also propose a follow-up integrate.
    propose = next(i for i in result.intents if i.type == IntentType.PROPOSE_ACTION)
    assert propose.payload["action_name"] == "integrate"


@pytest.mark.asyncio
async def test_kernel_opt_partial_failure_returns_succeeded(
    monkeypatch, session_dir, tmp_path,
):
    """If at least one backend succeeds, status is succeeded (best-of-N)."""
    ex = get_executor("kernel_opt")
    cand = tmp_path / "k.py"
    cand.write_text("# x")
    ctx = _make_ctx(
        name="kernel_opt", session_dir=session_dir,
        env=dict(INFERENCEX_PATH="/x", KERNEL_OPT_BACKENDS="geak"),
        params={"kernel_candidates": [str(cand)]},
    )
    state = {"calls": 0}

    async def flaky(cmd, *, env, log_path, timeout_s=None, cwd=None):
        state["calls"] += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("x\n")
        return 0  # the only call succeeds

    _patch_run_subprocess(monkeypatch, flaky)
    result = await ex.run(ctx)
    assert result.status == "succeeded"
    assert state["calls"] == 1


# ---------------------------------------------------------------------------
# SubAgentRunner integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sub_agent_runner_prefers_executor_over_llm(
    monkeypatch, session_dir,
):
    """When an executor is registered for the action, the LLM path is
    skipped and the executor's intents flow through ``intent_sink``."""
    from inference_optimizer.orchestrator.sub_agent_runner import (
        SubAgentRunner, TaskResult,
    )
    from inference_optimizer.orchestrator.action_executors.base import (
        ActionExecutor, ExecutorResult, _normalize,
    )
    from inference_optimizer.orchestrator.intent_parser import (
        Intent, IntentType,
    )

    class _StubBackend:
        called = False
        async def run(self, *args, **kwargs):
            type(self).called = True
            return []

    class _FakeLocks:
        async def acquire(self, *a, **kw):
            return object()
        async def release(self, lease):
            return None

    class _Tasks:
        states: list[tuple[str, str]] = []
        async def transition(self, task_id, new_state, *, evidence=None):
            type(self).states.append((task_id, new_state))

    class _Reg:
        actions = {"baseline": _make_action("baseline")}
        def get(self, name):
            return self.actions.get(name)
        def system_prompt_for(self, name):
            return ""
        def __len__(self):
            return len(self.actions)

    class _Exec(ActionExecutor):
        name = "baseline"
        async def run(self, ctx):
            return ExecutorResult(
                status="succeeded",
                metrics={"tput_per_gpu": 4321.0},
                artifacts=[],
                intents=[
                    Intent(IntentType.UPDATE_STATE, payload={
                        "changes": {"baseline_tput": 4321.0,
                                    "current_tput": 4321.0},
                    }),
                ],
                notes="executor ok",
            )

    sink_calls: list[tuple[str, IntentType]] = []

    async def sink(agent, intent):
        sink_calls.append((agent, intent.type))

    runner = SubAgentRunner(
        backend=_StubBackend(),
        policy=object(),  # validate_intent never called via executor path
        locks=_FakeLocks(),
        action_registry=_Reg(),
        tasks=_Tasks(),
        workspace=session_dir,
        agent_name="sub-agent",
        env={},
        executor_registry={_normalize("baseline"): _Exec()},
        intent_sink=sink,
    )

    task = _FakeTask(task_id="t1",
                     params={"action_name": "baseline", "params": {}})
    result = await runner.run(task)

    assert isinstance(result, TaskResult)
    assert result.status == "succeeded"
    assert _StubBackend.called is False, "LLM backend must not be called"
    # executor's intent went through sink
    assert sink_calls == [("sub-agent", IntentType.UPDATE_STATE)]
    # task transitioned through running -> succeeded
    assert ("t1", "running") in _Tasks.states
    assert ("t1", "succeeded") in _Tasks.states


@pytest.mark.asyncio
async def test_sub_agent_runner_falls_back_to_llm_on_env_error(
    monkeypatch, session_dir,
):
    """When the executor raises ExecutorEnvError, runner uses the LLM
    backend instead and the LLM intents flow through intent_sink."""
    from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
    from inference_optimizer.orchestrator.action_executors.base import (
        ActionExecutor, _normalize,
    )
    from inference_optimizer.orchestrator.intent_parser import (
        Intent, IntentType,
    )

    class _StubBackend:
        called = False
        async def run(self, *args, **kwargs):
            type(self).called = True
            return [Intent(IntentType.SEND_MESSAGE,
                           payload={"topic": "heartbeat",
                                    "body_md": "llm fallback"})]

    class _FakeLocks:
        async def acquire(self, *a, **kw):
            return object()
        async def release(self, lease):
            return None

    class _Tasks:
        async def transition(self, *a, **kw):
            return None

    class _Reg:
        actions = {"baseline": _make_action("baseline")}
        def get(self, n):
            return self.actions.get(n)
        def system_prompt_for(self, n):
            return ""

    class _OptOutExec(ActionExecutor):
        name = "baseline"
        async def run(self, ctx):
            raise ExecutorEnvError("MODEL missing")

    runner = SubAgentRunner(
        backend=_StubBackend(),
        policy=object(),
        locks=_FakeLocks(),
        action_registry=_Reg(),
        tasks=_Tasks(),
        workspace=session_dir,
        agent_name="sub-agent",
        env={},  # missing required vars on purpose
        executor_registry={_normalize("baseline"): _OptOutExec()},
    )

    task = _FakeTask(task_id="t2",
                     params={"action_name": "baseline", "params": {}})
    result = await runner.run(task)
    assert _StubBackend.called is True, "LLM backend must be invoked on fallback"
    assert result.status == "succeeded"
    assert any(i.type == IntentType.SEND_MESSAGE for i in result.intents)


# ---------------------------------------------------------------------------
# Conductor cumulative_gain auto-derivation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_conductor_auto_derives_cumulative_gain(session_dir, db):
    """An update_state that sets baseline_tput then current_tput should
    leave cumulative_gain populated automatically."""
    from inference_optimizer.orchestrator.conductor import Conductor
    from inference_optimizer.orchestrator.intent_parser import (
        Intent, IntentType,
    )

    conductor = Conductor(
        session_dir,
        env={"MODEL_PATH": "fake/m", "MAX_HOURS": "0.5"},
        db=db,
    )
    await conductor._bootstrap()
    assert conductor.ctx is not None

    # 1) baseline write — cumulative_gain stays 0
    await conductor._handle_update_state(
        "executor", {"changes": {"baseline_tput": 5000.0,
                                  "current_tput": 5000.0}},
    )
    assert conductor.ctx.state.baseline_tput == 5000.0
    assert conductor.ctx.state.cumulative_gain == 0.0

    # 2) bench_runner improvement — gain should be 10%
    await conductor._handle_update_state(
        "executor", {"changes": {"current_tput": 5500.0}},
    )
    assert conductor.ctx.state.current_tput == 5500.0
    assert conductor.ctx.state.cumulative_gain == pytest.approx(10.0)

    # 3) regression — negative gain works
    await conductor._handle_update_state(
        "executor", {"changes": {"current_tput": 4500.0}},
    )
    assert conductor.ctx.state.cumulative_gain == pytest.approx(-10.0)
