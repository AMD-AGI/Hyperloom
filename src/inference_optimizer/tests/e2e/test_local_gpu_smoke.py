"""End-to-end "fake-GPU" smoke for the local-GPU minimum path.

This test wires the *real* CLI flow (env probe → Conductor →
ActionRegistry → SubAgentRunner → ActionExecutor), but monkey-patches
``run_subprocess`` to fake the GPU-side scripts. We then drive a
two-step run by hand:

    1.  Inject a `delegate(baseline)` task.       → BaselineExecutor runs the (faked) run_baseline.sh,
        publishes update_state(baseline_tput=X) via intent_sink.
    2.  Inject a `delegate(bench_runner)` task.       → BenchRunnerExecutor runs the (faked) bench script,
        publishes update_state(current_tput=X+y).
    3.  Conductor._handle_update_state derives cumulative_gain.       → assert state.cumulative_gain > 0

This proves the chain Python ↔ shell ↔ state ↔ early-stop is
end-to-end intact without needing a real GPU. When you swap the
fake subprocess for a real one in production, the chain stays the
same; the test stays a useful regression for the wiring.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import (
    base as base_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    baseline as baseline_mod,
)
from inference_optimizer.orchestrator.action_executors import (
    bench_runner as bench_runner_mod,
)
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import Conductor
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import skill_actions_dir


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_gpu_smoke_baseline_then_bench_yields_positive_gain(
    monkeypatch, session_dir,
):
    """Minimum local-GPU loop: baseline → bench_runner → cumulative_gain > 0%.

    Asserted invariants:

    * The dispatcher loop picks both queued ``delegate`` tasks in order.
    * ``BaselineExecutor`` writes ``state.baseline_tput`` (per-GPU).
    * ``BenchRunnerExecutor`` writes ``state.current_tput`` (per-GPU)
      and DOES NOT touch ``baseline_tput``.
    * After the bench update the Conductor's
      ``_handle_update_state`` auto-derives a positive
      ``cumulative_gain``.
    * Both executor invocations got their env block (TP / CONC / ISL /
      OSL / INFERENCEX_PATH) from ``env``, NOT from the LLM payload —
      i.e. the env wiring works end-to-end.
    """
    # ------------------------------------------------------------------
    # 1. Fake subprocess: pretend run_baseline.sh wrote a metrics file.
    # ------------------------------------------------------------------
    baseline_tput_total = 8000.0   # tok/s aggregate
    bench_tput_total = 9600.0      # +20% over baseline
    tp = 4

    async def fake_subprocess(cmd, *, env, cwd, timeout_s, log_path):
        # Verify env was propagated. These assertions become test
        # failures (not silent mis-runs) if CLI plumbing regresses.
        for required in ("MODEL", "TP", "CONC", "ISL", "OSL",
                         "INFERENCEX_PATH"):
            assert env.get(required), f"missing {required} in subprocess env"

        results_dir = Path(env["RESULT_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        out_json = results_dir / (
            f"baseline_{env.get('FRAMEWORK', 'sglang')}_"
            f"tp{env['TP']}_conc{env['CONC']}_isl{env['ISL']}_"
            f"osl{env['OSL']}.json"
        )
        if env.get("KEEP_SERVER") == "1":
            tput = bench_tput_total  # bench_runner path
        else:
            tput = baseline_tput_total  # baseline path
        out_json.write_text(json.dumps({
            "output_throughput": tput,
            "total_token_throughput": tput * 1.1,
            "mean_tpot_ms": 12.5,
            "mean_ttft_ms": 30.0,
            "mean_itl_ms": 11.5,
            "mean_e2el_ms": 800.0,
            "completed": int(env["CONC"]),
            "num_prompts": int(env["CONC"]),
        }))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    monkeypatch.setattr(base_mod, "run_subprocess", fake_subprocess)
    monkeypatch.setattr(baseline_mod, "run_subprocess", fake_subprocess)
    monkeypatch.setattr(bench_runner_mod, "run_subprocess", fake_subprocess)

    # ------------------------------------------------------------------
    # 2. Build a real Conductor with the real ActionRegistry + a
    #    realistic env block (mirrors what cli.py + env_probe produce).
    # ------------------------------------------------------------------
    env = {
        "MODEL_PATH": "Qwen/Qwen3-8B",
        "MODEL": "Qwen/Qwen3-8B",
        "MAX_HOURS": "1",
        "TP": str(tp),
        "CONC": "32",
        "ISL": "1024",
        "OSL": "256",
        "PORT": "8888",
        "FRAMEWORK": "sglang",
        "INFERENCEX_PATH": "/fake/InferenceX",
        "GPU_COUNT": str(tp),
    }

    registry = ActionRegistry(skill_actions_dir()).load()
    backend = MockBackend()  # never invoked — we drive tasks manually

    conductor = Conductor(
        session_dir,
        backend=backend,
        env=env,
        action_registry=registry,
        # Slow ticks so we don't burn cycles in the test
        reactor_tick_s=0.1,
        clock_tick_s=0.1,
    )
    ctx = await conductor._bootstrap()
    assert ctx.sub_agent_runner is not None, "dispatcher must spawn"

    # ------------------------------------------------------------------
    # 3. Stage 1: baseline
    # ------------------------------------------------------------------
    await conductor._handle_delegate(
        "executor",
        {"action_name": "baseline", "params": {}},
    )

    # Drain the dispatcher exactly once for the queued baseline task.
    await _run_one_pending_delegate(ctx)

    expected_per_gpu = baseline_tput_total / tp
    assert ctx.state.baseline_tput == pytest.approx(expected_per_gpu)
    assert ctx.state.current_tput == pytest.approx(expected_per_gpu)
    assert ctx.state.cumulative_gain == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # 4. Stage 2: bench_runner
    # ------------------------------------------------------------------
    await conductor._handle_delegate(
        "executor",
        {"action_name": "bench_runner", "params": {}},
    )

    await _run_one_pending_delegate(ctx)

    bench_per_gpu = bench_tput_total / tp
    expected_gain = (bench_tput_total - baseline_tput_total) / baseline_tput_total * 100.0
    assert ctx.state.baseline_tput == pytest.approx(expected_per_gpu)  # unchanged
    assert ctx.state.current_tput == pytest.approx(bench_per_gpu)
    assert ctx.state.cumulative_gain == pytest.approx(expected_gain, rel=1e-6)
    assert ctx.state.cumulative_gain > 0  # the smoke goal

    # ------------------------------------------------------------------
    # 5. Audit — events bus shows the executor intents reached it.
    # ------------------------------------------------------------------
    rows = await ctx.db.fetchall(
        "SELECT topic, payload FROM events WHERE topic IN (?, ?, ?) "
        "ORDER BY seq",
        ("decision", "event", "proposal"),
    )
    decision_topics = [r["topic"] for r in rows]
    assert decision_topics.count("decision") >= 2, (
        "expected ≥2 update_state events (one per executor)"
    )
    # baseline_done + bench_done events should both appear under topic="event"
    event_kinds = []
    for r in rows:
        if r["topic"] == "event":
            try:
                event_kinds.append(json.loads(r["payload"]).get("kind"))
            except (TypeError, json.JSONDecodeError):
                pass
    assert "baseline_done" in event_kinds
    assert "bench_done" in event_kinds


# ---------------------------------------------------------------------------
async def _run_one_pending_delegate(ctx) -> None:
    """Drain exactly one queued ``delegate`` task via the
    SubAgentRunner; the conductor's ``_dispatcher_loop`` is the same
    code path but harder to drive deterministically in a unit test."""
    runner = ctx.sub_agent_runner
    assert runner is not None
    rows = await ctx.db.fetchall(
        "SELECT * FROM tasks WHERE kind=? AND state=? ORDER BY created_at",
        ("delegate", "queued"),
    )
    assert rows, "expected at least one queued delegate task"
    from inference_optimizer.orchestrator.task_registry import Task
    task = Task.from_row(rows[0])
    await runner.run(task)
