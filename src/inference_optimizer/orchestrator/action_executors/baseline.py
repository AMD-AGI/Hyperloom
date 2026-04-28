"""``baseline`` executor — first-light measurement of the unmodified server.

Wraps the bundled ``run_baseline.sh`` (under
``.cursor/skills/inference-optimizer/scripts/``). It launches the
target framework (sglang / vllm), waits for ``/health``, runs a
benchmark, then activates profiling via ``/start_profile`` and emits
filtered traces — all in a single process so the framework stays warm.

Result mapping (DESIGN §6.3 SharedState fields):

* ``output_throughput`` (tok/s aggregate) → metrics["tput"]
* ``output_throughput / TP`` → ``state.baseline_tput`` (tok/s/GPU)
* ``baseline_*.json`` artifact path → ``ExecutorResult.artifacts``

Required env vars (validated up-front via :meth:`require_env`):

    MODEL          model path / hub id
    TP             tensor-parallel size
    CONC           concurrent requests
    ISL            input sequence length
    OSL            output sequence length
    INFERENCEX_PATH path to the InferenceX checkout

If any of those is missing we raise :class:`ExecutorEnvError` so
:class:`SubAgentRunner` falls back to the LLM-driven path. This means a
mock-mode run on a developer laptop never tries to launch sglang.
"""
from __future__ import annotations

import logging

from ...paths import skill_script
from ._helpers import (
    find_first,
    merged_env,
    parse_serving_metrics,
    send_message_intent,
    update_state_intent,
)
from .base import (
    ActionExecutor,
    ExecutorContext,
    ExecutorResult,
    register_executor,
    run_subprocess,
)


log = logging.getLogger(__name__)


_REQUIRED_ENV = ("MODEL", "TP", "CONC", "ISL", "OSL", "INFERENCEX_PATH")


class BaselineExecutor(ActionExecutor):
    """Runs ``run_baseline.sh`` then publishes ``baseline_tput`` to state."""

    name = "baseline"
    timeout_s = 60 * 60  # 1 hour

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        env_block = ctx.require_env(*_REQUIRED_ENV)

        results_dir = ctx.results_dir()
        traces_dir = ctx.session_dir / "results" / ctx.task_id / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        env = merged_env(
            ctx.env,
            env_block,
            {
                "RESULT_DIR": str(results_dir),
                "TRACE_DIR": str(traces_dir),
                "PORT": ctx.env.get("PORT", "8888"),
                "FRAMEWORK": ctx.env.get("FRAMEWORK", "sglang"),
            },
        )

        script = skill_script("run_baseline.sh")
        log_path = results_dir / "run_baseline.log"

        rc = await run_subprocess(
            ["bash", str(script)],
            env=env,
            cwd=results_dir,
            timeout_s=self.timeout_s,
            log_path=log_path,
        )

        if rc != 0:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"run_baseline.sh exited rc={rc}; see {log_path}",
            )

        # Find the first ``baseline_*.json`` written by benchmark_serving.
        bench_json = find_first(results_dir, "baseline_*.json", "*.json")
        if bench_json is None:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"no baseline_*.json under {results_dir}",
            )

        m = parse_serving_metrics(bench_json)
        if not m or "output_throughput" not in m:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"could not parse output_throughput from {bench_json}",
            )

        try:
            tp = float(env_block["TP"])
        except ValueError:
            tp = 1.0
        tput_total = float(m["output_throughput"])
        tput_per_gpu = tput_total / tp if tp > 0 else tput_total

        artifacts = [str(bench_json), str(log_path)]
        # Filtered trace, if produced
        filtered = find_first(traces_dir, "filtered-TP-0.trace.json.gz")
        if filtered is not None:
            artifacts.append(str(filtered))

        intents = [
            update_state_intent(
                {
                    "baseline_tput": tput_per_gpu,
                    "current_tput": tput_per_gpu,
                    "current_action": "baseline",
                },
                rationale=(
                    f"baseline measured: total={tput_total:.2f} tok/s, "
                    f"per_gpu={tput_per_gpu:.2f} tok/s/GPU (tp={tp:g})"
                ),
            ),
            send_message_intent(
                topic="event",
                body_md=(
                    f"baseline complete: {tput_per_gpu:.2f} tok/s/GPU "
                    f"(p50_tpot={m.get('mean_tpot_ms', 0):.2f}ms, "
                    f"p50_ttft={m.get('mean_ttft_ms', 0):.2f}ms)"
                ),
                extras={
                    "kind": "baseline_done",
                    "tput_per_gpu": tput_per_gpu,
                    "tput_total": tput_total,
                    "metrics": m,
                    "artifact_path": str(bench_json),
                },
            ),
        ]

        return ExecutorResult(
            status="succeeded", rc=rc,
            metrics={
                "tput_total": tput_total,
                "tput_per_gpu": tput_per_gpu,
                **{k: v for k, v in m.items() if k != "output_throughput"},
            },
            artifacts=artifacts,
            intents=intents,
            notes=(
                f"baseline {tput_per_gpu:.2f} tok/s/GPU "
                f"(metrics from {bench_json.name})"
            ),
        )


register_executor(BaselineExecutor())


__all__ = ["BaselineExecutor"]
