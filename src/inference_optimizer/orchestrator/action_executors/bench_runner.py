"""``bench_runner`` executor — re-measure throughput on the existing server.

Differs from :class:`BaselineExecutor` in two ways:

1. **Reuses the running server** (``KEEP_SERVER=1``). It assumes a
   previous baseline / param / kernel-opt action already launched the
   framework on ``$PORT`` and left it healthy.
2. **Updates ``current_tput`` only**, never ``baseline_tput``. The
   Conductor's ``_handle_update_state`` then derives
   ``cumulative_gain`` from the diff.

If the server isn't reachable the executor returns ``failed`` instead
of falling back; the LLM should re-run baseline first.
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


class BenchRunnerExecutor(ActionExecutor):
    """Re-runs ``run_baseline.sh`` against the live server."""

    name = "bench_runner"
    timeout_s = 30 * 60  # 30 min — bench-only, much faster than baseline

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        env_block = ctx.require_env(*_REQUIRED_ENV)

        results_dir = ctx.results_dir()
        # Re-use the same trace dir as baseline (if known) so re-profile
        # data accumulates in one place. Defaults to a new sub-dir.
        traces_dir = results_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        env = merged_env(
            ctx.env,
            env_block,
            {
                "RESULT_DIR": str(results_dir),
                "TRACE_DIR": str(traces_dir),
                "PORT": ctx.env.get("PORT", "8888"),
                "FRAMEWORK": ctx.env.get("FRAMEWORK", "sglang"),
                # The downstream script honours this to skip the
                # cleanup/kill step in its EXIT trap.
                "KEEP_SERVER": "1",
            },
        )

        script = skill_script("run_baseline.sh")
        log_path = results_dir / "bench_runner.log"

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
                notes=f"bench_runner: run_baseline.sh exited rc={rc}",
            )

        bench_json = find_first(results_dir, "baseline_*.json", "*.json")
        if bench_json is None:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"no baseline_*.json under {results_dir}",
            )

        m = parse_serving_metrics(bench_json)
        if "output_throughput" not in m:
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

        intents = [
            update_state_intent(
                {
                    "current_tput": tput_per_gpu,
                    "current_action": "bench_runner",
                },
                rationale=(
                    f"bench_runner: total={tput_total:.2f} tok/s, "
                    f"per_gpu={tput_per_gpu:.2f} tok/s/GPU (tp={tp:g})"
                ),
            ),
            send_message_intent(
                topic="event",
                body_md=(
                    f"bench_runner complete: {tput_per_gpu:.2f} tok/s/GPU"
                ),
                extras={
                    "kind": "bench_done",
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
            artifacts=[str(bench_json), str(log_path)],
            intents=intents,
            notes=f"bench {tput_per_gpu:.2f} tok/s/GPU",
        )


register_executor(BenchRunnerExecutor())


__all__ = ["BenchRunnerExecutor"]
