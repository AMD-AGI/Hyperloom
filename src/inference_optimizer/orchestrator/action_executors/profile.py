"""``profile`` executor — capture filtered TP-0 trace via run_profile.sh.

Assumes a server is already running (just like
:class:`BenchRunnerExecutor`). Activates the framework's torch.profiler
through the ``/start_profile`` HTTP endpoint, drains for the configured
prompt count, then waits for traces to be flushed to NFS.

We don't try to parse trace contents here — that's TraceLens's job.
The executor only:

* runs the script,
* finds the filtered trace path,
* emits a ``send_message`` event so the LLM (executor reactor) can pick
  it up and propose follow-up profile-driven actions (e.g. kernel-opt
  candidates).
"""
from __future__ import annotations

import logging

from ...paths import asset_script
from ._helpers import find_first, merged_env, send_message_intent
from .base import (
    ActionExecutor,
    ExecutorContext,
    ExecutorResult,
    register_executor,
    run_subprocess,
)


log = logging.getLogger(__name__)


_REQUIRED_ENV = ("MODEL", "CONC", "ISL", "OSL", "INFERENCEX_PATH")


class ProfileExecutor(ActionExecutor):
    """Wraps ``run_profile.sh`` (assumes server already up)."""

    name = "profile"
    timeout_s = 30 * 60

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        env_block = ctx.require_env(*_REQUIRED_ENV)

        results_dir = ctx.results_dir()
        traces_dir = results_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        env = merged_env(
            ctx.env, env_block,
            {
                "RESULT_DIR": str(results_dir),
                "TRACE_DIR": str(traces_dir),
                "PORT": ctx.env.get("PORT", "8888"),
                "FRAMEWORK": ctx.env.get("FRAMEWORK", "sglang"),
            },
        )

        script = asset_script("run_profile.sh")
        log_path = results_dir / "run_profile.log"

        rc = await run_subprocess(
            ["bash", str(script)],
            env=env, cwd=results_dir,
            timeout_s=self.timeout_s, log_path=log_path,
        )

        if rc != 0:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"run_profile.sh exited rc={rc}; see {log_path}",
            )

        # Locate a filtered trace. Order of preference:
        #   1. filtered TP-0 in this task's traces dir
        #   2. filtered TP-0 anywhere under <session>/results/**/traces/
        #      (the live sglang server keeps writing to wherever
        #      ``SGLANG_TORCH_PROFILER_DIR`` was set at server-launch
        #      time — usually the baseline task's dir)
        filtered = find_first(traces_dir, "filtered-TP-0.trace.json.gz")
        if filtered is None:
            session_results = ctx.session_dir / "results"
            filtered = find_first(session_results, "filtered-TP-0.trace.json.gz")

        if filtered is None:
            # rc=0 means the script ran cleanly; no trace artefact this round
            # is treated as a soft skip (NOT a failure). This avoids the
            # pathological loop where the LLM keeps re-delegating profile
            # because the executor flags it failed despite a clean exit.
            # The executor reactor sees ``kind=profile_skipped`` and is
            # nudged toward param_sweep / kernel_opt / bench_runner.
            return ExecutorResult(
                status="succeeded", rc=rc,
                metrics={"trace_size_bytes": 0, "trace_skipped": 1},
                artifacts=[str(log_path)],
                intents=[
                    send_message_intent(
                        topic="event",
                        body_md=(
                            f"profile completed (rc=0) but no new trace was "
                            f"written this round; the running server still "
                            f"writes to its launch-time SGLANG_TORCH_PROFILER_DIR. "
                            f"Move on to bench_runner / param_sweep_run / "
                            f"kernel_opt — do NOT re-delegate profile."
                        ),
                        extras={
                            "kind": "profile_skipped",
                            "reason": "no_new_trace",
                            "results_dir": str(results_dir),
                            "next_actions_hint": [
                                "bench_runner", "param_sweep_run", "kernel_opt",
                            ],
                        },
                    ),
                ],
                notes=(
                    f"no filtered TP-0 trace under {traces_dir} or any "
                    f"sibling task dir; flagged as soft-skip (rc=0)"
                ),
            )

        intents = [
            send_message_intent(
                topic="event",
                body_md=(
                    f"profile complete; filtered trace at {filtered.name} "
                    f"({filtered.stat().st_size // 1024} KB)"
                ),
                extras={
                    "kind": "profile_done",
                    "trace_path": str(filtered),
                    "results_dir": str(results_dir),
                },
            ),
        ]

        return ExecutorResult(
            status="succeeded", rc=rc,
            metrics={
                "trace_size_bytes": filtered.stat().st_size,
            },
            artifacts=[str(filtered), str(log_path)],
            intents=intents,
            notes=f"profile resolved trace at {filtered}",
        )


register_executor(ProfileExecutor())


__all__ = ["ProfileExecutor"]
