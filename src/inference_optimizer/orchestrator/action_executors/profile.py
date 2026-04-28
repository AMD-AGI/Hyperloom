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

from ...paths import skill_script
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

        script = skill_script("run_profile.sh")
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

        # Locate the filtered trace.
        filtered = find_first(traces_dir, "filtered-TP-0.trace.json.gz")
        if filtered is None:
            return ExecutorResult(
                status="failed", rc=rc,
                notes=f"no filtered TP-0 trace produced under {traces_dir}",
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
            notes=f"profile written to {filtered}",
        )


register_executor(ProfileExecutor())


__all__ = ["ProfileExecutor"]
