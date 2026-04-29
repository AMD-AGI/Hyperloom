"""``kernel_opt`` executor — GEAK / OOB submission via Ray.

Wraps the bundled Ray submission scripts:

* ``geak_ray_submit.py``   — schedules ``geak`` CLI tasks with isolated
  GPUs (one task per kernel candidate).
* ``oob_ray_submit.py``    — schedules ``oob run -a {claude|codex}``
  tasks for the OOB-driven kernel rewrite path.

Backends are resolved from ``KERNEL_OPT_BACKENDS`` env (default
``"geak,codex"``) with the constraint that at least one must be present
in :data:`SUPPORTED_BACKENDS`. The executor *does not* benchmark the
optimised kernels — that's a follow-up :class:`BenchRunnerExecutor`
delegation. Here we only:

1. Resolve the candidate kernel paths from
   ``ctx.task_params["kernel_candidates"]`` (list of paths).
2. Submit one Ray task per backend × candidate, in parallel.
3. Wait for completion, gather output paths.
4. Emit a ``send_message`` event listing the produced kernel files +
   one ``propose_action`` intent suggesting an ``integrate`` follow-up
   so the Conductor → SubAgentRunner chain re-benchmarks.

Required env (validated up-front):

    KERNEL_OPT_BACKENDS  comma-separated subset of {"geak","codex","claude"}
    INFERENCEX_PATH      InferenceX checkout path
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ...paths import asset_script
from ..intent_parser import Intent, IntentType
from ._helpers import merged_env, send_message_intent
from .base import (
    ActionExecutor,
    ExecutorContext,
    ExecutorEnvError,
    ExecutorResult,
    register_executor,
    run_subprocess,
)


log = logging.getLogger(__name__)


SUPPORTED_BACKENDS = ("geak", "codex", "claude")


def _parse_backends(raw: str) -> list[str]:
    items = [b.strip().lower() for b in (raw or "").split(",") if b.strip()]
    out = [b for b in items if b in SUPPORTED_BACKENDS]
    if not out:
        # Default per design §4.6
        out = ["geak", "codex"]
    return out


async def _submit_geak(
    *, candidate: Path, env: dict[str, str], log_path: Path,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    """Submit one GEAK task. Returns (rc, telemetry)."""
    script = asset_script("geak_ray_submit.py")
    cmd = [
        "python3", str(script),
        "run", "-t", str(candidate), "--yolo",
    ]
    rc = await run_subprocess(
        cmd, env=env, timeout_s=timeout_s, log_path=log_path,
    )
    return rc, {"backend": "geak", "candidate": str(candidate),
                "log": str(log_path)}


async def _submit_oob(
    *, agent: str, prompt_file: Path, kernel_file: Path,
    env: dict[str, str], log_path: Path, timeout_s: float,
    max_turns: int,
) -> tuple[int, dict[str, Any]]:
    """Submit one OOB (codex/claude) round."""
    script = asset_script("oob_ray_submit.py")
    cmd = [
        "python3", str(script), "run",
        "-a", agent,
        "-p", f"@{prompt_file}",
        "-f", str(kernel_file),
        "--max-turns", str(max_turns),
        "--no-live", "--json",
    ]
    rc = await run_subprocess(
        cmd, env=env, timeout_s=timeout_s, log_path=log_path,
    )
    return rc, {"backend": agent, "candidate": str(kernel_file),
                "log": str(log_path)}


class KernelOptExecutor(ActionExecutor):
    """Wraps GEAK + OOB Ray submitters in parallel per candidate."""

    name = "kernel_opt"
    timeout_s = 60 * 60  # 1 hour total cap; per-task gets a smaller share

    GEAK_TASK_TIMEOUT_S = 30 * 60   # GEAK rounds: ~10-30 min
    OOB_ROUND_TIMEOUT_S = 15 * 60   # codex/claude per-iter: 2-15 min
    OOB_MAX_TURNS = 30

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        backends_raw = ctx.env.get("KERNEL_OPT_BACKENDS", "geak,codex")
        backends = _parse_backends(backends_raw)

        # Required env: just InferenceX path (Ray scripts also need
        # OOB_API_KEY / OPENAI_API_KEY but those are set per backend).
        ctx.require_env("INFERENCEX_PATH")

        candidates_raw = ctx.task_params.get("kernel_candidates")
        if not candidates_raw or not isinstance(candidates_raw, (list, tuple)):
            raise ExecutorEnvError(
                "kernel_opt requires task_params.kernel_candidates "
                "(list of kernel source file paths)"
            )
        candidates: list[Path] = []
        for c in candidates_raw:
            p = Path(str(c))
            if not p.is_file():
                log.warning("kernel candidate not found: %s", p)
                continue
            candidates.append(p)

        if not candidates:
            return ExecutorResult(
                status="failed",
                notes="no usable kernel candidates",
            )

        results_dir = ctx.results_dir()
        env = merged_env(ctx.env, {})

        # Per-candidate × per-backend tasks
        tasks: list[asyncio.Task] = []
        plan: list[tuple[str, Path]] = []  # (backend, candidate)
        for cand in candidates:
            for backend in backends:
                log_path = (
                    results_dir / f"{backend}_{cand.stem}.log"
                )
                if backend == "geak":
                    tasks.append(asyncio.create_task(
                        _submit_geak(
                            candidate=cand, env=env, log_path=log_path,
                            timeout_s=self.GEAK_TASK_TIMEOUT_S,
                        ),
                        name=f"geak-{cand.stem}",
                    ))
                else:  # codex / claude
                    prompt_file = ctx.task_params.get("prompt_file")
                    if not prompt_file:
                        log.warning(
                            "%s skipped: task_params.prompt_file required",
                            backend,
                        )
                        continue
                    tasks.append(asyncio.create_task(
                        _submit_oob(
                            agent=backend,
                            prompt_file=Path(str(prompt_file)),
                            kernel_file=cand,
                            env=env, log_path=log_path,
                            timeout_s=self.OOB_ROUND_TIMEOUT_S,
                            max_turns=self.OOB_MAX_TURNS,
                        ),
                        name=f"oob-{backend}-{cand.stem}",
                    ))
                plan.append((backend, cand))

        if not tasks:
            return ExecutorResult(
                status="failed",
                notes="no Ray tasks dispatched (missing prompt_file?)",
            )

        # Wait for all in parallel — each has its own internal timeout.
        per_task_results = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Aggregate
        succeeded: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for (backend, cand), outcome in zip(plan, per_task_results):
            if isinstance(outcome, Exception):
                failed.append({
                    "backend": backend, "candidate": str(cand),
                    "error": repr(outcome),
                })
                continue
            rc, telemetry = outcome
            telemetry["rc"] = rc
            if rc == 0:
                succeeded.append(telemetry)
            else:
                failed.append(telemetry)

        intents: list[Intent] = []
        artifacts: list[str] = []
        for t in (*succeeded, *failed):
            artifacts.append(t.get("log", ""))

        intents.append(send_message_intent(
            topic="event",
            body_md=(
                f"kernel_opt round complete: {len(succeeded)} succeeded, "
                f"{len(failed)} failed across {len(backends)} backend(s)"
            ),
            extras={
                "kind": "kernel_opt_done",
                "succeeded": succeeded,
                "failed": failed,
                "backends": backends,
            },
        ))

        # If anything succeeded, propose an integrate follow-up so the
        # next reactor turn can apply patches and re-benchmark.
        if succeeded:
            intents.append(Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={
                    "action_name": "integrate",
                    "predicted_gain_pct": 5.0,  # conservative placeholder
                    "params": {
                        "winning_outputs": succeeded,
                    },
                    "reason": "kernel_opt produced candidate patches",
                },
            ))

        status = "succeeded" if succeeded else "failed"
        notes = (
            f"kernel_opt: {len(succeeded)}/{len(plan)} task(s) succeeded "
            f"across backends={backends}"
        )

        return ExecutorResult(
            status=status,
            metrics={
                "n_succeeded": len(succeeded),
                "n_failed": len(failed),
                "n_candidates": len(candidates),
                "backends": ",".join(backends),
            },
            artifacts=[a for a in artifacts if a],
            intents=intents,
            notes=notes,
        )


register_executor(KernelOptExecutor())


__all__ = ["KernelOptExecutor", "SUPPORTED_BACKENDS"]
