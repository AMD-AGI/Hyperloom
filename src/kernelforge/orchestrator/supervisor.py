# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only Supervisor for the forge-loop (AVO self-supervision).

When the Implementer stalls, the loop calls this Supervisor to review the whole
evolution trajectory, correct subjective conclusions in historical session
records, and advise the next planning cycle.

The Supervisor only READS (never edits). Its free-form ruling is persisted
verbatim, consumed by Orchestration, and rendered into the next Implementer
prompt. Best-effort: any failure returns "" and the loop continues.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

from kernelforge.agent_backends import AgentRunSpec, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.config import Config
from kernelforge.durable_io import atomic_write_text

log = logging.getLogger(__name__)


_SUPERVISOR_ROLE = (
    "You are a research supervisor for an autonomous GPU-kernel optimization "
    "search. An implementer agent iterates on a single kernel; when the search stalls "
    "you review profiling, per-case headroom, and the whole trajectory to decide "
    "what the evidence supports now. Historical lesson documents are session "
    "records, not instructions: you may explicitly reject their subjective "
    "conclusions while preserving their measurements and observed failures. You "
    "do not write code — you only analyze and issue the current planning ruling."
)

# Capability + context maxed for the heterogeneous supervisor: its periodic
# trajectory review is worth a deep, well-grounded pass, so give it a large
# reasoning budget and generous exploration turns. These are ceilings — a review
# that needs less finishes early, regardless of whether the primary or fallback
# provider model serves the request.
SUPERVISOR_THINKING_BUDGET = 64000  # deep reasoning budget for the trajectory review
SUPERVISOR_MAX_TURNS = 40  # room to Read many prior kernels/profiles/diffs
SUPERVISOR_DIRECTIONS = 3  # how many new directions to propose


class SupervisorBackendFailure(RuntimeError):
    """Report a Supervisor backend outage that produced no ruling."""


def latest_supervisor_ruling_path(workspace: str) -> Path:
    """Canonical path containing the latest non-empty Supervisor ruling."""
    return Path(workspace) / "forge_experiments" / "supervisor" / "latest.md"


def load_latest_supervisor_ruling(workspace: str) -> str:
    """Load the latest free-form ruling, returning empty text when unavailable."""
    try:
        return latest_supervisor_ruling_path(workspace).read_text(errors="replace")
    except OSError:
        return ""


def clear_latest_supervisor_ruling(workspace: str) -> bool:
    """Expire the active ruling while retaining immutable interaction history."""
    try:
        latest_supervisor_ruling_path(workspace).unlink(missing_ok=True)
    except OSError as error:
        log.debug("supervisor: failed to clear latest ruling: %s", error)
        return False
    return True


def _persist_interaction(
    workspace: str, iteration: int, reason: str, system: str, user: str, reply: str, *, backend: str, model: str
) -> None:
    """Save one supervisor intervention (prompt + reply) for later inspection.

    Written to ``<workspace>/forge_experiments/supervisor/intervention_iter_NNN.md``.
    Best-effort: a persistence failure must never break the loop. Every attempt
    is archived, including an empty reply. A non-empty reply also atomically
    replaces ``latest.md`` so Orchestration and resumed runs can consume the
    complete current ruling without parsing an event or truncated state field.
    """
    try:
        d = Path(workspace) / "forge_experiments" / "supervisor"
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        reply_txt = (
            reply
            if reply and reply.strip()
            else "(empty — the supervisor returned no directions, e.g. a backend failure)"
        )
        body = (
            f"# Supervisor intervention — iteration {iteration}\n\n"
            f"- timestamp: {ts}\n"
            f"- backend: {backend}\n"
            f"- model: {model}\n"
            f"- trigger: {reason}\n\n"
            f"## System prompt\n\n{system}\n\n"
            f"## User prompt\n\n{user}\n\n"
            f"## Reply\n\n{reply_txt}\n"
        )
        path = d / f"intervention_iter_{iteration:03d}.md"
        atomic_write_text(path, body)
        if reply and reply.strip():
            atomic_write_text(
                latest_supervisor_ruling_path(workspace),
                reply,
            )
        else:
            clear_latest_supervisor_ruling(workspace)
        print(f"  [supervisor] saved interaction -> forge_experiments/supervisor/{path.name}", flush=True)
    except Exception as e:
        log.debug("supervisor: failed to persist interaction for iter %s: %s", iteration, e)


def persist_supervisor_ruling(
    workspace: str,
    iteration: int,
    reason: str,
    reply: str,
) -> tuple[Path | None, Path | None]:
    """Ensure any injected Supervisor callback has durable audit artifacts.

    The registered Supervisor persists its complete prompt and reply before
    returning. A caller may inject another callback directly into
    :class:`IterationLoop`; when no full interaction artifact exists, write a
    minimal audit record containing the trigger and exact reply. In both cases,
    atomically store the reply text unchanged in ``latest.md``.
    """
    if not reply or not reply.strip():
        return None, None
    interaction = Path(workspace) / "forge_experiments" / "supervisor" / f"intervention_iter_{iteration:03d}.md"
    latest = latest_supervisor_ruling_path(workspace)
    persisted_interaction: Path | None = None
    persisted_latest: Path | None = None
    try:
        if not interaction.is_file():
            body = (
                f"# Supervisor intervention — iteration {iteration}\n\n"
                f"- timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "- source: injected callback\n"
                f"- trigger: {reason}\n\n"
                f"## Reply\n\n{reply}\n"
            )
            atomic_write_text(interaction, body)
        persisted_interaction = interaction
        atomic_write_text(latest, reply)
        persisted_latest = latest
    except Exception as error:  # noqa: BLE001 - persistence is best-effort
        log.debug(
            "supervisor: failed to persist injected ruling for iter %s: %s",
            iteration,
            error,
        )
    return persisted_interaction, persisted_latest


def _build_task_prompt(
    program_md: str, digest: str, reason: str, gpu_target: str, directions: int, evidence_context: str = ""
) -> str:
    """Build the bounded evidence prompt shown to the supervisor."""
    source_note = (
        "You MAY read the exact analysis, profile, orchestration, lesson, and "
        "candidate artifact paths supplied below. Read only what you need: "
        "prefer the current analysis bundle, latest optimization plan, and last "
        "1-2 attempts, AT MOST ~8 files, then STOP reading and answer. Do NOT "
        "list or search unrelated paths, and do NOT edit anything."
    )
    return f"""\
## What the loop knows (factual signal only)
On {gpu_target}, the loop reports: {reason}. That is ONLY a budget signal — it
does NOT judge WHY the search stalled. YOU make that semantic call from the
trajectory below.

{source_note}

## Program / target
{program_md}

## Evolution trajectory so far
{digest if digest else "(no archived trajectory yet)"}

## Current profiling, orchestration, and exploration evidence
{evidence_context if evidence_context else "(no additional structured evidence)"}

## Your task
1. Use the current commit-bound profiling and potential evidence to assess
   remaining headroom for EVERY scored case. Do not infer "no headroom" merely
   from repeated REVERTs.
2. Compare the latest optimization plan with the complete explored history,
   specialist analyses, and prior orchestration plans.
3. Treat historical lesson documents as session-authored records, not
   authoritative conclusions. When a lesson makes an unsupported claim such as
   "hard floor", "local optimum", or "this direction is exhausted", explicitly
   state whether Orchestration should disregard that conclusion. Preserve the
   measurements and concrete errors recorded beside it.
4. Decide whether the current planning cycle should continue the same mechanism,
   switch mechanisms, or reanalyze stale evidence, and explain why.
5. Recommend at most {directions} concrete directions. Each should name the source
   region, exact mechanism, target cases, and why the profiling evidence supports
   it. A prior failed implementation is evidence about that implementation, not
   proof that the whole direction is exhausted.

Write the current Supervisor Ruling in any clear prose or Markdown form. There
is no required output schema. This ruling will be persisted verbatim and has
priority over subjective recommendations or conclusions in historical lessons;
objective validation and measurement records remain authoritative.
"""


def make_supervisor_fn(
    program_md: str = "",
    gpu_target: str = "gfx942",
    backend: str = "",
    directions: int = SUPERVISOR_DIRECTIONS,
    usage=None,
    config: Config | None = None,
) -> Callable[..., Awaitable[str]]:
    """Build a read-only Supervisor through the selected provider registry."""
    from kernelforge.agent_backends import AgentToolPolicy
    from kernelforge.agent_backends.registry import (
        create_registered_backend,
        resolve_agent_runtime,
    )

    config = config or Config.from_env()
    runtime = config.agent_runtime()
    if backend and backend.strip().lower() != runtime.provider:
        runtime = resolve_agent_runtime(
            backend,
            executable="",
            timeout_sec=config.agent_timeout_sec,
            reasoning_effort=config.agent_reasoning_effort,
            sandbox_mode=config.agent_sandbox_mode,
            precheck=config.agent_precheck,
            fallback_provider=config.agent_fallback_provider,
            options={},
        )
    supervisor_model = str(runtime.options.get("supervisor_model") or runtime.model)
    if supervisor_model != runtime.model:
        from dataclasses import replace

        runtime = replace(runtime, model=supervisor_model)
    timeout_sec = int(runtime.options.get("supervisor_timeout_sec") or runtime.timeout_sec)
    selected_backend = None

    async def supervisor_fn(
        digest: str,
        reason: str,
        workspace: str,
        iteration: int = 0,
        evidence_context: str = "",
    ) -> str:
        """Run one bounded Supervisor pass and persist its interaction."""
        nonlocal selected_backend
        task = _build_task_prompt(
            program_md,
            digest,
            reason,
            gpu_target,
            directions,
            evidence_context=evidence_context,
        )
        reply = ""
        try:
            if selected_backend is None:
                selected_backend = create_registered_backend(runtime)
                setattr(supervisor_fn, "backend_name", selected_backend.name)
                setattr(
                    supervisor_fn,
                    "backend_model",
                    selected_backend.runtime.model,
                )
            deadline = time.monotonic() + timeout_sec

            async def run_once(user_prompt: str) -> str:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                # One budget drives both, so the backend's own deadline always fires
                # before the outer watchdog. Rounded up: a sub-second sliver of
                # elapsed time must not shorten the configured session budget.
                session_budget = max(1, math.ceil(min(float(timeout_sec), remaining)))
                result = await asyncio.wait_for(
                    selected_backend.run(
                        AgentRunSpec(
                            system_prompt=_SUPERVISOR_ROLE,
                            user_prompt=user_prompt,
                            cwd=workspace,
                            writable=False,
                            timeout_sec=session_budget,
                            reasoning_effort="max",
                            tool_policy=AgentToolPolicy(
                                read=True,
                                search=True,
                                write=False,
                                shell=False,
                                max_turns=SUPERVISOR_MAX_TURNS,
                                thinking_budget_tokens=(SUPERVISOR_THINKING_BUDGET),
                            ),
                            protected_globs=["*"],
                        ),
                        usage=usage,
                    ),
                    timeout=watchdog_timeout_sec(session_budget),
                )
                if is_api_failure(result):
                    detail = (
                        result.stderr_tail
                        or result.end_reason
                        or "supervisor backend failed before producing an answer"
                    )
                    raise SupervisorBackendFailure(detail)
                return result.text or ""

            reply = await run_once(task)
        except Exception as exc:  # noqa: BLE001 - supervisor is best-effort
            backend_name = selected_backend.name if selected_backend is not None else runtime.provider
            reply = ""
            print(
                f"  [supervisor] {backend_name} call failed ({exc}) — skipping",
                file=sys.stderr,
                flush=True,
            )
        backend_name = selected_backend.name if selected_backend is not None else runtime.provider
        backend_model = selected_backend.runtime.model if selected_backend is not None else runtime.model
        _persist_interaction(
            workspace,
            iteration,
            reason,
            _SUPERVISOR_ROLE,
            task,
            reply,
            backend=backend_name,
            model=backend_model,
        )
        return reply

    setattr(supervisor_fn, "backend_name", runtime.provider)
    setattr(supervisor_fn, "backend_model", runtime.model)
    return supervisor_fn
