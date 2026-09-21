# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Executor for ``targeted_build`` task rows."""

from __future__ import annotations

import asyncio
import os
import subprocess
from concurrent.futures import CancelledError
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event

from ...enablement.recipe.build_inputs import build_driver_for, build_input_record
from ...enablement.runtime.build_actions import BuildResult, TargetedBuildAction
from ...enablement.runtime.stack_actions import FrameworkRuntime
from ...enablement.runtime.targeted_build import (
    _resolve_budget_sec,
    classify_build_exit,
    ensure_build_dead,
    spawn_build,
)
from ...loop.build_lifecycle import _driver_command
from ...loop.sub_agent_runner import ExecutionCleanupUnconfirmed, SubAgentResult
from ..cancel_channel import cancel_scope_listener
from ._subprocess_kill import STOP_GATE_POLL_SECONDS

if TYPE_CHECKING:
    from ...loop.sub_agent_runner import RunnerContext


class TargetedBuildExecutor:
    """Executor for ``targeted_build`` task rows."""

    async def __call__(self, ctx: "RunnerContext") -> dict[str, Any]:
        """Spawn and await a targeted build."""
        task = ctx.task
        action = TargetedBuildAction.from_state(task.params)
        session_dir = Path(ctx.extra["session_dir"])
        attempt_root = str(action.attempt_root)
        budget_sec = float(_resolve_budget_sec(action))
        shared_state = ctx.extra.get("shared_state")

        result: BuildResult | None = None
        with cancel_scope_listener() as scope:
            if scope is not None and scope.cancelled:
                raise CancelledError(scope.reason)
            handle = spawn_build(
                action,
                attempt_root=attempt_root,
                command=_driver_command(action, attempt_root),
            )

            # Every exit after spawn must reach teardown, including a failed sentinel write.
            try:
                if shared_state is not None:
                    shared_state.pending_targeted_build = handle.to_sentinel(task.task_id)
                    shared_state.save(session_dir)
                deadline = asyncio.get_running_loop().time() + budget_sec
                while True:
                    if scope is not None and scope.cancelled:
                        raise CancelledError(scope.reason)
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        rc = await asyncio.to_thread(handle.proc.wait, timeout=min(remaining, STOP_GATE_POLL_SECONDS))
                    except subprocess.TimeoutExpired:
                        continue
                    if scope is not None and scope.cancelled:
                        raise CancelledError(scope.reason)
                    result = classify_build_exit(handle, rc)
                    break
            except (asyncio.TimeoutError, CancelledError) as exc:
                cancelled = isinstance(exc, CancelledError)
                result = BuildResult(
                    ok=False,
                    attempt_root=handle.attempt_root,
                    runtime=FrameworkRuntime(),
                    build_log_path=handle.build_log_path,
                    failure_class="cancelled" if cancelled else "timeout",
                    failure_summary=str(exc) if cancelled else "targeted build exceeded wall-clock budget",
                    error="cancelled" if cancelled else "timeout",
                )
            finally:
                cleanup_result = (
                    SubAgentResult(
                        task_id=task.task_id,
                        state="cancelled"
                        if result.failure_class == "cancelled"
                        else "succeeded"
                        if result.ok
                        else "failed",
                        result=result.to_state(),
                        error=result.failure_summary or result.error or None,
                        error_class=result.failure_class,
                    )
                    if result is not None
                    else None
                )
                try:
                    confirmed_dead = ensure_build_dead(handle)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ExecutionCleanupUnconfirmed(
                        f"task={task.task_id}: targeted build cleanup failed: {exc}", result=cleanup_result
                    ) from exc
                if not confirmed_dead:
                    raise ExecutionCleanupUnconfirmed(
                        f"task={task.task_id}: targeted build cleanup unconfirmed", result=cleanup_result
                    )
                if shared_state is not None:
                    shared_state.pending_targeted_build = {}
                    shared_state.save(session_dir)

        self._record_result(
            result,
            shared_state,
            action=action,
            task_id=str(task.task_id or ""),
        )
        if result.failure_class == "cancelled":
            raise CancelledError(result.failure_summary)
        if not result.ok:
            raise RuntimeError(
                f"targeted_build failed: failure_class={result.failure_class!r}"
                f" summary={result.failure_summary or result.error!r}"
            )
        return result.to_state()

    @staticmethod
    def _record_result(
        result: Any,
        shared_state: Any,
        *,
        action: Any = None,
        task_id: str = "",
    ) -> None:
        """Append the build result to the manifest; record failure carrier.

        The inputs are recorded here because this is the one point where the
        action and the result are both in scope: the action's sentinel is
        cleared on finish, so a succeeded build's own recipe is otherwise
        unrecoverable from the row it leaves behind.
        """
        entry = result.to_state()
        if action is not None:
            entry["build_driver"] = build_driver_for(action)
            entry["build_inputs"] = build_input_record(
                action,
                installed_versions=getattr(result, "installed_versions", {}) or {},
                ambient_env=os.environ,
            )
        # Recorded on the timeline whether or not there is a SharedState to
        # append to: a build dispatched without one still ran, and the manifest
        # is only where the *lane* reads its own history from.
        enablement_event.record_build(task_id=task_id, entry=entry)
        if shared_state is None:
            return
        manifest = list(getattr(shared_state.enablement, "build_manifest", []) or [])
        manifest.append(entry)
        shared_state.enablement.build_manifest = manifest
        if not result.ok:
            shared_state.enablement.last_build_failure = {
                "failure_class": result.failure_class,
                "failure_summary": result.failure_summary or result.error,
            }
            from ._aiter_jit import sweep_stale_aiter_locks_if_dead

            sweep_stale_aiter_locks_if_dead(aiter_jit_dir=Path(str(result.attempt_root or "")) / "aiter_jit")
