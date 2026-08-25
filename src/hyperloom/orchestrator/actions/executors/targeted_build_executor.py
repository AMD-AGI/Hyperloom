# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Executor for ``targeted_build`` task rows.

Runs compiled-component builds through the standard sub_agent_runner path so
the in-flight asyncio.Task is registered in ``_inflight_actions`` (reachable
by ``cancel_inflight_actions`` at shutdown). This coroutine owns the build for
its whole life: ``asyncio.wait_for`` is the only wall-clock budget, and one
teardown covers every way out of the wait.

attempt_root is always ``session_dir / "enablement" / "builds" / task_id``.
This derivation must stay identical to the fallback in
``framework.py:_maybe_route_build_outcomes`` or succeeded builds are judged
``artifact_unreadable`` and reverted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...framework.build_actions import BuildResult, TargetedBuildAction
from ...framework.stack_actions import FrameworkRuntime
from ...framework.targeted_build import (
    _resolve_budget_sec,
    classify_build_exit,
    ensure_build_dead,
    spawn_build,
)
from ...loop.build_lifecycle import _driver_command

if TYPE_CHECKING:
    from ...loop.sub_agent_runner import RunnerContext


class TargetedBuildExecutor:
    """Executor for ``targeted_build`` task rows."""

    @staticmethod
    def _attempt_root(session_dir: Path, task_id: str) -> str:
        return str(session_dir / "enablement" / "builds" / task_id)

    async def __call__(self, ctx: "RunnerContext") -> dict[str, Any]:
        """Spawn and await a targeted build.

        Raises RuntimeError on a failed or timed-out build so
        sub_agent_runner writes the ``failed`` terminal state.
        """
        task = ctx.task
        action = TargetedBuildAction.from_state(task.params)
        session_dir = Path(ctx.extra["session_dir"])
        attempt_root = self._attempt_root(session_dir, task.task_id)
        budget_sec = float(_resolve_budget_sec(action))
        shared_state = ctx.extra.get("shared_state")

        handle = spawn_build(
            action,
            attempt_root=attempt_root,
            command=_driver_command(action, attempt_root),
        )

        # The build outlives this coroutine unless killed, and the lane is
        # released as it unwinds, so every exit from here must reach the
        # teardown -- cancel and a failed sentinel write included.
        try:
            if shared_state is not None:
                shared_state.pending_targeted_build = handle.to_sentinel(task.task_id)
                shared_state.save(session_dir)
            rc = await asyncio.wait_for(
                asyncio.to_thread(handle.proc.wait),
                timeout=budget_sec,
            )
            result = classify_build_exit(handle, rc)
        except asyncio.TimeoutError:
            result = BuildResult(
                ok=False,
                attempt_root=handle.attempt_root,
                runtime=FrameworkRuntime(),
                build_log_path=handle.build_log_path,
                failure_class="timeout",
                failure_summary="targeted build exceeded wall-clock budget",
                error="timeout",
            )
        finally:
            confirmed_dead = ensure_build_dead(handle)
            if shared_state is not None and confirmed_dead:
                shared_state.pending_targeted_build = {}
                shared_state.save(session_dir)

        self._record_result(result, shared_state)
        if not result.ok:
            raise RuntimeError(
                f"targeted_build failed: failure_class={result.failure_class!r}"
                f" summary={result.failure_summary or result.error!r}"
            )
        return result.to_state()

    @staticmethod
    def _record_result(result: Any, shared_state: Any) -> None:
        """Append the build result to the manifest; record failure carrier."""
        if shared_state is None:
            return
        manifest = list(getattr(shared_state.enablement, "build_manifest", []) or [])
        manifest.append(result.to_state())
        shared_state.enablement.build_manifest = manifest
        if not result.ok:
            shared_state.enablement.last_build_failure = {
                "failure_class": result.failure_class,
                "failure_summary": result.failure_summary or result.error,
            }
            from ._aiter_jit import sweep_stale_aiter_locks_if_dead

            sweep_stale_aiter_locks_if_dead(aiter_jit_dir=Path(str(result.attempt_root or "")) / "aiter_jit")
