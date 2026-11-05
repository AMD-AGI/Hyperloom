# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A delegate's idempotency key, and an explore grid narrowed to the variants new to the queue and launchable."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

__all__ = ["admit_explore_grid", "delegate_idempotency_key", "explore_grid_key"]

_EXPLORE = "explore"


def delegate_idempotency_key(request: Any, shared_state: Any, nested_key: str | None) -> str:
    """Grid content for explore -> top-level -> nested compat alias -> content-fingerprint auto-key."""
    from ..actions.executors.explore import framework_lever_grid

    raw_key = ""
    # A grid the executor grows with its pending rewrite levers is not all the round measures, so it is not keyed.
    if request.action_name == _EXPLORE and not framework_lever_grid(shared_state):
        raw_key = explore_grid_key(request.params, macro_cycle=shared_state.macro_cycle)
    raw_key = raw_key or request.intent.payload.get("idempotency_key") or nested_key
    if not raw_key:
        content_fp = hashlib.sha1(
            json.dumps(request.params, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()[:10]
        raw_key = f"{request.source}:{request.action_name}:t{int(shared_state.tick or 0)}:{content_fp}"
    return str(raw_key)


def explore_grid_key(params: Mapping[str, Any], *, macro_cycle: int) -> str:
    """Idempotency key naming the configurations, stack and macro cycle an explore grid measures, or ``""``.

    ``""`` (no grid, or a kernel-overlay variant whose args and envs do not name its kernel) keeps the proposer's key.
    """
    grid = params.get("grid")
    if not isinstance(grid, list) or not grid:
        return ""
    from ..actions.executors.measured_variants import stack_of

    prints: list[str] = []
    for variant in grid:
        if not isinstance(variant, Mapping) or variant.get("overlay_pythonpath"):
            return ""
        extra_args = variant.get("extra_args")
        prints.append(
            canonical_fingerprint(
                " ".join(map(str, extra_args)) if isinstance(extra_args, (list, tuple)) else str(extra_args or ""),
                dict(variant.get("extra_envs") or {}),
                remove_args=variant.get("remove_args"),
                unset_envs=variant.get("unset_envs"),
                args_mode=str(variant.get("args_mode") or "append"),
            )
        )
    body = json.dumps(
        {"cycle": macro_cycle, "stack": stack_of(params), "variants": sorted(prints)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(body.encode(), usedforsecurity=False).hexdigest()[:12]
    return f"explore-grid-c{macro_cycle}-{digest}"


async def admit_explore_grid(router: Any, request: Any, idempotency_key: str, nested_key: str | None) -> bool:
    """Narrow a delegated explore grid to new, launchable variants; ``False`` once it is refused or left empty.

    A re-sent key already in flight keeps its grid, so the enqueue duplicate rule answers it.
    """
    from ..actions.executors.explore import framework_lever_grid

    if framework_lever_grid(router.shared_state):
        # The executor appends its pending rewrite levers, so the proposed grid is not all the round measures.
        return True
    keys = {str(k) for k in (idempotency_key, nested_key) if k}
    active = [*await router.tasks.queued(), *await router.tasks.running()]
    if not any(task.kind == _EXPLORE and task.idempotency_key in keys for task in active):
        if not await _keep_unqueued_variants(router, request.source, request.params, active):
            return False
        if not await _keep_launchable_variants(router, request.source, request.params):
            return False
    return not await _explore_grid_covered(router, request, idempotency_key)


async def _observe(router: Any, source: str, kind: str, hint: str, **fields: Any) -> None:
    await router.bus.record_observation(
        "coordinator", "observation", {"kind": kind, "from_agent": source, **fields, "hint": hint}
    )


async def _keep_unqueued_variants(router: Any, source: str, params: dict[str, Any], active: list[Any]) -> bool:
    """Narrow ``params['grid']`` to the variants the queue is not already measuring; ``False`` when none is left."""
    from ..actions.executors.measured_variants import unqueued_variants

    before = list(params.get("grid") or ())
    if not before:
        # No grid supplied: the executor seeds its own.
        return True
    kept = unqueued_variants(before, params, active)
    params["grid"] = kept
    if len(kept) == len(before):
        return True
    await _observe(
        router,
        source,
        "explore_variants_deduped",
        "the queue already holds these configurations; propose a direction it does not, "
        "or read the measured result first",
        proposed=len(before),
        kept=len(kept),
    )
    return bool(kept)


async def _keep_launchable_variants(router: Any, source: str, params: dict[str, Any]) -> bool:
    """Drop the explore variants Magpie would refuse to launch, recording why; ``False`` when none is left."""
    from ..actions.executors.explore import _grid_variants_from_payload
    from ..actions.executors.magpie_agentx import drop_unlaunchable_variants

    refused = drop_unlaunchable_variants(params, router.shared_state.baseline_workspace, _grid_variants_from_payload)
    if not refused:
        return True
    await _observe(
        router,
        source,
        "explore_variants_refused_by_launch_contract",
        "Magpie refuses to launch these: the args and envs they change are owned by the AgentX recipe; "
        "spend variants on other knobs",
        refused=refused,
    )
    return bool(params["grid"])


async def _explore_grid_covered(router: Any, request: Any, idempotency_key: str) -> bool:
    """Refuse an explore grid that adds nothing new, recording the denial; report whether it was refused.

    A task already queued or running under ``idempotency_key`` is not counted against the grid: re-sending the key
    is answered as a duplicate by the enqueue, not as coverage.
    """
    from ..actions.executors.measured_variants import explore_grid_already_covered

    retry_prefix = f"{idempotency_key}-retry"
    active = [
        task
        for task in (*await router.tasks.queued(), *await router.tasks.running())
        if task.idempotency_key != idempotency_key and not task.idempotency_key.startswith(retry_prefix)
    ]
    covered = explore_grid_already_covered(request.params, active, router.shared_state.explore_search)
    if covered is None:
        return False
    await router.record_policy_denied(request.source, request.intent, covered, action_name=_EXPLORE)
    return True
