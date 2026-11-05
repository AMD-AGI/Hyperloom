# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist's worktree as tools on the session's MCP service: reset it to the branch head, check its change on
the CPU, and turn the change into a patch a test can carry.

The branch is the caller's token's, as for every specialist tool. The service reads the worktree where the specialist
left it and stores the patch in the session's blob store, so a test measures the bytes the worktree had when the patch
was taken, whatever the specialist edits afterwards.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from hyperloom.contracts.recipe import ComponentChange, RecipeDelta
from hyperloom.hosted.mcp_server import CallContext, InvalidParamsError, McpServer, ToolError
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    PackageLocator,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST
from hyperloom_multibranch.specialist.precheck import CheckFailed, check_change
from hyperloom_multibranch.specialist.worktree import BranchWorktree, installed_upstream

#: The worktree of a branch; the runtime adds it before the branch's specialist first runs.
WorktreeFor = Callable[[str], BranchWorktree]

_NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_PATCH: dict[str, Any] = {
    "type": "object",
    "properties": {"component": {"type": "string", "minLength": 1, "description": "The component the change is to."}},
    "required": ["component"],
    "additionalProperties": False,
}


async def sync_to_head(repo: ExperimentRepo, branch: str, worktree: BranchWorktree) -> str:
    """Reset ``worktree`` to ``branch``'s head and return that head."""
    head = await repo.head(branch)
    await sync_to_commit(repo, head, worktree)
    return head


async def sync_to_commit(repo: ExperimentRepo, commit: str, worktree: BranchWorktree) -> None:
    """Reset ``worktree`` to ``commit``: the installed packages with its recipe's patches applied."""
    recipe = await repo.read_recipe(commit)
    patches = [
        (await repo.patch_text(commit, patch.path)).encode()
        for component in recipe.components
        if worktree.holds(component.component)
        for patch in component.patches
    ]
    await asyncio.to_thread(worktree.sync, commit, patches)


def register_worktree_tools(
    server: McpServer,
    *,
    repo: ExperimentRepo,
    worktrees: WorktreeFor,
    locator: PackageLocator = locate_installed_package,
) -> None:
    own = {"scopes": frozenset({SCOPE_SPECIALIST})}
    server.register(
        "worktree_sync",
        description="Reset your worktree to your branch's head, dropping uncommitted edits; do it before each change.",
        schema=_NO_ARGUMENTS,
        handler=_sync(repo, worktrees),
        **own,
    )
    server.register(
        "worktree_check",
        description="Check your worktree's change to one component: its files compile and the package imports with it.",
        schema=_PATCH,
        handler=_check(worktrees, locator),
        **own,
    )
    server.register(
        "worktree_patch",
        description=(
            "Store your worktree's change to one component as a patch; returns the delta and parent to test. A component "
            "new on your head gets the upstream commit its installed package was built from."
        ),
        schema=_PATCH,
        handler=_patch(repo, worktrees, locator),
        **own,
    )


def _sync(repo: ExperimentRepo, worktrees: WorktreeFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        worktree = worktrees(ctx.principal.branch)
        head = await sync_to_head(repo, ctx.principal.branch, worktree)
        return {"head": head, "worktree": str(worktree.path)}

    return handle


def _check(worktrees: WorktreeFor, locator: PackageLocator) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        component = arguments["component"]
        worktree = worktrees(ctx.principal.branch)
        layout = COMPONENT_LAYOUTS.get(component)
        if layout is None or not worktree.holds(component):
            raise InvalidParamsError(f"your worktree holds no source of {component}")
        try:
            compiled = await asyncio.to_thread(check_change, worktree, component, layout, locator(layout))
        except (CheckFailed, UnsupportedComponentError) as error:
            raise ToolError(str(error)) from error
        return {"component": component, "compiled": compiled, "imported": layout.import_name}

    return handle


def _patch(
    repo: ExperimentRepo, worktrees: WorktreeFor, locator: PackageLocator
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        branch, component = ctx.principal.branch, arguments["component"]
        worktree = worktrees(branch)
        if not worktree.holds(component):
            raise InvalidParamsError(f"your worktree holds no source of {component}")
        head, synced = await repo.head(branch), await asyncio.to_thread(worktree.head)
        if synced != head:
            raise ToolError(f"your branch is at {head} and your worktree at {synced}; call worktree_sync and redo it")
        text = await asyncio.to_thread(worktree.diff, component)
        if not text:
            raise ToolError(f"your worktree has no change to {component}")
        ref = await repo.store_patch(component, text)
        change = ComponentChange(component, add_patches=(ref,))
        taken: dict[str, Any] = {"parent_commit": head, "patch": ref.to_document()}
        if all(present.component != component for present in (await repo.read_recipe(head)).components):
            upstream = worktree.native_upstream(component) or await asyncio.to_thread(
                installed_upstream, locator(COMPONENT_LAYOUTS[component])
            )
            if upstream is None:
                taken["note"] = (
                    f"{component} is new on your head and its installed package names no commit it was built from: "
                    "set upstream_repo and upstream_ref in the delta to the repository and full commit it came from"
                )
            else:
                change = replace(change, upstream_repo=upstream.repo, upstream_ref=upstream.commit)
        return {**taken, "delta": RecipeDelta(components=(change,)).to_document()}

    return handle


__all__ = ["WorktreeFor", "register_worktree_tools", "sync_to_commit", "sync_to_head"]
