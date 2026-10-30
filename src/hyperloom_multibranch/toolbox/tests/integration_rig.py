# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wiring of the toolbox integration test: the real toolbox, blackboard, repository, rule gate and MCP service.

Nothing here is faked. :class:`Variant` names the only two things that differ between the CPU and the GPU variant: the
planner (the real ``HostedPlanner``, in the CPU variant wrapped so that the fake runner is told what to report) and the
runner the sandboxes start. :func:`build_stack` returns everything a scripted specialist and the judge need.
"""

from __future__ import annotations

import json
import asyncio
import secrets
import sys
import urllib.request
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.contracts.recipe import Recipe
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PackageLocator, locate_installed_package
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.points import CanonicalMeasurement
from hyperloom_multibranch.contracts.fork import ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.mcp_tools import (
    McpSpecialistTokens,
    SpecialistToolOptions,
    register_specialist_tools,
)
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

Planner = Callable[[HostedPlanner], Any]


class NoScheduler:
    """The rule gate's scheduler: this round asks for no fork and no repair, so any request is a defect."""

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        raise AssertionError(f"unexpected fork request {request}")

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        raise AssertionError(f"unexpected repair request {request}")


@dataclass(frozen=True)
class Variant:
    """What differs between the CPU and the GPU variant.

    ``wrap_planner`` receives the real ``HostedPlanner`` and returns what the toolbox plans with; ``python`` is what
    every runner sandbox starts (a wrapper that starts the fake runner, or the real interpreter); ``locator`` finds the
    framework package the patches are applied to (``None``: the one installed in this image).
    """

    cards: tuple[int, ...]
    spec: SpecContext
    canonical: CanonicalMeasurement
    image_digest: str
    wrap_planner: Planner = lambda planner: planner
    python: str | None = None
    locator: PackageLocator = locate_installed_package
    runner_envs: Mapping[str, str] = field(default_factory=dict)
    timeout_sec: int = 600
    poll_interval_sec: float = 0.2


@dataclass
class Stack:
    """The running system: the specialists' side (``server``, ``tokens``) and the program's side."""

    root: Path
    toolbox: LocalHostedToolbox
    repo: GitExperimentRepo
    server: McpServer
    tokens: McpSpecialistTokens
    gate: RuleGate
    gate_blackboard: LocalBlackboard
    blobs: LocalBlobStore
    baseline_commit: str

    def specialist_blackboard(self, branch: str) -> LocalBlackboard:
        return LocalBlackboard(self.root / "blackboard", Writer(Role.SPECIALIST_RUNTIME, branch, branch))


class Specialist:
    """One scripted specialist: calls the campaign's MCP service over HTTP with the token of its own branch."""

    def __init__(self, stack: Stack, branch: str) -> None:
        self.branch = branch
        self._url = stack.server.url
        self._token = stack.tokens.issue(branch, frozenset({SCOPE_SPECIALIST})).token

    def _post(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
        request = urllib.request.Request(
            self._url,
            json.dumps(message).encode(),
            {"Content-Type": "application/json", "X-Hyperloom-Token": self._token},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())

    async def raw(self, tool: str, **arguments: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._post, tool, arguments)

    async def call(self, tool: str, **arguments: Any) -> str:
        """The text a tool answered; a refusal or a tool error fails the test with what the service said."""
        reply = await self.raw(tool, **arguments)
        assert "result" in reply and reply["result"]["isError"] is False, reply
        return str(reply["result"]["content"][0]["text"])

    async def refused(self, tool: str, **arguments: Any) -> str:
        reply = await self.raw(tool, **arguments)
        if "error" in reply:
            return str(reply["error"]["message"])
        assert reply["result"]["isError"] is True, reply
        return str(reply["result"]["content"][0]["text"])


@asynccontextmanager
async def build_stack(root: Path, variant: Variant, baseline: Recipe, *, k: int = 1) -> AsyncIterator[Stack]:
    """Build the system under ``root`` and close it, whatever the test did."""
    session = root / "session"
    repo = GitExperimentRepo(root / "experiments.git", variant.spec.blobs)
    await repo.init_root(baseline)
    blackboard = LocalBlackboard(root / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    compiler = EnvironmentCompiler(variant.spec.blobs, variant.image_digest, variant.locator)
    planner = variant.wrap_planner(HostedPlanner(compiler, variant.canonical, root / "points"))
    toolbox = build_local_toolbox(
        LocalSession(session, "w15", variant.cards),
        blackboard,
        AcceptanceSetup(repo, planner, variant.image_digest),
        variant.spec,
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=k, global_concurrency=len(variant.cards), branch_k={INTEGRATION_BRANCH: 2}),
            runner_envs=variant.runner_envs,
            python=variant.python or sys.executable,
            timeout_sec=variant.timeout_sec,
            poll_interval_sec=variant.poll_interval_sec,
        ),
    )
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w15")
    register_specialist_tools(
        server,
        toolbox=toolbox,
        blackboards=lambda branch: LocalBlackboard(
            root / "blackboard", Writer(Role.SPECIALIST_RUNTIME, branch, branch)
        ),
        options=SpecialistToolOptions(analysis=toolbox),
    )
    gate_blackboard = LocalBlackboard(root / "blackboard", Writer(Role.GATE, "gate"))
    stack = Stack(
        root,
        toolbox,
        repo,
        server,
        McpSpecialistTokens(server),
        RuleGate(gate_blackboard, repo, NoScheduler(), toolbox),
        gate_blackboard,
        variant.spec.blobs,  # type: ignore[arg-type]
        await repo.head(INTEGRATION_BRANCH),
    )
    server.start()
    try:
        await toolbox.start()
        yield stack
    finally:
        server.stop()
        await toolbox.close()
