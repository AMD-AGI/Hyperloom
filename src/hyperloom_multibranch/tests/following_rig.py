# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session whose integration head took in another branch's commit while ``s1`` committed one of its own, written
through the experiment repository and the blackboard as the gate and the integrator write them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.recipe import ConfigDelta, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import MergeAttempt
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import TTL, intent, judgement, request, result
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

CHUNKING = RecipeDelta(config=ConfigDelta(server_args="--enable-chunked-prefill"))
OWN = RecipeDelta(config=ConfigDelta(envs={"VLLM_ROCM_USE_AITER": "1"}))


@dataclass(frozen=True)
class Diverged:
    """``merged``, the other branch's commit the integration head ``onto`` took in; ``own``, the branch's commit."""

    merged: str
    onto: str
    own: str


def _as(root: Path, role: Role, branch: str = "") -> LocalBlackboard:
    return LocalBlackboard(root, Writer(role, branch or role.value, branch))


async def _measure(root: Path, branch: str, request_id: str, parent: str, change: RecipeDelta, target: str) -> None:
    of = intent(branch, parent, target, f"{target} change")
    claim = await _as(root, Role.SPECIALIST_RUNTIME, branch).acquire(of, change.fingerprint(), ttl_seconds=TTL)
    toolbox = _as(root, Role.TOOLBOX)
    await toolbox.write_request(request(request_id, of, change))
    await toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)
    await toolbox.write_result(result(request_id))


async def diverged(repo: GitExperimentRepo, blackboard_root: Path, root: str, base: Recipe, branch: str) -> Diverged:
    """``s0`` commits chunked prefill, which the integrator merges; ``branch`` commits an aiter env on the root."""

    gate, integrator = _as(blackboard_root, Role.GATE), _as(blackboard_root, Role.INTEGRATOR)
    await repo.create_branch("s0", root)
    merged = await repo.commit("s0", root, apply(base, CHUNKING), "chunked prefill")
    await _measure(blackboard_root, "s0", "r0", root, CHUNKING, "scheduler")
    await gate.write_judgement("r0", judgement(3.0, ("r0",)), Conclusion.COMMITTED, merged)
    own = await repo.commit(branch, root, apply(base, OWN), "aiter")
    await _measure(blackboard_root, branch, "r1", root, OWN, "attention")
    await gate.write_judgement("r1", judgement(2.0, ("r1",)), Conclusion.COMMITTED, own)
    onto = await repo.commit_integration(root, CHUNKING, (merged,))
    await _as(blackboard_root, Role.TOOLBOX).write_request(request("m0", intent("integration", root, "merge", merged)))
    await integrator.record_merge_attempt(MergeAttempt((merged,), root, request_id="m0"))
    await _as(blackboard_root, Role.TOOLBOX).write_result(result("m0"))
    await integrator.write_judgement("m0", judgement(3.1, ("m0",)), Conclusion.MERGED, onto)
    return Diverged(merged, onto, own)
