# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The archived search tree and the session breakdown carry an autonomous branch's agent manifest as the blackboard
keeps it, without its credentials."""

from __future__ import annotations

from pathlib import Path

from hyperloom_multibranch.contracts.blackboard import Role
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import BranchDriver, LineageKind
from hyperloom_multibranch.blackboard.knowledge.breakdown import build_breakdown
from hyperloom_multibranch.blackboard.local.agent_manifest import sanitized_manifest
from hyperloom_multibranch.blackboard.rsi.archive import tree_document, tree_from_document, tree_from_snapshot
from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.blackboard.tests.sessions import METRIC, Writers, scripted_session

MANIFEST: dict[str, object] = {
    "source": "https://bot:ghp_secret@github.com/org/agent.git",
    "launch": "python run.py --api-key sk-3",
}


async def test_the_tree_and_the_breakdown_carry_an_autonomous_branchs_agent(tmp_path: Path) -> None:
    root, clock = tmp_path / "blackboard", FakeClock()
    observer = await scripted_session(root, clock)
    w = Writers(root, clock)
    opened = LineageEvent(LineageKind.OPEN, "a1", from_commit="c0", reason="agent", driver=BranchDriver.AUTONOMOUS)
    await w.as_role(Role.SCHEDULER).record_lineage(opened)
    await w.as_role(Role.SPECIALIST_RUNTIME, "a1").record_agent_manifest("a1", MANIFEST)

    snapshot = await observer.snapshot()
    tree = tree_from_snapshot(snapshot, METRIC)

    (agent,) = (branch for branch in tree.branches if branch.name == "a1")
    assert agent.agent == sanitized_manifest(MANIFEST)
    assert all(branch.agent is None for branch in tree.branches if branch.name != "a1")
    assert tree_from_document(tree_document(tree)) == tree
    assert (
        build_breakdown(snapshot)["agent_manifests"]["a1"]["manifest"]["source"] == "https://github.com/org/agent.git"
    )
