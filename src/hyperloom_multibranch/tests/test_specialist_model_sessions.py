# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from hyperloom.measure.compile import PackageBase
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec
from hyperloom_multibranch.specialist.agent_sessions import AgentSessions, with_agents
from hyperloom_multibranch.specialist.autonomous_session import AutonomousSession
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession, AgentSetup
from hyperloom_multibranch.specialist.model_sessions import ModelSessions, model_sessions
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.specialist.worktree import SourceBase


def test_a_branch_s_sessions_and_its_worktree_tools_share_one_worktree_on_one_source_base(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    role = role_prompt("qwen3-8b on vllm, TP 1, concurrency 64")
    command = AgentCommand("claude", "claude", model="claude-opus-5-5")
    setup = AgentSetup(command, role, {}, object(), object(), object(), object())

    def built() -> ModelSessions:
        return model_sessions(
            setup,
            base=SourceBase(tmp_path / "MB-source", locator=lambda _layout: installed),
            components=("vllm",),
            branch_dir=lambda branch: tmp_path / "MB-branches" / branch,
            notebook_for=lambda branch: object(),
        )

    sessions = built()

    assert isinstance(sessions.factory(SessionStart("s1", "c0", 2, BranchTask("leads"))), AgentSession)
    worktree = sessions.worktrees("s1")
    assert worktree.path == tmp_path / "MB-branches" / "s1" / "worktree"
    assert (worktree.path / "vllm" / "config.py").read_bytes() == b"max_num_seqs = 128\n"
    assert built().worktrees("s1").path == worktree.path


def _agent_sessions() -> AgentSessions:
    spec = AgentSpec("/models/m", "vllm", 1, "bf16", "mi355x", {}, "output_throughput", "", {}, "", (), 3600.0, 31100)
    command = AgentCommand("claude", "claude", model="claude-opus-5-5")
    return AgentSessions(spec, command, {}, object(), object(), object())


def test_a_branch_whose_task_names_an_agent_runs_it_on_ports_of_its_own_in_the_sandbox_it_leases(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    setup = AgentSetup(AgentCommand("claude", "claude"), "role", {}, object(), object(), object(), object())
    base = SourceBase(tmp_path / "MB-source", locator=lambda _layout: installed)

    def branch_dir(branch: str) -> Path:
        return tmp_path / "MB-branches" / branch

    def notebook_for(branch: str) -> object:
        return object()

    sessions = model_sessions(setup, base=base, components=("vllm",), branch_dir=branch_dir, notebook_for=notebook_for)
    built = with_agents(sessions, _agent_sessions(), branch_dir=branch_dir, notebook_for=notebook_for)
    first, second, third = (
        built.factory(SessionStart(branch, "c0", 1, BranchTask(agent="https://example.com/agent.git")))
        for branch in ("a1", "a2", "a3")
    )
    again = built.factory(SessionStart("a1", "c0", 1, BranchTask(agent="https://example.com/agent.git")))

    assert all(isinstance(session, AutonomousSession) for session in (first, second, third))
    specs = [session._setup.spec for session in (first, second, third, again)]
    assert [spec.port for spec in specs] == [31100, 31110, 31120, 31100]
    lease = BranchLease("lease-1", "a1", (6,), tmp_path / "lease-1", {"SGLANG_USE_AITER": "1"}, 1)
    sandbox = first._setup.sandbox(lease)
    assert (sandbox.cards, sandbox.root) == ((6,), lease.root)
    assert sandbox.env({})["SGLANG_USE_AITER"] == "1"
    assert isinstance(built.factory(SessionStart("s1", "c0", 1, BranchTask("leads"))), AgentSession)
    with pytest.raises(ValueError, match="no specialist stands for"):
        sessions.factory(SessionStart("a1", "c0", 1, BranchTask(agent="https://example.com/agent.git")))
