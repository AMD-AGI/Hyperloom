# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from hyperloom.measure.compile import PackageBase
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession
from hyperloom_multibranch.specialist.model_sessions import ModelSessions, model_sessions


def test_a_branch_s_sessions_and_its_worktree_tools_share_one_worktree_on_one_source_base(tmp_path):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)

    def built() -> ModelSessions:
        return model_sessions(
            source=tmp_path / "MB-source",
            branch_dir=lambda branch: tmp_path / "MB-branches" / branch,
            components=("vllm",),
            workload="qwen3-8b on vllm, TP 1, concurrency 64",
            command=AgentCommand("claude", "claude"),
            env={},
            runner=object(),
            tokens=object(),
            processes=object(),
            repo=object(),
            notebook_for=lambda branch: object(),
            locator=lambda _layout: installed,
        )

    sessions = built()

    assert isinstance(sessions.factory("s1", "c0", 2, "leads"), AgentSession)
    worktree = sessions.worktrees("s1")
    assert worktree.path == tmp_path / "MB-branches" / "s1" / "worktree"
    assert (worktree.path / "vllm" / "config.py").read_bytes() == b"max_num_seqs = 128\n"
    assert built().worktrees("s1").path == worktree.path
