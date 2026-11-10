# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's specialist launched through the agent runner in a real bubblewrap sandbox: it reads the session's
references and model, and applies a patch from its own dev lease's root to its worktree with git, and it reads
neither the app's login, nor another branch's lease, nor a file a link in a reference's source pointed at."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists.agent_runner import LocalAgentRunner
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.toolbox import SpecialistGrant
from hyperloom_multibranch.scheduling.reference import ReferenceSource, copy_references
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession, AgentSetup, BranchPlace
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.tests.branch_rig import branch_rig

pytestmark = pytest.mark.agent_isolation

BRANCH, OTHER, TOKEN_ENV = "s1", "s2", "HYPERLOOM_SPECIALIST_MCP_TOKEN"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)
#: A Forge result as its lease's root holds it: a new file in the worktree's package.
FORGE_PATCH = """diff --git a/forge_result.txt b/forge_result.txt
new file mode 100644
--- /dev/null
+++ b/forge_result.txt
@@ -0,0 +1 @@
+fused
"""
#: The stand-in claude CLI: it notes what it reaches, applies the Forge patch from its lease's root in its worktree,
#: then waits for its stdin to close.
_REACH = """#!/bin/sh
{{
  echo "reference=$(cat {reference} 2>/dev/null || echo unreadable)"
  echo "alias=$(cat {alias} 2>/dev/null || echo unreadable)"
  echo "linked_key=$(cat {linked_key} 2>/dev/null || echo unreadable)"
  echo "model=$(cat {model} 2>/dev/null || echo unreadable)"
  git apply {patch} 2>{out}.git && echo "applied=yes" || echo "applied=no"
  cat {login} >/dev/null 2>&1 && echo "login=read" || echo "login=unreachable"
  cat {other} >/dev/null 2>&1 && echo "other_lease=read" || echo "other_lease=unreachable"
  test -e {other_area} && echo "other_area=seen" || echo "other_area=absent"
}} > {out}.partial
mv {out}.partial {out}
cat > /dev/null
"""


class _Tokens:
    def issue(self, branch, scopes):
        return SpecialistGrant("http://127.0.0.1:9/mcp", TOKEN_ENV, f"token-of-{branch}")

    def revoke(self, branch):
        pass


class _Processes:
    def adopt(self, pid, *, branch, argv):
        pass

    def forget(self, pid):
        pass


async def _until(condition, what: str) -> None:
    for _ in range(600):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def test_a_sandboxed_specialist_reads_its_references_model_and_lease_and_neither_the_login_nor_another_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolation = BwrapIsolation(usable_bwrap())
    monkeypatch.setattr(launch_env, "choose_agent_isolation", lambda **_kwargs: isolation)
    # The app's home holds its claude login and, below it, the model; the directory above the home is shown too.
    above = tmp_path / "above"
    home = above / "home"
    login = home / ".claude" / ".credentials.json"
    login.parent.mkdir(parents=True)
    login.write_text('{"claudeAiOauth": "APP-LOGIN"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    model = home / "models" / "llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text("weights-config", encoding="utf-8")
    # The reference's source holds a link to its own tuning notes and one to the operator's key outside it; the
    # session copies it in as a run does.
    key = tmp_path / "operator" / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True)
    key.write_text("PRIVATE-KEY", encoding="utf-8")
    source = tmp_path / "GEAK"
    source.mkdir()
    (source / "tuning.md").write_text("tune the GEMMs", encoding="utf-8")
    (source / "alias.md").symlink_to("tuning.md")
    (source / "innocent.md").symlink_to(key)
    (tmp_path / "MB-reference").mkdir()
    copy_references((ReferenceSource("geak", str(source), "c" * 40),), tmp_path / "MB-reference")
    reference = tmp_path / "MB-reference" / "geak"
    area = tmp_path / "MB-toolbox" / "branch-sandboxes"
    lease_root = area / BRANCH / "lease-mine"
    lease_root.mkdir(parents=True)
    (lease_root / "forge.patch").write_text(FORGE_PATCH, encoding="utf-8")
    other = area / OTHER / "lease-theirs" / "forge.patch"
    other.parent.mkdir(parents=True)
    other.write_text("THEIR-PATCH", encoding="utf-8")

    built = await branch_rig(tmp_path, ROOT, BRANCH)
    worktree = built.base.worktree(BRANCH, tmp_path / BRANCH / "worktree")
    workspace = tmp_path / BRANCH / "session"
    out = workspace / "reach.txt"
    cli = tmp_path / "cli" / "bin" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        _REACH.format(
            reference=reference / "tuning.md",
            alias=reference / "alias.md",
            linked_key=reference / "innocent.md",
            model=model / "config.json",
            patch=lease_root / "forge.patch",
            login=login,
            other=other,
            other_area=area / OTHER,
            out=out,
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    runner = LocalAgentRunner(
        session_dir=tmp_path / "agents", poll_interval_seconds=0.05, version_probe=lambda a, e: "x"
    )
    setup = AgentSetup(
        command=AgentCommand("claude", str(cli)),
        role="You are a long-running optimization specialist.",
        env={"PATH": os.environ.get("PATH", ""), "CLAUDE_CODE_OAUTH_TOKEN": "token-for-the-test"},
        runner=runner,
        tokens=_Tokens(),
        processes=_Processes(),
        repo=built.repo,
        poll_seconds=0.01,
        min_session_seconds=0.0,
        readable=(reference, model, above),
        lease_area=lambda branch: area / branch,
    )
    place = BranchPlace(worktree, built.blackboard, workspace)
    session = AgentSession(SessionStart(BRANCH, built.root, 2, BranchTask("leads: fp8 KV cache")), setup, place)
    running = asyncio.create_task(session.run())
    try:
        await _until(out.exists, "the stand-in CLI to note what it reached")
    finally:
        await session.stop()
        await asyncio.wait_for(running, 30)

    reach = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    assert reach == {
        "reference": "tune the GEMMs",
        "alias": "tune the GEMMs",
        "linked_key": "unreadable",
        "model": "weights-config",
        "applied": "yes",
        "login": "unreachable",
        "other_lease": "unreachable",
        "other_area": "absent",
    }
    assert (worktree.path / "forge_result.txt").read_text(encoding="utf-8") == "fused\n"
    assert login.read_text(encoding="utf-8") == '{"claudeAiOauth": "APP-LOGIN"}'
