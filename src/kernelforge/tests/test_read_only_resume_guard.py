# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A read-only resume protects the workspace, not the ledger the loop keeps writing under its output tree."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError
from kernelforge.workspace.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME, FORGE_PROTECTED_PATHS


@pytest.fixture
def resumed(tmp_path: Path) -> AgentRunSpec:
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Forge Test"),
        ("config", "user.email", "forge@example.invalid"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    (tmp_path / "kernel.py").write_text("VALUE = 1\n")
    (tmp_path / "driver.py").write_text("ORACLE = 1\n")
    (tmp_path / ".gitignore").write_text(f"{FORGE_LOOP_OUTPUT_DIRNAME}/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "baseline"], check=True, capture_output=True)
    output = tmp_path / FORGE_LOOP_OUTPUT_DIRNAME
    output.mkdir()
    (output / "llm_usage.json").write_text(json.dumps({"cost_usd": 10.0}))
    (output / "events.jsonl").write_text('{"iter": 1}\n')
    # The spec `OrchestrationAgent._run_spec(read_only_resume=True)` builds.
    return AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        writable=False,
        read_only_resume=True,
        allow_dirty_targets=True,
        allow_untracked=True,
        tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=False, max_turns=4),
        protected_path_rules=FORGE_PROTECTED_PATHS,
        protected_globs=["*"],
    )


def test_the_loop_ledger_written_during_a_read_only_resume_survives_it(resumed: AgentRunSpec):
    """Spend and events the loop records during a read-only resume are kept, neither restored nor deleted."""
    output = Path(resumed.cwd) / FORGE_LOOP_OUTPUT_DIRNAME
    guard = WorkspaceGuard(resumed, dirty_baseline_default=True)
    guard.prepare()

    (output / "llm_usage.json").write_text(json.dumps({"cost_usd": 13.5}))
    (output / "events.jsonl").write_text('{"iter": 1}\n{"type": "plan_critic"}\n')
    (output / "new_record.json").write_text("{}")

    assert guard.verify() == []
    assert json.loads((output / "llm_usage.json").read_text()) == {"cost_usd": 13.5}
    assert (output / "events.jsonl").read_text().count("\n") == 2
    assert (output / "new_record.json").is_file()


def test_a_read_only_resume_still_rejects_a_change_to_the_workspace(resumed: AgentRunSpec):
    """The output tree is the loop's; everything else is still the session's to leave alone."""
    kernel = Path(resumed.cwd) / "kernel.py"
    guard = WorkspaceGuard(resumed, dirty_baseline_default=True)
    guard.prepare()

    kernel.write_text("VALUE = 2\n")

    with pytest.raises(WorkspaceSafetyError, match="read-only session changed the workspace"):
        guard.verify()
    assert kernel.read_text() == "VALUE = 1\n"
